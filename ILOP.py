from __future__ import annotations

from collections.abc import Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.segresnet_block import ResBlock, get_conv_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer


class ILOPEncoder(nn.Module):
    """
    Image-Lesion Encoder for outcome prediction.
    Based on SegResNet architecture but modified for dual-channel input (image + mask).
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 2,  # Changed default to 2 for image + mask
        dropout_prob: float | None = None,
        act: tuple | str = ("RELU", {"inplace": True}),
        norm: tuple | str = ("GROUP", {"num_groups": 8}),
        blocks_down: tuple = (1, 2, 2, 4),
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.dropout_prob = dropout_prob
        self.act = act
        self.act_mod = get_act_layer(act)
        self.norm = norm

        self.convInit = get_conv_layer(spatial_dims, in_channels, init_filters)
        self.down_layers = self._make_down_layers()

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_down_layers(self):
        down_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm = (
            self.blocks_down,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )

        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2**i
            pre_conv = (
                get_conv_layer(
                    spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2
                )
                if i > 0
                else nn.Identity()
            )
            down_layer = nn.Sequential(
                pre_conv,
                *[
                    ResBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act)
                    for _ in range(item)
                ],
            )
            down_layers.append(down_layer)
        return down_layers

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)

        down_x = []
        for down in self.down_layers:
            x = down(x)
            down_x.append(x)

        return x, down_x


class ResLinearBlock(nn.Module):
    """ResNet-style block for fully connected layers"""

    def __init__(self, in_features: int, dropout_prob: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.block(x)
        out += identity
        out = self.relu(out)
        return out


class ILOPDecoder(nn.Module):
    """Outcome prediction decoder using ResNet-style fully connected layers"""

    def __init__(
        self,
        input_features: int,
        clinical_features: int,
        hidden_dim: int = 512,
        num_res_blocks: int = 3,
        dropout_prob: float = 0.1,
    ):
        super().__init__()

        self.total_features = input_features + clinical_features

        self.input_layer = nn.Sequential(
            nn.Linear(self.total_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
        )

        self.res_blocks = nn.ModuleList(
            [ResLinearBlock(hidden_dim, dropout_prob) for _ in range(num_res_blocks)]
        )

        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, clinical_data: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, clinical_data], dim=1)
        x = self.input_layer(x)

        for res_block in self.res_blocks:
            x = res_block(x)

        x = self.final_layer(x)
        return x


class ILOPVAE(nn.Module):
    """
    Complete VAE model for Image-Lesion Outcome Prediction combining encoder, VAE components,
    reconstruction decoder, and outcome prediction.
    """

    def __init__(
        self,
        input_image_size: Sequence[int],
        clinical_features: int,
        vae_estimate_std: bool = False,
        vae_default_std: float = 0.3,
        vae_nz: int = 256,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 2,
        dropout_prob: float = 0.1,
        act: tuple | str = ("RELU", {"inplace": True}),
        norm: tuple | str = ("GROUP", {"num_groups": 8}),
        blocks_down: tuple = (1, 2, 2, 4),
    ):
        super().__init__()

        self.input_image_size = input_image_size
        self.clinical_features = clinical_features
        self.smallest_filters = 16
        self.vae_estimate_std = vae_estimate_std
        self.vae_default_std = vae_default_std
        self.vae_nz = vae_nz

        # Initialize encoder
        self.encoder = ILOPEncoder(
            spatial_dims=spatial_dims,
            init_filters=init_filters,
            in_channels=in_channels,
            dropout_prob=dropout_prob,
            act=act,
            norm=norm,
            blocks_down=blocks_down,
        )

        # Calculate VAE input size
        zoom = 2 ** (len(blocks_down) - 1)
        self.fc_insize = [s // (2 * zoom) for s in input_image_size]

        # Initialize VAE components
        self._prepare_vae_modules()

        # Initialize reconstruction decoders
        self.recon_conv_image = self._make_final_conv(1)
        self.recon_conv_mask = self._make_final_conv(1)

        # Initialize outcome predictor
        self.outcome_decoder = ILOPDecoder(
            input_features=vae_nz, clinical_features=clinical_features
        )

    def _prepare_vae_modules(self):
        """Initialize VAE-specific modules"""
        total_elements = int(self.smallest_filters * np.prod(self.fc_insize))
        v_filters = self.encoder.init_filters * 2 ** (len(self.encoder.blocks_down) - 1)

        self.vae_down = nn.Sequential(
            get_norm_layer(
                name=self.encoder.norm,
                spatial_dims=self.encoder.spatial_dims,
                channels=v_filters,
            ),
            self.encoder.act_mod,
            get_conv_layer(
                self.encoder.spatial_dims,
                v_filters,
                self.smallest_filters,
                stride=2,
                bias=True,
            ),
            get_norm_layer(
                name=self.encoder.norm,
                spatial_dims=self.encoder.spatial_dims,
                channels=self.smallest_filters,
            ),
            self.encoder.act_mod,
        )

        self.vae_fc1 = nn.Linear(total_elements, self.vae_nz)  # mean
        if self.vae_estimate_std:
            self.vae_fc2 = nn.Linear(total_elements, self.vae_nz)  # std
        self.vae_fc3 = nn.Linear(self.vae_nz, total_elements)  # decoder input

    def _make_final_conv(self, out_channels: int):
        """Create final convolution layer for reconstruction"""
        v_filters = self.encoder.init_filters * 2 ** (len(self.encoder.blocks_down) - 1)

        return nn.Sequential(
            get_norm_layer(
                name=self.encoder.norm,
                spatial_dims=self.encoder.spatial_dims,
                channels=v_filters,
            ),
            self.encoder.act_mod,
            get_conv_layer(
                self.encoder.spatial_dims,
                v_filters,
                out_channels,
                kernel_size=1,
                bias=True,
            ),
        )

    def encode_vae(self, x: torch.Tensor):
        """Encode input through VAE"""
        x_vae = self.vae_down(x)
        x_vae = x_vae.view(-1, self.vae_fc1.in_features)

        z_mean = self.vae_fc1(x_vae)
        z_mean_rand = torch.randn_like(z_mean)

        if self.vae_estimate_std:
            z_sigma = F.softplus(self.vae_fc2(x_vae))
            z = z_mean + z_sigma * z_mean_rand
            return z, z_mean, z_sigma
        else:
            z = z_mean + self.vae_default_std * z_mean_rand
            return z, z_mean, None

    def forward(self, x: torch.Tensor, clinical_data: torch.Tensor | None = None):
        """Forward pass through the entire model"""
        # Split input into image and mask
        image, mask = x[:, 0:1], x[:, 1:2]

        # Encode
        encoded, _ = self.encoder(x)

        # VAE encoding
        z, z_mean, z_sigma = self.encode_vae(encoded)

        # Reconstruction
        recon_image = self.recon_conv_image(encoded)
        recon_mask = self.recon_conv_mask(encoded)
        reconstruction = torch.cat([recon_image, recon_mask], dim=1)

        # Outcome prediction - modified to handle no clinical features
        if self.clinical_features > 0 and clinical_data is not None:
            outcome = self.outcome_decoder(z_mean, clinical_data)
        else:
            outcome = self.outcome_decoder(z_mean, torch.zeros((z_mean.size(0), 0), device=z_mean.device))

        return {
            "reconstruction": reconstruction,
            "outcome": outcome,
            "z_mean": z_mean,
            "z_sigma": z_sigma,
            "z": z,
            "image": image,
            "mask": mask,
        }


class ILOPDirect(nn.Module):
    def __init__(
            self,
            input_image_size: Sequence[int],
            clinical_features: int = 0,  # Default to 0
            spatial_dims: int = 3,
            init_filters: int = 8,
            in_channels: int = 2,
            dropout_prob: float = 0.1,
            act: tuple | str = ("RELU", {"inplace": True}),
            norm: tuple | str = ("GROUP", {"num_groups": 8}),
            blocks_down: tuple = (1, 2, 2, 4),
    ):
        super().__init__()

        self.clinical_features = clinical_features

        # Encoder
        self.encoder = ILOPEncoder(
            spatial_dims=spatial_dims,
            init_filters=init_filters,
            in_channels=in_channels,
            dropout_prob=dropout_prob,
            act=act,
            norm=norm,
            blocks_down=blocks_down,
        )

        # Calculate encoded dimension
        zoom = 2 ** (len(blocks_down) - 1)
        encoded_size = [s // zoom for s in input_image_size]
        encoded_channels = init_filters * 2 ** (len(blocks_down) - 1)
        self.encoded_dim = int(encoded_channels * np.prod(encoded_size))

        # Reconstruction branches
        self.recon_conv_image = self._make_final_conv(1)
        self.recon_conv_mask = self._make_final_conv(1)

        # Outcome prediction
        self.outcome_decoder = ILOPDecoder(
            input_features=self.encoded_dim,
            clinical_features=clinical_features
        )

    def _make_final_conv(self, out_channels: int):
        """Create final convolution layer for reconstruction"""
        v_filters = self.encoder.init_filters * 2 ** (len(self.encoder.blocks_down) - 1)

        return nn.Sequential(
            get_norm_layer(
                name=self.encoder.norm,
                spatial_dims=self.encoder.spatial_dims,
                channels=v_filters,
            ),
            self.encoder.act_mod,
            get_conv_layer(
                self.encoder.spatial_dims,
                v_filters,
                out_channels,
                kernel_size=1,
                bias=True,
            ),
        )

    def forward(self, x: torch.Tensor, clinical_data: torch.Tensor | None = None):
        # Split input into image and mask
        image, mask = x[:, 0:1], x[:, 1:2]

        # Encode
        encoded, _ = self.encoder(x)
        encoded_flat = encoded.view(encoded.size(0), -1)

        # Reconstruction
        recon_image = self.recon_conv_image(encoded)
        recon_mask = self.recon_conv_mask(encoded)
        reconstruction = torch.cat([recon_image, recon_mask], dim=1)

        # Outcome prediction
        if self.clinical_features > 0 and clinical_data is not None:
            outcome = self.outcome_decoder(encoded_flat, clinical_data)
        else:
            outcome = self.outcome_decoder(encoded_flat,
                                           torch.zeros((encoded_flat.size(0), 0), device=encoded_flat.device))

        return {
            "reconstruction": reconstruction,
            "outcome": outcome,
            "encoded": encoded_flat,
        }


class ILOPDirectCompact(nn.Module):
    def __init__(
            self,
            input_image_size: Sequence[int],
            clinical_features: int = 0,
            spatial_dims: int = 3,
            init_filters: int = 8,
            in_channels: int = 2,
            dropout_prob: float = 0.1,
            latent_dim: int = 256,
            act: tuple | str = ("RELU", {"inplace": True}),
            norm: tuple | str = ("GROUP", {"num_groups": 8}),
            blocks_down: tuple = (1, 2, 2, 4),
    ):
        super().__init__()

        self.clinical_features = clinical_features
        self.latent_dim = latent_dim
        self.smallest_filters = 16

        # Encoder
        self.encoder = ILOPEncoder(
            spatial_dims=spatial_dims,
            init_filters=init_filters,
            in_channels=in_channels,
            dropout_prob=dropout_prob,
            act=act,
            norm=norm,
            blocks_down=blocks_down,
        )

        # Calculate dimensions after encoder - matching ILOPVAE
        zoom = 2 ** (len(blocks_down) - 1)
        self.fc_insize = [s // (2 * zoom) for s in input_image_size]
        total_elements = int(self.smallest_filters * np.prod(self.fc_insize))
        v_filters = init_filters * 2 ** (len(blocks_down) - 1)

        # Feature reduction pathway
        self.feature_down = nn.Sequential(
            get_norm_layer(
                name=norm,
                spatial_dims=spatial_dims,
                channels=v_filters,
            ),
            self.encoder.act_mod,
            get_conv_layer(
                spatial_dims,
                v_filters,
                self.smallest_filters,
                stride=2,
                bias=True,
            ),
            get_norm_layer(
                name=norm,
                spatial_dims=spatial_dims,
                channels=self.smallest_filters,
            ),
            self.encoder.act_mod,
        )

        # Linear projection to latent space
        self.fc_encode = nn.Linear(total_elements, latent_dim)

        # Reconstruction branches directly from encoder
        self.recon_conv_image = self._make_final_conv(1)
        self.recon_conv_mask = self._make_final_conv(1)

        # Outcome prediction
        self.outcome_decoder = ILOPDecoder(
            input_features=latent_dim,
            clinical_features=clinical_features
        )

    def _make_final_conv(self, out_channels: int):
        v_filters = self.encoder.init_filters * 2 ** (len(self.encoder.blocks_down) - 1)
        return nn.Sequential(
            get_norm_layer(
                name=self.encoder.norm,
                spatial_dims=self.encoder.spatial_dims,
                channels=v_filters,
            ),
            self.encoder.act_mod,
            get_conv_layer(
                self.encoder.spatial_dims,
                v_filters,
                out_channels,
                kernel_size=1,
                bias=True,
            ),
        )

    def forward(self, x: torch.Tensor, clinical_data: torch.Tensor | None = None):
        # Encode
        encoded, _ = self.encoder(x)

        # Generate reconstructions directly from encoder
        recon_image = self.recon_conv_image(encoded)
        recon_mask = self.recon_conv_mask(encoded)
        reconstruction = torch.cat([recon_image, recon_mask], dim=1)

        # Feature reduction and latent encoding for outcome prediction
        x_down = self.feature_down(encoded)
        x_flat = x_down.view(x_down.size(0), -1)
        latent = self.fc_encode(x_flat)

        # Outcome prediction
        if self.clinical_features > 0 and clinical_data is not None:
            outcome = self.outcome_decoder(latent, clinical_data)
        else:
            outcome = self.outcome_decoder(latent,
                                         torch.zeros((latent.size(0), 0), device=latent.device))

        return {
            "reconstruction": reconstruction,
            "outcome": outcome,
            "encoded": latent,
        }


class ILOPVAENoReconstruction(nn.Module):
    """
    VAE model for Image-Lesion Outcome Prediction without reconstruction branch,
    combining encoder, VAE components, and outcome prediction only.
    """

    def __init__(
        self,
        input_image_size: Sequence[int],
        clinical_features: int,
        vae_estimate_std: bool = False,
        vae_default_std: float = 0.3,
        vae_nz: int = 256,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 2,
        dropout_prob: float = 0.1,
        act: tuple | str = ("RELU", {"inplace": True}),
        norm: tuple | str = ("GROUP", {"num_groups": 8}),
        blocks_down: tuple = (1, 2, 2, 4),
    ):
        super().__init__()

        self.input_image_size = input_image_size
        self.clinical_features = clinical_features
        self.smallest_filters = 16
        self.vae_estimate_std = vae_estimate_std
        self.vae_default_std = vae_default_std
        self.vae_nz = vae_nz

        # Initialize encoder
        self.encoder = ILOPEncoder(
            spatial_dims=spatial_dims,
            init_filters=init_filters,
            in_channels=in_channels,
            dropout_prob=dropout_prob,
            act=act,
            norm=norm,
            blocks_down=blocks_down,
        )

        # Calculate VAE input size
        zoom = 2 ** (len(blocks_down) - 1)
        self.fc_insize = [s // (2 * zoom) for s in input_image_size]

        # Initialize VAE components
        self._prepare_vae_modules()

        # Initialize outcome predictor
        self.outcome_decoder = ILOPDecoder(
            input_features=vae_nz, clinical_features=clinical_features
        )

    def _prepare_vae_modules(self):
        """Initialize VAE-specific modules"""
        total_elements = int(self.smallest_filters * np.prod(self.fc_insize))
        v_filters = self.encoder.init_filters * 2 ** (len(self.encoder.blocks_down) - 1)

        self.vae_down = nn.Sequential(
            get_norm_layer(
                name=self.encoder.norm,
                spatial_dims=self.encoder.spatial_dims,
                channels=v_filters,
            ),
            self.encoder.act_mod,
            get_conv_layer(
                self.encoder.spatial_dims,
                v_filters,
                self.smallest_filters,
                stride=2,
                bias=True,
            ),
            get_norm_layer(
                name=self.encoder.norm,
                spatial_dims=self.encoder.spatial_dims,
                channels=self.smallest_filters,
            ),
            self.encoder.act_mod,
        )

        self.vae_fc1 = nn.Linear(total_elements, self.vae_nz)  # mean
        if self.vae_estimate_std:
            self.vae_fc2 = nn.Linear(total_elements, self.vae_nz)  # std
        self.vae_fc3 = nn.Linear(self.vae_nz, total_elements)  # decoder input

    def encode_vae(self, x: torch.Tensor):
        """Encode input through VAE"""
        x_vae = self.vae_down(x)
        x_vae = x_vae.view(-1, self.vae_fc1.in_features)

        z_mean = self.vae_fc1(x_vae)
        z_mean_rand = torch.randn_like(z_mean)

        if self.vae_estimate_std:
            z_sigma = F.softplus(self.vae_fc2(x_vae))
            z = z_mean + z_sigma * z_mean_rand
            return z, z_mean, z_sigma
        else:
            z = z_mean + self.vae_default_std * z_mean_rand
            return z, z_mean, None

    def forward(self, x: torch.Tensor, clinical_data: torch.Tensor | None = None):
        """Forward pass through the entire model without reconstruction branch"""
        # Split input into image and mask (for potential use in loss calculation)
        image, mask = x[:, 0:1], x[:, 1:2]

        # Encode
        encoded, _ = self.encoder(x)

        # VAE encoding
        z, z_mean, z_sigma = self.encode_vae(encoded)

        # Outcome prediction - modified to handle no clinical features
        if self.clinical_features > 0 and clinical_data is not None:
            outcome = self.outcome_decoder(z_mean, clinical_data)
        else:
            outcome = self.outcome_decoder(z_mean, torch.zeros((z_mean.size(0), 0), device=z_mean.device))

        return {
            "outcome": outcome,
            "z_mean": z_mean,
            "z_sigma": z_sigma,
            "z": z,
            "image": image,
            "mask": mask,
        }
