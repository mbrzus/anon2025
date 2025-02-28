from typing import Any

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss, SSIMLoss
from monai.config import print_config
from monai.visualize import plot_2d_or_3d_image
from monai.data import CacheDataset
from monai.transforms import (
    Compose,
    ToTensord,
    RandGaussianNoised,
)
from torchmetrics import Recall, Accuracy, Specificity, AUROC
from ILOP import ILOPVAE
from src.cnn_transforms import *
import os
import pandas as pd


class ILOPVAEModule(pl.LightningModule):
    def __init__(
        self,
        model_params: dict,
        learning_rate: float = 1e-3,
        loss_weights=None,
        ssim_weight: float = 0.5,  # Weight for SSIM vs MSE loss
        optimizer_config: dict = None,
    ):
        super().__init__()
        if loss_weights is None:
            loss_weights = {
                "image_recon": 1.0,
                "mask_recon": 1.0,
                "vae_kl": 0.1,
                "outcome": 1.0,
            }
        self.save_hyperparameters()

        self.model = ILOPVAE(**model_params)
        self.learning_rate = learning_rate
        self.loss_weights = loss_weights
        self.optimizer_config = optimizer_config or {"type": "single"}
        self.ssim_weight = ssim_weight

        self.sigmoid = nn.Sigmoid()
        self.threshold = 0.5

        self.downsample = nn.AvgPool3d(kernel_size=8)

        self._init_loss_functions()
        self._init_metrics()

    def _init_loss_functions(self):
        """Initialize all loss functions"""
        # Combined image reconstruction loss (MSE + SSIM)
        self.ssim_loss = SSIMLoss(
            spatial_dims=3,  # For 3D images
            data_range=1.0,
            kernel_type="gaussian",
            win_size=11,
            kernel_sigma=1.5,
            k1=0.01,
            k2=0.03,
            reduction="mean",
        )
        self.image_recon_loss = self._compute_image_recon_loss

        # Mask reconstruction loss (Dice + BCE)
        self.mask_recon_loss = DiceCELoss(weight=torch.tensor(10.0), sigmoid=True)

        # Outcome prediction loss
        self.outcome_loss = nn.BCEWithLogitsLoss()

    def forward(self, x, c) -> Any:
        return self.model(x, c)

    def _compute_image_recon_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Combined MSE and SSIM loss for image reconstruction"""
        mse_loss = F.mse_loss(pred, target)
        ssim_loss = self.ssim_loss(pred, target)

        return (1 - self.ssim_weight) * mse_loss + self.ssim_weight * ssim_loss

    def _init_metrics(self):
        """Initialize separate metrics for training and validation"""
        # DICE metrics for mask reconstruction
        self.train_dice_metric = DiceMetric(
            reduction="mean", num_classes=2, ignore_empty=False
        )
        self.val_dice_metric = DiceMetric(
            reduction="mean", num_classes=2, ignore_empty=False
        )

        # F1 score for outcome prediction
        # self.train_f1 = BinaryF1Score()
        # self.val_f1 = BinaryF1Score()

        # Training metrics
        self.train_accuracy = Accuracy(task="binary", average="macro")
        self.train_auroc = AUROC(task="binary", average="macro")
        self.train_recall = Recall(task="binary", average="macro")
        self.train_specificity = Specificity(task="binary", average="macro")

        # Validation metrics
        self.val_accuracy = Accuracy(task="binary", average="macro")
        self.val_auroc = AUROC(task="binary", average="macro")
        self.val_recall = Recall(task="binary", average="macro")
        self.val_specificity = Specificity(task="binary", average="macro")

    def _get_binary_mask(self, mask_logits):
        """Convert logits to binary mask"""
        mask_probs = self.sigmoid(mask_logits)
        return (mask_probs > self.threshold).float()

    def _get_predictions(self, y_hat):
        y_prob = torch.sigmoid(y_hat).squeeze()  # Changed from softmax
        y_pred = (y_prob > 0.5).float()
        return y_prob, y_pred

    def _compute_vae_kl_loss(
        self, z_mean: torch.Tensor, z_sigma: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence loss for VAE"""
        if z_sigma is None:  # Using fixed std
            return 0.5 * torch.mean(z_mean.pow(2))

        # Full KL divergence with learned std
        return -0.5 * torch.mean(
            1 + torch.log(z_sigma.pow(2)) - z_mean.pow(2) - z_sigma.pow(2)
        )

    def _compute_all_losses(
        self, outputs: dict, image: torch.tensor, label: torch.tensor
    ) -> dict:
        # Downsample input image to match reconstruction size
        downsampled_img = self.downsample(image[:, 0:1])
        downsampled_mask = self.downsample(image[:, 1:2])

        # Extract predictions
        recon_img = outputs["reconstruction"][:, 0:1]
        recon_mask = outputs["reconstruction"][:, 1:2]

        # Add epoch-dependent loss weighting
        reconstruction_weight = min(
            1.0, self.current_epoch / 10
        )  # Gradually increase from 0 to 1
        outcome_weight = max(
            0.1, min(1.0, (self.current_epoch - 5) / 10)
        )  # Start later, gradually increase

        losses = {
            "image_recon": self.image_recon_loss(recon_img, downsampled_img),
            "mask_recon": self.mask_recon_loss(recon_mask, downsampled_mask),
            "vae_kl": self._compute_vae_kl_loss(outputs["z_mean"], outputs["z_sigma"]),
            "outcome": self.outcome_loss(outputs["outcome"], label.view(-1, 1)),
        }

        losses["total"] = (
            reconstruction_weight
            * (
                self.loss_weights["image_recon"] * losses["image_recon"]
                + self.loss_weights["mask_recon"] * losses["mask_recon"]
            )
            + self.loss_weights["vae_kl"] * losses["vae_kl"]
            + outcome_weight * self.loss_weights["outcome"] * losses["outcome"]
        )
        return losses

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        # Forward pass and loss computation
        x, y = batch["image"], batch["label"]
        c = batch["age"].unsqueeze(-1)
        outputs = self.model(x, c)
        losses = self._compute_all_losses(outputs, x, y)

        # Get predictions using built-in function
        y_prob, y_pred = self._get_predictions(outputs["outcome"])

        # Compute mask predictions
        mask_pred = self._get_binary_mask(outputs["reconstruction"][:, 1:2])
        true_mask = (self.downsample(batch["image"][:, 1:2]) > 0.2).float()

        # Update metrics
        with torch.no_grad():
            # Mask reconstruction metrics
            self.train_dice_metric(mask_pred, true_mask)

            # Outcome prediction metrics
            self.train_accuracy(y_pred, y.view(-1))
            self.train_auroc(y_prob, y.view(-1))
            self.train_recall(y_pred, y.view(-1))
            self.train_specificity(y_pred, y.view(-1))

        # Logging
        # Loss logging
        self.log("train_loss", losses["total"], on_step=True, on_epoch=True)
        for name, value in losses.items():
            self.log(f"train_{name}_loss", value, on_step=True, on_epoch=True)

        # Metric logging
        self.log("train_dice", self.train_dice_metric.aggregate(), on_epoch=True)
        self.log("train_accuracy", self.train_accuracy, on_epoch=True)
        self.log("train_auroc", self.train_auroc, on_epoch=True)
        self.log("train_recall", self.train_recall, on_epoch=True)
        self.log("train_specificity", self.train_specificity, on_epoch=True)

        return losses["total"]

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        # Forward pass and loss computation
        x, y = batch["image"], batch["label_binary"]
        c = batch["age"].unsqueeze(-1)
        outputs = self.model(x, c)
        losses = self._compute_all_losses(outputs, x, y)

        # Get predictions using built-in function
        y_prob, y_pred = self._get_predictions(outputs["outcome"])

        # Compute mask predictions
        mask_pred = self._get_binary_mask(outputs["reconstruction"][:, 1:2])
        true_mask = (self.downsample(batch["image"][:, 1:2]) > 0.2).float()

        # # Create visualizations
        # plot_2d_or_3d_image(
        #     data=self.downsample(x[0:1, 0:1]),
        #     tag="Val True_Image",
        #     step=self.current_epoch,
        #     writer=self.logger.experiment,
        #     max_channels=1,
        # )
        # plot_2d_or_3d_image(
        #     data=true_mask[0:1],
        #     tag="Val True_Mask",
        #     step=self.current_epoch,
        #     writer=self.logger.experiment,
        #     max_channels=1,
        # )
        # plot_2d_or_3d_image(
        #     data=outputs["reconstruction"][0:1, 0:1],
        #     tag="Val Reconstructed_Image",
        #     step=self.current_epoch,
        #     writer=self.logger.experiment,
        #     max_channels=1,
        # )
        # plot_2d_or_3d_image(
        #     data=mask_pred[0:1],
        #     tag="Val Reconstructed_Mask",
        #     step=self.current_epoch,
        #     writer=self.logger.experiment,
        #     max_channels=1,
        # )

        # Update metrics
        self.val_dice_metric(mask_pred, true_mask)


        self.val_accuracy(y_pred, y.view(-1))
        self.val_auroc(y_prob, y.view(-1))
        self.val_recall(y_pred, y.view(-1))
        self.val_specificity(y_pred, y.view(-1))

        # Logging
        # Loss logging
        self.log("val_loss", losses["total"], on_epoch=True, prog_bar=True)
        for name, value in losses.items():
            self.log(f"val_{name}_loss", value, on_epoch=True)

        # Metric logging
        self.log(
            "val_dice", self.val_dice_metric.aggregate(), on_epoch=True, prog_bar=True
        )
        self.log(
            "val_accuracy",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_auroc",
            self.val_auroc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_recall",
            self.val_recall,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_specificity",
            self.val_specificity,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return losses["total"]

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-4,  # Lower initial learning rate
            weight_decay=0.01,
            eps=1e-8,  # Increase epsilon for better numerical stability
        )

        # Use more gradual LR reduction
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.2,  # More gradual reduction
            patience=15,
            verbose=True,
            min_lr=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    def prepare_data(self):
        """
        This function sets up data preprocessing for the model training
        """
        # access data from prepared csv files
        csv_path = "spreadsheets/final_train_test_data_mni_registry_only.csv"
        df = pd.read_csv(csv_path)
        split = 'split2'
        train_df = df[df[split] == "train"]
        # train_df = train_df[
        #     ~train_df["lesion_mask"].str.contains(r"augmented\d+\.nii\.gz$")
        # ]
        # train_df = train_df.drop_duplicates(subset=["subject"])

        test_df = df[df[split] == "test"]
        test_df = test_df[
            ~test_df["lesion_mask"].str.contains(r"augmented\d+\.nii\.gz$")
        ]
        test_df = test_df.drop_duplicates(subset=["subject"])
        # organize the data into list of dictionaries structure which is compatible with MONAI preprocessing workflow
        train_files = [
            {
                "t1": t1_name,
                "lesion": lesion_name,
                "bnt_norm": bnt_norm,
                "binary_weight": binary_weight,
                "source": source,
                "age": age,
            }
            for t1_name, lesion_name, bnt_norm, binary_weight, source, age in zip(
                train_df["t1w_synthsr_ss"],
                train_df["lesion_mask"],
                # Labels are already Z-scores ranging from -16.85 to +1.7
                # No further normalization is needed
                train_df["bnt_norm"],
                train_df["bnt_adaptive_weights"],
                train_df["source"],
                train_df["age"],
            )
        ]

        test_files = [
            {
                "t1": t1_name,
                "lesion": lesion_name,
                "bnt_norm": bnt_norm,
                "label_binary": label_binary,
                "binary_weight": binary_weight,
                "source": source,
                "age": age,
            }
            for t1_name, lesion_name, bnt_norm, label_binary, binary_weight, source, age in zip(
                test_df["t1w_synthsr_ss"],
                test_df["lesion_mask"],
                test_df["bnt_norm"],
                test_df["bnt_binary"],
                test_df["bnt_adaptive_weights"],
                test_df["source"],
                test_df["age"],
            )
        ]
        # train_files = train_files[:5]
        # test_files = test_files[:2]

        # Define transforms
        train_transforms = Compose(
            [
                LoadITKImaged(
                    keys=["t1", "lesion"],  # "sca", "fca"],
                    pixel_types=[itk.F, itk.UC, itk.US, itk.US],
                ),
                DownsampleCustomSingleImaged(keys=["t1"], spacing=[0.5, 0.5, 6]),
                ITKResampled(
                    keys=["t1", "lesion"],  # , "sca", "fca"],
                    pixel_types=[itk.F, itk.UC, itk.US, itk.US],
                    spacing=[1.1, 1.1, 1.1],
                    size=[160, 192, 160],
                ),
                ITKImageToNumpyd(keys=["t1", "lesion"]),  # "sca", "fca"]),
                ScaleIntensityRangePercentiles2d(
                    keys=["t1"],
                    lower=2.0,
                    upper=98.0,
                    b_min=0.0,
                    b_max=1.0,
                    clip_min=True,
                    clip_max=True,
                    relative=False,
                    exclude_background=True,
                ),
                RandGaussianNoised(
                    keys=["t1"],
                    prob=1.0,  # apply to every image while testing
                    mean=0.0,  # centered noise
                    std=0.05,  # start with 5% noise - adjust based on results
                    sample_std=True,  # will randomly sample std between 0 and 0.05
                ),
                CombineImagesd(keys=["t1", "lesion"]),  # "sca", "fca"]),
                AugmentAndBinarizeLabeld(keys=["bnt_norm"]),
                ToTensord(
                    keys=["image", "label", "binary_weight", "age"], dtype=torch.float32
                ),
            ]
        )

        test_transforms = Compose(
            [
                LoadITKImaged(
                    keys=["t1", "lesion"],  # "sca", "fca"],
                    pixel_types=[itk.F, itk.UC, itk.US, itk.US],
                ),
                ITKResampled(
                    keys=["t1", "lesion"],  # "sca", "fca"],
                    pixel_types=[itk.F, itk.UC, itk.US, itk.US],
                    spacing=[1.1, 1.1, 1.1],
                    size=[160, 192, 160],
                    use_moments=False,
                ),
                ITKImageToNumpyd(keys=["t1", "lesion"]),  # "sca", "fca"]),
                ScaleIntensityRangePercentiles2d(
                    keys=["t1"],
                    lower=2.0,
                    upper=98.0,
                    b_min=0.0,
                    b_max=1.0,
                    clip_min=True,
                    clip_max=True,
                    relative=False,
                    exclude_background=True,
                ),
                CombineImagesd(keys=["t1", "lesion"]),  # "sca", "fca"]),
                ToTensord(
                    keys=["image", "label_binary", "binary_weight", "age"],
                    dtype=torch.float32,
                ),
            ]
        )

        # # we use cached datasets - these are 10x faster than regular datasets
        self.train_dataset = CacheDataset(
            data=train_files, transform=train_transforms, num_workers=16, cache_num=2000
        )
        self.val_dataset = CacheDataset(
            data=test_files, transform=test_transforms, num_workers=8
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset, batch_size=20, shuffle=True, num_workers=16
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset, batch_size=11, num_workers=8)
        return val_loader


def setup_training(log_dir: str):
    """Setup training configuration and create trainer"""

    exp_name = "ILOPVAE_mni_no_age_split2"

    # Model parameters
    model_params = {
        "input_image_size": [160, 192, 160],
        "clinical_features": 1,
        "vae_nz": 256,
        "spatial_dims": 3,
        "init_filters": 8,
        "in_channels": 2,
        "vae_estimate_std": True,  # Enable learned std for VAE
        "dropout_prob": 0.2,  # Add dropout for regularization
    }

    # Create logger
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=log_dir, name=exp_name, default_hp_metric=False
    )

    # Create callbacks
    checkpoint_callback_loss = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "checkpoints"),
        filename=exp_name + "-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )

    checkpoint_callback_auroc = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "checkpoints"),
        filename=exp_name + "-{epoch:02d}-{val_auroc:.2f}",
        monitor="val_auroc",  # Changed from val_auroc_epoch
        mode="max",
        save_top_k=2,
        save_last=True,
        verbose=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,  # Require more significant improvement (was 0.00)
        patience=40,  # Double the patience (was 20)
        verbose=True,
        mode="min",
    )

    # Create learning rate monitor
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=300,
        accelerator="gpu",
        devices=[1],
        callbacks=[
            checkpoint_callback_loss,
            checkpoint_callback_auroc,
            early_stop_callback,
            lr_monitor,
        ],
        logger=tb_logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,  # Add gradient clipping
        accumulate_grad_batches=2,  # Gradient accumulation for larger effective batch size
    )

    return model_params, trainer


if __name__ == "__main__":
    # Configuration
    root_dir = ""
    log_dir = os.path.join(root_dir, "dl_logs")
    os.makedirs(log_dir, exist_ok=True)

    # Print system configuration
    print_config()

    # Setup training
    model_params, trainer = setup_training(log_dir)

    # Create model
    model = ILOPVAEModule(
        model_params=model_params,
        learning_rate=1e-3,
        loss_weights={
            "image_recon": 1.0,
            "mask_recon": 1.0,
            "vae_kl": 0.1,
            "outcome": 1.0,
        },
        ssim_weight=0.7,  # Weight for SSIM loss in image reconstruction
    )

    # Train model
    trainer.fit(model=model)

    print("Training completed!")
