from copy import deepcopy
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch import nn
import torchvision
from torchvision.models import Inception_V3_Weights, inception_v3

from lightning import LightningModule
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error

class InceptionV3FacenetModule(LightningModule):
    INCEPTION_OUT_DIMS = 2048
    INCEPTION_AUX_DIMS = 1000
    INCEPTION_IN_CHANNELS = 3
    MIN_SIZE = 299

    def __init__(self,
                 *args: Any,
                 embedding_size: int,
                 margin: float,
                 squared: bool,
                 lr: float,
                 lr_gamma: float,
                 loss: Any,
                 y_encoder: OrdinalEncoder,
                 **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.embedding_size = embedding_size
        self.margin = margin
        self.squared = squared
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.loss = loss
        self.y_encoder = y_encoder

        self.inception = torchvision.models.inception_v3(
            weights=torchvision.models.Inception_V3_Weights(
                torchvision.models.Inception_V3_Weights.DEFAULT
            )
        )

        # Freeze base model
        for param in self.inception.parameters():
            param.requires_grad = False
        self.inception.eval()

        self.inception.fc = torch.nn.Identity()

        self.conversion_layer = torch.nn.Conv2d(
            in_channels=10,
            out_channels=10,
            kernel_size=(3,3),
            padding='same')
        self.conversion_layer_2 = torch.nn.Conv2d(
            in_channels=10,
            out_channels=7,
            kernel_size=(3,3),
            padding='same')
        self.conversion_layer_1 = torch.nn.Conv2d(
            in_channels=7,
            out_channels=3,
            kernel_size=(1,1),
            padding='same')

        self.activation = torch.nn.ReLU()

        self.embedding_layer = torch.nn.Linear(
            in_features=2048,
            out_features=self.embedding_size
        )

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Notes:
            - Min input size: 299 x 299.
            - Batch size: >= 2.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
        """

        if (
            any(s == 1 for s in x.shape)
            or x.shape[2] < self.MIN_SIZE
            or x.shape[3] < self.MIN_SIZE
        ):
            raise RuntimeError(
                f"Input shape {x.shape} must adhere to the following:\n"
                f" - No singleton dimensions\n"
                f" - Size >= {self.MIN_SIZE}\n"
            )

        x = self.conversion_layer(x)
        x = self.activation(x)
        x = self.conversion_layer_2(x)
        x = self.activation(x)
        x = self.conversion_layer_1(x)
        x = self.activation(x)
        x = self.base_model(x)
        x = self.embedding_layer(x)

        return x

    def configure_optimizers(self):
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimiser := torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
            ),
            gamma=self.lr_gamma,
        )
        lr_scheduler_config = {
            "optimizer": optimiser,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }
        return lr_scheduler_config

    def training_step(self, batch, batch_idx):
        images, labels = batch

        embeddings = self(images)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1, eps=1e-10)

        loss = self.loss(labels, embeddings, self.margin, self.squared)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        embeddings = self(images)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1, eps=1e-10)

        loss = self.loss(labels, embeddings, self.margin, self.squared)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch

        embeddings = self(images)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1, eps=1e-10)

        loss = self.loss(labels, embeddings, self.margin, self.squared)
        return loss
