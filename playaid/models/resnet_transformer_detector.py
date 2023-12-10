from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import statistics
import numpy as np
import einops
import torch
import timm

from playaid.ult_action_dataset import UltActionRecogDataset


@torch.jit.script
def time_encoding(x: torch.Tensor, num_freq: int, dim: int = 1):
    out = [x]
    for i in range(num_freq):
        out.extend((torch.cos(np.pi * x * (2**i)), torch.sin(np.pi * x * (2**i))))
    return torch.cat(out, dim=dim)


class ResFormer(nn.Module):
    def __init__(
        self, num_actions=61, sequence_length=7, hidden_dim=247, num_heads=8, num_layers=3
    ):
        super().__init__()
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim

        # Load a pre-trained ResNet model
        # self.resnet = resnet50(pretrained=True)
        self.resnet = timm.create_model("resnet50", num_classes=0, pretrained=True)

        # Pass the resnet features into an encoder to reduce its dimensionality.
        # 2048 -> 247
        self.resnet_ffn = nn.Linear(2048, self.hidden_dim)

        self.register_buffer(
            "freq_encoding",
            time_encoding(
                torch.linspace(0, 1, sequence_length).reshape(-1, 1),
                4,  # num of frequencies used to encode the value.
            ),
        )

        self.output_transformer_dim = self.hidden_dim + self.freq_encoding.shape[1]

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.output_transformer_dim,
            nhead=num_heads,
        )
        self.transformer = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers,
        )

        # Classifier layer
        # 1000 features * 7 images in the sequences.
        self.resnet_classifier = nn.Linear(self.hidden_dim, num_actions)
        self.classifier = nn.Linear(self.output_transformer_dim, num_actions)

    def forward(self, frames, return_single_frame_preds=False):
        batch_size, sequence_length, C, H, W = frames.size()

        x = einops.rearrange(frames, "b s c h w -> (b s) c h w")

        # Extract features using the ResNet
        # output - [images x 2048]
        cnn_features = self.resnet(x)
        flattened_resnet_features = self.resnet_ffn(cnn_features)
        sequenced_features = einops.rearrange(
            flattened_resnet_features, "(b s) f -> b s f", b=batch_size
        )

        transformer_features = torch.cat(
            (sequenced_features, einops.repeat(self.freq_encoding, "s f -> b s f", b=batch_size)),
            dim=2,
        )
        # Pass through the transformer
        transformer_features = self.transformer(transformer_features)

        # Classify the output
        transformer_preds = einops.rearrange(
            self.classifier(einops.rearrange(transformer_features, "b s f -> (b s) f")),
            "(b s) c -> b s c",
            b=batch_size,
            c=self.num_actions,
        )

        return transformer_preds


class ResnetTransformerDetector(LightningModule):
    def __init__(
        self,
        actions: list,
        batch_size: int = 64,
        sequence_length: int = 4,
        learning_rate: float = 2e-4,
        num_samples: int = 1024,
        freeze_encoder=False,
        **kwargs,
    ):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()

        # Set our init args as class attributes
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.actions = actions
        self.num_actions = len(self.actions)
        self.num_samples = num_samples
        self.sequence_length = sequence_length

        self.smash_action_dataset_train = None
        self.smash_action_dataset_val = None
        self.smash_action_dataset_test = None

        self.dataset_kwargs = kwargs

        self.model = ResFormer(self.num_actions, self.sequence_length)

        self.action_train_accuracy = Accuracy(task="multiclass", num_classes=self.num_actions)
        self.action_val_accuracy = Accuracy(task="multiclass", num_classes=self.num_actions)
        self.action_test_accuracy = Accuracy(task="multiclass", num_classes=self.num_actions)

        # Keeps running tabs on batch accuracy for an epoch
        self.training_epoch_acc = []

    def forward(self, frames):
        """
        batch is [batch_size, frames_per_sequence, channel, height, width]
        """
        action_logits = self.model(frames)
        # output is [batch_size, sequence_size, num_actions]

        return F.log_softmax(action_logits, dim=2)

    def training_step(self, batch, batch_idx):
        input, char_label, action_label, _ = batch
        batch_size, seq_length, c, h, w = input.shape

        action_logits = self(input)
        action_logits = einops.rearrange(action_logits, "b s f -> (b s) f")
        action_label = einops.rearrange(action_label, "b s -> (b s)")

        action_loss = F.nll_loss(action_logits, action_label)
        action_preds = torch.argmax(action_logits, dim=1)
        self.action_train_accuracy.update(action_preds, action_label)

        self.training_epoch_acc.append(
            float(self.action_train_accuracy(action_preds, action_label))
        )

        self.log("training_action_loss", action_loss)
        self.log("training_action_acc", self.action_train_accuracy)

        return action_loss

    def training_epoch_end(self, training_step_outputs):
        # If we're getting 90% of the epoch correct, then increase the synth difficulty
        # Yes... average of averages is bad form.
        epoch_acc = statistics.mean(self.training_epoch_acc)
        # print(f"Epoch accuracy: {epoch_acc}")
        if epoch_acc > 0.85:
            self.smash_action_dataset_train.make_synth_more_challenging()

        # Clear it out each epoch.
        self.training_epoch_acc = []
        self.smash_action_dataset_train.switch_num_frames_per_sample()
        pass

    def validation_step(self, batch, batch_idx):
        input, char_label, action_label, _ = batch
        batch_size, seq_length, c, h, w = input.shape

        action_logits = self(input)
        action_logits = einops.rearrange(action_logits, "b s f -> (b s) f")
        action_label = einops.rearrange(action_label, "b s -> (b s)")

        action_loss = F.nll_loss(action_logits, action_label)
        action_preds = torch.argmax(action_logits, dim=1)
        self.action_val_accuracy.update(action_preds, action_label)
        self.log("val_action_loss", action_loss)
        self.log("val_action_acc", self.action_val_accuracy)

    def test_step(self, batch, batch_idx):
        input, char_label, action_label, _ = batch
        batch_size, seq_length, c, h, w = input.shape

        action_logits = self(input)
        action_logits = einops.rearrange(action_logits, "b s f -> (b s) f")
        action_label = einops.rearrange(action_label, "b s -> (b s)")

        action_loss = F.nll_loss(action_logits, action_label)
        action_preds = torch.argmax(action_logits, dim=1)
        self.action_test_accuracy.update(action_preds, action_label)
        self.log("test_action_loss", action_loss)
        self.log("test_action_acc", self.action_test_accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def train_dataloader(self):
        if not self.smash_action_dataset_train:
            self.smash_action_dataset_train = UltActionRecogDataset(
                split="train",
                num_samples=self.num_samples,
                img_dimension=128,
                anim_subset=self.actions,
                **self.dataset_kwargs,
            )

        return DataLoader(
            self.smash_action_dataset_train, batch_size=self.batch_size, num_workers=1
        )

    def val_dataloader(self):
        if not self.smash_action_dataset_val:
            self.smash_action_dataset_val = UltActionRecogDataset(
                split="validation",
                num_samples=int(self.num_samples / 4),
                img_dimension=128,
                anim_subset=self.actions,
                **self.dataset_kwargs,
            )
        return DataLoader(self.smash_action_dataset_val, batch_size=self.batch_size, num_workers=1)

    def test_dataloader(self):
        if not self.smash_action_dataset_test:
            self.smash_action_dataset_test = UltActionRecogDataset(
                split="test",
                num_samples=int(self.num_samples / 4),
                img_dimension=128,
                anim_subset=self.actions,
                **self.dataset_kwargs,
            )
        return DataLoader(self.smash_action_dataset_test, batch_size=self.batch_size, num_workers=1)
