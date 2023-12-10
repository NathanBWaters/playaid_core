from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.models import resnet18
import statistics

from playaid.ult_action_dataset import UltActionRecogDataset


class SpatialStreamCNN(nn.Module):
    def __init__(self, num_actions, sequence_length):
        super(SpatialStreamCNN, self).__init__()
        self.cnn2d = resnet18(pretrained=True)

        # Do I need to set output size?
        # self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))

        resnet_feature_size = 1000
        self.cnn1d = nn.Sequential(
            nn.Conv1d(resnet_feature_size, 512, kernel_size=sequence_length, stride=1),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, num_actions))

    def forward(self, x):
        batch_size, sequence_length, C, H, W = x.size()
        x = x.view(batch_size * sequence_length, C, H, W)
        x = self.cnn2d(x)
        # [sequence_length, resnet_feature_size]
        x = x.view(batch_size, sequence_length, -1).permute(0, 2, 1)
        # [batch_size, resnet_feature_size, sequence_length]
        # Q: should this be [batch_size, sequence_length, resnet_feature_size]?
        x = self.cnn1d(x)
        # [batch_size, 512, 1]
        x = x.view(x.size(0), -1)
        # [batch_size, 512]
        x = self.classifier(x)
        # [batch_size, num_actions]
        return x


class CNNActionDetector(LightningModule):
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

        self.model = SpatialStreamCNN(self.num_actions, self.sequence_length)

        self.action_train_accuracy = Accuracy(task="multiclass", num_classes=self.num_actions)
        self.action_val_accuracy = Accuracy(task="multiclass", num_classes=self.num_actions)
        self.action_test_accuracy = Accuracy(task="multiclass", num_classes=self.num_actions)

        # Keeps running tabs on batch accuracy for an epoch
        self.training_epoch_acc = []

    def forward(self, x):
        """
        batch is [batch_size, frames_per_sequence, channel, height, width]
        """
        action_logits = self.model(x)

        return F.log_softmax(action_logits, dim=1)

    def training_step(self, batch, batch_idx):
        input, char_label, action_label, _ = batch
        batch_size, seq_length, c, h, w = input.shape
        
        center_index = seq_length // 2
        centered_action_label = action_label[:, center_index]

        # action_logits is [batch_size, num_classes]
        action_logits = self(input)

        # action_label is [batch_size]
        action_loss = F.nll_loss(action_logits, centered_action_label)
        action_preds = torch.argmax(action_logits, dim=1)
        self.action_train_accuracy.update(action_preds, centered_action_label)

        self.training_epoch_acc.append(
            float(self.action_train_accuracy(action_preds, centered_action_label))
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
        
        center_index = seq_length // 2
        centered_action_label = action_label[:, center_index]

        # action_logits is [batch_size, num_classes]
        action_logits = self(input)

        # action_label is [batch_size]
        action_loss = F.nll_loss(action_logits, centered_action_label)
        action_preds = torch.argmax(action_logits, dim=1)
        self.action_val_accuracy.update(action_preds, centered_action_label)
        self.log("val_action_loss", action_loss)
        self.log("val_action_acc", self.action_val_accuracy)

    def test_step(self, batch, batch_idx):
        input, char_label, action_label, _ = batch
        batch_size, seq_length, c, h, w = input.shape
        
        center_index = seq_length // 2
        centered_action_label = action_label[:, center_index]

        # action_logits is [batch_size, num_classes]
        action_logits = self(input)

        # action_label is [batch_size]
        action_loss = F.nll_loss(action_logits, centered_action_label)
        action_preds = torch.argmax(action_logits, dim=1)
        self.action_test_accuracy.update(action_preds, centered_action_label)
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
