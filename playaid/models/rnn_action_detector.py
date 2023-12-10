from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.models import resnet18
import statistics

from playaid.ult_action_dataset import UltActionRecogDataset


class RNNActionDetector(LightningModule):
    def __init__(
        self,
        fighter_name: str,
        actions: list,
        batch_size: int = 8,
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

        self.num_chars = 1
        self.actions = actions
        self.num_actions = len(self.actions)
        self.fighter_name = fighter_name
        self.num_samples = num_samples

        self.smash_action_dataset_train = None
        self.smash_action_dataset_val = None
        self.smash_action_dataset_test = None

        self.dataset_kwargs = kwargs

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))

        # Setting this to True seems to hurt performance and generalization.
        if freeze_encoder:
            for param in self.resnet.parameters():
                param.requires_grad = False

        self.lstm = nn.LSTM(input_size=300, hidden_size=512, num_layers=3)

        self.action_decoder = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, self.num_actions)
        )

        self.action_train_accuracy = Accuracy(task="multiclass", num_classes=self.num_actions)
        self.action_val_accuracy = Accuracy(task="multiclass", num_classes=self.num_actions)
        self.action_test_accuracy = Accuracy(task="multiclass", num_classes=self.num_actions)

        # Keeps running tabs on batch accuracy for an epoch
        self.training_epoch_acc = []

    def forward(self, x):
        """
        batch is [batch_size, frames_per_sequence, channel, height, width]
        """
        batch_size, seq_length, c, h, w = x.shape
        # Operates 2D conv over each image easily here.
        # Way better than a for loop.
        x = x.view(batch_size * seq_length, c, h, w)
        # Input: [32, 3, 128, 128]
        # Output: [32, 300]
        x = self.resnet(x)
        # Output: [8, 4, 300]
        x = x.view(batch_size, seq_length, -1)
        # Output: [8, 4, 512]
        (x, hidden_state) = self.lstm(x, None)

        # Now fan out each frame to be decoded.
        # Output: [32, num_actions]
        x = x.view(batch_size * seq_length, -1)
        action_logits = self.action_decoder(x)

        return F.log_softmax(action_logits, dim=1)

    def training_step(self, batch, batch_idx):
        input, char_label, action_label, _ = batch
        batch_size, seq_length, c, h, w = input.shape
        action_label = action_label.view(batch_size * seq_length)

        # action_logits is [batch_size * frames_per_sequence, num_classes]
        action_logits = self(input)

        # action_label is [batch_size]
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
        action_label = action_label.view(batch_size * seq_length)

        # action_logits is [batch_size * frames_per_sequence, num_classes]
        action_logits = self(input)

        # action_label is [batch_size]
        action_loss = F.nll_loss(action_logits, action_label)
        action_preds = torch.argmax(action_logits, dim=1)
        self.action_val_accuracy.update(action_preds, action_label)
        self.log("val_action_loss", action_loss)
        self.log("val_action_acc", self.action_val_accuracy)

    def test_step(self, batch, batch_idx):
        input, char_label, action_label, _ = batch
        batch_size, seq_length, c, h, w = input.shape
        action_label = action_label.view(batch_size * seq_length)

        # action_logits is [batch_size * frames_per_sequence, num_classes]
        action_logits = self(input)

        # action_label is [batch_size]
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
                char_subset=[self.fighter_name],
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
                char_subset=[self.fighter_name],
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
                char_subset=[self.fighter_name],
                anim_subset=self.actions,
                **self.dataset_kwargs,
            )
        return DataLoader(self.smash_action_dataset_test, batch_size=self.batch_size, num_workers=1)
