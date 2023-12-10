import torch
import click
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import RichModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

from playaid.models.resnet_transformer_detector import ResnetTransformerDetector
from playaid.anim_ontology import MOVE_TO_CLASS_ID

# https://github.com/pytorch/pytorch/issues/11201#issuecomment-421146936
torch.multiprocessing.set_sharing_strategy("file_system")


@click.command()
@click.option("--ckpt", default=None, help="initial weights path")
@click.option("--fighters", "-f", multiple=True, help="fighter(s) names")
@click.option("--batch_size", default=8, type=int, help="batch size")
@click.option("--num_epochs", default=1000, type=int, help="num epochs")
@click.option("--name", default=None, help="name of the run")
@click.option("--num_samples", default=1024, help="simulated number of items in the dataset")
@click.option("--num_frames_per_sample", default=7, help="number of frames per sample")
@click.option("--frame_delta", default=[1, 2, 3, 4, 5, 6], multiple=True, help="ƒrame delta")
@click.option("--skip_wandb", is_flag=True, help="skip wandb logging")
def train(
    ckpt,
    fighters,
    batch_size,
    num_epochs,
    name,
    num_samples,
    num_frames_per_sample,
    frame_delta,
    skip_wandb,
):
    char_subset = list(fighters)
    print("char_subset: ", char_subset)
    # Roughly 60 actions currently.
    actions = list(MOVE_TO_CLASS_ID.keys())
    batch_size = batch_size
    model_args = {
        "char_subset": char_subset,
        "actions": actions,
        "batch_size": batch_size,
        "learning_rate": 3e-4,
        # this model only allows for one.
        "num_frames_per_sample": [num_frames_per_sample],
        "sequence_length": num_frames_per_sample,
        "frame_delta": list(frame_delta),
        "num_samples": num_samples,
    }
    if ckpt:
        print("Going to try and load from ckpt")
        model = ResnetTransformerDetector.load_from_checkpoint(ckpt, **model_args)
    else:
        model = ResnetTransformerDetector(**model_args)

    loggers = [TensorBoardLogger(save_dir="logs", name=f"action_recog/{name}")]

    if not skip_wandb:
        wandb_logger = WandbLogger(project="action-recognition", name=name)
        loggers.append(wandb_logger)
        # log gradients and model topology
        wandb_logger.watch(model)

    trainer = Trainer(
        accelerator="auto",
        log_every_n_steps=1,
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=num_epochs,
        callbacks=[TQDMProgressBar(refresh_rate=20), RichModelSummary()],
        logger=loggers,
    )

    trainer.fit(model)
    trainer.test()


if __name__ == "__main__":
    train()
