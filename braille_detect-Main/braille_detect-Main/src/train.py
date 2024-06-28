from __future__ import annotations
import shutil     #use for high level management
import warnings   #ignore warnings during the execution

from pytorch_lightning import Trainer   #Main class for training models in PyTorch Lightning.
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint   #save model check points
from pytorch_lightning.loggers import WandbLogger

from dataset import BrailleDataModule
from model import BrailleTagger


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    datamodule = BrailleDataModule()   #handle the model

    checkpoint = ModelCheckpoint(
        monitor="val/accuracy", mode="max", save_weights_only=True, save_top_k=1
    )

    trainer = Trainer(   
        accelerator="cpu",   #before here use gpu with [0] devices
        devices=1,
        precision=16,    #16 bit floating points
        #amp_backend="native",
        max_steps=6000,
        log_every_n_steps=40,
        val_check_interval=40,
        logger=WandbLogger("multilabel-braille", project="multilabel-braille"),   #project name wiith loggiing wandb
        callbacks=[checkpoint, LearningRateMonitor("step")],
    )

    trainer.fit((BrailleTagger(n_training_steps=6000, n_warmup_steps=600)), datamodule)

    trainer.test(
        model=BrailleTagger(),
        ckpt_path=checkpoint.best_model_path,
        datamodule=datamodule,   #same data module
    )
