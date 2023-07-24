from __future__ import annotations

import wandb
from gnn_tracking.utils.loading import TrackingDataModule
from gnn_tracking.utils.nomenclature import random_trial_name
from gnn_tracking.training.callbacks import PrintValidationMetrics

from pytorch_lightning.callbacks import TQDMProgressBar, RichProgressBar
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from wandb_osh.lightning_hooks import TriggerWandbSyncLightningCallback

name = random_trial_name()


logger = WandbLogger(
    project="mdmm",
    group="graph-construction",
    offline=True,
    version=name,
)

tb_logger = TensorBoardLogger(".", version=name)


def cli_main():
    # noinspection PyUnusedLocal
    cli = LightningCLI(  # noqa: F841
        datamodule_class=TrackingDataModule,
        trainer_defaults={
            "callbacks": [
                PrintValidationMetrics(),
                TQDMProgressBar(refresh_rate=5),
                TriggerWandbSyncLightningCallback(),
            ],
            "logger": [tb_logger, logger],
            "plugins": [SLURMEnvironment()],
        },
    )

if __name__ == "__main__":
    cli_main()
