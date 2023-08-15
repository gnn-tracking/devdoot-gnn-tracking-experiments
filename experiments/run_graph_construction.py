from __future__ import annotations

import wandb
from gnn_tracking.utils.loading import TrackingDataModule
from gnn_tracking.utils.nomenclature import random_trial_name
from gnn_tracking.training.callbacks import PrintValidationMetrics, ExpandWandbConfig

from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint, RichProgressBar
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from wandb_osh.lightning_hooks import TriggerWandbSyncLightningCallback


name = random_trial_name()

early_stopping_callback = EarlyStopping(
    monitor='n_edges_frac_segment50_95',
    patience=20,
    mode='min',
    check_finite=False
)

checkpoint_callback = ModelCheckpoint(
    filename='{epoch}-{n_edges_frac_segment50_95:.2f}',
    monitor='n_edges_frac_segment50_95',
    mode='min',
    save_top_k=1,
    save_last=True
)

logger = WandbLogger(
    project="mdmm",
    group="graph-construction-k=(1-10)",
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
                early_stopping_callback,
                checkpoint_callback,
                PrintValidationMetrics(),
                TQDMProgressBar(refresh_rate=5),
                TriggerWandbSyncLightningCallback(),
                ExpandWandbConfig()
            ],
            "logger": [tb_logger, logger],
            "plugins": [SLURMEnvironment()],
        },
    )

if __name__ == "__main__":
    cli_main()
