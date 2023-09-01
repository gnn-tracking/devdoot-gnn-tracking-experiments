"""Pytorch lightning module with training and validation step for the metric learning
approach to graph construction using MDMM.
"""

# Ignore unused arguments because of save_hyperparameters
# ruff: noqa: ARG002

from typing import Any

from torch_geometric.data import Data

from gnn_tracking.graph_construction.k_scanner import GraphConstructionKNNScanner
from mdmm import MDMMModule
from gnn_tracking.metrics.losses import GraphConstructionHingeEmbeddingLoss
from gnn_tracking.training.base import TrackingModule
from gnn_tracking.utils.dictionaries import add_key_suffix, to_floats
from gnn_tracking.utils.lightning import obj_from_or_to_hparams
from gnn_tracking.utils.oom import tolerate_some_oom_errors


class MDMMMLModule(MDMMModule):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        *,
        loss_fcts= {"embedding_loss": GraphConstructionHingeEmbeddingLoss},
        main_losses = {"embedding_loss": {"attractive":1.0}},
        constraint_losses = {"embedding_loss": {"repulsive": {"type":"equal",
                                                              "weight":1.0,
                                                              "epsilon":10.0,
                                                              "damping_factor":1.0}}},
        gc_scanner: GraphConstructionKNNScanner | None = None,
        **kwargs,
    ):
        super().__init__(loss_fcts=loss_fcts, main_losses=main_losses, constraint_losses=constraint_losses, **kwargs)

        self.gc_scanner = obj_from_or_to_hparams(self, "gc_scanner", gc_scanner)

    def validation_step(self, batch: Data, batch_idx: int):
        out = self(batch)
        loss, loss_dct = self.get_losses(out, batch)
        self.log_dict_with_errors(
            loss_dct, batch_size=self.trainer.val_dataloaders.batch_size
        )
        if self.gc_scanner is not None:
            self.gc_scanner(batch, batch_idx, latent=out["H"])
    
    def on_validation_epoch_end(self) -> None:
        if self.gc_scanner is not None:
            self.log_dict(
                self.gc_scanner.get_foms(),
                on_step=False,
                on_epoch=True,
                batch_size=self.trainer.val_dataloaders.batch_size,
            )

    def highlight_metric(self, metric: str) -> bool:
        return metric in [
            "n_edges_frac_segment50_95",
            "total",
            "attractive",
            "repulsive",
            "max_frac_segment50",
        ]
