"""Pytorch lightning module with training and validation step for the metric learning
approach to graph construction.
"""

from typing import Any

import torch
from torch import Tensor
from torch import Tensor as T
import torch.nn as nn
from torch_geometric.data import Data

from gnn_tracking.metrics.losses import GraphConstructionHingeEmbeddingLoss
from gnn_tracking.training.base import TrackingModule
from gnn_tracking.utils.dictionaries import add_key_suffix, to_floats
from gnn_tracking.utils.lightning import obj_from_or_to_hparams
from gnn_tracking.utils.oom import tolerate_some_oom_errors


class MDMMMLModule(TrackingModule):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        *,
        lr_params,
        lr_lambda,
        loss_fct: GraphConstructionHingeEmbeddingLoss,
        main_losses = {"attractive":1.0},
        constraint_losses = {"repulsive": {"weight":1.0,
                                           "epsilon":0.0003,
                                           "damping_factor":1.0}},
        **kwargs,
    ):
        """Pytorch lightning module with training and validation step for the metric
        learning approach to graph construction.
        """
        super().__init__(**kwargs)

        assert lr_lambda < 0, "Learning rate for Lagrange multipliers must be negative."

        self.lr_params = lr_params
        self.lr_lambda = lr_lambda

        self.save_hyperparameters(
                                  "main_losses",
                                  "constraint_losses",
                                )
        
        self.l_multipliers_mapper = {}
        for n_constraint, (key, _) in enumerate(constraint_losses.items()):
            self.l_multipliers_mapper[key] = n_constraint
        
        self.l_multipliers = [nn.Parameter((torch.tensor(0, dtype=torch.float32))) for _ in range(len(constraint_losses))]
        self.loss_fct = obj_from_or_to_hparams(self, "loss_fct", loss_fct)

    # noinspection PyUnusedLocal
    def get_losses(self, out: dict[str, Any], data: Data) -> tuple[T, dict[str, float]]:
        loss_dct = self.loss_fct(
            x=out["H"],
            particle_id=data.particle_id,
            batch=data.batch,
            edge_index=data.edge_index,
            pt=data.pt,
        )

        lws_main = {}
        for key, weight in self.hparams.main_losses.items():
            lws_main[key] = weight

        constraints = {}
        for key, val in self.hparams.constraint_losses.items():
            constraints[key] = val

        loss = sum(loss_dct[k] * v for k, v in lws_main.items())

        for k, v in constraints.items():
            infeasibility = loss_dct[k] - v["epsilon"]
            l_term = self.l_multipliers[self.l_multipliers_mapper[k]] * infeasibility
            damp = v["damping_factor"] * (infeasibility**2 / 2)
            loss += v["weight"]*(l_term + damp)

        # loss_dct |= {f"{k}_weighted": v * lws[k] for k, v in loss_dct.items()}
        loss_dct["total"] = loss
        return loss, to_floats(loss_dct)

    @tolerate_some_oom_errors
    def training_step(self, batch: Data, batch_idx: int) -> Tensor | None:
        batch = self.data_preproc(batch)
        out = self(batch)
        loss, loss_dct = self.get_losses(out, batch)
        self.log_dict(add_key_suffix(loss_dct, "_train"), prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch: Data, batch_idx: int):
        batch = self.data_preproc(batch)
        out = self(batch)
        loss, loss_dct = self.get_losses(out, batch)
        self.log_dict_with_errors(
            loss_dct, batch_size=self.trainer.val_dataloaders.batch_size
        )
        # todo: add graph analysis
    
    def configure_optimizers(self) -> Any:
        optimizer = self.optimizer([{'params':self.model.parameters(), 'lr':self.lr_params},
                                    {'params':self.l_multipliers, 'lr':self.lr_lambda}])

        scheduler = self.scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
