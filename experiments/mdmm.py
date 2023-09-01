"""Base class for MDMM based pytorch lightning modules.
"""

# Ignore unused arguments because of save_hyperparameters
# ruff: noqa: ARG002

from typing import Any, Union

import torch
from torch import Tensor
from torch import Tensor as T
import torch.nn as nn
from torch_geometric.data import Data

from gnn_tracking.training.base import TrackingModule
from gnn_tracking.utils.dictionaries import add_key_suffix, to_floats
from gnn_tracking.utils.lightning import obj_from_or_to_hparams
from gnn_tracking.utils.oom import tolerate_some_oom_errors
from gnn_tracking.metrics.losses import (
    LossFctType,
    loss_weight_type,
    unpack_loss_returns,
)

constraint_hparams = dict[str, Union[float, str]]
loss_constraint_type = Union[constraint_hparams, dict[str, constraint_hparams]]

class MDMMModule(TrackingModule):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        *,
        lr_params,
        lr_lambda,
        loss_fcts: dict[str, LossFctType],
        main_losses: dict[str, loss_weight_type],
        constraint_losses: dict[str, loss_constraint_type],
        **kwargs,
    ):
        """Pytorch lightning module with training and validation step for the metric
        learning approach to graph construction.

        Args:
            lr_params: The learning rate for the model paramers and the slack variables (to satisfy the
                inequality constraints).
            lr_lambda: The learning rate for the Lagrange multiplier (has to be a negative floating point number).
            loss_fcts: Dictionary of loss functions keyed by loss name.
                For example: {"embedding_loss": GraphConstructionHingeEmbeddingLoss}
            main_losses: Dictionary of main loss functions and their weights, keyed by loss name.
                The weights can be specified as-
                1. float: If the loss function has only one component.
                2. dict: If there are several components, the weight(s) for any component(s) can be specified in a
                         dictionary, keyed by component name.
                         For example: {"embedding_loss": {"attractive":1.0}}
            constraint_losses: Dictionary of constraint loss functions and their respective hyperparameters
                (type, weights, epsilons, and damping factors) keyed by loss name.
                The hyperparameters for a particular constrained loss are to be specified as a dictionary
                consisting of 4 key-value pairs:
                1. type (string): The constraint type can be "equal", "min", or "max".
                2. epsilon (float): The constraint function value. MDMM ensures that the constraint loss converges
                       to this value.
                3. damping_factor (float): A quadratic damping term is also added to ensure smooth convergence.
                4. weight (float): Scales the entire loss term.
                If a constraint loss has more than one component, the hyperparameters for any component(s)
                can be specified in a dictionary, keyed by component name.
                For example: {
                    "embedding_loss": {"repulsive": {"type":"equal",
                                                     "weight":1.0,
                                                     "epsilon":10.0,
                                                     "damping_factor":1.0}}
                    }
        """
        super().__init__(**kwargs)

        assert lr_lambda < 0, "Learning rate for Lagrange multipliers must be negative."

        self.lr_params = lr_params
        self.lr_lambda = lr_lambda

        main_losses_hparams = {}
        constraint_losses_hparams = {}

        for key, val in main_losses.items():
            main_losses_hparams.update(unpack_loss_returns(key, val))
        
        for key, val in constraint_losses.items():
            constraint_losses_hparams.update(unpack_loss_returns(key, val))

        self.save_hyperparameters({"main_losses_hparams": main_losses_hparams,
                                   "constraint_losses_hparams": constraint_losses_hparams}
                                )
        
         # Maps the constraint loss functions to the index of their respective Lagrangian multipliers.
        self.l_multipliers_mapper = {}

        # Maps the constraint loss functions to the index of their respective slack variables.
        self.max_slack_mapper = {}
        self.min_slack_mapper = {}

        max_constraint = 0
        min_constraint = 0
        for n_constraint, (key, val) in enumerate(constraint_losses_hparams.items()):
            self.l_multipliers_mapper[key] = n_constraint
            if val["type"] == "max":
                self.max_slack_mapper[key] = int(max_constraint)
                max_constraint += 1
            if val["type"] == "min":
                self.min_slack_mapper[key] = int(min_constraint)
                min_constraint += 1
        
        # List of Lagrangian multipliers for the constraint loss functions.
        self.l_multipliers = [nn.Parameter((torch.tensor(0, dtype=torch.float32))) for _ in range(len(constraint_losses_hparams))]
        
        # List of slack variables for the constraint loss functions with a maximum constraint.
        self.max_slacks = [nn.Parameter(torch.as_tensor(float('nan'), device=self.device)) for _ in range(max_constraint)]
        
        # List of slack variables for the constraint loss functions with a minimum constraint.
        self.min_slacks = [nn.Parameter(torch.as_tensor(float('nan'), device=self.device)) for _ in range(min_constraint)]
        
        self.loss_fcts = obj_from_or_to_hparams(
            self, "loss_fcts", loss_fcts
        )

    # noinspection PyUnusedLocal
    def get_losses(self, out: dict[str, Any], data: Data) -> tuple[T, dict[str, float]]:
        if not hasattr(data, "true_edge_index"):
            # For the point cloud data, we unfortunately saved the true edges
            # simply as edge_index.
            data.true_edge_index = data.edge_index

        loss_dct = {}

        for key, loss_func in self.loss_fcts.items():
            loss_dct.update(unpack_loss_returns(key, loss_func(
                                                        x=out["H"],
                                                        particle_id=data.particle_id,
                                                        batch=data.batch,
                                                        true_edge_index=data.true_edge_index,
                                                        pt=data.pt,
                                                    )))

        lws_main = {}
        for key, weight in self.hparams.main_losses_hparams.items():
            lws_main[key] = weight

        constraints = {}
        for key, val in self.hparams.constraint_losses_hparams.items():
            constraints[key] = val

        loss = sum(loss_dct[k] * v for k, v in lws_main.items())

        for k, v in constraints.items():
            infeasibility = loss_dct[k] - v["epsilon"] # infeasibility = g(x) - ε

            if v["type"]=="max":

                if self.max_slacks[self.max_slack_mapper[k]].isnan():
                    with torch.no_grad():
                        self.max_slacks[self.max_slack_mapper[k]].copy_((v["epsilon"]-loss_dct[k]).relu().pow(1/2))

                # g(x) <= ε
                # => g(x) - ε <= 0
                # => g(x) - ε + slack^2 = 0
                infeasibility += self.max_slacks[self.max_slack_mapper[k]]**2

            if v["type"]=="min":

                if self.min_slacks[self.min_slack_mapper[k]].isnan():
                    with torch.no_grad():
                        self.min_slacks[self.min_slack_mapper[k]].copy_((loss_dct[k]-v["epsilon"]).relu().pow(1/2))

                # g(x) >= ε
                # => g(x) - ε >= 0
                # => g(x) - ε - slack^2 = 0
                infeasibility -= self.min_slacks[self.min_slack_mapper[k]]**2

            l_term = self.l_multipliers[self.l_multipliers_mapper[k]] * infeasibility # Lagrangian term = λ*infeasibility

            # Quadratic damping term to reduce oscillations.
            damp = v["damping_factor"] * (infeasibility**2 / 2) # c*(infeasibility^2 / 2)
            loss += v["weight"]*(l_term + damp)

        loss_dct["total"] = loss
        return loss, to_floats(loss_dct)

    @tolerate_some_oom_errors
    def training_step(self, batch: Data, batch_idx: int) -> Tensor | None:
        out = self(batch)
        loss, loss_dct = self.get_losses(out, batch)
        self.log_dict(
            add_key_suffix(loss_dct, "_train"),
            prog_bar=True,
            on_step=True,
            batch_size=self.trainer.train_dataloader.batch_size,
        )
        return loss
    
    def configure_optimizers(self) -> Any:
        # 'params' is a list of all the parameters- model parameters, Lagrange multipliers, and slack variables.
        params = [{'params':self.model.parameters(), 'lr':self.lr_params},]
        
        if len(self.l_multipliers):
            params.append({'params':self.l_multipliers, 'lr':self.lr_lambda})
        if len(self.max_slacks):
            params.append({'params':self.max_slacks, 'lr':self.lr_params})
        if len(self.min_slacks):
            params.append({'params':self.min_slacks, 'lr':self.lr_params})
        
        optimizer = self.optimizer(params)

        scheduler = self.scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
