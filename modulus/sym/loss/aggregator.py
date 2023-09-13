import paddle
import logging
import numpy as np
from typing import Dict, List, Optional, Callable, Union
from modulus.sym.eq.derivatives import gradient
from modulus.sym.hydra import to_absolute_path, add_hydra_run_path
logger = logging.getLogger(__name__)


class Aggregator(paddle.nn.Layer):
    """
    Base class for loss aggregators
    """

    def __init__(self, params, num_losses, weights):
        super().__init__()
        self.params: List[paddle.Tensor] = list(params)
        self.num_losses: int = num_losses
        self.weights: Optional[Dict[str, float]] = weights
        self.place: str
        self.place = list(set(p.place for p in self.params))[0]
        self.init_loss: paddle.Tensor = paddle.to_tensor(data=0.0, place=
            self.place)

        def weigh_losses_initialize(weights: Optional[Dict[str, float]]
            ) ->Callable[[Dict[str, paddle.Tensor], Optional[Dict[str,
            float]]], Dict[str, paddle.Tensor]]:
            if weights is None:

                def weigh_losses(losses: Dict[str, paddle.Tensor], weights:
                    None) ->Dict[str, paddle.Tensor]:
                    return losses
            else:

                def weigh_losses(losses: Dict[str, paddle.Tensor], weights:
                    Dict[str, float]) ->Dict[str, paddle.Tensor]:
                    for key in losses.keys():
                        if key not in weights.keys():
                            weights.update({key: 1.0})
                    losses = {key: (weights[key] * losses[key]) for key in
                        losses.keys()}
                    return losses
            return weigh_losses
        self.weigh_losses = weigh_losses_initialize(self.weights)


class Sum(Aggregator):
    """
    Loss aggregation by summation
    """

    def __init__(self, params, num_losses, weights=None):
        super().__init__(params, num_losses, weights)

    def forward(self, losses: Dict[str, paddle.Tensor], step: int
        ) ->paddle.Tensor:
        """
        Aggregates the losses by summation

        Parameters
        ----------
        losses : Dict[str, torch.Tensor]
            A dictionary of losses.
        step : int
            Optimizer step.

        Returns
        -------
        loss : torch.Tensor
            Aggregated loss.
        """
        losses = self.weigh_losses(losses, self.weights)
        loss: paddle.Tensor = paddle.zeros_like(x=self.init_loss)
        for key in losses.keys():
            loss += losses[key]
        return loss


class GradNorm(Aggregator):
    """
    GradNorm for loss aggregation
    Reference: "Chen, Z., Badrinarayanan, V., Lee, C.Y. and Rabinovich, A., 2018, July.
    Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks.
    In International Conference on Machine Learning (pp. 794-803). PMLR."
    """

    def __init__(self, params, num_losses, alpha=1.0, weights=None):
        super().__init__(params, num_losses, weights)
        self.alpha: float = alpha
        out_93 = paddle.create_parameter(shape=paddle.zeros(shape=
            num_losses).shape, dtype=paddle.zeros(shape=num_losses).numpy()
            .dtype, default_initializer=paddle.nn.initializer.Assign(paddle
            .zeros(shape=num_losses)))
        out_93.stop_gradient = not True
        self.lmbda: paddle.Tensor = out_93
        self.register_buffer(name='init_losses', tensor=paddle.zeros(shape=
            self.num_losses))

    def forward(self, losses: Dict[str, paddle.Tensor], step: int
        ) ->paddle.Tensor:
        """
        Weights and aggregates the losses using the gradNorm algorithm

        Parameters
        ----------
        losses : Dict[str, torch.Tensor]
            A dictionary of losses.
        step : int
            Optimizer step.

        Returns
        -------
        loss : torch.Tensor
            Aggregated loss.
        """
        losses = self.weigh_losses(losses, self.weights)
        if step == 0:
            for i, key in enumerate(losses.keys()):
                self.init_losses[i] = losses[key].clone().detach()
        with paddle.no_grad():
            normalizer: paddle.Tensor = self.num_losses / paddle.exp(x=self
                .lmbda).sum()
            for i in range(self.num_losses):
                self.lmbda[i] = self.lmbda[i].clone() + paddle.log(x=
                    normalizer.detach())
        lmbda_exp: paddle.Tensor = paddle.exp(x=self.lmbda)
        losses_stacked: paddle.Tensor = paddle.stack(x=list(losses.values()))
        with paddle.no_grad():
            relative_losses: paddle.Tensor = paddle.divide(x=losses_stacked,
                y=paddle.to_tensor(self.init_losses))
            inverse_rate: paddle.Tensor = (relative_losses /
                relative_losses.mean())
            gradnorm_coef: paddle.Tensor = paddle.pow(x=inverse_rate, y=
                self.alpha)
        grads_norm: paddle.Tensor = paddle.zeros_like(x=self.init_losses)
        shared_params: paddle.Tensor = self.params[-2]
        for i, key in enumerate(losses.keys()):
            grads: paddle.Tensor = gradient(losses[key], [shared_params])[0]
            grads_norm[i] = paddle.linalg.norm(x=lmbda_exp[i] * grads.
                detach(), p=2)
        avg_grad: paddle.Tensor = grads_norm.detach().mean()
        loss_gradnorm: paddle.Tensor = paddle.abs(x=grads_norm - avg_grad *
            gradnorm_coef).sum()
        loss_model: paddle.Tensor = (lmbda_exp.detach() * losses_stacked).sum()
        loss: paddle.Tensor = loss_gradnorm + loss_model
        return loss


class ResNorm(Aggregator):
    """
    Residual normalization for loss aggregation
    Contributors: T. Nandi, D. Van Essendelft, M. A. Nabian
    """

    def __init__(self, params, num_losses, alpha=1.0, weights=None):
        super().__init__(params, num_losses, weights)
        self.alpha: float = alpha
        out_94 = paddle.create_parameter(shape=paddle.zeros(shape=
            num_losses).shape, dtype=paddle.zeros(shape=num_losses).numpy()
            .dtype, default_initializer=paddle.nn.initializer.Assign(paddle
            .zeros(shape=num_losses)))
        out_94.stop_gradient = not True
        self.lmbda: paddle.Tensor = out_94
        self.register_buffer(name='init_losses', tensor=paddle.zeros(shape=
            self.num_losses))

    def forward(self, losses: Dict[str, paddle.Tensor], step: int
        ) ->paddle.Tensor:
        """
        Weights and aggregates the losses using the ResNorm algorithm

        Parameters
        ----------
        losses : Dict[str, torch.Tensor]
            A dictionary of losses.
        step : int
            Optimizer step.

        Returns
        -------
        loss : torch.Tensor
            Aggregated loss.
        """
        losses = self.weigh_losses(losses, self.weights)
        if step == 0:
            for i, key in enumerate(losses.keys()):
                self.init_losses[i] = losses[key].clone().detach()
        with paddle.no_grad():
            normalizer: paddle.Tensor = self.num_losses / paddle.exp(x=self
                .lmbda).sum()
            for i in range(self.num_losses):
                self.lmbda[i] = self.lmbda[i].clone() + paddle.log(x=
                    normalizer.detach())
        lmbda_exp: paddle.Tensor = paddle.exp(x=self.lmbda)
        losses_stacked: paddle.Tensor = paddle.stack(x=list(losses.values()))
        with paddle.no_grad():
            relative_losses: paddle.Tensor = paddle.divide(x=losses_stacked,
                y=paddle.to_tensor(self.init_losses))
            inverse_rate: paddle.Tensor = (relative_losses /
                relative_losses.mean())
            resnorm_coef: paddle.Tensor = paddle.pow(x=inverse_rate, y=self
                .alpha)
        residuals: paddle.Tensor = paddle.zeros_like(x=self.init_losses)
        for i, key in enumerate(losses.keys()):
            residuals[i] = lmbda_exp[i] * losses[key].detach()
        avg_residuals: paddle.Tensor = losses_stacked.detach().mean()
        loss_resnorm: paddle.Tensor = paddle.abs(x=residuals - 
            avg_residuals * resnorm_coef).sum()
        loss_model: paddle.Tensor = (lmbda_exp.detach() * losses_stacked).sum()
        loss: paddle.Tensor = loss_resnorm + loss_model
        return loss


class HomoscedasticUncertainty(Aggregator):
    """
    Homoscedastic task uncertainty for loss aggregation
    Reference: "Reference: Kendall, A., Gal, Y. and Cipolla, R., 2018.
    Multi-task learning using uncertainty to weigh losses for scene geometry and semantics.
    In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7482-7491)."
    """

    def __init__(self, params, num_losses, weights=None):
        super().__init__(params, num_losses, weights)
        out_95 = paddle.create_parameter(shape=paddle.zeros(shape=self.
            num_losses).shape, dtype=paddle.zeros(shape=self.num_losses).
            numpy().dtype, default_initializer=paddle.nn.initializer.Assign
            (paddle.zeros(shape=self.num_losses)))
        out_95.stop_gradient = not True
        self.log_var: paddle.Tensor = out_95

    def forward(self, losses: Dict[str, paddle.Tensor], step: int
        ) ->paddle.Tensor:
        """
        Weights and aggregates the losses using homoscedastic task uncertainty

        Parameters
        ----------
        losses : Dict[str, torch.Tensor]
            A dictionary of losses.
        step : int
            Optimizer step.

        Returns
        -------
        loss : torch.Tensor
            Aggregated loss.
        """
        losses = self.weigh_losses(losses, self.weights)
        loss: paddle.Tensor = paddle.zeros_like(x=self.init_loss)
        precision: paddle.Tensor = paddle.exp(x=-self.log_var)
        for i, key in enumerate(losses.keys()):
            loss += precision[i] * losses[key]
        loss += self.log_var.sum()
        loss /= 2.0
        return loss


class LRAnnealing(Aggregator):
    """
    Learning rate annealing for loss aggregation
    References: "Wang, S., Teng, Y. and Perdikaris, P., 2020.
    Understanding and mitigating gradient pathologies in physics-informed
    neural networks. arXiv preprint arXiv:2001.04536.", and
    "Jin, X., Cai, S., Li, H. and Karniadakis, G.E., 2021.
    NSFnets (Navier-Stokes flow nets): Physics-informed neural networks for the
    incompressible Navier-Stokes equations. Journal of Computational Physics, 426, p.109951."
    """

    def __init__(self, params, num_losses, update_freq=1, alpha=0.01,
        ref_key=None, eps=1e-08, weights=None):
        super().__init__(params, num_losses, weights)
        self.update_freq: int = update_freq
        self.alpha: float = alpha
        self.ref_key: Union[str, None] = ref_key
        self.eps: float = eps
        self.register_buffer(name='lmbda_ema', tensor=paddle.ones(shape=
            self.num_losses))

    def forward(self, losses: Dict[str, paddle.Tensor], step: int
        ) ->paddle.Tensor:
        """
        Weights and aggregates the losses using the learning rate annealing algorithm

        Parameters
        ----------
        losses : Dict[str, torch.Tensor]
            A dictionary of losses.
        step : int
            Optimizer step.

        Returns
        -------
        loss : torch.Tensor
            Aggregated loss.
        """
        losses = self.weigh_losses(losses, self.weights)
        loss: paddle.Tensor = paddle.zeros_like(x=self.init_loss)
        if self.ref_key is None:
            ref_idx = 0
        else:
            for i, key in enumerate(losses.keys()):
                if self.ref_key in key:
                    ref_idx = i
                    break
        if step % self.update_freq == 0:
            grads_mean: List[paddle.Tensor] = []
            for key in losses.keys():
                grads: List[paddle.Tensor] = gradient(losses[key], self.params)
                grads_flattened: List[paddle.Tensor] = []
                for i in range(len(grads)):
                    if grads[i] is not None:
                        grads_flattened.append(paddle.abs(x=paddle.flatten(
                            x=grads[i])))
                grads_mean.append(paddle.mean(x=paddle.concat(x=
                    grads_flattened)))
            for i, key in enumerate(losses.keys()):
                with paddle.no_grad():
                    self.lmbda_ema[i] *= 1.0 - self.alpha
                    self.lmbda_ema[i] += self.alpha * grads_mean[ref_idx] / (
                        grads_mean[i] + self.eps)
                loss += self.lmbda_ema[i].clone() * losses[key]
        else:
            for i, key in enumerate(losses.keys()):
                loss += self.lmbda_ema[i] * losses[key]
        return loss


class SoftAdapt(Aggregator):
    """
    SoftAdapt for loss aggregation
    Reference: "Heydari, A.A., Thompson, C.A. and Mehmood, A., 2019.
    Softadapt: Techniques for adaptive loss weighting of neural networks with multi-part loss functions.
    arXiv preprint arXiv: 1912.12355."
    """

    def __init__(self, params, num_losses, eps=1e-08, weights=None):
        super().__init__(params, num_losses, weights)
        self.eps: float = eps
        self.register_buffer(name='prev_losses', tensor=paddle.zeros(shape=
            self.num_losses))

    def forward(self, losses: Dict[str, paddle.Tensor], step: int
        ) ->paddle.Tensor:
        """
        Weights and aggregates the losses using the original variant of the softadapt algorithm

        Parameters
        ----------
        losses : Dict[str, torch.Tensor]
            A dictionary of losses.
        step : int
            Optimizer step.

        Returns
        -------
        loss : torch.Tensor
            Aggregated loss.
        """
        losses = self.weigh_losses(losses, self.weights)
        loss: paddle.Tensor = paddle.zeros_like(x=self.init_loss)
        if step == 0:
            for i, key in enumerate(losses.keys()):
                loss += losses[key]
                self.prev_losses[i] = losses[key].clone().detach()
        else:
            lmbda: paddle.Tensor = paddle.ones_like(x=self.prev_losses)
            lmbda_sum: paddle.Tensor = paddle.zeros_like(x=self.init_loss)
            losses_stacked: paddle.Tensor = paddle.stack(x=list(losses.
                values()))
            normalizer: paddle.Tensor = (losses_stacked / self.prev_losses
                ).max()
            for i, key in enumerate(losses.keys()):
                with paddle.no_grad():
                    lmbda[i] = paddle.exp(x=losses[key] / (self.prev_losses
                        [i] + self.eps) - normalizer)
                    lmbda_sum += lmbda[i]
                loss += lmbda[i].clone() * losses[key]
                self.prev_losses[i] = losses[key].clone().detach()
            loss *= self.num_losses / (lmbda_sum + self.eps)
        return loss


class Relobralo(Aggregator):
    """
    Relative loss balancing with random lookback
    Reference: "Bischof, R. and Kraus, M., 2021.
    Multi-Objective Loss Balancing for Physics-Informed Deep Learning.
    arXiv preprint arXiv:2110.09813."
    """

    def __init__(self, params, num_losses, alpha=0.95, beta=0.99, tau=1.0,
        eps=1e-08, weights=None):
        super().__init__(params, num_losses, weights)
        self.alpha: float = alpha
        self.beta: float = beta
        self.tau: float = tau
        self.eps: float = eps
        self.register_buffer(name='init_losses', tensor=paddle.zeros(shape=
            self.num_losses))
        self.register_buffer(name='prev_losses', tensor=paddle.zeros(shape=
            self.num_losses))
        self.register_buffer(name='lmbda_ema', tensor=paddle.ones(shape=
            self.num_losses))

    def forward(self, losses: Dict[str, paddle.Tensor], step: int
        ) ->paddle.Tensor:
        """
        Weights and aggregates the losses using the ReLoBRaLo algorithm

        Parameters
        ----------
        losses : Dict[str, torch.Tensor]
            A dictionary of losses.
        step : int
            Optimizer step.

        Returns
        -------
        loss : torch.Tensor
            Aggregated loss.
        """
        losses = self.weigh_losses(losses, self.weights)
        loss: paddle.Tensor = paddle.zeros_like(x=self.init_loss)
        if step == 0:
            for i, key in enumerate(losses.keys()):
                loss += losses[key]
                self.init_losses[i] = losses[key].clone().detach()
                self.prev_losses[i] = losses[key].clone().detach()
        else:
            losses_stacked: paddle.Tensor = paddle.stack(x=list(losses.
                values()))
            normalizer_prev: paddle.Tensor = (losses_stacked / (self.tau *
                self.prev_losses)).max()
            normalizer_init: paddle.Tensor = (losses_stacked / (self.tau *
                self.init_losses)).max()
            rho: paddle.Tensor = paddle.bernoulli(x=paddle.to_tensor(data=
                self.beta))
            with paddle.no_grad():
                lmbda_prev: paddle.Tensor = paddle.exp(x=losses_stacked / (
                    self.tau * self.prev_losses + self.eps) - normalizer_prev)
                lmbda_init: paddle.Tensor = paddle.exp(x=losses_stacked / (
                    self.tau * self.init_losses + self.eps) - normalizer_init)
                lmbda_prev *= self.num_losses / (lmbda_prev.sum() + self.eps)
                lmbda_init *= self.num_losses / (lmbda_init.sum() + self.eps)
            for i, key in enumerate(losses.keys()):
                with paddle.no_grad():
                    self.lmbda_ema[i] = self.alpha * (rho * self.lmbda_ema[
                        i].clone() + (1.0 - rho) * lmbda_init[i])
                    self.lmbda_ema[i] += (1.0 - self.alpha) * lmbda_prev[i]
                loss += self.lmbda_ema[i].clone() * losses[key]
                self.prev_losses[i] = losses[key].clone().detach()
        return loss


class NTK(paddle.nn.Layer):

    def __init__(self, run_per_step: int=1000, save_name: Union[str, None]=None
        ):
        super(NTK, self).__init__()
        self.run_per_step = run_per_step
        self.if_csv_head = True
        self.save_name = to_absolute_path(add_hydra_run_path(save_name)
            ) if save_name else None
        if self.save_name:
            logger.warning(
                'Cuda graphs does not work when saving NTK values to file! Set `cuda_graphs` to false.'
                )

    def group_ntk(self, model, losses):
        ntk_value = dict()
        for key, loss in losses.items():
            grad = paddle.grad(outputs=paddle.sqrt(x=paddle.abs(x=loss)),
                inputs=model.parameters(), retain_graph=True, allow_unused=True
                )
            ntk_value[key] = paddle.sqrt(x=paddle.sum(x=paddle.stack(x=[
                paddle.sum(x=t.detach() ** 2) for t in grad if t is not
                None], axis=0)))
        return ntk_value

    def save_ntk(self, ntk_dict, step):
        import pandas as pd
        output_dict = {}
        for key, value in ntk_dict.items():
            output_dict[key] = value.cpu().numpy()
        df = pd.DataFrame(output_dict, index=[step])
        df.to_csv(self.save_name + '.csv', mode='a', header=self.if_csv_head)
        self.if_csv_head = False

    def forward(self, constraints, ntk_weights, step):
        losses = dict()
        dict_constraint_losses = dict()
        ntk_sum = 0
        for key, constraint in constraints.items():
            paddle.framework.core.nvprof_nvtx_push(f'Running Constraint {key}')
            constraint.forward()
            paddle.framework.core.nvprof_nvtx_pop()
        for key, constraint in constraints.items():
            constraint_losses = constraint.loss(step)
            if step % self.run_per_step == 0 and step > 0:
                ntk_dict = self.group_ntk(constraint.model, constraint_losses)
            else:
                ntk_dict = None
            if ntk_dict is not None:
                ntk_weights[key] = ntk_dict
            if ntk_weights.get(key) is not None:
                ntk_sum += paddle.sum(x=paddle.stack(x=list(ntk_weights[key
                    ].values()), axis=0))
            dict_constraint_losses[key] = constraint_losses
        if step == 0:
            ntk_sum = 1.0
        if self.save_name and step % self.run_per_step == 0 and step > 0:
            self.save_ntk({(d_key + '_' + k): v for d_key, d in ntk_weights
                .items() for k, v in d.items()}, step)
        for key, constraint_losses in dict_constraint_losses.items():
            for loss_key, value in constraint_losses.items():
                if ntk_weights.get(key) is None or ntk_weights[key].get(
                    loss_key) is None:
                    ntk_weight = ntk_sum / 1.0
                else:
                    ntk_weight = ntk_sum / ntk_weights[key][loss_key]
                if loss_key not in list(losses.keys()):
                    losses[loss_key] = ntk_weight * value
                else:
                    losses[loss_key] += ntk_weight * value
        return losses, ntk_weights
