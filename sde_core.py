# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SDE core used by ``sde.py``: types, tensor helpers, ForwardSDE adapters, Euler–Maruyama, and input checks."""

from __future__ import annotations

import abc
import warnings
from typing import Sequence, Union, Optional, Any, Dict, Tuple, Callable  # noqa: F401

import torch
from torch import nn


Tensor = torch.Tensor
Tensors = Sequence[Tensor]
TensorOrTensors = Union[Tensor, Tensors]

Scalar = Union[float, Tensor]
Vector = Union[Sequence[float], Tensor]

Module = torch.nn.Module
Modules = Sequence[Module]
ModuleOrModules = Union[Module, Modules]

Size = torch.Size
Sizes = Sequence[Size]



def assert_no_grad(names, maybe_tensors):
    for name, maybe_tensor in zip(names, maybe_tensors):
        if torch.is_tensor(maybe_tensor) and maybe_tensor.requires_grad:
            raise ValueError(f"Argument {name} must not require gradient.")


def handle_unused_kwargs(unused_kwargs, msg=None):
    if len(unused_kwargs) > 0:
        if msg is not None:
            warnings.warn(f"{msg}: Unexpected arguments {unused_kwargs}")
        else:
            warnings.warn(f"Unexpected arguments {unused_kwargs}")


def flatten(sequence):
    """Concatenate tensors in ``sequence`` into a 1-D tensor (empty → scalar empty)."""
    return torch.cat([p.reshape(-1) for p in sequence]) if len(sequence) > 0 else torch.tensor([])


def convert_none_to_zeros(sequence, like_sequence):
    return [torch.zeros_like(q) if p is None else p for p, q in zip(sequence, like_sequence)]


def make_seq_requires_grad(sequence):
    return [p if p.requires_grad else p.detach().requires_grad_(True) for p in sequence]


def is_strictly_increasing(ts):
    return all(x < y for x, y in zip(ts[:-1], ts[1:]))


def is_nan(t):
    return torch.any(torch.isnan(t))


def seq_add(*seqs):
    return [sum(seq) for seq in zip(*seqs)]


def seq_sub(xs, ys):
    return [x - y for x, y in zip(xs, ys)]


def batch_mvp(m, v):
    """Batched matrix–vector product: ``m`` is ``(B, D, M)``, ``v`` is ``(B, M)``."""
    return torch.bmm(m, v.unsqueeze(-1)).squeeze(dim=-1)


def stable_division(a, b, epsilon=1e-7):
    b = torch.where(b.abs().detach() > epsilon, b, torch.full_like(b, fill_value=epsilon) * b.sign())
    return a / b


def vjp(outputs, inputs, **kwargs):
    """Vector–Jacobian product; ``None`` gradients become zeros (PyTorch #39784 workaround)."""
    if torch.is_tensor(inputs):
        inputs = [inputs]
    _dummy_inputs = [torch.as_strided(i, (), ()) for i in inputs]  # Workaround for PyTorch bug #39784.  # noqa: 74

    if torch.is_tensor(outputs):
        outputs = [outputs]
    outputs = make_seq_requires_grad(outputs)

    _vjp = torch.autograd.grad(outputs, inputs, **kwargs)
    return convert_none_to_zeros(_vjp, inputs)


def jvp(outputs, inputs, grad_inputs=None, **kwargs):
    """Jacobian–vector product without repeating the forward map (unlike ``torch.autograd.functional.jvp``)."""
    if torch.is_tensor(inputs):
        inputs = [inputs]
    _dummy_inputs = [torch.as_strided(i, (), ()) for i in inputs]  # Workaround for PyTorch bug #39784.  # noqa: 88

    if torch.is_tensor(outputs):
        outputs = [outputs]
    outputs = make_seq_requires_grad(outputs)

    dummy_outputs = [torch.zeros_like(o, requires_grad=True) for o in outputs]
    _vjp = torch.autograd.grad(outputs, inputs, grad_outputs=dummy_outputs, create_graph=True, allow_unused=True)
    _vjp = make_seq_requires_grad(convert_none_to_zeros(_vjp, inputs))

    _jvp = torch.autograd.grad(_vjp, dummy_outputs, grad_outputs=grad_inputs, **kwargs)
    return convert_none_to_zeros(_jvp, dummy_outputs)


def flat_to_shape(flat_tensor, shapes):
    """Convert a flat tensor to a list of tensors with specified shapes.

    `flat_tensor` must have exactly the number of elements as stated in `shapes`.
    """
    numels = [shape.numel() for shape in shapes]
    return [flat.reshape(shape) for flat, shape in zip(flat_tensor.split(split_size=numels), shapes)]


def linear_interp(t0, y0, t1, y1, t):
    """Linear interpolation of ``y`` onto time ``t`` with ``t0 <= t <= t1``."""
    assert t0 <= t <= t1, f"Incorrect time order for linear interpolation: t0={t0}, t={t}, t1={t1}."
    y = (t1 - t) / (t1 - t0) * y0 + (t - t0) / (t1 - t0) * y1
    return y


class BaseSDE(abc.ABC, nn.Module):
    """Base class for all SDEs.

    Inheriting from this class ensures ``noise_type`` and ``sde_type`` are valid
    attributes, which the solver depends on.
    """

    def __init__(self, noise_type, sde_type):
        super(BaseSDE, self).__init__()
        # Making these Python properties breaks `torch.jit.script`.
        self.noise_type = noise_type
        self.sde_type = sde_type


class RenameMethodsSDE(BaseSDE):
    """Expose ``f`` / ``g`` / … as aliases of the names used on the wrapped SDE."""

    def __init__(self, sde, drift='f', diffusion='g', prior_drift='h', diffusion_prod='g_prod',
                 drift_and_diffusion='f_and_g', drift_and_diffusion_prod='f_and_g_prod'):
        super(RenameMethodsSDE, self).__init__(noise_type=sde.noise_type, sde_type=sde.sde_type)
        self._base_sde = sde
        for name, value in zip(('f', 'g', 'h', 'g_prod', 'f_and_g', 'f_and_g_prod'),
                               (drift, diffusion, prior_drift, diffusion_prod, drift_and_diffusion,
                                drift_and_diffusion_prod)):
            try:
                setattr(self, name, getattr(sde, value))
            except AttributeError:
                pass


class ForwardSDE(BaseSDE):
    """Normalize an SDE object so the Euler step can always call ``f_and_g_prod``."""

    def __init__(self, sde, fast_dg_ga_jvp_column_sum=False):
        super(ForwardSDE, self).__init__(sde_type=sde.sde_type, noise_type=sde.noise_type)
        self._base_sde = sde
        self.f_and_g_prod = self.f_and_g_prod_default2

        self.f = getattr(sde, 'f', self.f_default)
        self.g = getattr(sde, 'g', self.g_default)
        self.f_and_g = getattr(sde, 'f_and_g', self.f_and_g_default)
        self.g_prod = getattr(sde, 'g_prod', self.g_prod_default)
        self.prod = {
            'diagonal': self.prod_diagonal
        }.get(sde.noise_type, self.prod_default)

        self.g_prod_and_gdg_prod = {
            'diagonal': self.g_prod_and_gdg_prod_diagonal,
            'additive': self.g_prod_and_gdg_prod_additive,
        }.get(sde.noise_type, self.g_prod_and_gdg_prod_default)
        self.dg_ga_jvp_column_sum = {
            'general': (
                self.dg_ga_jvp_column_sum_v2 if fast_dg_ga_jvp_column_sum else self.dg_ga_jvp_column_sum_v1
            )
        }.get(sde.noise_type, self._return_zero)

    ########################################
    #                  f                   #
    ########################################
    def f_default(self, t, y):
        raise RuntimeError("Method `f` has not been provided, but is required for this method.")

    ########################################
    #                  g                   #
    ########################################
    def g_default(self, t, y):
        raise RuntimeError("Method `g` has not been provided, but is required for this method.")

    ########################################
    #               f_and_g                #
    ########################################

    def f_and_g_default(self, t, y):
        return self.f(t, y), self.g(t, y)

    ########################################
    #                prod                  #
    ########################################

    def prod_diagonal(self, g, v):
        """Elementwise ``g * v`` when diffusion is diagonal."""
        return g * v

    def prod_default(self, g, v):
        """Batched matrix–vector product ``g @ v``."""
        return batch_mvp(g, v)

    ########################################
    #                g_prod                #
    ########################################

    def g_prod_default(self, t, y, v):
        return self.prod(self.g(t, y), v)

    ########################################
    #             f_and_g_prod             #
    ########################################

    def f_and_g_prod_default1(self, t, y, v):
        return self.f(t, y), self.g_prod(t, y, v)

    def f_and_g_prod_default2(self, t, y, v):
        with torch.enable_grad():
            f, g = self.f_and_g(t, y)
        return f, self.prod(g, v)

    ########################################
    #          g_prod_and_gdg_prod         #
    ########################################

    # Computes: g_prod and sum_{j, l} g_{j, l} d g_{j, l} d x_i v2_l.
    def g_prod_and_gdg_prod_default(self, t, y, v1, v2):
        requires_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            y = y if y.requires_grad else y.detach().requires_grad_(True)
            g = self.g(t, y)
            vg_dg_vjp, = vjp(
                outputs=g,
                inputs=y,
                grad_outputs=g * v2.unsqueeze(-2),
                retain_graph=True,
                create_graph=requires_grad,
                allow_unused=True
            )
        return self.prod(g, v1), vg_dg_vjp

    def g_prod_and_gdg_prod_diagonal(self, t, y, v1, v2):
        requires_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            y = y if y.requires_grad else y.detach().requires_grad_(True)
            g = self.g(t, y)
            vg_dg_vjp, = vjp(
                outputs=g,
                inputs=y,
                grad_outputs=g * v2,
                retain_graph=True,
                create_graph=requires_grad,
                allow_unused=True
            )
        return self.prod(g, v1), vg_dg_vjp

    def g_prod_and_gdg_prod_additive(self, t, y, v1, v2):
        return self.g_prod(t, y, v1), 0.

    ########################################
    #              dg_ga_jvp               #
    ########################################

    # Computes: sum_{j,k,l} d g_{i,l} / d x_j g_{j,k} A_{k,l}.
    def dg_ga_jvp_column_sum_v1(self, t, y, a):
        requires_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            y = y if y.requires_grad else y.detach().requires_grad_(True)
            g = self.g(t, y)
            ga = torch.bmm(g, a)
            dg_ga_jvp = [
jvp(
                    outputs=g[..., col_idx],
                    inputs=y,
                    grad_inputs=ga[..., col_idx],
                    retain_graph=True,
                    create_graph=requires_grad,
                    allow_unused=True
                )[0]
                for col_idx in range(g.size(-1))
            ]
            dg_ga_jvp = sum(dg_ga_jvp)
        return dg_ga_jvp

    def dg_ga_jvp_column_sum_v2(self, t, y, a):
        # Faster, but more memory intensive.
        requires_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            y = y if y.requires_grad else y.detach().requires_grad_(True)
            g = self.g(t, y)
            ga = torch.bmm(g, a)

            batch_size, d, m = g.size()
            y_dup = torch.repeat_interleave(y, repeats=m, dim=0)
            g_dup = self.g(t, y_dup)
            ga_flat = ga.transpose(1, 2).flatten(0, 1)
            dg_ga_jvp, = jvp(
                outputs=g_dup,
                inputs=y_dup,
                grad_inputs=ga_flat,
                create_graph=requires_grad,
                allow_unused=True
            )
            dg_ga_jvp = dg_ga_jvp.reshape(batch_size, m, d, m).permute(0, 2, 1, 3)
            dg_ga_jvp = dg_ga_jvp.diagonal(dim1=-2, dim2=-1).sum(-1)
        return dg_ga_jvp

    def _return_zero(self, t, y, v):  # noqa
        return 0.


class ABCMeta(abc.ABCMeta):
    """Metaclass that also rejects instances missing abstract *attributes*."""

    def __call__(cls, *args, **kwargs):
        instance = super(ABCMeta, cls).__call__(*args, **kwargs)
        abstract_attributes = {
            name
            for name in dir(instance)
            if getattr(getattr(instance, name), '__is_abstract_attribute__', False)
        }
        if abstract_attributes:
            raise NotImplementedError(
                "Can't instantiate abstract class {} with"
                " abstract attributes: {}".format(
                    cls.__name__,
                    ', '.join(abstract_attributes)
                )
            )
        return instance


class BaseSDESolver(metaclass=ABCMeta):
    """API for solvers with possibly adaptive time stepping."""

    def __init__(self,
                 sde: BaseSDE,
                 bm,
                 dt: Scalar,
                 adaptive: bool,
                 rtol: Scalar,
                 atol: Scalar,
                 dt_min: Scalar,
                 options: Dict,
                 **kwargs):
        super(BaseSDESolver, self).__init__(**kwargs)
        self.sde = sde
        self.bm = bm
        self.dt = dt
        self.adaptive = adaptive
        self.rtol = rtol
        self.atol = atol
        self.dt_min = dt_min
        self.options = options

    def init_extra_solver_state(self, t0, y0) -> Tensors:
        """Optional solver state carried between steps (unused by Euler)."""
        return ()

    @abc.abstractmethod
    def step(self, t0: Scalar, t1: Scalar, y0: Tensor, extra0: Tensors) -> Tuple[Tensor, Tensors]:
        raise NotImplementedError

    def integrate(self, y0: Tensor, ts: Tensor, extra0: Tensors) -> Tuple[Tensor, Tensors]:
        """Integrate from ``ts[0]`` to each later ``ts`` value; return stacked ``y``."""
        step_size = self.dt
        prev_t = curr_t = ts[0]
        prev_y = curr_y = y0
        curr_extra = extra0
        ys = [y0]
        for out_t in ts[1:]:
            while curr_t < out_t:
                next_t = min(curr_t + step_size, ts[-1])
                prev_t, prev_y = curr_t, curr_y
                curr_y, curr_extra = self.step(curr_t, next_t, curr_y, curr_extra)
                curr_t = next_t
            ys.append(linear_interp(t0=prev_t, y0=prev_y, t1=curr_t, y1=curr_y, t=out_t))

        return torch.stack(ys, dim=0), curr_extra


class Euler(BaseSDESolver):
    """Euler–Maruyama: ``y1 = y0 + f(t0, y0) * dt + g(t0, y0) · ΔW``."""

    def __init__(self, sde, **kwargs):
        super(Euler, self).__init__(sde=sde, **kwargs)
        self.strong_order = 1.0 if sde.noise_type == 'additive' else 0.5

    def step(self, t0, t1, y0, extra0):
        del extra0
        dt = t1 - t0
        I_k = self.bm(t0, t1)

        f, g_prod = self.sde.f_and_g_prod(t0, y0, I_k)
        y1 = y0 + f * dt + g_prod
        return y1, ()


def check_contract(sde, y0, ts, bm, method, adaptive, options, names, logqp):
    """Validate inputs and return ``(sde, y0, ts, bm, method, options)`` ready for Euler.

    Unused arguments ``method``, ``adaptive``, and ``logqp`` are kept for the
    historical ``sdeint`` signature.
    """
    from brownian import BrownianInterval

    if names is None:
        names_to_change = {}
    else:
        names_to_change = {key: names[key] for key in ("drift", "diffusion", "prior_drift", "drift_and_diffusion",
                                                       "drift_and_diffusion_prod") if key in names}
    if len(names_to_change) > 0:
        sde = RenameMethodsSDE(sde, **names_to_change)

    if not torch.is_tensor(y0):
        raise ValueError("`y0` must be a torch.Tensor.")
    if y0.dim() != 2:
        raise ValueError("`y0` must be a 2-dimensional tensor of shape (batch, channels).")

    if not torch.is_tensor(ts):
        if not isinstance(ts, (tuple, list)) or not all(isinstance(t, (float, int)) for t in ts):
            raise ValueError("Evaluation times `ts` must be a 1-D Tensor or list/tuple of floats.")
        ts = torch.tensor(ts, dtype=y0.dtype, device=y0.device)
    if not is_strictly_increasing(ts):
        raise ValueError("Evaluation times `ts` must be strictly increasing.")

    batch_sizes = []
    state_sizes = []
    noise_sizes = []
    batch_sizes.append(y0.size(0))
    state_sizes.append(y0.size(1))
    if bm is not None:
        if len(bm.shape) != 2:
            raise ValueError("`bm` must be of shape (batch, noise_channels).")
        batch_sizes.append(bm.shape[0])
        noise_sizes.append(bm.shape[1])

    def _check_2d(name, shape):
        if len(shape) != 2:
            raise ValueError(f"{name} must be of shape (batch, state_channels), but got {shape}.")
        batch_sizes.append(shape[0])
        state_sizes.append(shape[1])

    def _check_2d_or_3d(name, shape):
        if sde.noise_type == 'diagonal':
            if len(shape) != 2:
                raise ValueError(f"{name} must be of shape (batch, state_channels), but got {shape}.")
            batch_sizes.append(shape[0])
            state_sizes.append(shape[1])
            noise_sizes.append(shape[1])
        else:
            if len(shape) != 3:
                raise ValueError(f"{name} must be of shape (batch, state_channels, noise_channels), but got {shape}.")
            batch_sizes.append(shape[0])
            state_sizes.append(shape[1])
            noise_sizes.append(shape[2])

    has_f = False
    has_g = False
    if hasattr(sde, 'f'):
        has_f = True
        f_drift_shape = tuple(sde.f(ts[0], y0).size())
        _check_2d('Drift', f_drift_shape)
    if hasattr(sde, 'g'):
        has_g = True
        g_diffusion_shape = tuple(sde.g(ts[0], y0).size())
        _check_2d_or_3d('Diffusion', g_diffusion_shape)
    if hasattr(sde, 'f_and_g'):
        has_f = True
        has_g = True
        _f, _g = sde.f_and_g(ts[0], y0)
        f_drift_shape = tuple(_f.size())
        g_diffusion_shape = tuple(_g.size())
        _check_2d('Drift', f_drift_shape)
        _check_2d_or_3d('Diffusion', g_diffusion_shape)

    for batch_size in batch_sizes[1:]:
        if batch_size != batch_sizes[0]:
            raise ValueError("Batch sizes not consistent.")
    for state_size in state_sizes[1:]:
        if state_size != state_sizes[0]:
            raise ValueError("State sizes not consistent.")
    for noise_size in noise_sizes[1:]:
        if noise_size != noise_sizes[0]:
            raise ValueError("Noise sizes not consistent.")

    if sde.noise_type == 'scalar':
        if noise_sizes[0] != 1:
            raise ValueError(f"Scalar noise must have only one channel; the diffusion has {noise_sizes[0]} noise "
                             f"channels.")

    sde = ForwardSDE(sde)

    if bm is None:
        bm = BrownianInterval(t0=ts[0], t1=ts[-1], size=(batch_sizes[0], noise_sizes[0]), dtype=y0.dtype,
                              device=y0.device, levy_area_approximation='none')

    if options is None:
        options = {}
    else:
        options = options.copy()

    return sde, y0, ts, bm, method, options
