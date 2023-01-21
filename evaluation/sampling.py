"""Various sampling methods."""
import functools
import torch
import numpy as np
import abc

from models.utils import from_flattened_numpy, to_flattened_numpy, get_predict_fn
from scipy import integrate
from models import utils as mutils
from tqdm import tqdm
from PIL import Image
from torchvision.utils import make_grid, save_image
from torchdiffeq import odeint
import os
from methods import gt_substituion

_ODESOLVER = {}


def register_odesolver(cls=None, *, name=None):
    """A decorator for registering ode solver classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _ODESOLVER:
            raise ValueError(f'Already registered model with name: {local_name}')
        _ODESOLVER[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_ode_solver(name):
    return _ODESOLVER[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps, model=None):
    """Create a sampling function.

    Args:
      config: A `ml_collections.ConfigDict` object that contains all configuration information.
      sde: A `methods.SDE` object that represents the forward SDE.
      shape: A sequence of integers representing the expected shape of a single sample.
      inverse_scaler: The inverse data normalizer function.
      eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

    Returns:
      A function that takes random states and a replicated training state and outputs samples with the
        trailing dimensions matching `shape`.
    """

    sampler_name = config.sampling.method
    # ODE sampling with black-box ODE solvers
    if sampler_name.lower() == 'ode':
        if config.sampling.ode_solver == 'rk45':
            # RK45 ode sampler for PFGM
            sampling_fn = get_rk45_sampler_pfgm(sde=sde,
                                                shape=shape,
                                                inverse_scaler=inverse_scaler,
                                                eps=eps,
                                                device=config.device)
        elif config.sampling.ode_solver == 'torchdiffeq':
            # sampling_fn = get_torchdiffeq_sampler_pfgm(sde=sde, shape=shape, inverse_scaler=inverse_scaler,
            # eps=eps,device=config.device)
            sampling_fn = OdeTorch(
                sde=sde, shape=shape,
                inverse_scaler=inverse_scaler,
                eps=eps,
                device=config.device,
                model=model
            ).to(config.device)
        else:
            ode_solver = get_ode_solver(config.sampling.ode_solver.lower())
            sampling_fn = get_ode_sampler(sde=sde,
                                          shape=shape,
                                          ode_solver=ode_solver,
                                          inverse_scaler=inverse_scaler,
                                          eps=eps,
                                          device=config.device)
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn


class OdeSolverABC(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, net_fn, eps=None):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        if sde.config.training.sde != 'poisson':
            self.rsde = sde.reverse(net_fn, probability_flow=True)
        self.net_fn = net_fn
        self.eps = eps

    @abc.abstractmethod
    def update_fn(self, x, t, t_list=None, idx=None):
        """One update of the predictor.

        Args:
          x: A PyTorch tensor representing the current state
          t: A Pytorch tensor representing the current time step.

        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


@register_odesolver(name='forward_euler')
class ForwardEulerPredictor(OdeSolverABC):
    def __init__(self, sde, net_fn, eps=None):
        super().__init__(sde, net_fn, eps)

    def update_fn(self, x, t, t_list=None, idx=None):

        if self.sde.config.training.sde == 'poisson':
            # dt = - (np.log(self.sde.config.sampling.z_max) - np.log(self.eps)) / self.sde.N
            drift = self.sde.ode(self.net_fn, x, t)
            if t_list is None:
                dt = - (np.log(self.sde.config.sampling.z_max) - np.log(self.eps)) / self.sde.N
            else:
                # integration over z
                dt = - (1 - torch.exp(t_list[idx + 1] - t_list[idx]))
                dt = float(dt.cpu().numpy())
        else:
            dt = -1. / self.sde.N
            drift, _ = self.rsde.sde(x, t)
        x = x + drift * dt
        return x


@register_odesolver(name='improved_euler')
class ImprovedEulerPredictor(OdeSolverABC):
    def __init__(self, sde, net_fn, eps=None):
        super().__init__(sde, net_fn, eps)

    def update_fn(self, x, t, t_list=None, idx=None):
        if t_list is None:
            dt = - (torch.log(self.sde.config.sampling.z_max) - torch.log(self.eps)) / self.sde.N
        else:
            # integration over z
            dt = (torch.exp(t_list[idx + 1] - t_list[idx]) - 1)
        drift = self.sde.ode(self.net_fn, x, t)
        x_new = x + drift * dt

        if idx == self.sde.N - 1:
            return x_new
        else:
            idx_new = idx + 1
            t_new = t_list[idx_new]
            t_new = torch.ones(len(t), device=t.device) * t_new

            if t_list is None:
                dt_new = - (torch.log(self.sde.config.sampling.z_max) - torch.log(self.eps)) / self.sde.N
            else:
                # integration over z
                dt_new = (1 - torch.exp(t_list[idx] - t_list[idx + 1]))
                # dt_new = float(dt_new.cpu().numpy())
            drift_new = self.sde.ode(self.net_fn, x_new, t_new)

            x = x + (0.5 * drift * dt + 0.5 * drift_new * dt_new)
            return x


def shared_ode_solver_update_fn(x, t, sde, model, ode_solver, eps, t_list=None, idx=None):
    """A wrapper that configures and returns the update function of ODE solvers."""
    net_fn = mutils.get_predict_fn(sde, model, train=False, continuous=True)
    ode_solver_obj = ode_solver(sde, net_fn, eps)
    return ode_solver_obj.update_fn(x, t, t_list=t_list, idx=idx)


def get_ode_sampler(sde, shape, ode_solver, inverse_scaler, eps=1e-3, device='cuda'):
    """Create a ODE sampler, for foward Euler or Improved Euler method.

    Args:
      sde: An `methods.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      ode_solver: A subclass of `sampling.OdeSolverABC` representing the predictor algorithm.
      inverse_scaler: The inverse data normalizer.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      device: PyTorch device.

    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    # Create predictor & corrector update functions
    ode_update_fn = functools.partial(shared_ode_solver_update_fn,
                                      sde=sde,
                                      ode_solver=ode_solver,
                                      eps=eps)

    def ode_sampler(model):
        """ The ODE sampler funciton.

        Args:
          model: A PFGM or score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample
            x = sde.prior_sampling(shape).to(device).float()
            timesteps = torch.linspace(np.log(sde.config.sampling.z_max), np.log(eps), sde.N + 1, device=device).float()
            imgs = []
            for i in tqdm(range(sde.N)):
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device).float() * t
                x = ode_update_fn(x, vec_t, model=model, t_list=timesteps, idx=i)

                if sde.config.eval.show_sampling == True:
                    image_grid = make_grid(inverse_scaler(x), nrow=int(np.sqrt(len(x))))
                    im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu',
                                                                                                           torch.uint8).numpy())
                    imgs.append(im)
                    imgs[0].save(os.path.join("movie.gif"), save_all=True, append_images=imgs[1:], duration=1, loop=0)
            return inverse_scaler(x), 2 * sde.N - 1 if sde.config.sampling.ode_solver == 'improved_euler' else sde.N

    return ode_sampler


def get_rk45_sampler_pfgm(sde, shape, inverse_scaler, rtol=1e-4, atol=1e-4,
                          method='RK45', eps=1e-3, device='cuda'):
    """RK45 ODE sampler for PFGM.

    Args:
      sde: An `methods.SDE` object that represents PFGM.
      shape: A sequence of integers. The expected shape of a single sample.
      inverse_scaler: The inverse data normalizer.
      rtol: A `float` number. The relative tolerance level of the ODE solver.
      atol: A `float` number. The absolute tolerance level of the ODE solver.
      method: A `str`. The algorithm used for the black-box ODE solver.
        See the documentation of `scipy.integrate.solve_ivp`.
      eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
      device: PyTorch device.

    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """

    def ode_sampler(model, x=None):

        with torch.no_grad():
            # Initial sample
            if x is None:
                x = sde.prior_sampling(shape).to(device)

            z = torch.ones((len(x), 1, 1, 1)).to(x.device)
            z = z.repeat((1, 1, sde.config.data.image_height, sde.config.data.image_width)) * sde.config.sampling.z_max
            x = x.view(shape)
            # Augment the samples with extra dimension z
            # We concatenate the extra dimension z as an addition channel to accomondate this solver
            x = torch.cat((x, z), dim=1)
            x = x.float()
            new_shape = (
                len(x), sde.config.data.channels + 1, sde.config.data.image_height, sde.config.data.image_width)

            def ode_func(t, x):

                if sde.config.sampling.vs:
                    (np.exp(t))
                x = from_flattened_numpy(x, new_shape).to(device).type(torch.float32)
                # Change-of-variable z=exp(t)
                z = np.exp(t)
                net_fn = get_predict_fn(sde, model, train=False)

                x_drift, z_drift = net_fn(x[:, :-1], torch.ones((len(x))).cuda() * z)
                x_drift = x_drift.view(len(x_drift), -1)

                # Substitute the predicted z with the ground-truth
                # Please see Appendix B.2.3 in PFGM paper (https://arxiv.org/abs/2209.11178) for details
                z_exp = sde.config.sampling.z_exp
                if z < z_exp and sde.config.training.gamma > 0:
                    data_dim = sde.config.data.image_height * sde.config.data.image_width * sde.config.data.channels
                    sqrt_dim = np.sqrt(data_dim)
                    norm_1 = x_drift.norm(p=2, dim=1) / sqrt_dim
                    x_norm = sde.config.training.gamma * norm_1 / (1 - norm_1)
                    x_norm = torch.sqrt(x_norm ** 2 + z ** 2)
                    z_drift = -sqrt_dim * torch.ones_like(z_drift) * z / (x_norm + sde.config.training.gamma)

                # Predicted normalized Poisson field
                v = torch.cat([x_drift, z_drift[:, None]], dim=1)
                dt_dz = 1 / (v[:, -1] + 1e-5)
                dx_dt = v[:, :-1].view(shape)

                # Get dx/dz
                dx_dz = dx_dt * dt_dz.view(-1, *([1] * len(x.size()[1:])))
                # drift = z * (dx/dz, dz/dz) = z * (dx/dz, 1)
                drift = torch.cat([z * dx_dz, torch.ones(
                    (len(dx_dz), 1, sde.config.data.image_height, sde.config.data.image_width)).to(dx_dz.device) * z],
                                  dim=1)
                return to_flattened_numpy(drift)

            # Black-box ODE solver for the probability flow ODE.
            # Note that we use z = exp(t) for change-of-variable to accelearte the ODE simulation
            solution = integrate.solve_ivp(ode_func, (np.log(sde.config.sampling.z_max), np.log(eps)),
                                           to_flattened_numpy(x), rtol=rtol, atol=atol, method='RK45')

            nfe = solution.nfev
            x = torch.tensor(solution.y[:, -1]).reshape(new_shape).to(device).type(torch.float32)

            # Detach augmented z dimension
            x = x[:, :-1]
            x = inverse_scaler(x)
            return x, nfe

    return ode_sampler


class OdeFunct(torch.nn.Module):
    def __init__(self, sde, shape, new_shape, model, inverse_scaler=None):
        super(OdeFunct, self).__init__()
        self.sde = sde
        self.shape = shape
        self.new_shape = new_shape
        self.model = model
        self.nfe = 0
        self.inverse_scaler = inverse_scaler
        self.samples = []

    @torch.no_grad()
    def forward(self, t, x):
        x = x.reshape(self.new_shape)
        z = torch.exp(t)
        x_drift, z_drift = self.model(x[:, :-1], torch.ones((len(x))).cuda() * z)

        if self.sde.config.eval.show_sampling:
            image_grid = make_grid(self.inverse_scaler(x), nrow=int(np.sqrt(len(x))))
            im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu',
                                                                                                   torch.uint8).numpy())
            self.samples.append(im)

        drift = self.calculate_drift(x, z, x_drift, z_drift).flatten()
        self.nfe += 1
        return drift

    def calculate_drift(self, x, z, x_drift, z_drift):
        x_drift = x_drift.view(len(x_drift), -1)

        # Substitute the predicted z with the ground-truth
        # Please see Appendix B.2.3 in PFGM paper (https://arxiv.org/abs/2209.11178) for details
        z_exp = self.sde.config.sampling.z_exp
        if z < z_exp and self.sde.config.training.gamma > 0:
            data_dim = self.sde.config.data.image_height * self.sde.config.data.image_width * \
                       self.sde.config.data.channels
            z_drift = gt_substituion(x_drift, z_drift, z, torch.tensor(data_dim),
                                     torch.tensor(self.sde.config.training.gamma))

        # Predicted normalized Poisson field
        v = torch.cat([x_drift, z_drift[:, None]], dim=1)
        dt_dz = 1 / (v[:, -1] + 1e-5)
        dx_dt = v[:, :-1].view(self.shape)

        # Get dx/dz
        dx_dz = dx_dt * dt_dz.view(-1, *([1] * len(x.size()[1:])))
        # drift = z * (dx/dz, dz/dz) = z * (dx/dz, 1)
        drift = torch.cat(
            [z * dx_dz,
             torch.ones((len(dx_dz), 1, self.sde.config.data.image_height, self.sde.config.data.image_width)).to(
                 dx_dz.device) * z], dim=1)
        return drift


class OdeTorch(torch.nn.Module):
    "Ode solver using torchdiffeq"

    def __init__(self, sde, shape, inverse_scaler, rtol=1e-4, atol=1e-4, eps=1e-3, device='cuda', model=None):
        super(OdeTorch, self).__init__()
        self.sde = sde
        self.shape = shape
        self.inverse_scaler = inverse_scaler
        self.rtol = rtol
        self.atol = atol
        self.eps = eps
        self.device = device
        self.model = model
        self.new_shape = None
        assert model is not None, "No Model given"

    @torch.no_grad()
    def forward(self, input=None):
        sde = self.sde
        shape = self.shape
        # Initial sample
        x = sde.prior_sampling(shape).to(self.device, non_blocking=True)

        z = torch.ones((len(x), 1, 1, 1)).to(x.device, non_blocking=True)
        z = z.repeat((1, 1, sde.config.data.image_height, sde.config.data.image_width)) * sde.config.sampling.z_max
        x = x.view(shape)
        # Augment the samples with extra dimension z
        # We concatenate the extra dimension z as an addition channel to accomondate this solver
        x = torch.cat((x, z), dim=1).float()
        new_shape = (len(x), sde.config.data.channels + 1, sde.config.data.image_height, sde.config.data.image_width)
        # Black-box ODE solver for the probability flow ODE.
        # Note that we use z = exp(t) for change-of-variable to accelearte the ODE simulation
        t_start = np.log(sde.config.sampling.z_max)
        t_end = np.log(self.eps)
        # make a spaced sampling strategy
        time_span = torch.linspace(t_start, t_end, sde.config.sampling.N + 1).to(x.device).float()
        # time_span = torch.tensor((t_start, t_end)).to(x.device)

        Ode = OdeFunct(sde, shape, new_shape, self.model, self.inverse_scaler).to(self.device, non_blocking=True)

        solution = odeint(
            Ode,
            y0=torch.flatten(x),
            t=time_span,
            rtol=self.rtol,
            atol=self.atol,
            method='rk4', options=dict(step_size=sde.config.sampling.rk_stepsize, perturb=False)
            # method='dopri5', options=dict(max_num_steps=sde.config.sampling.N+1)
        )

        if self.sde.config.eval.show_sampling:
            Ode.samples[0].save(os.path.join("movie.gif"), save_all=True, append_images=Ode.samples[1:], duration=5,
                                loop=0)
        x = solution[-1].reshape(new_shape)

        # Detach augmented z dimension
        x = x[:, :-1]
        x = self.inverse_scaler(x)
        return x, Ode.nfe
