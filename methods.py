"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
import torch
import numpy as np
import time


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct an SDE.

        Args:
          N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """Generate one sample from the prior distribution, $p_T(x)$."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
          z: latent code
        Returns:
          log probability density
        """
        pass

    def discretize(self, x, t):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
          x: a torch tensor
          t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
          f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(self, net_fn, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
          net_fn: a z-dependent PFGM that takes x and z and returns the normalized Poisson field.
            Or a time-dependent score-based model that takes x and t and returns the score.
          probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""

                drift, diffusion = sde_fn(x, t)
                score = net_fn(x.float(), t.float())
                drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
                # Set the diffusion function to zero for ODEs.
                diffusion = torch.zeros_like(diffusion) if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t)
                rev_f = f - G[:, None, None, None] ** 2 * net_fn(x, t) * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()


class Poisson():
    def __init__(self, args):
        """Construct a PFGM.

        Args:
          config: configurations
        """
        self.config = args.config
        self.N = args.config.sampling.N
        self.DDP = args.DDP

    @property
    def M(self):
        return self.config.training.M

    def prior_sampling(self, shape):
        """
        Sampling initial data from p_prior on z=z_max hyperplane.
        See Section 3.3 in PFGM paper
        """

        # Sample the radius from p_radius (details in Appendix A.4 in the PFGM paper)
        max_z = self.config.sampling.z_max
        N = self.config.data.channels * self.config.data.image_height * self.config.data.image_width + 1
        # Sampling form inverse-beta distribution
        samples_norm = np.random.beta(a=N / 2. - 0.5, b=0.5, size=shape[0])
        inverse_beta = samples_norm / (1 - samples_norm)
        # Sampling from p_radius(R) by change-of-variable
        samples_norm = np.sqrt(max_z ** 2 * inverse_beta)
        # clip the sample norm (radius)
        samples_norm = np.clip(samples_norm, 1, self.config.sampling.upper_norm)
        samples_norm = torch.from_numpy(samples_norm).cuda().view(len(samples_norm), -1)

        # Uniformly sample the angle direction
        gaussian = torch.randn(shape[0], N - 1).cuda()
        unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)

        # Radius times the angle direction
        init_samples = unit_gaussian * samples_norm

        return init_samples.float().view(len(init_samples), self.config.data.num_channels,
                                         self.config.data.image_height, self.config.data.image_width)

    def ode(self, net_fn, x, t):
        z = torch.exp(t.mean())
        if self.config.sampling.vs:
            print(z)
        x_drift, z_drift = net_fn(x, torch.ones((len(x))).cuda() * z)
        x_drift = x_drift.view(len(x_drift), -1)

        # Substitute the predicted z with the ground-truth
        # Please see Appendix B.2.3 in PFGM paper (https://arxiv.org/abs/2209.11178) for details
        z_exp = self.config.sampling.z_exp

        if z < z_exp and self.config.training.gamma > 0:
            data_dim = self.config.data.image_height * self.config.data.image_width * self.config.data.channels
            z_drift = gt_substituion(x_drift, z_drift, z, torch.tensor(data_dim),
                                     torch.tensor(self.config.training.gamma))

        # Predicted normalized Poisson field
        v = torch.cat([x_drift, z_drift[:, None]], dim=1)

        dt_dz = 1 / (v[:, -1] + 1e-5)
        dx_dt = v[:, :-1].view(len(x), self.config.data.num_channels, self.config.data.image_height,
                               self.config.data.image_width)
        dx_dz = dx_dt * dt_dz.view(-1, *([1] * len(x.size()[1:])))

        # dx/dt_prime =  z * dx/dz
        dx_dt_prime = z * dx_dz
        return dx_dt_prime


@torch.jit.script
def gt_substituion(x_drift, z_drift, z, data_dim, gamma):
    sqrt_dim = torch.sqrt(torch.tensor(data_dim))
    norm_1 = x_drift.norm(p=2, dim=1) / sqrt_dim
    x_norm = gamma * norm_1 / (1 - norm_1)
    x_norm = torch.sqrt(x_norm ** 2 + z ** 2)
    z_drift = -sqrt_dim * torch.ones_like(z_drift) * z / (x_norm + gamma)
    return z_drift
