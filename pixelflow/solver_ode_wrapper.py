import torch
from torchdiffeq import odeint


# https://github.com/willisma/SiT/blob/main/transport/integrators.py#L77
class ODE:
    """ODE solver class"""
    def __init__(
        self,
        *,
        t0,
        t1,
        sampler_type,
        num_steps,
        atol,
        rtol,
    ):
        assert t0 < t1, "ODE sampler has to be in forward time"

        self.t = torch.linspace(t0, t1, num_steps)
        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type

    def time_linear_to_timesteps(self, t, t_start, t_end, T_start, T_end):
        # T = k * t + b
        k = (T_end - T_start) / (t_end - t_start)
        b = T_start - t_start * k
        return k * t + b

    def sample(self, x, model, T_start, T_end, **model_kwargs):
        device = x[0].device if isinstance(x, tuple) else x.device
        def _fn(t, x):
            t = torch.ones(x[0].size(0)).to(device) * t if isinstance(x, tuple) else torch.ones(x.size(0)).to(device) * t
            model_output = model(x, self.time_linear_to_timesteps(t, 0, 1, T_start, T_end), **model_kwargs)
            assert model_output.shape == x.shape, "Output shape from ODE solver must match input shape"
            return model_output

        t = self.t.to(device)
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        samples = odeint(
            _fn,
            x,
            t,
            method=self.sampler_type,
            atol=atol,
            rtol=rtol
        )
        return samples
