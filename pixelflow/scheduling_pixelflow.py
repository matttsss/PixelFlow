import math
import numpy as np
import torch


def cal_rectify_ratio(start_t, gamma):
    return 1 / (math.sqrt(1 - (1 / gamma)) * (1 - start_t) + start_t)


class PixelFlowScheduler:
    def __init__(self, num_train_timesteps, num_stages, gamma=-1 / 3):
        assert num_stages > 0, f"num_stages must be positive, got {num_stages}"
        self.num_stages = num_stages
        self.gamma = gamma

        self.Timesteps = torch.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=torch.float32)

        self.t = self.Timesteps / num_train_timesteps  # normalized time in [0, 1]

        self.stage_range = [x / num_stages for x in range(num_stages + 1)]

        self.original_start_t = torch.zeros(num_stages, dtype=torch.float64)
        self.start_t = torch.zeros(num_stages, dtype=torch.float64)
        self.end_t = torch.zeros(num_stages, dtype=torch.float64)
        self.t_window_per_stage = torch.empty((num_stages, num_train_timesteps), dtype=torch.float64)
        self.Timesteps_per_stage = torch.empty((num_stages, num_train_timesteps), dtype=torch.float64)
        stage_distance = torch.empty(num_stages, dtype=torch.float64)

        # stage_idx = 0: min t, min resolution, most noisy
        # stage_idx = num_stages - 1 : max t, max resolution, most clear
        for stage_idx in range(num_stages):
            start_idx = max(int(num_train_timesteps * self.stage_range[stage_idx]), 0)
            end_idx = min(int(num_train_timesteps * self.stage_range[stage_idx + 1]), num_train_timesteps)

            start_t = self.t[start_idx].item()
            end_t = self.t[end_idx].item() if end_idx < num_train_timesteps else 1.0

            self.original_start_t[stage_idx] = start_t

            if stage_idx > 0:
                start_t *= cal_rectify_ratio(start_t, gamma)

            self.start_t[stage_idx] = start_t
            self.end_t[stage_idx] = end_t
            stage_distance[stage_idx] = end_t - start_t

        total_stage_distance = stage_distance.sum().item()
        stage_distance_cumsum = torch.cumsum(stage_distance, dim=0)
        t_within_stage = torch.linspace(0, 1, num_train_timesteps + 1, dtype=torch.float64)[:-1]
        self.t_window_per_stage[:] = t_within_stage

        for stage_idx in range(num_stages):
            start_ratio = 0.0 if stage_idx == 0 else (stage_distance_cumsum[stage_idx - 1].item() / total_stage_distance)
            end_ratio = 1.0 if stage_idx == num_stages - 1 else (stage_distance_cumsum[stage_idx].item() / total_stage_distance)

            Timestep_start = self.Timesteps[int(num_train_timesteps * start_ratio)]
            Timestep_end = self.Timesteps[min(int(num_train_timesteps * end_ratio), num_train_timesteps - 1)]

            if stage_idx == num_stages - 1:
                self.Timesteps_per_stage[stage_idx] = torch.linspace(Timestep_start.item(), Timestep_end.item(), num_train_timesteps, dtype=torch.float64)
            else:
                self.Timesteps_per_stage[stage_idx] = torch.linspace(Timestep_start.item(), Timestep_end.item(), num_train_timesteps + 1, dtype=torch.float64)[:-1]

    @staticmethod
    def time_linear_to_Timesteps(t, t_start, t_end, T_start, T_end):
        """
        linearly map t to T: T = k * t + b
        """
        k = (T_end - T_start) / (t_end - t_start)
        b = T_start - t_start * k
        return k * t + b

    def set_timesteps(self, num_inference_steps, stage_index, device=None, shift=1.0):
        self.num_inference_steps = num_inference_steps

        stage_T_start = self.Timesteps_per_stage[stage_index][0].item()
        stage_T_end = self.Timesteps_per_stage[stage_index][-1].item()

        t_start = self.t_window_per_stage[stage_index][0].item()
        t_end = self.t_window_per_stage[stage_index][-1].item()

        t = np.linspace(t_start, t_end, num_inference_steps, dtype=np.float64)
        t = t / (shift  + (1 - shift) * t)

        Timesteps = self.time_linear_to_Timesteps(t, t_start, t_end, stage_T_start, stage_T_end)
        self.Timesteps = torch.from_numpy(Timesteps).to(device=device)

        self.t = torch.from_numpy(np.append(t, 1.0)).to(device=device, dtype=torch.float64)
        self._step_index = None

    def step(self, model_output, sample):
        if self.step_index is None:
            self._step_index = 0

        sample = sample.to(torch.float32)
        t = self.t[self.step_index].float()
        t_next = self.t[self.step_index + 1].float()

        prev_sample = sample + (t_next - t) * model_output
        self._step_index += 1

        return prev_sample.to(model_output.dtype)

    @property
    def step_index(self):
        """Current step index for the scheduler."""
        return self._step_index
