"""Bounded nonlinear MPC with control blocking and a recurrent ESN rollout.

Publication repair of the thesis controller. See RELEASE_NOTES.md for changes.
"""

import numpy as np
from scipy.optimize import minimize

from core_utils import DataTools, normalize_angle
from robot import DiffDriveSim


class MPCController:
    def __init__(self, esn, stats, horizon=8, n_blocks=4, dt=0.05,
                 v_limits=(0.0, 0.6), omega_limits=(-1.5, 1.5),
                 Q=None, R=None, warm_start=True, maxiter=40, ftol=1e-8):
        if horizon <= 0 or n_blocks <= 0 or horizon % n_blocks:
            raise ValueError("horizon must be positive and divisible by n_blocks")
        self.esn, self.stats = esn, stats
        self.N, self.n_blocks, self.dt = horizon, n_blocks, dt
        self.steps_per_block = horizon // n_blocks
        self.v_limits, self.omega_limits = v_limits, omega_limits
        self.Q = np.diag([15., 15., 8.]) if Q is None else np.asarray(Q)
        self.R = np.diag([0.1, 0.1]) if R is None else np.asarray(R)
        self.warm_start, self.maxiter, self.ftol = warm_start, maxiter, ftol
        self.bounds = [v_limits, omega_limits] * n_blocks
        self.u_prev_opt = None

    def _cost(self, z, x0_n, reservoir, reference, u_prev):
        controls = np.repeat(z.reshape(self.n_blocks, 2), self.steps_per_block, axis=0)
        normalized = DataTools.normalize(controls, self.stats['u_mean'], self.stats['u_std'])
        predicted, _ = self.esn.rollout(x0_n, normalized, reservoir, stats=self.stats)
        physical = DataTools.denormalize(predicted, self.stats['y_mean'], self.stats['y_std'])
        states = np.column_stack([physical[:, :2], np.arctan2(physical[:, 2], physical[:, 3])])
        # Current-state error is constant in the decision; charge future states only.
        errors = states[1:] - reference[1:]
        angles = normalize_angle(errors[:, 2])
        periodic_error = 1.0 - np.cos(angles)
        stage = (self.Q[0, 0] * errors[:, 0] ** 2
                 + self.Q[1, 1] * errors[:, 1] ** 2
                 + self.Q[2, 2] * (periodic_error + 0.1 * periodic_error ** 2))
        stage[-1] *= 2.0
        changes = np.diff(np.vstack([u_prev, controls]), axis=0)
        control_cost = np.einsum('ni,ij,nj->', changes, self.R, changes)
        return float(stage.sum() + control_cost)

    def solve(self, x_curr, r_curr, x_ref_horizon, u_prev):
        reference = np.asarray(x_ref_horizon)
        if reference.shape != (self.N + 1, 3):
            raise ValueError("reference must have shape (horizon + 1, 3)")
        if self.warm_start and self.u_prev_opt is not None:
            # Shift the previous solution by one simulation step, then re-block.
            previous = np.repeat(self.u_prev_opt.reshape(-1, 2), self.steps_per_block, axis=0)
            shifted = np.vstack([previous[1:], previous[-1]])
            initial = shifted[::self.steps_per_block].ravel()
        else:
            v = np.linalg.norm(reference[1, :2] - reference[0, :2]) / self.dt
            omega = normalize_angle(reference[1, 2] - reference[0, 2]) / self.dt
            initial = np.tile([np.clip(v, *self.v_limits), np.clip(omega, *self.omega_limits)], self.n_blocks)
        x0_n = DataTools.normalize(x_curr, self.stats['x_mean'], self.stats['x_std'])
        result = minimize(self._cost, initial,
                          args=(x0_n, r_curr, reference, u_prev),
                          method='L-BFGS-B', bounds=self.bounds,
                          options={'maxiter': self.maxiter, 'ftol': self.ftol})
        if not np.isfinite(result.fun) or not np.all(np.isfinite(result.x)):
            raise RuntimeError("MPC produced a non-finite solution")
        # A finite iterate may be usable at the iteration limit; the benchmark
        # records result.success and result.message rather than hiding the limit.
        self.u_prev_opt = result.x.copy()
        return result.x[:2].copy(), result


class KinematicPredictor:
    """Nominal-model ablation using the identical MPC objective and optimizer."""
    n_reservoir = 0

    def __init__(self, dt=0.05):
        self.dt = dt

    def rollout(self, x0, U_seq, r0=None, stats=None):
        state = DataTools.denormalize(x0, stats['x_mean'], stats['x_std'])
        controls = DataTools.denormalize(U_seq, stats['u_mean'], stats['u_std'])
        states = [state.copy()]
        for control in controls:
            state = DiffDriveSim.step(state, control, self.dt)
            states.append(state.copy())
        states = np.asarray(states)
        output = np.column_stack([states[:, :2], np.sin(states[:, 2]), np.cos(states[:, 2])])
        return DataTools.normalize(output, stats['y_mean'], stats['y_std']), np.empty(0)
