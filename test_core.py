"""Regression tests for publication repairs and the experimental protocol."""

import unittest

import numpy as np

from baseline import collect_dataset_from_baseline
from core_utils import DataTools, normalize_angle
from esn_enhanced import ESNEnhanced
from experiment import make_dataset, physical_rollout, simulate, tracking_metrics, train_model
from mpc import KinematicPredictor, MPCController
from robot import DiffDriveSim
from trajectory import TrajectoryGenerator


class CoreTests(unittest.TestCase):
    def test_normalization_round_trip_with_constant_feature(self):
        values = np.array([[1., 0.], [2., 0.]])
        mean, std = DataTools.compute_stats(values)
        np.testing.assert_allclose(DataTools.denormalize(DataTools.normalize(values, mean, std), mean, std), values)

    def test_wrapped_heading_metric(self):
        states = [[0., 0., np.pi - 0.01]]
        target = [[0., 0., -np.pi + 0.01]]
        self.assertAlmostEqual(tracking_metrics(states, target)['heading_rmse_deg'], np.degrees(0.02))

    def test_robot_stationary_and_forward_motion(self):
        state = np.array([1., 2., 0.])
        np.testing.assert_allclose(DiffDriveSim.step(state, [0., 0.], 0.05), state)
        np.testing.assert_allclose(DiffDriveSim.step(state, [0.4, 0.], 0.05), [1.02, 2., 0.])

    def test_angle_wrap_at_robot_boundary(self):
        result = DiffDriveSim.step([0., 0., np.pi - 0.01], [0., 1.], 0.05)
        self.assertAlmostEqual(result[2], -np.pi + 0.04)

    def test_data_generator_reproducibility(self):
        first = make_dataset(12, 1, 30, 0.05)
        second = make_dataset(12, 1, 30, 0.05)
        for x, y in zip(first, second):
            np.testing.assert_array_equal(x, y)
        different = make_dataset(13, 1, 30, 0.05)
        self.assertFalse(np.array_equal(first[0], different[0]))

    def test_ideal_simulator_has_zero_residual(self):
        X, U, Y, _ = make_dataset(12, 1, 30, 0.05)
        model = ESNEnhanced(5, n_reservoir=8, use_residual=True)
        targets = model._compute_incremental_target(X, Y, U)
        np.testing.assert_allclose(targets[:, :3], 0., atol=1e-12)
        np.testing.assert_allclose(targets[:, 3], 1., atol=1e-12)

    def test_unit_circle_is_applied_after_decoding(self):
        model = ESNEnhanced(5, n_reservoir=8, use_residual=False)
        model.delta_mean = np.array([0., 0., 0.01, 0.99])
        model.delta_std = np.array([0.01, 0.01, 0.02, 0.03])
        normalized = np.array([0., 0., 2., -1.])
        physical = normalized * model.delta_std + model.delta_mean
        expected_angle = np.arctan2(physical[2], physical[3])
        decoded = model._decode_increment(normalized)
        self.assertAlmostEqual(np.arctan2(decoded[2], decoded[3]), expected_angle)
        self.assertAlmostEqual(np.linalg.norm(decoded[2:]), 1., places=6)


class ModelAndControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = {'train_seed': 0, 'trials_per_type': 1, 'train_steps': 60,
                        'dt': 0.05, 'reservoir_size': 16, 'model_seed': 42, 'washout': 5,
                        'horizon': 4, 'n_blocks': 2, 'v_ref': 0.3}
        cls.model, cls.stats, _ = train_model(cls.settings)

    def test_scaling_statistics_use_training_data(self):
        X, U, _, Y4 = make_dataset(0, 1, 60, 0.05)
        for key, values in [('x', X), ('u', U), ('y', Y4)]:
            np.testing.assert_allclose(self.stats[key + '_mean'], np.mean(values, axis=0))

    def test_independent_trials_reset_recurrent_prediction(self):
        X, U, _, _ = make_dataset(19, 1, 30, 0.05)
        Xn = DataTools.normalize(X, self.stats['x_mean'], self.stats['x_std'])
        Un = DataTools.normalize(U, self.stats['u_mean'], self.stats['u_std'])
        together = self.model.predict_sequence(Xn, Un, stats=self.stats, sequence_starts=[0, 30, 60])
        separate = np.vstack([self.model.predict_sequence(Xn[k:k+30], Un[k:k+30], stats=self.stats)
                              for k in [0, 30, 60]])
        np.testing.assert_allclose(together, separate)

    def test_rollout_is_finite_and_preserves_initial_state(self):
        initial = np.array([0.2, -0.1, np.pi - 0.01])
        predicted, _ = physical_rollout(self.model, self.stats, initial, np.tile([0.3, 0.2], (5, 1)))
        np.testing.assert_allclose(predicted[0], initial)
        self.assertTrue(np.all(np.isfinite(predicted)))
        self.assertLessEqual(np.max(np.abs(normalize_angle(np.diff(predicted[:, 2])))), 0.075 + 1e-9)

    def test_mpc_cost_depends_on_candidate_future_states(self):
        controller = MPCController(KinematicPredictor(), self.stats, horizon=4, n_blocks=2, R=np.zeros((2, 2)))
        state = np.zeros(3)
        reference = TrajectoryGenerator.generate_line(0.2, 0.05, 0.3)
        normalized = DataTools.normalize(state, self.stats['x_mean'], self.stats['x_std'])
        stopped = controller._cost(np.zeros(4), normalized, np.empty(0), reference, np.zeros(2))
        moving = controller._cost(np.array([0.3, 0., 0.3, 0.]), normalized, np.empty(0), reference, np.zeros(2))
        self.assertLess(moving, stopped)

    def test_terminal_cost_is_counted_once_at_double_weight(self):
        controller = MPCController(KinematicPredictor(), self.stats, horizon=4, n_blocks=2, R=np.zeros((2, 2)))
        reference = np.zeros((5, 3))
        reference[-1, 0] = 1.
        normalized = DataTools.normalize(np.zeros(3), self.stats['x_mean'], self.stats['x_std'])
        cost = controller._cost(np.zeros(4), normalized, np.empty(0), reference, np.zeros(2))
        self.assertAlmostEqual(cost, 30.)

    def test_mpc_obeys_bounds_and_uses_predictor(self):
        class CountingPredictor(KinematicPredictor):
            calls = 0
            def rollout(self, *args, **kwargs):
                self.calls += 1
                return super().rollout(*args, **kwargs)
        predictor = CountingPredictor()
        controller = MPCController(predictor, self.stats, horizon=4, n_blocks=2)
        reference = TrajectoryGenerator.generate_line(0.2, 0.05, 0.3)
        action, result = controller.solve(np.array([0., 0.05, 0.]), np.empty(0), reference, np.zeros(2))
        self.assertGreater(predictor.calls, 1)
        self.assertGreaterEqual(action[0], 0.)
        self.assertLessEqual(action[0], 0.6)
        self.assertLessEqual(abs(action[1]), 1.5)
        self.assertTrue(np.isfinite(result.fun))

    def test_horizon_validation(self):
        with self.assertRaises(ValueError):
            MPCController(KinematicPredictor(), self.stats, horizon=3, n_blocks=2)

    def test_all_controllers_run_from_shared_initial_state(self):
        reference = TrajectoryGenerator.generate_line(0.25, 0.05, 0.3)
        initial = np.array([0., 0.03, 0.02])
        for name in ['baseline', 'pid', 'kinematic_mpc', 'esn_mpc']:
            metrics, states, controls = simulate(name, reference, initial, self.model, self.stats, self.settings)
            np.testing.assert_array_equal(states[0], initial)
            self.assertEqual(states.shape, (6, 3))
            self.assertTrue(np.isfinite(metrics['position_rmse_m']))
            self.assertTrue(np.all(np.isfinite(controls)))


if __name__ == '__main__':
    unittest.main()
