"""Generate simulation data, train an ESN, and compare tracking controllers."""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy

from baseline import BaselineController, collect_dataset_from_baseline
from core_utils import DataTools, normalize_angle
from esn_enhanced import ESNEnhanced
from mpc import KinematicPredictor, MPCController
from pid_controller import PIDController
from robot import DiffDriveSim
from trajectory import TrajectoryGenerator

TRAJECTORIES = ('line', 'circle', 's_curve')
CONTROLLERS = ('baseline', 'pid', 'kinematic_mpc', 'esn_mpc')


def tracking_metrics(states, reference):
    errors = np.asarray(states) - np.asarray(reference)
    if errors.ndim != 2 or errors.shape[1] != 3 or not len(errors):
        raise ValueError('expected nonempty, aligned (T, 3) states')
    position = np.linalg.norm(errors[:, :2], axis=1)
    heading = normalize_angle(errors[:, 2])
    return {'position_rmse_m': float(np.sqrt(np.mean(position ** 2))),
            'heading_rmse_deg': float(np.degrees(np.sqrt(np.mean(heading ** 2)))),
            'max_position_error_m': float(position.max())}


def make_dataset(seed, trials, steps, dt):
    X, U, Y, Y4 = collect_dataset_from_baseline(
        dt=dt, num_trials_per_type=trials, T_per_trial=steps, seed=seed)
    return X, U, Y, Y4


def train_model(settings):
    X, U, Y, Y4 = make_dataset(settings['train_seed'], settings['trials_per_type'],
                              settings['train_steps'], settings['dt'])
    stats = {}
    for key, values in [('x', X), ('u', U), ('y', Y4)]:
        stats[key + '_mean'], stats[key + '_std'] = DataTools.compute_stats(values)
    model = ESNEnhanced(n_inputs=5, n_reservoir=settings['reservoir_size'],
                        spectral_radius=0.8, sparsity=0.15, leak_rate=0.5,
                        ridge_alpha=5e-4, use_incremental=True, use_residual=False,
                        dt=settings['dt'], seed=settings['model_seed'])
    model.fit(DataTools.normalize(X, stats['x_mean'], stats['x_std']),
              DataTools.normalize(U, stats['u_mean'], stats['u_std']), Y,
              stats=stats, washout=settings['washout'], noise_std=0.01,
              sequence_starts=range(0, len(X), settings['train_steps']))
    return model, stats, len(X)


def physical_rollout(model, stats, state, controls, reservoir=None):
    output, next_reservoir = model.rollout(
        DataTools.normalize(state, stats['x_mean'], stats['x_std']),
        DataTools.normalize(controls, stats['u_mean'], stats['u_std']),
        reservoir, stats=stats)
    output = DataTools.denormalize(output, stats['y_mean'], stats['y_std'])
    return np.column_stack([output[:, :2], np.arctan2(output[:, 2], output[:, 3])]), next_reservoir


def evaluate_prediction(model, stats, settings):
    steps = settings['train_steps']
    X, U, Y, _ = make_dataset(settings['validation_seed'], 2, steps, settings['dt'])
    one_step, rollout_predictions, rollout_truth = [], [], []
    for start in range(0, len(X), steps):
        reservoir = np.zeros(model.n_reservoir)
        for k in range(start, start + steps):
            predicted, reservoir = physical_rollout(model, stats, X[k], U[k:k+1], reservoir)
            one_step.append(predicted[-1])
        # Nonoverlapping 20-step windows; reset reservoir at every window.
        for k in range(start, start + steps, 20):
            stop = min(k + 20, start + steps)
            predicted, _ = physical_rollout(model, stats, X[k], U[k:stop])
            rollout_predictions.extend(predicted[1:])
            rollout_truth.extend(Y[k:stop])
    return {'one_step': tracking_metrics(one_step, Y),
            'cold_start_rollout_20_steps': tracking_metrics(rollout_predictions, rollout_truth),
            'validation_transitions': len(X),
            'validation_trials': 6,
            'validation_seed': settings['validation_seed']}


def simulate(name, reference, initial, model, stats, settings):
    dt, steps, horizon = settings['dt'], len(reference) - 1, settings['horizon']
    predictor = model if name == 'esn_mpc' else KinematicPredictor(dt)
    controller = None
    if name.endswith('_mpc'):
        controller = MPCController(predictor, stats, horizon=horizon,
                                   n_blocks=settings['n_blocks'], dt=dt)
    reservoir = np.zeros(predictor.n_reservoir)
    integral, previous_error = np.zeros(3), np.zeros(3)
    state, previous_action = initial.copy(), np.zeros(2)
    states, controls, latencies, statuses = [state.copy()], [], [], []
    for k in range(steps):
        before = perf_counter()
        if name == 'baseline':
            action = BaselineController.get_action(state, reference[k + 1], settings['v_ref'])
        elif name == 'pid':
            action, integral, previous_error = PIDController.get_action(
                state, reference[k + 1], settings['v_ref'], dt=dt,
                integral_state=integral, last_error_state=previous_error)
        else:
            indices = np.minimum(np.arange(k, k + horizon + 1), steps)
            action, result = controller.solve(state, reservoir, reference[indices], previous_action)
            statuses.append({'success': bool(result.success), 'message': str(result.message)})
            if name == 'esn_mpc':
                normalized_state = DataTools.normalize(state, stats['x_mean'], stats['x_std'])
                normalized_action = DataTools.normalize(action, stats['u_mean'], stats['u_std'])
                reservoir = model._reservoir_step(reservoir, np.r_[normalized_state, normalized_action])
        latencies.append((perf_counter() - before) * 1000)
        state = DiffDriveSim.step(state, action, dt)
        states.append(state.copy())
        controls.append(action.copy())
        previous_action = action
    states, controls = np.asarray(states), np.asarray(controls)
    # Exclude the shared initial state from the error average.
    metrics = tracking_metrics(states[1:], reference[1:])
    changes = np.diff(np.vstack([np.zeros(2), controls]), axis=0)
    metrics.update({'action_latency_mean_ms': float(np.mean(latencies)),
                    'action_latency_p95_ms': float(np.percentile(latencies, 95)),
                    'deadline_miss_fraction': float(np.mean(np.asarray(latencies) > dt * 1000)),
                    'linear_control_delta_rms_m_per_s': float(np.sqrt(np.mean(changes[:, 0] ** 2))),
                    'angular_control_delta_rms_rad_per_s': float(np.sqrt(np.mean(changes[:, 1] ** 2))),
                    'optimizer_nonconvergence_count': sum(not item['success'] for item in statuses),
                    'optimizer_messages': sorted(set(item['message'] for item in statuses)),
                    'steps': steps})
    return metrics, states, controls


def run(settings):
    before = perf_counter()
    model, stats, train_count = train_model(settings)
    training_seconds = perf_counter() - before
    prediction = evaluate_prediction(model, stats, settings)
    runs, examples = [], {}
    for trajectory in TRAJECTORIES:
        for seed in settings['evaluation_seeds']:
            rng = np.random.default_rng(seed)
            reference = TrajectoryGenerator.generate(trajectory, settings['steps'] * settings['dt'],
                                                      settings['dt'], settings['v_ref'], rng=rng)
            initial = reference[0] + rng.normal(0.0, [0.05, 0.05, 0.05])
            initial[2] = normalize_angle(initial[2])
            for name in CONTROLLERS:
                metrics, states, controls = simulate(name, reference, initial, model, stats, settings)
                runs.append({'trajectory': trajectory, 'seed': seed, 'controller': name, **metrics})
                if seed == settings['evaluation_seeds'][0]:
                    entry = examples.setdefault(trajectory, {'reference': reference.tolist(), 'states': {}})
                    entry['states'][name] = states.tolist()
                print(f'{trajectory:8} seed={seed} {name:14} RMSE={metrics["position_rmse_m"]:.5f} m', flush=True)
    summaries = []
    for trajectory in TRAJECTORIES:
        for name in CONTROLLERS:
            subset = [r for r in runs if r['trajectory'] == trajectory and r['controller'] == name]
            row = {'trajectory': trajectory, 'controller': name, 'n_runs': len(subset)}
            for metric in ['position_rmse_m', 'heading_rmse_deg', 'action_latency_mean_ms',
                           'action_latency_p95_ms', 'deadline_miss_fraction']:
                values = [r[metric] for r in subset]
                row[metric + '_mean'] = float(np.mean(values))
                row[metric + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row['optimizer_nonconvergence_count'] = sum(r['optimizer_nonconvergence_count'] for r in subset)
            summaries.append(row)
    import matplotlib
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(Path(__file__).parent.glob('*.py'))}
    report = {'schema_version': 1, 'edition': 'publication repair; regenerated simulation benchmark',
              'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                              'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__,
                              'os': platform.system(), 'architecture': platform.machine()},
              'settings': settings, 'source_sha256': source_hashes,
              'training_transitions': train_count, 'training_seconds': training_seconds,
              'prediction_metrics': prediction, 'summary': summaries, 'runs': runs,
              'limitations': ['Ideal kinematic simulation only; no robot hardware validation.',
                              'Validation uses new trajectories from the same generator distribution.',
                              'One training/model seed; evaluation seeds vary trajectories and initial states.',
                              'Residual learning disabled because simulator equals the nominal model.',
                              'Short finite-horizon benchmark; latency depends on this machine.',
                              'Baseline/PID use the next waypoint; MPC receives the full future horizon.']}
    return report, examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='shorter smoke benchmark')
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    settings = {'dt': 0.05, 'v_ref': 0.3, 'train_seed': 0, 'validation_seed': 1000,
                'model_seed': 42, 'evaluation_seeds': [101, 202, 303],
                'reservoir_size': 32 if args.quick else 80,
                'trials_per_type': 2 if args.quick else 4,
                'train_steps': 80 if args.quick else 120, 'washout': 10,
                'steps': 30 if args.quick else 80,
                'horizon': 4 if args.quick else 8, 'n_blocks': 2 if args.quick else 4,
                'use_residual': False, 'profile': 'quick' if args.quick else 'standard'}
    args.output.mkdir(parents=True, exist_ok=True)
    report, examples = run(settings)
    (args.output / 'benchmark_results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (args.output / 'benchmark_trajectories.json').write_text(json.dumps(examples, allow_nan=False) + '\n')
    from plot_results import plot_results
    plot_results(report, examples, args.output)
    print(f'Wrote results to {args.output}')


if __name__ == '__main__':
    main()
