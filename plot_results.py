"""Plots derived exclusively from the saved benchmark observations."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LABELS = {'baseline': 'Geometric baseline', 'pid': 'PID',
          'kinematic_mpc': 'Kinematic MPC', 'esn_mpc': 'ESN-MPC'}
COLORS = {'baseline': '#999999', 'pid': '#b7791f',
          'kinematic_mpc': '#37648d', 'esn_mpc': '#29735e'}


def plot_results(report, examples, output):
    output = Path(output)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout='constrained')
    for ax, (trajectory, entry) in zip(axes, examples.items()):
        reference = np.asarray(entry['reference'])
        ax.plot(reference[:, 0], reference[:, 1], 'k--', label='Reference', linewidth=1.4)
        for name, values in entry['states'].items():
            states = np.asarray(values)
            ax.plot(states[:, 0], states[:, 1], label=LABELS[name], color=COLORS[name], linewidth=1.3)
        ax.set(title=trajectory.replace('_', ' ').title(), xlabel='x (m)', ylabel='y (m)')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=0.2)
    axes[-1].legend(fontsize=8)
    seed = report['settings']['evaluation_seeds'][0]
    duration = report['settings']['steps'] * report['settings']['dt']
    fig.suptitle(f'Closed-loop trajectories | evaluation seed {seed} | {duration:g} s simulation')
    fig.savefig(output / 'tracking_comparison.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
    trajectories = list(examples)
    x = np.arange(len(trajectories))
    for index, name in enumerate(LABELS):
        rows = [next(r for r in report['summary'] if r['trajectory'] == t and r['controller'] == name)
                for t in trajectories]
        offset = (index - 1.5) * 0.19
        axes[0].bar(x + offset, [100 * r['position_rmse_m_mean'] for r in rows], width=0.18,
                    yerr=[100 * r['position_rmse_m_std'] for r in rows], capsize=2,
                    label=LABELS[name], color=COLORS[name])
        axes[1].plot(x, [r['action_latency_p95_ms_mean'] for r in rows], marker='o',
                     label=LABELS[name], color=COLORS[name])
    axes[0].set(title='Position RMSE: mean and sample SD across seeds', ylabel='Position RMSE (cm)')
    axes[1].set(title='Mean of per-run p95 action latency', ylabel='Action latency (ms)')
    axes[1].axhline(report['settings']['dt'] * 1000, color='black', linestyle='--', label='Simulation step budget')
    axes[1].set_yscale('log')
    for ax in axes:
        ax.set_xticks(x, [t.replace('_', ' ') for t in trajectories])
        ax.set_xlabel('Trajectory type')
        ax.grid(axis='y', alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Source: benchmark_results.json | {len(report['settings']['evaluation_seeds'])} evaluation seeds | {duration:g} s per run")
    fig.savefig(output / 'benchmark_comparison.png', dpi=160)
    plt.close(fig)
