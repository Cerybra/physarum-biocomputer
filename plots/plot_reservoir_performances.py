import os

import itertools

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error


def run_experiment(
        recordings: np.ndarray,
        model: callable,
        **kwargs
) -> tuple[float, float]:
    mask = np.all(np.abs(recordings) < 20.0, axis=1)
    recordings = recordings[mask]

    split = int(recordings.shape[0] * kwargs.get('train_size', 0.8))
    train_split, test_split = recordings[:split], recordings[split:]

    x_train_features, x_test_features = train_split[:, [0, 2]], test_split[:, [0, 2]]
    x_train_states, x_test_states = train_split[:, 4:], test_split[:, 4:]
    y_train, y_test = train_split[:, 3], test_split[:, 3]

    baseline = model()
    baseline.fit(x_train_features, y_train)

    readout = model()
    readout.fit(np.hstack([x_train_features, x_train_states]), y_train)

    y_pred_baseline = baseline.predict(x_test_features)
    y_pred_readout = readout.predict(np.hstack([x_test_features, x_test_states]))

    return (
        mean_squared_error(y_true=y_test, y_pred=y_pred_baseline),
        mean_squared_error(y_true=y_test, y_pred=y_pred_readout)
    )


def plot_dumbbell_improvement(
        ax: plt.Axes,
        baseline_data: np.ndarray,
        readout_data: np.ndarray,
        color: str = 'blue',
        label: str = 'Reservoir'
) -> None:
    ax.set_facecolor('white')

    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['left'].set_color('black')

    ax.set_xlabel('Root Mean Squared Error', fontsize=6)
    ax.set_ylabel('Counts', fontsize=6)

    ax.tick_params(colors='black', labelsize=5, which='both')

    ax.xaxis.get_offset_text().set_fontsize(6)
    ax.yaxis.get_offset_text().set_fontsize(6)

    indices = np.argsort(baseline_data)

    base_sorted = np.array(baseline_data)[indices]
    read_sorted = np.array(readout_data)[indices]

    y_axis = np.arange(len(base_sorted))

    ax.hlines(y_axis, base_sorted, read_sorted, color='gray', alpha=0.5, linewidth=1)

    ax.scatter(base_sorted, y_axis, color='black', s=20, label='Baseline', zorder=3)
    ax.scatter(read_sorted, y_axis, color=color, s=20, label=label, zorder=3)

    ax.grid(axis='x', linestyle='--', alpha=0.3)


def plot_interleaved_dumbbell(
        ax: plt.Axes,
        group_a: tuple[np.ndarray, np.ndarray, str, str],
        group_b: tuple[np.ndarray, np.ndarray, str, str],
) -> None:
    ax.set_facecolor('white')

    ax.grid(axis='x', ls='--', alpha=0.2)

    ax.spines[['top', 'left', 'right', 'bottom']].set_visible(False)

    ax.set_xlabel('Root Mean Squared Error', fontsize=6)

    ax.set_yticks([])
    ax.tick_params(colors='black', labelsize=5, which='both')

    ax.xaxis.get_offset_text().set_fontsize(6)
    ax.yaxis.get_offset_text().set_fontsize(6)

    baseline_a, readout_a, colour_a, label_a = group_a
    baseline_b, readout_b, colour_b, label_b = group_b

    index_a = np.argsort(baseline_a)
    index_b = np.argsort(baseline_b)

    max_len = max(len(index_a), len(index_b))
    total_len = len(index_a) + len(index_b)

    ax.vlines(x=np.mean(baseline_a), ymin=0.0, ymax=total_len, color='black', alpha=0.9)
    ax.text(s='Baseline', x=(np.mean(baseline_a) - 0.05), y=(total_len + 0.25), fontsize=6)

    y_index = 0
    for i in range(max_len):
        if i < len(index_a):
            ii = index_a[i]

            ax.hlines(y_index, baseline_a[ii], readout_a[ii], color='gray', alpha=0.2, lw=0.8)

            ax.scatter(
                baseline_a[ii],
                y_index,
                color='gray',
                alpha=0.85,
                s=12,
                zorder=3,
                label=(
                    'Baseline' if y_index == 0 else ''
                )
            )

            ax.scatter(
                readout_a[ii],
                y_index,
                color=colour_a,
                s=12,
                zorder=3,
                label=(
                    label_a if y_index == 0 else ''
                )
            )

            y_index += 1

        if i < len(index_b):
            ii = index_b[i]
            ax.hlines(y_index, baseline_b[ii], readout_b[ii], color='gray', alpha=0.2, lw=0.8)
            ax.scatter(
                baseline_b[ii],
                y_index,
                color='gray',
                alpha=0.85,
                s=12,
                zorder=3
            )
            ax.scatter(
                readout_b[ii],
                y_index,
                color=colour_b,
                s=12,
                zorder=3,
                label=(
                    label_b if y_index == 1 else ''
                )
            )
            y_index += 1

    ax.legend(fontsize=7, frameon=False, loc='upper right', bbox_to_anchor=(1, 1))


if __name__ == '__main__':
    matplotlib.use('TkAgg')

    plt.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['pdf.fonttype'] = 42

    palette = {
        'main': '#FFCB05',
        'deep': '#B6862C',
        'error': 'gray',
        'target': 'black'
    }

    NULL_COUNT = 5
    RESERVOIR_1_COUNT = 9
    RESERVOIR_2_COUNT = 11

    null_reservoir_recordings = map(
        lambda file_path: np.load(f'./data/null/{file_path}')['recordings'], filter(
            lambda file_path: file_path.endswith('.npz'), os.listdir('./data/null')
        )
    )
    reservoir_1_reservoir_recordings = map(
        lambda file_path: np.load(f'./data/reservoir-1/{file_path}')['recordings'], filter(
            lambda file_path: file_path.endswith('.npz'), os.listdir('./data/reservoir-1')
        )
    )
    reservoir_2_reservoir_recordings = map(
        lambda file_path: np.load(f'./data/reservoir-2/{file_path}')['recordings'], filter(
            lambda file_path: file_path.endswith('.npz'), os.listdir('./data/reservoir-2')
        )
    )

    chained = itertools.chain(
        null_reservoir_recordings,
        reservoir_1_reservoir_recordings,
        reservoir_2_reservoir_recordings
    )

    baseline_results, readout_results = [], []
    for reservoir_recordings in itertools.chain(
            null_reservoir_recordings,
            reservoir_1_reservoir_recordings,
            reservoir_2_reservoir_recordings
    ):
        baseline_performance, readout_performance = run_experiment(
            reservoir_recordings, LinearRegression
        )

        baseline_results.append(baseline_performance)
        readout_results.append(readout_performance)

    null_baseline_results = np.array(baseline_results[:NULL_COUNT])
    null_readout_results = np.array(readout_results[:NULL_COUNT])

    reservoir_baseline_results = np.array(baseline_results[NULL_COUNT:])
    reservoir_readout_results = np.array(readout_results[NULL_COUNT:])

    print(f'Top reservoir performance: {np.min(reservoir_readout_results)}')

    print(f'Average baseline performance: {np.mean(reservoir_baseline_results)}')
    print(f'Average reservoir performance: {np.mean(reservoir_readout_results)}')

    figure, axes = plt.subplots(figsize=(7.08, 6.7))

    null_group = (null_baseline_results, null_readout_results, 'gray', 'Null Reservoir')
    phys_group = (reservoir_baseline_results, reservoir_readout_results, '#FFCB05', 'Physical Reservoir')

    plot_interleaved_dumbbell(axes, null_group, phys_group)

    plt.tight_layout()
    plt.savefig('supplementary-reservoir-performances.pdf', dpi=600, bbox_inches='tight')
