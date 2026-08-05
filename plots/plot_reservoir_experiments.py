import os

import ast

import csv

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from matplotlib.transforms import ScaledTranslation

from sklearn.linear_model import (
    LinearRegression, LogisticRegression
)
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from typing import Any, Optional


def load_temporal_xor_recordings(file_path: str) -> np.ndarray:
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=',')

        next(reader)
        for row in reader:
            data.append([
                *map(lambda value: ast.literal_eval(value), row)
            ])

    return np.vstack(data)


class Figure6:

    def __init__(
            self,
            figure: plt.Figure,
            **kwargs
    ):
        self.figure = figure

        self.gs = GridSpec(5, 4, figure=figure)

        self.ax_a = figure.add_subplot(self.gs[0:2, 0:2])
        self.ax_b = figure.add_subplot(self.gs[2:3, 0:2])

        self.ax_c = figure.add_subplot(self.gs[0:1, 2:4])
        self.ax_d = figure.add_subplot(self.gs[1:2, 2:4])
        self.ax_e = figure.add_subplot(self.gs[2:3, 2:4])

        self.palette = kwargs.get(
            'palette', {
                'main': '#FFCB05',
                'deep': '#B6862C',
                'error': 'gray',
                'target': 'black'
            }
        )

    @staticmethod
    def plot_temporal_xor(
            steps: np.ndarray,
            series: np.ndarray,
            figure: plt.Figure,
            ax: plt.Axes,
            label: str,
            palette: dict[str, str]
    ) -> None:
        ax.set_facecolor('white')

        ax.set_xlabel('Timesteps', fontsize=6)
        ax.set_ylabel('Value', fontsize=6)

        ax.tick_params(colors='black', labelsize=5, which='both')

        ax.xaxis.set_label_position('bottom')
        ax.xaxis.set_tick_params(
            pad=2,
            labelbottom=True,
            bottom=True,
            labelsize=5,
            labelrotation=0,
            color='black'
        )

        ax.spines[['top', 'right', 'bottom']].set_visible(False)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['left'].set_color('black')

        ax.step(steps, series, alpha=0.9, color=palette['target'], linewidth=1.5)

        ax.text(
            0.0,
            1.0,
            label,
            transform=(
                    ax.transAxes +
                    ScaledTranslation(-10 / 72, 7 / 72, figure.dpi_scale_trans)
            ),
            fontsize=7,
            fontweight='bold',
            va='bottom',
            ha='right'
        )

    @staticmethod
    def plot_temporal_xor_results(
            steps: np.ndarray,
            baseline: np.ndarray,
            predictions: np.ndarray,
            figure: plt.Figure,
            ax: plt.Axes,
            label: str,
            palette: dict[str, str],
            title: Optional[str] = None
    ) -> None:
        ax.set_facecolor('white')

        if isinstance(title, str):
            ax.set_title(title, fontsize=7)

        ax.set_xlabel('Timesteps', fontsize=6)
        ax.set_ylabel('Value', fontsize=6)

        ax.tick_params(colors='black', labelsize=5, which='both')

        ax.xaxis.set_label_position('bottom')
        ax.xaxis.set_tick_params(
            pad=2,
            labelbottom=True,
            bottom=True,
            labelsize=5,
            labelrotation=0,
            color='black'
        )

        ylims = ax.get_ylim()
        ax.set_yticks([ylims[0], ylims[1]])

        ax.spines[['top', 'right', 'bottom']].set_visible(False)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['left'].set_color('black')

        ax.step(steps, predictions, alpha=0.9, color=palette['main'], linewidth=1.5)
        ax.step(
            steps,
            baseline,
            color=palette['error'],
            alpha=0.8,
            linewidth=0.75,
            linestyle='--'
        )

        ax.text(
            0.0,
            1.0,
            label,
            transform=(
                    ax.transAxes +
                    ScaledTranslation(-10 / 72, 7 / 72, figure.dpi_scale_trans)
            ),
            fontsize=7,
            fontweight='bold',
            va='bottom',
            ha='right'
        )

    @staticmethod
    def plot_temporal_xor_trial_performances(
            figure: plt.Figure,
            ax: plt.Axes,
            label: str,
            palette: dict[str, str],
    ) -> None:
        ax.set_facecolor('white')

        ax.set_title('Temporal XOR Trials', fontsize=7)

        ax.set_xlabel('Accuracy', fontsize=6)
        ax.set_ylabel('Counts', fontsize=6)

        ax.tick_params(colors='black', labelsize=5, which='both')

        ax.xaxis.set_label_position('bottom')
        ax.xaxis.set_tick_params(
            pad=2,
            labelbottom=True,
            bottom=True,
            labelsize=5,
            labelrotation=0,
            color='black'
        )

        ax.spines[['top', 'right', 'bottom']].set_visible(False)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['left'].set_color('black')

        # Taken directly from the training results run on the microcontroller.
        ax.hist([
            [0.5017, 0.5017, 0.66, 0.66, 0.66],
            [1.0, 1.0, 1.0, 0.92, 0.93, 0.90]
        ],
            bins=15,
            color=[palette['error'], palette['main']],
            density=True
        )

        ax.text(
            0.0,
            1.0,
            label,
            transform=(
                    ax.transAxes +
                    ScaledTranslation(-10 / 72, 7 / 72, figure.dpi_scale_trans)
            ),
            fontsize=7,
            fontweight='bold',
            va='bottom',
            ha='right'
        )

    @staticmethod
    def plot_ode_prediction(
            x_true: np.ndarray,
            y_true: np.ndarray,
            baseline_predictions: np.ndarray,
            readout_predictions: np.ndarray,
            figure: plt.Figure,
            ax: plt.Axes,
            label: str,
            palette: dict[str, str],
            title: Optional[str] = None
    ) -> None:
        ax.set_facecolor('white')

        if isinstance(title, str):
            ax.set_title(title, fontsize=7)

        ax.set_xlabel('X Coordinate', fontsize=6)
        ax.set_ylabel('Y Coordinate', fontsize=6)

        ax.tick_params(colors='black', labelsize=5, which='both')

        ax.xaxis.set_label_position('bottom')
        ax.xaxis.set_tick_params(
            pad=2,
            labelbottom=True,
            bottom=True,
            labelsize=5,
            labelrotation=0,
            color='black'
        )

        ax.spines[['top', 'right', 'bottom']].set_visible(False)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['left'].set_color('black')

        ax.plot(
            x_true,
            y_true,
            color=palette['error'],
            alpha=0.45,
            linewidth=5
        )

        ax.plot(
            x_true,
            baseline_predictions,
            color='black',
            alpha=0.35,
            linewidth=1.0,
            linestyle='--'
        )

        ax.plot(
            x_true,
            readout_predictions,
            color=palette['main'],
            alpha=0.9,
            linewidth=1.5
        )

        ax.text(
            0.0,
            1.0,
            label,
            transform=(
                    ax.transAxes +
                    ScaledTranslation(-10 / 72, 7 / 72, figure.dpi_scale_trans)
            ),
            fontsize=7,
            fontweight='bold',
            va='bottom',
            ha='right'
        )

    @staticmethod
    def plot_ode_prediction_series(
            times: np.ndarray,
            y_true: np.ndarray,
            baseline_predictions: np.ndarray,
            readout_predictions: np.ndarray,
            figure: plt.Figure,
            ax: plt.Axes,
            label: str,
            palette: dict[str, str],
            title: Optional[str] = None
    ) -> None:
        ax.set_facecolor('white')

        ax.set_title(title, fontsize=7)

        ax.set_xlabel('Times', fontsize=6)
        ax.set_ylabel('Y Coordinate', fontsize=6)

        ax.tick_params(colors='black', labelsize=5, which='both')

        ax.xaxis.set_label_position('bottom')
        ax.xaxis.set_tick_params(
            pad=2,
            labelbottom=True,
            bottom=True,
            labelsize=5,
            labelrotation=0,
            color='black'
        )

        ax.spines[['top', 'right', 'bottom']].set_visible(False)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['left'].set_color('black')

        ax.plot(
            times,
            y_true,
            color=palette['error'],
            alpha=0.45,
            linewidth=5
        )

        ax.plot(
            times,
            baseline_predictions,
            color='black',
            alpha=0.35,
            linewidth=1.0,
            linestyle='--'
        )

        ax.plot(
            times,
            readout_predictions,
            color=palette['main'],
            alpha=0.9,
            linewidth=1.5
        )

        ax.text(
            0.0,
            1.0,
            label,
            transform=(
                    ax.transAxes +
                    ScaledTranslation(-10 / 72, 7 / 72, figure.dpi_scale_trans)
            ),
            fontsize=7,
            fontweight='bold',
            va='bottom',
            ha='right'
        )

    def plot(self) -> None:
        self.plot_temporal_xor(
            steps=range(50),
            series=self.reservoir_recordings_data[0][:, 0][:50],
            figure=self.figure,
            ax=self.ax_a,
            label='(b)',
            palette=self.palette
        )


def load_and_evaluate_ensemble(
        directories: list[str],
        readout_model: Any = None,
        baseline_model: Any = None,
        split_ratio: float = 0.8
) -> tuple[Any, Any, Any, np.ndarray]:
    if readout_model is None:
        readout_model = LinearRegression()
    if baseline_model is None:
        baseline_model = LinearRegression()

    file_paths = []
    for directory in directories:
        file_paths.extend([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npz')])

    raw_recordings = [np.load(f)['recordings'] for f in file_paths]

    if not raw_recordings:
        raise ValueError("No valid .npz files found.")

    global_mask = np.ones(raw_recordings[0].shape[0], dtype=bool)
    for rec in raw_recordings:
        global_mask &= (np.abs(rec).max(axis=1) < 20.0)

    recordings = [rec[global_mask] for rec in raw_recordings]

    base_features = recordings[0][:, [0, 2]]
    target = recordings[0][:, 3]
    split = int(len(target) * split_ratio)

    baseline_model.fit(base_features[:split], target[:split])
    y_baseline = baseline_model.predict(base_features)

    reservoir_predictions = []
    for rec in recordings:
        features = np.hstack([base_features, rec[:, 4:]])
        readout_model.fit(features[:split], target[:split])
        reservoir_predictions.append(readout_model.predict(features))

    return base_features, target, y_baseline, np.array(reservoir_predictions)


def apply_publication_style(
        ax: plt.Axes,
        xlabel: str = 'Time Steps',
        ylabel: str = 'Predicted'
) -> None:
    ax.set_facecolor('white')
    ax.set_xlabel(xlabel, fontsize=6)
    ax.set_ylabel(ylabel, fontsize=6)
    ax.tick_params(colors='black', labelsize=5, which='both')

    ax.xaxis.set_label_position('bottom')
    ax.xaxis.set_tick_params(pad=2, bottom=True, labelbottom=True, labelsize=5, color='black')

    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['left'].set_color('black')


def plot_average_ode_prediction(
        ax: plt.Axes,
        palette: dict[str, str],
        directories: list[str] = None
) -> None:
    if directories is None:
        directories = ['./data/reservoir-1', './data/reservoir-2']

    apply_publication_style(ax, xlabel='Base Feature X', ylabel='Predicted Y')

    base_features, target, y_baseline, res_preds = load_and_evaluate_ensemble(directories)
    x_coords = base_features[:, 0]

    ax.plot(x_coords, target, color='black', alpha=0.45, lw=5.0, label='Target', zorder=5)
    ax.plot(x_coords, y_baseline, color=palette['deep'], alpha=1.0, lw=1.5, ls='--', label='Baseline', zorder=4)
    ax.plot(x_coords, np.mean(res_preds, axis=0), color=palette['main'], lw=1.5, ls='-', label='Reservoir Average', zorder=6)

    ax.legend(fontsize=6, frameon=False)


def plot_average_ode_prediction_series(
        ax: plt.Axes,
        palette: dict[str, str],
        directories: list[str] = None
) -> None:
    if directories is None:
        directories = ['./data/reservoir-1', './data/reservoir-2']

    apply_publication_style(ax, xlabel='Time Steps', ylabel='Predicted Y')

    _, target, y_baseline, reservoir_predictions = load_and_evaluate_ensemble(directories)

    for i, pred in enumerate(reservoir_predictions):
        ax.plot(pred, color=palette['error'], alpha=0.1, lw=2.5, label='Individual Reservoirs' if i == 0 else None)

    ax.plot(target, color='black', lw=1.5, label='Target', zorder=5)
    ax.plot(y_baseline, color=palette['deep'], lw=2.5, ls='-', label='Linear Baseline', zorder=4)
    ax.plot(np.mean(reservoir_predictions, axis=0), color=palette['main'], lw=1.5, ls='-', label='Reservoir Average', zorder=6)

    ax.legend(fontsize=6, frameon=False)


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

    fig = plt.figure(figsize=(7.08, 6.7))

    plotter = Figure6(figure=fig)

    temporal_xor_data = load_temporal_xor_recordings('./data/temporal-xor-recordings.csv')
    van_der_pol_data = np.load('./data/reservoir-2/recordings-1773464097.530412-2.npz')['recordings']

    x_data, y_data = np.hstack(
        [temporal_xor_data[:, 0].reshape(-1, 1), temporal_xor_data[:, 2:]]
    ), temporal_xor_data[:, 1]

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        train_size=0.8,
        shuffle=False
    )

    baseline = LogisticRegression(penalty='l2')
    baseline.fit(x_train[:, 0].reshape(-1, 1), y_train)

    readout = LogisticRegression(penalty='l2')
    readout.fit(x_train[:, 1:], y_train)

    temporal_xor_baseline_score = accuracy_score(y_true=y_test, y_pred=baseline.predict(x_test[:, 0].reshape(-1, 1)))
    temporal_xor_reservoir_score = accuracy_score(y_true=y_test, y_pred=readout.predict(x_test[:, 1:]))

    print(f'Temporal XOR performance without reservoir: {temporal_xor_baseline_score}')
    print(f'Temporal XOR performance with reservoir: {temporal_xor_reservoir_score}')

    plot_average_ode_prediction(
        plotter.ax_a, palette
    )

    plot_average_ode_prediction_series(
        plotter.ax_b, palette
    )

    plotter.plot_temporal_xor_results(
        steps=range(50),
        baseline=x_test[:, 0][:50],
        predictions=y_test.reshape(-1,)[:50],
        figure=plotter.figure,
        ax=plotter.ax_c,
        label='(c)',
        palette=palette,
        title='Temporal XOR'
    )

    plotter.plot_temporal_xor_results(
        steps=range(50),
        baseline=baseline.predict(x_test[:, 0].reshape(-1, 1))[:50],
        predictions=readout.predict(x_test[:, 1:])[:50],
        figure=plotter.figure,
        ax=plotter.ax_d,
        label='(d)',
        palette=palette,
        title='Temporal XOR Predictions'
    )

    plotter.plot_temporal_xor_trial_performances(
        figure=plotter.figure,
        ax=plotter.ax_e,
        label='(e)',
        palette=palette
    )

    fig.set_layout_engine('constrained')
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01)

    plt.savefig('reservoir-experiments-figure.pdf', dpi=600)
