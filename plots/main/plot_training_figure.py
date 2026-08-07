import csv

import dataclasses

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import ScaledTranslation

from matplotlib.gridspec import GridSpec

from mpl_toolkits.axes_grid1 import make_axes_locatable


def load_matching_recordings_data(filename: str) -> tuple[list[float], list[float], list[float]]:
    weights, updates, targets = [], [], []

    with open(filename, 'r') as file:
        reader = csv.DictReader(file)

        for row in reader:
            for key, value in row.items():
                if 'Weights' in key:
                    weights.append(float(value))
                elif 'Updates' in key:
                    updates.append(float(value))
                else:
                    pass

            targets.append(row['Targets'])

    return weights, updates, targets


def load_regression_recordings_data(filename: str) -> tuple[list[float], list[float], list[float]]:
    predictions, weights, targets = [], [], []

    with open(filename, 'r') as file:
        for i, line in enumerate(file.readlines()):
            if i == 0:
                continue

            prediction, target, weight, _ = line.split(',')

            predictions.append(float(prediction))
            weights.append(float(weight))
            targets.append(float(target))

    return predictions, weights, targets


@dataclasses.dataclass
class FileNames:
    training_weight_conductance_files: list[str]
    training_weight_conductance_targets: list[float]

    regression_weight_conductance_files: list[str]
    regression_weight_conductance_targets: list[float]

    regression_line_files: list[str]
    regression_line_targets: list[float]


@dataclasses.dataclass
class WeightMatchingExperiment:
    predictions: list[float]
    updates: list[float]
    targets: list[float]


@dataclasses.dataclass
class RegressionExperiment:
    predictions: list[float]
    weights: list[float]
    targets: list[float]


class Figure5:

    def __init__(
            self,
            figure: plt.Figure,
            weight_matching_files: list[str],
            weight_matching_targets: list[float],
            regression_files: list[str],
            regression_line_slopes: list[float],
            **kwargs
    ):
        self.weight_matching_files = weight_matching_files
        self.weight_matching_data = []
        for filename in self.weight_matching_files:
            predictions, updates, targets = load_matching_recordings_data(filename)
            self.weight_matching_data.append(
                WeightMatchingExperiment(
                    predictions=predictions,
                    updates=updates,
                    targets=targets
                )
            )
        self.weight_matching_targets = weight_matching_targets

        self.regression_files = regression_files
        self.regression_data = []
        for filename in self.regression_files:
            predictions, weights, targets = load_regression_recordings_data(filename)
            self.regression_data.append(
                RegressionExperiment(
                    predictions=predictions,
                    weights=weights,
                    targets=targets
                )
            )
        self.regression_line_slopes = regression_line_slopes

        self.gs = GridSpec(4, 4, figure=figure)

        self.ax_a = figure.add_subplot(self.gs[0, 0])
        self.ax_b = figure.add_subplot(self.gs[0, 1])
        self.ax_c = figure.add_subplot(self.gs[0, 2])
        self.ax_d = figure.add_subplot(self.gs[0, 3])

        self.ax_e = figure.add_subplot(self.gs[1, 0])
        self.ax_f = figure.add_subplot(self.gs[1, 1])
        self.ax_g = figure.add_subplot(self.gs[1, 2])
        self.ax_h = figure.add_subplot(self.gs[1, 3])

        self.ax_i = figure.add_subplot(self.gs[2, 0:2])
        self.ax_j = figure.add_subplot(self.gs[2, 2:])

        self.ax_k = fig.add_subplot(self.gs[3, 0:2])
        self.ax_l = fig.add_subplot(self.gs[3, 2:])

        self.palette = kwargs.get(
            'palette',  {
                'main': '#FFCB05',
                'deep': '#B6862C',
                'error': 'gray',
                'target': 'black'
            }
        )

    @staticmethod
    def plot_training_weight_conductance(
            weights: np.ndarray,
            axes: tuple[plt.Axes, plt.Axes],
            labels: tuple[str, str],
            palette: dict[str, str],
            target: float,
            log_scale: bool = False
    ) -> None:
        gold_cmap = LinearSegmentedColormap.from_list(
            'gold_gradient',
            [palette['deep'], palette['main']],
            N=256
        )

        for i, ax in enumerate(axes):
            ax.set_facecolor('white')

            ax.spines[['top', 'right', 'bottom']].set_visible(False)
            ax.spines['left'].set_linewidth(1.0)
            ax.spines['left'].set_color('black')

            ax.tick_params(colors='black', labelsize=5, which='both')

            ax.set_ylabel('Conductance', fontsize=6)

            ax.text(
                0.0,
                1.0,
                labels[i],
                transform=(
                        ax.transAxes +
                        ScaledTranslation(-10 / 72, 7 / 72, fig.dpi_scale_trans)
                ),
                fontsize=7,
                fontweight='bold',
                va='bottom',
                ha='right'
            )

        lower_bound = 0.9 * target
        upper_bound = 1.1 * target
        for ax in axes:
            ax.axhspan(
                lower_bound,
                upper_bound,
                color=palette['error'],
                alpha=0.05,
                linewidth=0
            )

        n = len(weights)
        for i in range(1, n):
            color_intensity = i / n
            opacity = 0.2 + 0.8 * color_intensity
            axes[0].plot(
                weights[(i - 1):(i + 1)],
                color=gold_cmap(color_intensity),
                linewidth=1.5,
                alpha=opacity,
                solid_capstyle='round'
            )

        axes[0].set_title('Conductance Steps', fontdict={'fontsize': 6})

        axes[0].axhline(
            target,
            color=palette['target'],
            linestyle='--',
            linewidth=0.75,
            alpha=0.8
        )

        divider = make_axes_locatable(axes[0])

        gradient = np.linspace(0, 1, 256).reshape(-1, 1)

        cax = divider.append_axes('right', size='3%')
        cax.imshow(gradient, aspect='auto', cmap=gold_cmap, extent=[0, 1, 0, 1])
        cax.set_axis_off()

        x = np.arange(len(weights))

        axes[1].set_title('Conductance Values', fontdict={'fontsize': 6})

        axes[1].set_xlabel('Iteration', fontsize=6)

        axes[1].plot(x, weights, color=palette['main'], linewidth=1.5)
        axes[1].axhline(
            target,
            color=palette['target'],
            linestyle='--',
            linewidth=0.75,
            alpha=0.8
        )

        if log_scale:
            axes[1].set_yscale('log')

    @staticmethod
    def plot_regression_weight_conductance(
            weights: list[float],
            ax: plt.Axes,
            label: str,
            palette: dict[str, str],
            target: float
    ) -> None:
        ax.set_facecolor('white')

        ax.set_xlabel('Timesteps', fontsize=6)
        ax.set_ylabel('Weight Value', fontsize=6)

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

        ax.plot(weights, alpha=0.9, color=palette['main'], linewidth=1.5)
        ax.plot(
            target * np.ones(100,),
            color='black',
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
                    ScaledTranslation(-10 / 72, 7 / 72, fig.dpi_scale_trans)
            ),
            fontsize=7,
            fontweight='bold',
            va='bottom',
            ha='right'
        )

    @staticmethod
    def plot_regression_line(
            predictions: list[float],
            targets: list[float],
            ax: plt.Axes,
            label: str,
            palette: dict[str, str],
            target: float
    ) -> None:
        ax.set_facecolor('white')

        ax.set_xlabel('Input (X)', fontsize=6)
        ax.set_ylabel('Response (Y)', fontsize=6)

        ax.tick_params(colors='black', labelsize=5, which='both')

        ax.xaxis.set_label_position('bottom')
        ax.xaxis.set_tick_params(
            pad=2,
            labelbottom=True,
            bottom=True,
            labelsize=5,
            labelrotation=0,
            color='black',
        )

        ax.spines[['top', 'right', 'bottom']].set_visible(False)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['left'].set_color('black')

        x = np.linspace(0, 1, 100)

        ax.scatter(
            x=x,
            y=targets,
            alpha=0.3,
            color=palette['main'],
            edgecolor='none',
            label='Data',
            s=10
        )
        ax.scatter(
            x=x,
            y=predictions,
            alpha=0.8,
            color=palette['main'],
            edgecolor='none',
            label='Predictions',
            s=10
        )

        ax.plot(
            x,
            (target * x),
            alpha=0.8,
            color='black',
            linewidth=0.75,
            linestyle='--',
            label='Initial Line'
        )

        ax.text(
            0.0,
            1.0,
            label,
            transform=(
                    ax.transAxes +
                    ScaledTranslation(-10 / 72, 7 / 72, fig.dpi_scale_trans)
            ),
            fontsize=7,
            fontweight='bold',
            va='bottom',
            ha='right'
        )

    def plot(self) -> None:
        self.plot_training_weight_conductance(
            np.reshape(
                np.array(self.weight_matching_data[0].predictions),
                (-1, 4)
            )[:, 0],
            (self.ax_a, self.ax_b),
            ('(a)', '(b)'),
            self.palette,
            self.weight_matching_targets[0],
            log_scale=True
        )
        self.plot_training_weight_conductance(
            np.reshape(
                np.array(self.weight_matching_data[2].updates),
                (-1, 4)
            )[:, 0],
            (self.ax_c, self.ax_d),
            ('(c)', '(d)'),
            self.palette,
            self.weight_matching_targets[0],
            log_scale=False
        )
        self.plot_training_weight_conductance(
            np.reshape(
                np.array(self.weight_matching_data[1].predictions),
                (-1, 4)
            )[:, 2],
            (self.ax_e, self.ax_f),
            ('(e)', '(f)'),
            self.palette,
            self.weight_matching_targets[1],
            log_scale=True
        )
        self.plot_training_weight_conductance(
            np.reshape(
                np.array(self.weight_matching_data[2].updates),
                (-1, 4)
            )[:, 1],
            (self.ax_g, self.ax_h),
            ('(g)', '(h)'),
            self.palette,
            self.weight_matching_targets[1],
            log_scale=False
        )

        self.plot_regression_line(
            self.regression_data[0].predictions,
            self.regression_data[0].targets,
            self.ax_i,
            '(i)',
            self.palette,
            self.regression_line_slopes[0]
        )
        self.plot_regression_weight_conductance(
            self.regression_data[0].weights,
            self.ax_j,
            '(j)',
            self.palette,
            self.regression_line_slopes[0]
        )
        self.plot_regression_line(
            self.regression_data[1].predictions,
            self.regression_data[1].targets,
            self.ax_k,
            '(k)',
            self.palette,
            self.regression_line_slopes[1]
        )
        self.plot_regression_weight_conductance(
            self.regression_data[1].weights,
            self.ax_l,
            '(l)',
            self.palette,
            self.regression_line_slopes[1]
        )


if __name__ == '__main__':
    matplotlib.use('TkAgg')

    plt.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ytick.labelsize'] = 6
    plt.rcParams['xtick.labelsize'] = 6

    fig = plt.figure(figsize=(7.08, 6.7))

    plotter = Figure5(
        figure=fig,
        weight_matching_files=[
            './data/device-regression-recordings/experiment-2/regression-with-grad-target-0.43.csv',
            './data/device-regression-recordings/experiment-2/regression-with-grad-target-0.72.csv',
            './data/device-regression-recordings/experiment-2/regression-without-grad.csv',
        ],
        weight_matching_targets=[0.43, 0.72, 0.43, 0.72],
        regression_weight_conductance_files=[
            './data/device-regression-recordings/experiment-1/regression-with-grad-target-0.44.csv',
            './data/device-regression-recordings/experiment-1/regression-with-grad-target-0.44.csv',
        ],
        regression_files=[
            './data/device-regression-recordings/experiment-1/regression-with-grad-target-0.44.csv',
            './data/device-regression-recordings/experiment-1/regression-without-grad-target-0.44.csv',
        ],
        regression_line_slopes=[0.43, 0.43]
    )
    plotter.plot()

    fig.set_layout_engine('constrained')
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01)

    plt.savefig('figure-5.pdf', dpi=600)
