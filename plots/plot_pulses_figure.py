import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from matplotlib.pyplot import GridSpec
from matplotlib.transforms import ScaledTranslation


def load_pulse_data(filename: str, device_number: int) -> np.ndarray:
    df = pd.read_csv(filename)

    return df[df['Device Number'] ==
              device_number].sort_values('Pulse Number')['Value'].values


class Figure4:

    def __init__(
            self,
            figure: plt.Figure,
            **kwargs
    ):
        self.gs = GridSpec(12, 16, figure=figure)

        self.ax_a = figure.add_subplot(self.gs[0:4, :])
        self.ax_b = figure.add_subplot(self.gs[4:8, :])

        self.ax_c = figure.add_subplot(self.gs[8:12, :4])
        self.ax_d = figure.add_subplot(self.gs[8:12, 5:9])

        self.ax_e = figure.add_subplot(self.gs[8:12, 10:])

        self.palette = kwargs.get(
            'palette',  {
                'main': '#FFCB05',
                'deep': '#B6862C',
                'error': 'gray',
                'target': 'black'
            }
        )

    @staticmethod
    def plot_pulse_cycles(
            cycles: np.ndarray,
            ax: plt.Axes,
            label: str,
            palette: dict[str, str],
            swap_label: bool = False
    ) -> None:
        ax.set_facecolor('white')

        ax.grid(False)

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

        step = 50
        gray_band = 'gray'
        yellow_band = '#FFF7CC'
        for i in range(0, cycles.shape[0], step):
            color = gray_band if (i // step) % 2 else yellow_band
            ax.axvspan(
                i,
                min(i + step, cycles.shape[0]),
                facecolor=color,
                alpha=0.05,
                zorder=0
            )

        ax.plot(
            cycles,
            'o',
            color=palette['main'],
            alpha=1.0,
            markersize=2
        )

        ax.set_xlabel('Pulse Number', fontsize=6)
        ax.set_ylabel('Measured Current (A)', fontsize=6)

        ax.text(
            (1.0 if swap_label else 0.0),
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
    def plot_conductance_changes(
            cycles: np.ndarray,
            ax: plt.Axes,
            label: str,
            palette: dict[str, str]
    ) -> None:
        ax.set_facecolor('white')

        ax.grid(False)

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
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())

        ax.spines[['top', 'right', 'bottom']].set_visible(False)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['left'].set_color('black')

        ax.tick_params(colors='black', labelsize=5, which='both')

        ax.plot(
            cycles,
            'o',
            color=palette['main'],
            alpha=1.0,
            markersize=2
        )

        ax.set_xlabel('Pulse Number', fontsize=6)
        ax.set_ylabel('Conductance Change %', fontsize=6)

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

    def plot(
        self, 
        file_path: str = './data/device-pulse-recordings/legacy/device-pulse-recordings.csv', 
        scale_current: float = 1e-8
    ) -> None:
        self.plot_pulse_cycles(
            load_pulse_data(file_path, 13) * scale_current,
            self.ax_a,
            '(a)',
            self.palette
        )
        self.plot_pulse_cycles(
            load_pulse_data(file_path, 15) * scale_current,
            self.ax_b,
            '(b)',
            self.palette
        )

        positive_conductance_changes = load_pulse_data(file_path, 1)
        negative_conductance_changes = load_pulse_data(file_path, 23)

        self.plot_conductance_changes(
            -((positive_conductance_changes - positive_conductance_changes[0]) / positive_conductance_changes[0]) * 100,
            self.ax_c,
            '(c)',
            self.palette
        )
        self.plot_conductance_changes(
            ((negative_conductance_changes - negative_conductance_changes[0]) / negative_conductance_changes[0]) * 100,
            self.ax_d,
            '(d)',
            self.palette
        )

        self.plot_pulse_cycles(
            load_pulse_data(file_path, 27),
            self.ax_e,
            '(e)',
            self.palette,
            swap_label=True
        )


if __name__ == '__main__':
    matplotlib.use('TkAgg')

    plt.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['pdf.fonttype'] = 42

    plt.rcParams['ytick.labelsize'] = 6
    plt.rcParams['xtick.labelsize'] = 6

    fig = plt.figure(figsize=(7.08, 6.7))

    plotter = Figure4(figure=fig)
    plotter.plot()

    fig.subplots_adjust(wspace=0.5, hspace=6.0)

    plt.savefig(
        'figure-4.pdf', dpi=600, bbox_inches='tight'
    )
