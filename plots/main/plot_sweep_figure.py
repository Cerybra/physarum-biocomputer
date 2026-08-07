import json

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from matplotlib.transforms import ScaledTranslation

from typing import Iterable, Optional


def load_json_sweep_data(filename: str) -> dict:
    with open(filename, 'r') as file:
        return json.load(file)


def get_device_sweeps(
        json_data: dict[str, list[dict[str, int | list[float]]]],
        device: str
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    trials = json_data.get(device, [])

    voltages = [np.array(trial.get('Voltages', [])) for trial in trials]
    currents = [np.array(trial.get('Currents', [])) for trial in trials]

    return voltages, currents


class Figure1:

    def __init__(self, figure: plt.Figure, **kwargs):
        self.figure = figure
        self.gs = GridSpec(7, 9, figure=figure)

        self.ax_a = figure.add_subplot(self.gs[0:2, 0:3])
        self.ax_b = figure.add_subplot(self.gs[0:2, 3:6])
        self.ax_c = figure.add_subplot(self.gs[0:2, 6:9])
        self.ax_d = figure.add_subplot(self.gs[2:4, 0:3])
        self.ax_e = figure.add_subplot(self.gs[2:4, 3:6])
        self.ax_f = figure.add_subplot(self.gs[2:4, 6:9])

        self.ax_g = figure.add_subplot(self.gs[4:7, 0:9])

        self.palette = kwargs.get(
            'palette',
            {
                'main': '#FFCB05',
                'deep': '#B6862C',
                'error': 'gray',
                'target': 'black',
            }
        )

    @staticmethod
    def plot_single_sweep(
        v: np.ndarray,
        i: np.ndarray,
        ax: plt.Axes,
        label: str,
        palette: dict[str, str],
        scale_current: float = 1e-8,
    ) -> None:
        ax.set_facecolor('white')

        ax.spines[['top', 'right', 'bottom']].set_visible(False)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['left'].set_color('black')

        ax.set_xlabel('Voltage (V)', fontsize=6)
        ax.set_ylabel('Current (A)', fontsize=6)

        ax.tick_params(colors='black', labelsize=5, which='both')

        ax.xaxis.get_offset_text().set_fontsize(6)
        ax.yaxis.get_offset_text().set_fontsize(6)

        ax.plot(v, i * scale_current, color=palette['main'], linewidth=1.5)

        ax.text(
            0.0,
            1.0,
            label,
            transform=(
                ax.transAxes
                + ScaledTranslation(-10 / 72, 7 / 72, ax.figure.dpi_scale_trans)
            ),
            fontsize=7,
            fontweight='bold',
            va='bottom',
            ha='right',
        )

    @staticmethod
    def plot_multiple_sweep(
        voltages: Iterable[np.ndarray],
        currents: Iterable[np.ndarray],
        ax: plt.Axes,
        label: str,
        palette: dict[str, str],
        trial: Optional[int] = None,
        scale_current: float = 1e-8,
    ) -> None:
        ax.set_facecolor('white')

        ax.spines[['top', 'right', 'bottom']].set_visible(False)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['left'].set_color('black')

        ax.set_xlabel('Voltage (V)', fontsize=6)
        ax.set_ylabel('Current (A)', fontsize=6)

        ax.tick_params(colors='black', labelsize=5, which='both')

        ax.xaxis.get_offset_text().set_fontsize(6)
        ax.yaxis.get_offset_text().set_fontsize(6)

        voltages_list = list(voltages)
        currents_list = list(currents)

        if trial is None:
            for trial_voltages, trial_currents in zip(
                voltages_list[:-1], currents_list[:-1]
            ):
                ax.plot(
                    trial_voltages,
                    trial_currents * scale_current,
                    color=palette['main'],
                    alpha=0.2,
                    linewidth=1.5,
                )
            trial = -1

        ax.plot(
            voltages_list[trial],
            currents_list[trial] * scale_current,
            color=palette['main'],
            linewidth=1.5,
        )

        ax.text(
            0.0,
            1.0,
            label,
            transform=(
                ax.transAxes
                + ScaledTranslation(-10 / 72, 7 / 72, ax.figure.dpi_scale_trans)
            ),
            fontsize=7,
            fontweight='bold',
            va='bottom',
            ha='right',
        )

    @staticmethod
    def plot_multi_sweep_average(
        voltages: Iterable[np.ndarray],
        currents: Iterable[np.ndarray],
        ax: plt.Axes,
        label: str,
        palette: dict[str, str],
        scale_current: float = 1e-8,
    ) -> None:
        ax.set_facecolor('white')

        ax.spines[['top', 'right', 'bottom']].set_visible(False)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['left'].set_color('black')

        ax.set_xlabel('Voltage (V)', fontsize=6)
        ax.set_ylabel('Current (A)', fontsize=6)

        ax.tick_params(colors='black', labelsize=5, which='both')

        ax.xaxis.get_offset_text().set_fontsize(6)
        ax.yaxis.get_offset_text().set_fontsize(6)

        v_matrix = np.array(list(voltages))
        i_matrix = np.array(list(currents)) * scale_current

        average = np.mean(i_matrix, axis=0)
        std = np.std(i_matrix, axis=0)

        upper = average + std
        lower = average - std

        ax.plot(v_matrix[0], average, color=palette['main'], linewidth=1.5)

        ax.fill_between(
            v_matrix[0],
            lower,
            upper,
            color=palette['main'],
            alpha=0.2,
        )

        ax.text(
            0.0,
            1.0,
            label,
            transform=(
                ax.transAxes
                + ScaledTranslation(-10 / 72, 7 / 72, ax.figure.dpi_scale_trans)
            ),
            fontsize=7,
            fontweight='bold',
            va='bottom',
            ha='right',
        )

    def plot(self, json_data: dict) -> None:
        self.plot_multiple_sweep(
            *get_device_sweeps(json_data, 'Device 60'),
            self.ax_a,
            label='(a)',
            palette=self.palette,
            trial=0
        )

        self.plot_multiple_sweep(
            *get_device_sweeps(json_data, 'Device 3'),
            self.ax_b,
            label='(b)',
            palette=self.palette,
            trial=4
        )

        self.plot_multiple_sweep(
            *get_device_sweeps(json_data, 'Device 4'),
            self.ax_c,
            label='(c)',
            palette=self.palette,
            trial=3
        )

        self.plot_multi_sweep_average(
            *get_device_sweeps(json_data, 'Device 55'),
            self.ax_d,
            label='(d)',
            palette=self.palette
        )

        self.plot_multi_sweep_average(
            *get_device_sweeps(json_data, 'Device 59'),
            self.ax_e,
            label='(e)',
            palette=self.palette
        )

        self.plot_multiple_sweep(
            *get_device_sweeps(json_data, 'Device 15'),
            self.ax_f,
            label='(f)',
            palette=self.palette,
            trial=4
        )


if __name__ == '__main__':
    matplotlib.use('TkAgg')

    plt.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['pdf.fonttype'] = 42

    json_dataset = load_json_sweep_data('./data/device-sweep-recordings/device-sweep-recordings.json')

    fig = plt.figure(figsize=(7.08, 6.7))

    plotter = Figure1(figure=fig)
    plotter.plot(json_dataset)

    fig.subplots_adjust(wspace=2.0, hspace=1.0)

    plt.savefig('figure-1.pdf', dpi=600, bbox_inches='tight')
