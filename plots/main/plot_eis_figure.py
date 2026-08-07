import os
import re

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from matplotlib.transforms import ScaledTranslation

from typing import Any


def get_sort_key(filename: str) -> Any:
    numbers = re.findall(r'\d+', filename)

    return int(re.findall(r'\d+', filename)[-1]) if numbers else filename


class Figure3:

    def __init__(
            self,
            figure: plt.Figure,
            **kwargs
    ):
        self.figure = figure

        self.gs_top = GridSpec(
            1, 4, figure=figure, left=0.08, right=0.98, top=0.96, bottom=0.81, wspace=0.5
        )
        self.gs_bottom = GridSpec(
            3, 4, figure=figure, left=0.08, right=0.98, top=0.73, bottom=0.04, hspace=0.25, wspace=0.85
        )

        self.ax_a = figure.add_subplot(self.gs_top[0, 0])
        self.ax_b = figure.add_subplot(self.gs_top[0, 1])
        self.ax_c = figure.add_subplot(self.gs_top[0, 2])
        self.ax_d = figure.add_subplot(self.gs_top[0, 3])

        self.ax_e = figure.add_subplot(self.gs_bottom[0, 0:2])
        self.ax_f = figure.add_subplot(self.gs_bottom[0, 2:4])

        self.ax_g = figure.add_subplot(self.gs_bottom[1, 0:2])
        self.ax_h = figure.add_subplot(self.gs_bottom[1, 2:4])

        self.ax_i = figure.add_subplot(self.gs_bottom[2, 0:2])
        self.ax_j = figure.add_subplot(self.gs_bottom[2, 2:4])

        self.palette = kwargs.get(
            'palette', [
                '#FFCB05',
                '#E5A800',
                '#B6862C',
                '#8C621A',
                '#5F410E'
            ]
        )

    @staticmethod
    def apply_axis_style(ax: plt.Axes) -> None:
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

    @staticmethod
    def set_titles_and_labels(
            ax_a: plt.Axes,
            ax_b: plt.Axes,
            label: str,
            figure: plt.Figure,
            is_top_row: bool = False,
            hide_titles: bool = False,
            hide_xlabels: bool = False
    ) -> None:
        if not hide_titles:
            ax_a.set_title('Nyquist Plot', fontsize=7)
            ax_b.set_title('Bode Plot', fontsize=7)
        else:
            ax_a.set_title('')
            ax_b.set_title('')

        if not hide_xlabels:
            ax_a.set_xlabel("Z' (M\u03a9)", fontsize=6)
            ax_b.set_xlabel('Frequency (Hz)', fontsize=6)
        else:
            ax_a.set_xlabel('')
            ax_b.set_xlabel('')

        ax_a.set_ylabel("-Z'' (M\u03a9)", fontsize=6)

        if is_top_row:
            ax_b.set_ylabel('', fontsize=6)
            ax_b.yaxis.set_tick_params(left=True, labelleft=True, which='both')
            ax_b.yaxis.set_major_locator(
                matplotlib.ticker.LogLocator(base=10.0, subs='auto', numticks=3)
            )
            ax_b.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        else:
            ax_b.set_ylabel('Impedance Magnitude (M\u03a9)', fontsize=6)

        ax_a.text(
            0.0,
            1.0,
            label,
            transform=(
                    ax_a.transAxes +
                    ScaledTranslation(-10 / 72, 7 / 72, figure.dpi_scale_trans)
            ),
            fontsize=7,
            fontweight='bold',
            va='bottom',
            ha='right'
        )

    @staticmethod
    def plot_subset(
            subset: str,
            ax_a: plt.Axes,
            ax_b: plt.Axes,
            palette: list[str],
            source_directory: str = './data/eis',
            show_legend: bool = True
    ) -> None:
        subset_path = os.path.join(source_directory, subset.strip('/'))

        final_frequencies = None

        final_z_real = None
        final_z_imag = None

        raw_files = os.listdir(subset_path)
        data_files = [f for f in raw_files if f.endswith(('.csv', '.txt'))]

        sorted_files = sorted(data_files, key=get_sort_key)

        for i, filename in enumerate(sorted_files):
            colour = palette[i % len(palette)]
            filepath = os.path.join(subset_path, filename)

            raw_data = np.loadtxt(filepath, delimiter=',', skiprows=1)

            file_frequencies = raw_data[:, 0]

            z_real = raw_data[:, 1] / 1e6
            z_imag = raw_data[:, 2] / 1e6

            z_mag = np.sqrt(z_real ** 2 + z_imag ** 2)

            numbers = re.findall(r'\d+', filename)
            if numbers:
                name = f'Run {numbers[-1]}'
            else:
                name = os.path.splitext(filename)[0]

            ax_a.plot(z_real, -z_imag, 'o-', color=colour, lw=1.5, markersize=4, alpha=0.85, label=name)
            ax_b.loglog(file_frequencies, z_mag, 's-', color=colour, lw=1.5, markersize=3, alpha=0.85)

            final_frequencies = file_frequencies

            final_z_real = z_real
            final_z_imag = z_imag

        if final_frequencies is not None:
            for f, rx, ix in zip(final_frequencies, final_z_real, final_z_imag):
                is_milestone = any(
                    np.isclose(f, target, atol=1e-5) for target in [0.1, 1.0, 10.0, 100.0, 1000.0]
                )
                if is_milestone:
                    target_val = next(
                        target for target in [0.1, 1.0, 10.0, 100.0, 1000.0] if np.isclose(f, target, atol=1e-5)
                    )
                    ax_a.annotate(
                        f' {target_val} Hz',
                        (rx, -ix),
                        textcoords='offset points',
                        xytext=(4, 4),
                        ha='left',
                        fontsize=5,
                        color='#444444',
                        fontweight='bold'
                    )

        if show_legend:
            ax_a.legend(
                fontsize=5,
                ncols=1,
                frameon=False,
                loc='best',
            )


if __name__ == '__main__':
    matplotlib.use('TkAgg')

    plt.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['pdf.fonttype'] = 42

    plt.rcParams['ytick.labelsize'] = 6
    plt.rcParams['xtick.labelsize'] = 6

    palette = [
        '#FFCB05',
        '#E5A800',
        '#B6862C',
        '#8C621A',
        '#5F410E'
    ]

    fig = plt.figure(figsize=(7.08, 6.7))

    plotter = Figure3(figure=fig, palette=palette)

    for ax in [
        plotter.ax_a,
        plotter.ax_b,
        plotter.ax_c,
        plotter.ax_d
    ]:
        plotter.apply_axis_style(ax)

    plotter.set_titles_and_labels(
        plotter.ax_a,
        plotter.ax_b,
        '(a)',
        figure=fig,
        is_top_row=True,
        hide_titles=False,
        hide_xlabels=False
    )
    plotter.set_titles_and_labels(
        plotter.ax_c,
        plotter.ax_d,
        '(b)',
        figure=fig,
        is_top_row=True,
        hide_titles=False,
        hide_xlabels=False
    )

    plotter.plot_subset(
        'resistor',
        plotter.ax_a,
        plotter.ax_b,
        palette=plotter.palette,
        show_legend=False
    )
    plotter.plot_subset(
        'open-air-agar',
        plotter.ax_c,
        plotter.ax_d,
        palette=plotter.palette,
        show_legend=False
    )

    plotter.apply_axis_style(plotter.ax_e)
    plotter.apply_axis_style(plotter.ax_f)
    plotter.set_titles_and_labels(
        plotter.ax_e,
        plotter.ax_f,
        '(c)',
        figure=fig,
        is_top_row=False,
        hide_titles=False,
        hide_xlabels=True
    )
    plotter.plot_subset(
        'slime-mould/device-2/before-bias',
        plotter.ax_e,
        plotter.ax_f,
        palette=plotter.palette,
        show_legend=True
    )

    plotter.apply_axis_style(plotter.ax_g)
    plotter.apply_axis_style(plotter.ax_h)
    plotter.set_titles_and_labels(
        plotter.ax_g,
        plotter.ax_h,
        '(d)',
        figure=fig,
        is_top_row=False,
        hide_titles=True,
        hide_xlabels=True
    )
    plotter.plot_subset(
        'slime-mould/device-2/after-bias',
        plotter.ax_g,
        plotter.ax_h,
        palette=plotter.palette,
        show_legend=True
    )

    plotter.apply_axis_style(plotter.ax_i)
    plotter.apply_axis_style(plotter.ax_j)
    plotter.set_titles_and_labels(
        plotter.ax_i,
        plotter.ax_j,
        '(e)',
        figure=fig,
        is_top_row=False,
        hide_titles=True,
        hide_xlabels=False
    )
    plotter.plot_subset(
        'slime-mould/device-2/after-second-bias',
        plotter.ax_i,
        plotter.ax_j,
        palette=plotter.palette,
        show_legend=True
    )

    fig.align_ylabels([plotter.ax_e, plotter.ax_g, plotter.ax_i])
    fig.align_ylabels([plotter.ax_f, plotter.ax_h, plotter.ax_j])

    plt.savefig('figure-3.pdf', dpi=600, bbox_inches='tight')
