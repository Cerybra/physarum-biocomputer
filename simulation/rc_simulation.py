import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from matplotlib.transforms import ScaledTranslation


if __name__ == '__main__':
    matplotlib.use('TkAgg')

    plt.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['pdf.fonttype'] = 42

    plt.rcParams['ytick.labelsize'] = 6
    plt.rcParams['xtick.labelsize'] = 6

    colours = [
        '#FFCB05',
        '#E5A800',
        '#B6862C',
        '#8C621A',
        '#5F410E'
    ]

    frequencies = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 250.0, 500.0, 1000.0])
    omega = 2 * np.pi * frequencies

    r1 = 3000e3
    c2 = 40e-6
    c1 = 8e-6
    r2 = 1000e3

    j = 1j

    # Impedances of the individual components.
    z_r1 = r1
    z_c2 = 1 / (j * omega * c2)
    z_r2 = r2
    z_c1 = 1 / (j * omega * c1)

    # Impedance of the parallel block (R2 || C1).
    z_parallel = (z_r2 * z_c1) / (z_r2 + z_c1)

    # Total circuit impedance.
    z_total = z_r1 + z_c2 + z_parallel

    z_total_momega = z_total / 1e6

    z_real = z_total_momega.real
    z_imag = z_total_momega.imag
    z_mag = np.abs(z_total_momega)
    z_phase = np.angle(z_total, deg=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.08, 6.7 / 2))

    for ax in [ax1, ax2]:
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

    ax1.plot(z_real, -z_imag, 'o-', color=colours[0], lw=1.5, markersize=4, alpha=0.85, label='Simulation')
    ax1.set_title('Nyquist Plot', fontsize=7)
    ax1.set_xlabel("Z' Real Impedance (M\u03a9)", fontsize=6)
    ax1.set_ylabel("-Z'' Imaginary Impedance (M\u03a9)", fontsize=6)

    # Plot (a): Nyquist plot.
    ax1.text(
        0.0,
        1.0,
        '(a)',
        transform=(
                ax1.transAxes +
                ScaledTranslation(-10 / 72, 7 / 72, fig.dpi_scale_trans)
        ),
        fontsize=7,
        fontweight='bold',
        va='bottom',
        ha='right'
    )

    # Plot (b): Bode plot.
    ax2.loglog(frequencies, z_mag, 's-', color=colours[0], lw=1.5, markersize=3, alpha=0.85)
    ax2.set_title('Bode Plot', fontsize=7)
    ax2.set_xlabel('Frequency (Hz)', fontsize=6)
    ax2.set_ylabel('Impedance Magnitude (M\u03a9)', fontsize=6)

    ax2.text(
        0.0,
        1.0,
        '(b)',
        transform=(
                ax2.transAxes +
                ScaledTranslation(-10 / 72, 7 / 72, fig.dpi_scale_trans)
        ),
        fontsize=7,
        fontweight='bold',
        va='bottom',
        ha='right'
    )

    for f in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        idx = np.abs(frequencies - f).argmin()
        rx = z_real[idx]
        ix = z_imag[idx]

        ax1.annotate(
            f' {f} Hz',
            (rx, -ix),
            textcoords='offset points',
            xytext=(4, 4),
            ha='left',
            fontsize=5,
            color='#444444',
            fontweight='bold'
        )

    ax1.legend(
        fontsize=7,
        ncols=1,
        frameon=False,
        loc='upper left',
    )

    plt.tight_layout()
    plt.savefig('rc-circuit-simulation.pdf', dpi=600, bbox_inches='tight')

    plt.show()
