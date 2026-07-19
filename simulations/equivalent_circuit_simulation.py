import ast

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from matplotlib.transforms import ScaledTranslation
from scipy.optimize import differential_evolution


def simulate_memristor(params, inputs, targets):
    simulated = np.zeros(len(inputs))

    g_off, g_on, v_th, aval_k, decay, ltp_rate, ltd_rate, c_off = params

    g_active, prev_v = g_off, 0.0
    for i, (input_v, target_v) in enumerate(zip(inputs, targets)):
        dv = input_v - prev_v

        current = (g_active * input_v) + (c_off * dv)
        simulated[i] = current

        error = current - target_v
        update_v = np.clip(-3.0 * input_v * error, -0.1, 0.1)

        if update_v > 0:
            dg_update = ltp_rate * update_v
        else:
            dg_update = ltd_rate * update_v

        if input_v > v_th:
            overvoltage = input_v - v_th
            step_factor = np.clip(aval_k * overvoltage, 0.0, 1.0)
            dg_avalanche = step_factor * (g_on - g_active)
        else:
            dg_avalanche = 0.0

        dg_decay = decay * (g_off - g_active)

        g_active = np.clip(
            g_active + dg_update + dg_avalanche + dg_decay, g_off, g_on
        )
        prev_v = input_v

    return simulated


def objective(params, inputs, targets, empirical):
    simulated = simulate_memristor(params, inputs, targets)

    return np.mean((simulated - empirical) ** 2)


def main():
    matplotlib.use('TkAgg')

    plt.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ytick.labelsize'] = 6
    plt.rcParams['xtick.labelsize'] = 6

    colours = ['#FFCB05', '#99621E', '#444444']
    palette = {'main': '#FFCB05', 'sim': '#99621E', 'target': 'black'}

    fig = plt.figure(figsize=(7.08, 6.7))
    fig.patch.set_facecolor('white')

    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1.2])

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, :])
    ax_c = fig.add_subplot(gs[2, :])
    ax_d = fig.add_subplot(gs[3, 0])
    ax_e = fig.add_subplot(gs[3, 1])
    ax_f = fig.add_subplot(gs[3, 2])

    j = 1j

    with open('./data/selector-regression-recordings-1.txt', 'r') as file:
        data = np.vstack(ast.literal_eval(file.read().strip()))

    empirical_inputs = data[:, 0]
    empirical_targets_raw = data[:, 1]
    empirical_predictions_raw = data[:, 2]

    samples = 100
    epochs = len(empirical_inputs) // samples

    bounds = [
        (0.1, 0.8),
        (1.5, 6.0),
        (0.6, 0.95),
        (0.1, 10.0),
        (0.0001, 0.02),
        (0.0, 0.05),
        (0.5, 5.0),
        (0.0, 0.1),
    ]

    # Running the evolution algorithm to extract optimized model parameters.
    result = differential_evolution(
        objective,
        bounds,
        args=(
            empirical_inputs,
            empirical_targets_raw,
            empirical_predictions_raw,
        ),
        strategy='best1bin',
        maxiter=100,
        popsize=15,
        tol=1e-4,
    )
    best_params = result.x

    mem_recordings_opt = simulate_memristor(
        best_params, empirical_inputs, empirical_targets_raw
    )

    static_eis_params = [0.625, 4.0, 0.8, 1.0, 0.01, 0.01, 1.0, 0.012]
    mem_recordings_eis = simulate_memristor(
        static_eis_params, empirical_inputs, empirical_targets_raw
    )

    target_current = empirical_targets_raw * 1e-8
    empirical_current = empirical_predictions_raw * 1e-8
    simulated_current_opt = mem_recordings_opt * 1e-8
    simulated_current_eis = mem_recordings_eis * 1e-8

    # Defining the logarithmic spectrum of frequencies for EIS analysis.
    frequencies = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 250.0, 500.0, 1000.0])
    omega = 2 * np.pi * frequencies

    r_s_uni = 100000.0
    c_memb_uni = 120e-12
    g_off_uni = 0.625e-6
    sigma_baseline_uni = 250000
    alpha_decay_uni = 1.875
    mu_i_uni = -2.7e-6

    # Solving the initial baseline zero-bias state impedance vectors.
    v0_initial = 1e-9

    r_state_init = alpha_decay_uni / (v0_initial * mu_i_uni)
    l_state_init = 1.0 / (v0_initial * mu_i_uni)

    y_memb_initial = (
            g_off_uni
            + (j * omega * c_memb_uni)
            + 1.0 / (r_state_init + j * omega * l_state_init)
    )

    z_total_initial = (
            r_s_uni
            + (1.0 / y_memb_initial)
            + (sigma_baseline_uni / np.sqrt(j * omega))
    )

    # Simulating the time-domain conductance evolution under constant DC bias.
    duration, dt = 60.0, 0.01
    steps = np.arange(0, duration, dt)

    g_trajectory = np.zeros(len(steps))
    g_current = g_off_uni
    v_bias = 0.1

    for index, t in enumerate(steps):
        g_trajectory[index] = g_current
        dg_dt = -alpha_decay_uni * (g_current - g_off_uni) + mu_i_uni * v_bias
        g_current += dg_dt * dt

    g_final = g_current

    v0_biased = 1e-9

    r_state_read = alpha_decay_uni / (v0_biased * mu_i_uni)
    l_state_read = 1.0 / (v0_biased * mu_i_uni)

    sigma_biased_uni = sigma_baseline_uni * (g_off_uni / g_final)

    y_memb_read = (
            g_final
            + (j * omega * c_memb_uni)
            + 1.0 / (r_state_read + j * omega * l_state_read)
    )

    z_total_read = (
            r_s_uni
            + (1.0 / y_memb_read)
            + (sigma_biased_uni / np.sqrt(j * omega))
    )

    z_mag_initial = np.abs(z_total_initial) / 1e6
    z_mag_read = np.abs(z_total_read) / 1e6

    axes = [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f]
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    for ax in axes:
        ax.set_facecolor('white')
        ax.grid(False)
        ax.tick_params(colors='black', labelsize=5, which='both')
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.set_tick_params(
            pad=2, labelbottom=True, bottom=True, labelsize=5, color='black'
        )
        ax.spines[['top', 'right', 'bottom']].set_visible(False)
        ax.spines['left'].set_linewidth(1.0)

    for i, ax in enumerate(axes):
        ax.text(
            0.0,
            1.0,
            labels[i],
            transform=(
                    ax.transAxes
                    + ScaledTranslation(-10 / 72, 7 / 72, fig.dpi_scale_trans)
            ),
            fontsize=7,
            fontweight='bold',
            va='bottom',
            ha='right',
        )

    x_axis = np.arange(len(empirical_inputs))

    # Panel (a): Plotting raw empirical observations.
    ax_a.plot(
        x_axis,
        target_current,
        color=palette['target'],
        lw=0.8,
        label='Target',
    )
    ax_a.plot(
        x_axis,
        empirical_current,
        color=palette['main'],
        lw=1.2,
        label='Empirical',
    )
    ax_a.set_xlabel('Time Steps', fontsize=6)
    ax_a.set_ylabel('Current (A)', fontsize=6)

    for epoch in range(1, epochs):
        ax_a.axvline(
            x=epoch * samples, color='gray', linestyle=':', lw=0.6, alpha=0.5
        )

    # Panel (b): Plotting model programming with empirically matched parameters.
    ax_b.plot(
        x_axis,
        target_current,
        color=palette['target'],
        lw=0.8,
        label='Target',
    )
    ax_b.plot(
        x_axis,
        simulated_current_opt,
        color=palette['sim'],
        lw=1.2,
        label='Optimized Fit',
    )
    ax_b.set_xlabel('Time Steps', fontsize=6)
    ax_b.set_ylabel('Current (A)', fontsize=6)

    for epoch in range(1, epochs):
        ax_b.axvline(
            x=epoch * samples, color='gray', linestyle=':', lw=0.6, alpha=0.5
        )

    # Panel (c): Plotting model programming with static EIS parameters.
    ax_c.plot(
        x_axis,
        target_current,
        color=palette['target'],
        lw=0.8,
        label='Target',
    )
    ax_c.plot(
        x_axis,
        simulated_current_eis,
        color='#777777',
        lw=1.2,
        label='EIS Param Baseline',
    )
    ax_c.set_xlabel('Time Steps', fontsize=6)
    ax_c.set_ylabel('Current (A)', fontsize=6)

    for epoch in range(1, epochs):
        ax_c.axvline(
            x=epoch * samples, color='gray', linestyle=':', lw=0.6, alpha=0.5
        )

    # Panel (d): Plotting the time-domain conductance trajectory.
    ax_d.plot(
        steps[:1000], g_trajectory[:1000] * 1e6, color=colours[2], lw=1.2
    )
    ax_d.set_title('Conductance Evolution', fontsize=7, fontweight='bold')
    ax_d.set_xlabel('Time (seconds)', fontsize=6)
    ax_d.set_ylabel('Conductance G (\u03bcS)', fontsize=6)

    # Panel (e): Plotting complex impedance in a Nyquist chart.
    ax_e.plot(
        z_total_initial.real / 1e6,
        -z_total_initial.imag / 1e6,
        'o-',
        color=colours[0],
        lw=1.1,
        markersize=2.5,
        label='0.0V Baseline',
    )
    ax_e.plot(
        z_total_read.real / 1e6,
        -z_total_read.imag / 1e6,
        'o-',
        color=colours[1],
        lw=1.1,
        markersize=2.5,
        label='Post-Bias',
    )
    ax_e.set_title(
        'Nyquist Plot', fontsize=7, fontweight='bold'
    )
    ax_e.set_xlabel("Z' Real Impedance (M\u03a9)", fontsize=6)
    ax_e.set_ylabel("-Z'' Imaginary Impedance (M\u03a9)", fontsize=6)
    ax_e.set_ylim(-0.1, 1.3)
    ax_e.set_xlim(0.0, 2.8)

    for f in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        idx = np.abs(frequencies - f).argmin()
        ax_e.annotate(
            f'{f} Hz',
            (z_total_read[idx].real / 1e6, -z_total_read[idx].imag / 1e6),
            textcoords='offset points',
            xytext=(3, 3),
            fontsize=4,
            color='#555555',
            fontweight='bold',
        )

    # Panel (f): Plot logarithmic impedance magnitudes.
    ax_f.loglog(
        frequencies,
        z_mag_initial,
        'o-',
        color=colours[0],
        lw=1.1,
        markersize=2.5,
        label='0.0V Baseline',
    )
    ax_f.loglog(
        frequencies,
        z_mag_read,
        'o-',
        color=colours[1],
        lw=1.1,
        markersize=2.5,
        label='Post-Bias',
    )
    ax_f.set_title('Bode Plot', fontsize=7, fontweight='bold')
    ax_f.set_xlabel('Frequency (Hz)', fontsize=6)
    ax_f.set_ylabel('Impedance Magnitude (M\u03a9)', fontsize=6)

    # Plotting the final figure.
    plt.tight_layout()
    plt.savefig(
        'equivalent-circuit-simulation.pdf', dpi=600, bbox_inches='tight'
    )

    plt.show()


if __name__ == '__main__':
    main()
