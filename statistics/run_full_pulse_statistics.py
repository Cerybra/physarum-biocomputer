import pathlib
import warnings

import numpy as np
import pandas as pd

from scipy.optimize import curve_fit

from typing import Dict, Optional, Tuple, Union


DEFAULT_PULSE_FILEPATH = pathlib.Path(
    'data/device-pulse-recordings/legacy/device-pulse-recordings.csv'
)
FALLBACK_PULSE_FILEPATH = pathlib.Path('device-pulse-recordings.csv')

PULSE_STATISTICS_CAPTION = (
    r'\textbf{Classification of pulsed responses across $N = 32$ device'
    r' experiments ($9,776$ pulse events).} The recordings are assigned to'
    r' mutually exclusive categories: Switching (active'
    r' potentiation/depression cycles with delayed threshold onset, unipolar'
    r' resets, or mid-train polarity reversals; $9 / 28 = 32.14\%$ of'
    r' colonized channels), Monotonic Non-RC (non-switching conductance'
    r' manipulation behavior not fit by an exponential RC curve;'
    r' $0 / 28 = 0.00\%$), RC / Ohmic (monotonic exponential capacitive'
    r' relaxation after accounting for initial ADC rail clipping, or passive'
    r' step tracking; $11 / 28 = 39.29\%$), and Dead / Control (static'
    r' rail-clamped or uncolonized open-circuit channels).'
)


def load_pulse_data(filepath: Union[str, pathlib.Path]) -> pd.DataFrame:
    return pd.read_csv(filepath, index_col=False)


def smooth_segment(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(values) < 3:
        return values.copy(), values.copy()

    despiked = (
        pd.Series(values).rolling(3, center=True, min_periods=1).median()
    )
    smoothed = (
        pd.Series(despiked).rolling(5, center=True, min_periods=1).mean()
    )

    return despiked.to_numpy(dtype=float), smoothed.to_numpy(dtype=float)


def trim_initial_saturation_rail(
        values: np.ndarray,
        saturation_threshold: float = 2.64
) -> Tuple[np.ndarray, int, bool]:
    if len(values) == 0:
        return values, 0, False

    unsaturated_indices = np.where(np.abs(values) < saturation_threshold)[0]
    if len(unsaturated_indices) == 0:
        return values, len(values), True

    first_unsaturated_index = int(unsaturated_indices[0])
    started_on_saturation_rail = first_unsaturated_index > 0 and (
        abs(values[0]) >= saturation_threshold
    )
    if (
        started_on_saturation_rail
        and len(values) - first_unsaturated_index >= 8
    ):
        return (
            values[first_unsaturated_index:],
            first_unsaturated_index,
            True
        )

    return values, 0, started_on_saturation_rail


def evaluate_clipped_rc_hypothesis(
        values: np.ndarray,
        saturation_threshold: float = 2.64
) -> Dict[str, Union[float, int, bool]]:
    unsaturated_indices = np.where(np.abs(values) < saturation_threshold)[0]
    if len(unsaturated_indices) < 5:
        return {
            'is_valid_rc': False,
            'r_squared': 0.0,
            'tau': 1.0,
            'asymptote': np.nan,
            'extrapolated_initial': np.nan,
            'onset_index': len(values),
            'crosses_zero': False,
            'hits_both_rails': False
        }

    start_index, end_index = (
        int(unsaturated_indices[0]),
        int(unsaturated_indices[-1])
    )

    active_indices = np.arange(start_index, end_index + 1, dtype=float)
    active_values = values[start_index:end_index + 1]

    despiked_full, _ = smooth_segment(values)
    _, smoothed_active = smooth_segment(active_values)

    crosses_zero = bool(
        np.min(despiked_full) < -0.25 and np.max(despiked_full) > 0.25
    )
    hits_both_rails = bool(
        np.min(despiked_full) <= -saturation_threshold
        and np.max(despiked_full) >= saturation_threshold
    )

    if len(smoothed_active) < 5 or np.ptp(smoothed_active) < 1e-6:
        return {
            'is_valid_rc': False,
            'r_squared': 1.0,
            'tau': 1.0,
            'asymptote': np.mean(smoothed_active),
            'extrapolated_initial': smoothed_active[0],
            'onset_index': start_index,
            'crosses_zero': crosses_zero,
            'hits_both_rails': hits_both_rails
        }

    def shifted_exponential(
            x_index: np.ndarray,
            asymptote: float,
            amplitude: float,
            tau: float
    ) -> np.ndarray:
        return asymptote + amplitude * np.exp(
            -(x_index - start_index) / np.maximum(tau, 1e-6)
        )

    best_r_squared = -np.inf
    best_tau = 25.0
    best_asymptote = smoothed_active[-1]
    best_extrapolated_initial = smoothed_active[0]

    is_increasing = smoothed_active[-1] > smoothed_active[0]
    if is_increasing:
        parameter_bounds = ([-2.6, -25.0, 0.5], [25.0, 0.0, 500.0])
        asymptote_guess = float(
            np.clip(smoothed_active[-1] + 0.2, -2.59, 24.99)
        )
        amplitude_guess = float(
            np.clip(smoothed_active[0] - asymptote_guess, -24.99, -1e-6)
        )
    else:
        parameter_bounds = ([-25.0, 0.0, 0.5], [2.6, 25.0, 500.0])
        asymptote_guess = float(
            np.clip(smoothed_active[-1] - 0.1, -24.99, 2.59)
        )
        amplitude_guess = float(
            np.clip(smoothed_active[0] - asymptote_guess, 1e-6, 24.99)
        )

    for tau_initial in (5.0, 15.0, 35.0, 65.0):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')

                optimal_parameters, _ = curve_fit(
                    shifted_exponential,
                    active_indices,
                    smoothed_active,
                    p0=[asymptote_guess, amplitude_guess, tau_initial],
                    bounds=parameter_bounds,
                    maxfev=4000
                )

            asymptote_fit, amplitude_fit, tau_fit = (
                optimal_parameters[0],
                optimal_parameters[1],
                optimal_parameters[2]
            )

            predicted = shifted_exponential(
                active_indices, asymptote_fit, amplitude_fit, tau_fit
            )

            residuals = smoothed_active - predicted
            ss_residuals = np.sum(residuals ** 2)
            ss_total = np.sum(
                (smoothed_active - np.mean(smoothed_active)) ** 2
            )
            r_squared = 1.0 - (ss_residuals / (ss_total + 1e-12))

            if r_squared > best_r_squared:
                best_r_squared = r_squared
                best_tau = tau_fit
                best_asymptote = asymptote_fit
                best_extrapolated_initial = shifted_exponential(
                    np.array([0.0]), asymptote_fit, amplitude_fit, tau_fit
                )[0]

        except Exception:
            continue

    is_valid_rc = bool(
        (not crosses_zero)
        and (not hits_both_rails)
        and (best_r_squared >= 0.85)
        and (abs(best_extrapolated_initial) <= 8.0)
        and (abs(best_asymptote) <= saturation_threshold)
    )

    return {
        'is_valid_rc': is_valid_rc,
        'r_squared': best_r_squared,
        'tau': best_tau,
        'asymptote': best_asymptote,
        'extrapolated_initial': best_extrapolated_initial,
        'onset_index': start_index,
        'crosses_zero': crosses_zero,
        'hits_both_rails': hits_both_rails
    }


def _exponential_model(
        x_index: np.ndarray,
        asymptote: float,
        amplitude: float,
        tau: float
) -> np.ndarray:
    return asymptote + amplitude * np.exp(-x_index / np.maximum(tau, 1e-6))


def fit_exponential_curve(
        values: np.ndarray,
        trim_saturation: bool = True
) -> Tuple[float, float, float]:
    active_segment = values
    if trim_saturation:
        active_segment, _, _ = trim_initial_saturation_rail(values)

    _, smoothed_segment = smooth_segment(active_segment)

    number_of_points = len(smoothed_segment)
    if number_of_points < 5:
        return 1.0, 0.0, 1.0

    pulse_indices = np.arange(number_of_points, dtype=float)

    value_min, value_max = np.min(smoothed_segment), np.max(smoothed_segment)
    value_span = value_max - value_min
    if value_span < 1e-6:
        return 1.0, 0.0, 1.0

    normalized_values = (smoothed_segment - value_min) / value_span

    initial_level, final_level = normalized_values[0], normalized_values[-1]
    amplitude_guess = initial_level - final_level
    tau_guess = max(number_of_points / 3.0, 2.0)

    best_r_squared, best_nrmse, best_tau = -np.inf, 100.0, tau_guess
    for tau_initial in (
        2.0,
        8.0,
        tau_guess,
        number_of_points / 2.0,
        number_of_points
    ):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                optimal_parameters, _ = curve_fit(
                    _exponential_model,
                    pulse_indices,
                    normalized_values,
                    p0=[final_level, amplitude_guess, tau_initial],
                    bounds=(
                        [-2.0, -4.0, 0.3],
                        [3.0, 4.0, number_of_points * 10.0]
                    ),
                    maxfev=2500
                )

            predicted = _exponential_model(pulse_indices, *optimal_parameters)
            residuals = normalized_values - predicted

            ss_residuals = np.sum(residuals ** 2)
            ss_total = np.sum(
                (normalized_values - np.mean(normalized_values)) ** 2
            )

            r_squared = 1.0 - (ss_residuals / (ss_total + 1e-12))
            nrmse_percent = np.sqrt(np.mean(residuals ** 2)) * 100.0

            if r_squared > best_r_squared:
                best_r_squared, best_nrmse, best_tau = (
                    r_squared,
                    nrmse_percent,
                    optimal_parameters[2]
                )

        except Exception:
            continue

    return best_r_squared, best_nrmse, best_tau


def compute_unsaturated_onset_latency(
        values: np.ndarray,
        threshold_fraction: float = 0.10
) -> int:
    rc_hypothesis = evaluate_clipped_rc_hypothesis(values)
    if bool(rc_hypothesis['is_valid_rc']):
        return 0

    _, smoothed_full = smooth_segment(values)
    full_span = np.ptp(smoothed_full)
    if full_span < 0.05:
        return 0

    initial_value = smoothed_full[0]
    deviations = np.abs(smoothed_full - initial_value)

    exceeded_indices = np.where(deviations > threshold_fraction * full_span)[0]
    if len(exceeded_indices) == 0:
        return len(values)

    return int(exceeded_indices[0])


def evaluate_constant_bias_windows(
        values: np.ndarray,
        window_length: int = 50
) -> Dict[str, Union[int, float, bool]]:
    number_of_pulses = len(values)
    if number_of_pulses < 200:
        windows = [values]
    else:
        windows = []
        for start_index in range(0, number_of_pulses, window_length):
            segment = values[start_index:start_index + window_length]
            if len(segment) >= 15:
                windows.append(segment)

    mid_window_zero_crossings = 0
    delayed_onset_windows = 0
    valid_rc_windows = 0
    interior_drift_ratios = []
    window_exponential_r_squared_values = []

    _, smoothed_full = smooth_segment(values)
    boundary_steps = []
    for start_index in range(window_length, number_of_pulses, window_length):
        boundary_steps.append(
            abs(smoothed_full[start_index] - smoothed_full[start_index - 1])
        )

    mean_boundary_step = np.mean(boundary_steps) if boundary_steps else 0.0

    for raw_segment in windows:
        despiked_segment, smoothed_segment = smooth_segment(raw_segment)
        segment_span = np.ptp(smoothed_segment)

        rc_window_check = evaluate_clipped_rc_hypothesis(raw_segment)
        if bool(rc_window_check['is_valid_rc']):
            valid_rc_windows += 1

        if segment_span >= 0.02:
            segment_r_squared, _, _ = fit_exponential_curve(
                raw_segment, trim_saturation=True
            )
            window_exponential_r_squared_values.append(segment_r_squared)

        interior_drift = abs(smoothed_segment[-1] - smoothed_segment[1])
        denominator = max(mean_boundary_step, segment_span, 1e-6)
        interior_drift_ratios.append(interior_drift / denominator)

        onset_index = compute_unsaturated_onset_latency(
            raw_segment, threshold_fraction=0.10
        )
        if (
            onset_index >= 5
            and segment_span > 1.0
            and not bool(rc_window_check['is_valid_rc'])
        ):
            delayed_onset_windows += 1

        interior_segment = despiked_segment[1:]
        if len(interior_segment) >= 6 and segment_span > 1.50:
            significant_values = interior_segment[
                np.abs(interior_segment) > 0.25
            ]

            if len(significant_values) >= 6:
                positive_count = int(np.sum(significant_values > 0.25))
                negative_count = int(np.sum(significant_values < -0.25))

                if positive_count >= 2 and negative_count >= 2:
                    mid_window_zero_crossings += 1

    mean_window_r_squared = (
        np.mean(window_exponential_r_squared_values)
        if window_exponential_r_squared_values
        else 1.0
    )
    max_window_r_squared = (
        np.max(window_exponential_r_squared_values)
        if window_exponential_r_squared_values
        else 1.0
    )
    max_interior_drift_ratio = (
        np.max(interior_drift_ratios) if interior_drift_ratios else 0.0
    )

    return {
        'mid_window_zero_crossings': int(mid_window_zero_crossings),
        'delayed_onset_windows': int(delayed_onset_windows),
        'valid_rc_windows': int(valid_rc_windows),
        'mean_window_exponential_r2': mean_window_r_squared,
        'max_window_exponential_r2': max_window_r_squared,
        'max_interior_drift_ratio': max_interior_drift_ratio,
        'mean_boundary_step': mean_boundary_step
    }


def classify_pulse_channel(
        device_id: int,
        values: np.ndarray
) -> Dict[str, Union[str, int, float, bool]]:
    number_of_pulses = len(values)
    despiked_values, smoothed_values = smooth_segment(values)

    initial_value, final_value = values[0], values[-1]
    min_value, max_value = np.min(values), np.max(values)

    mean_value = np.mean(values)
    standard_deviation = (
        np.std(values, ddof=1) if number_of_pulses > 1 else 0.0
    )

    peak_to_peak = np.ptp(values)
    despiked_peak_to_peak = np.ptp(despiked_values)

    if device_id < 8:
        cohort = '50-Pulse Step/Ramp'
    elif device_id < 16:
        cohort = '499-Pulse Rail-to-Rail Cyclical'
    elif device_id < 20:
        cohort = '249-Pulse Unipolar Depression'
    elif device_id < 24:
        cohort = '99-Pulse Unipolar Depression'
    else:
        cohort = '499-Pulse Unsaturated Cyclical'

    reference_scale = max(
        abs(smoothed_values[0]),
        np.max(np.abs(smoothed_values)),
        0.10
    )
    relative_change_percent = (despiked_peak_to_peak / reference_scale) * 100.0
    end_to_end_change_percent = (
        abs(smoothed_values[-1] - smoothed_values[0]) / reference_scale
    ) * 100.0

    rc_hypothesis = evaluate_clipped_rc_hypothesis(values)
    full_exponential_r_squared, _, fitted_tau_pulses = fit_exponential_curve(
        values, trim_saturation=True
    )
    onset_latency = compute_unsaturated_onset_latency(
        values, threshold_fraction=0.10
    )
    window_metrics = evaluate_constant_bias_windows(values, window_length=50)

    if number_of_pulses < 100:
        polarity = 'Unipolar'
    else:
        despiked_min, despiked_max = (
            np.min(despiked_values),
            np.max(despiked_values)
        )

        is_unipolar_positive = (
            (despiked_min >= -0.08) and (despiked_max > 0.10)
        )
        is_unipolar_negative = (
            (despiked_max <= 0.08) and (despiked_min < -0.10)
        )

        polarity = (
            'Unipolar'
            if (is_unipolar_positive or is_unipolar_negative)
            else 'Bipolar'
        )

    is_uncolonized_noise = (
        despiked_peak_to_peak < 0.05 and np.max(np.abs(despiked_values)) < 0.05
    )
    is_rail_clamped_static = (
        despiked_peak_to_peak < 0.05 or relative_change_percent < 5.0
    )

    if is_uncolonized_noise:
        return {
            'device': f'Device {device_id}',
            'cohort': cohort,
            'pulses': number_of_pulses,
            'colonized': False,
            'active_switching': False,
            'classification': 'Dead / Control',
            'mechanism': 'Uncolonized Control',
            'polarity': 'None',
            'relative_change_percent': round(relative_change_percent, 2),
            'exponential_r2': round(full_exponential_r_squared, 4),
            'fitted_tau_pulses': round(fitted_tau_pulses, 2),
            'onset_latency_pulses': int(onset_latency),
            'initial': round(initial_value, 4),
            'final': round(final_value, 4),
            'minimum': round(min_value, 4),
            'maximum': round(max_value, 4),
            'range_ptp': round(peak_to_peak, 4),
            'mean': round(mean_value, 4),
            'standard_deviation': round(standard_deviation, 4)
        }

    if is_rail_clamped_static:
        return {
            'device': f'Device {device_id}',
            'cohort': cohort,
            'pulses': number_of_pulses,
            'colonized': True,
            'active_switching': False,
            'classification': 'Dead / Control',
            'mechanism': 'Static / Rail-Clamped',
            'polarity': 'Unipolar',
            'relative_change_percent': round(relative_change_percent, 2),
            'exponential_r2': round(full_exponential_r_squared, 4),
            'fitted_tau_pulses': round(fitted_tau_pulses, 2),
            'onset_latency_pulses': int(onset_latency),
            'initial': round(initial_value, 4),
            'final': round(final_value, 4),
            'minimum': round(min_value, 4),
            'maximum': round(max_value, 4),
            'range_ptp': round(peak_to_peak, 4),
            'mean': round(mean_value, 4),
            'standard_deviation': round(standard_deviation, 4)
        }

    effective_exponential_r_squared = (
        rc_hypothesis['r_squared']
        if number_of_pulses <= 250
        else max(
            full_exponential_r_squared,
            window_metrics['mean_window_exponential_r2'],
            window_metrics['max_window_exponential_r2']
        )
    )

    is_exponential_rc_curve = bool(rc_hypothesis['is_valid_rc']) or (
        effective_exponential_r_squared >= 0.85
        and int(window_metrics['mid_window_zero_crossings']) == 0
        and onset_latency <= 3
    )
    is_ohmic_step = (
        number_of_pulses >= 200
        and int(window_metrics['mid_window_zero_crossings']) == 0
        and (
            window_metrics['max_interior_drift_ratio'] < 0.35
            or window_metrics['max_window_exponential_r2'] >= 0.75
        )
    )

    is_switching = bool(
        relative_change_percent > 25.0
        and not is_exponential_rc_curve
        and not is_ohmic_step
        and (
            int(window_metrics['mid_window_zero_crossings']) >= 1
            or onset_latency >= 5
            or int(window_metrics['delayed_onset_windows']) >= 1
        )
    )

    if is_switching:
        if int(window_metrics['mid_window_zero_crossings']) >= 1:
            switching_mechanism = 'Mid-Train Zero-Crossing'
        else:
            switching_mechanism = 'Delayed-Onset Switching'

        return {
            'device': f'Device {device_id}',
            'cohort': cohort,
            'pulses': number_of_pulses,
            'colonized': True,
            'active_switching': True,
            'classification': 'Switching',
            'mechanism': switching_mechanism,
            'polarity': polarity,
            'relative_change_percent': round(relative_change_percent, 2),
            'exponential_r2': round(effective_exponential_r_squared, 4),
            'fitted_tau_pulses': round(fitted_tau_pulses, 2),
            'onset_latency_pulses': int(onset_latency),
            'initial': round(initial_value, 4),
            'final': round(final_value, 4),
            'minimum': round(min_value, 4),
            'maximum': round(max_value, 4),
            'range_ptp': round(peak_to_peak, 4),
            'mean': round(mean_value, 4),
            'standard_deviation': round(standard_deviation, 4)
        }

    has_non_rc_plasticity = (
        relative_change_percent > 25.0
        and end_to_end_change_percent > 25.0
        and not is_exponential_rc_curve
        and not is_ohmic_step
    )

    if has_non_rc_plasticity:
        return {
            'device': f'Device {device_id}',
            'cohort': cohort,
            'pulses': number_of_pulses,
            'colonized': True,
            'active_switching': True,
            'classification': 'Plasticity',
            'mechanism': f'{polarity} Non-Exponential Plasticity',
            'polarity': polarity,
            'relative_change_percent': round(relative_change_percent, 2),
            'exponential_r2': round(effective_exponential_r_squared, 4),
            'fitted_tau_pulses': round(fitted_tau_pulses, 2),
            'onset_latency_pulses': int(onset_latency),
            'initial': round(initial_value, 4),
            'final': round(final_value, 4),
            'minimum': round(min_value, 4),
            'maximum': round(max_value, 4),
            'range_ptp': round(peak_to_peak, 4),
            'mean': round(mean_value, 4),
            'standard_deviation': round(standard_deviation, 4)
        }

    rc_mechanism = (
        'Exponential RC Relaxation'
        if is_exponential_rc_curve
        else 'Passive RC / Ohmic Step'
    )

    return {
        'device': f'Device {device_id}',
        'cohort': cohort,
        'pulses': number_of_pulses,
        'colonized': True,
        'active_switching': False,
        'classification': 'RC / Ohmic',
        'mechanism': rc_mechanism,
        'polarity': polarity,
        'relative_change_percent': round(relative_change_percent, 2),
        'exponential_r2': round(effective_exponential_r_squared, 4),
        'fitted_tau_pulses': round(fitted_tau_pulses, 2),
        'onset_latency_pulses': int(onset_latency),
        'initial': round(initial_value, 4),
        'final': round(final_value, 4),
        'minimum': round(min_value, 4),
        'maximum': round(max_value, 4),
        'range_ptp': round(peak_to_peak, 4),
        'mean': round(mean_value, 4),
        'standard_deviation': round(standard_deviation, 4)
    }


def compute_pulse_statistics(
        dataframe_raw: pd.DataFrame
) -> pd.DataFrame:
    device_column = (
        'Experiment Number'
        if 'Experiment Number' in dataframe_raw.columns
        else 'Device Number'
    )

    records = []
    for device_id in sorted(dataframe_raw[device_column].unique()):
        sub_dataframe = dataframe_raw[
            dataframe_raw[device_column] == device_id
        ]
        if 'Pulse Number' in sub_dataframe.columns:
            sub_dataframe = sub_dataframe.sort_values('Pulse Number')

        values = sub_dataframe['Value'].to_numpy(dtype=float)
        records.append(classify_pulse_channel(int(device_id), values))

    return pd.DataFrame(records)


def format_proportion(
        count: int,
        total: int,
        escape_percent: bool = True
) -> str:
    percent_value = (count / total) * 100.0 if total > 0 else 0.0
    unit = r'\%' if escape_percent else '%'

    return f'{count} / {total} ({percent_value:.2f}{unit})'


def save_pulse_statistics(
        target_filepath: Union[str, pathlib.Path],
        dataframe: pd.DataFrame,
        caption: Optional[str] = ''
) -> pd.DataFrame:
    target_path = pathlib.Path(target_filepath)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    is_tex = target_path.suffix.lower() == '.tex'

    dataframe_colonized = dataframe[dataframe['colonized']]
    dataframe_active = dataframe[dataframe['active_switching']]

    total_channels = len(dataframe)
    colonized_channels = len(dataframe_colonized)
    active_channels = len(dataframe_active)

    switching_total = int((dataframe['classification'] == 'Switching').sum())
    switching_unipolar = int(
        (
            (dataframe['classification'] == 'Switching')
            & (dataframe['polarity'] == 'Unipolar')
        ).sum()
    )
    switching_bipolar = int(
        (
            (dataframe['classification'] == 'Switching')
            & (dataframe['polarity'] == 'Bipolar')
        ).sum()
    )

    plasticity_total = int((dataframe['classification'] == 'Plasticity').sum())

    rc_total = int((dataframe['classification'] == 'RC / Ohmic').sum())
    rc_exponential = int(
        (dataframe['mechanism'] == 'Exponential RC Relaxation').sum()
    )
    rc_step = int((dataframe['mechanism'] == 'Passive RC / Ohmic Step').sum())

    dead_control_total = int(
        (dataframe['classification'] == 'Dead / Control').sum()
    )
    dead_static = int(
        (dataframe['mechanism'] == 'Static / Rail-Clamped').sum()
    )
    uncolonized_control = int(
        (dataframe['mechanism'] == 'Uncolonized Control').sum()
    )

    all_header = f'All Channels (N={total_channels})'
    colonized_header = f'Colonized (N={colonized_channels})'
    active_header = f'Active / Switching (N={active_channels})'

    records = [
        {
            'Category': 'Total Recorded Channels',
            'Mechanism': 'All Tested Channels',
            all_header: str(total_channels),
            colonized_header: str(colonized_channels),
            active_header: str(active_channels)
        },
        {
            'Category': 'Switching',
            'Mechanism': 'Total Active / Threshold Switching',
            all_header: format_proportion(
                switching_total, total_channels, is_tex
            ),
            colonized_header: format_proportion(
                switching_total, colonized_channels, is_tex
            ),
            active_header: format_proportion(
                switching_total, active_channels, is_tex
            )
        },
        {
            'Category': '',
            'Mechanism': 'Unipolar Threshold Switching',
            all_header: format_proportion(
                switching_unipolar, total_channels, is_tex
            ),
            colonized_header: format_proportion(
                switching_unipolar, colonized_channels, is_tex
            ),
            active_header: format_proportion(
                switching_unipolar, active_channels, is_tex
            )
        },
        {
            'Category': '',
            'Mechanism': 'Bipolar Cyclical Switching',
            all_header: format_proportion(
                switching_bipolar, total_channels, is_tex
            ),
            colonized_header: format_proportion(
                switching_bipolar, colonized_channels, is_tex
            ),
            active_header: format_proportion(
                switching_bipolar, active_channels, is_tex
            )
        },
        {
            'Category': 'Monotonic Non-RC',
            'Mechanism': 'Non-Exponential Drift',
            all_header: format_proportion(
                plasticity_total, total_channels, is_tex
            ),
            colonized_header: format_proportion(
                plasticity_total, colonized_channels, is_tex
            ),
            active_header: format_proportion(
                plasticity_total, active_channels, is_tex
            )
        },
        {
            'Category': 'RC / Ohmic',
            'Mechanism': 'Total Passive RC / Ohmic',
            all_header: format_proportion(rc_total, total_channels, is_tex),
            colonized_header: format_proportion(
                rc_total, colonized_channels, is_tex
            ),
            active_header: '---'
        },
        {
            'Category': '',
            'Mechanism': 'Exponential RC Relaxation',
            all_header: format_proportion(
                rc_exponential, total_channels, is_tex
            ),
            colonized_header: format_proportion(
                rc_exponential, colonized_channels, is_tex
            ),
            active_header: '---'
        },
        {
            'Category': '',
            'Mechanism': 'Passive RC / Ohmic Step',
            all_header: format_proportion(rc_step, total_channels, is_tex),
            colonized_header: format_proportion(
                rc_step, colonized_channels, is_tex
            ),
            active_header: '---'
        },
        {
            'Category': 'Dead / Control',
            'Mechanism': 'Total Inactive / Control',
            all_header: format_proportion(
                dead_control_total, total_channels, is_tex
            ),
            colonized_header: format_proportion(
                dead_static, colonized_channels, is_tex
            ),
            active_header: '---'
        },
        {
            'Category': '',
            'Mechanism': 'Static / Rail-Clamped',
            all_header: format_proportion(dead_static, total_channels, is_tex),
            colonized_header: format_proportion(
                dead_static, colonized_channels, is_tex
            ),
            active_header: '---'
        },
        {
            'Category': '',
            'Mechanism': 'Uncolonized Control',
            all_header: format_proportion(
                uncolonized_control, total_channels, is_tex
            ),
            colonized_header: '---',
            active_header: '---'
        }
    ]

    dataframe_summary = pd.DataFrame(records)

    if is_tex:
        all_tex_header = f'All Channels ($N={total_channels}$)'
        colonized_tex_header = f'Colonized ($N={colonized_channels}$)'
        active_tex_header = f'Active / Switching ($N={active_channels}$)'

        latex_lines = [
            r'\begin{table*}[htbp]',
            r'\centering',
            rf'\caption{{{caption}}}',
            r'\label{tab:pulse-statistics}',
            r'\resizebox{\textwidth}{!}{%',
            r'\begin{tabular}{llccc}',
            r'\toprule',
            (
                r'\textbf{Category} & \textbf{Mechanism} &'
                rf' \textbf{{{all_tex_header}}} &'
                rf' \textbf{{{colonized_tex_header}}} &'
                rf' \textbf{{{active_tex_header}}} \\'
            ),
            r'\midrule'
        ]

        for _, row in dataframe_summary.iterrows():
            category_cell = str(row['Category'])

            mechanism_cell = str(row['Mechanism']).replace('%', r'\%')
            mechanism_cell = mechanism_cell.replace('<', '$<$')
            mechanism_cell = mechanism_cell.replace('>=', r'$\ge$')
            mechanism_cell = mechanism_cell.replace('>', '$>$')

            all_cell = str(row[all_header])
            colonized_cell = str(row[colonized_header])
            active_cell = str(row[active_header])

            if category_cell in (
                'Switching',
                'Monotonic Non-RC',
                'RC / Ohmic',
                'Dead / Control'
            ):
                latex_lines.append(r'\midrule')

            latex_lines.append(
                f'{category_cell} & {mechanism_cell} & {all_cell} &'
                f' {colonized_cell} & {active_cell} \\\\'
            )

        latex_lines.extend([
            r'\bottomrule',
            r'\end{tabular}%',
            r'}',
            r'\end{table*}'
        ])
        target_path.write_text('\n'.join(latex_lines), encoding='utf-8')
    else:
        dataframe_summary.to_csv(target_path, index=False)

    return dataframe_summary


if __name__ == '__main__':
    input_filepath = (
        DEFAULT_PULSE_FILEPATH
        if DEFAULT_PULSE_FILEPATH.exists()
        else FALLBACK_PULSE_FILEPATH
    )

    dataframe_raw_pulses = load_pulse_data(input_filepath)
    dataframe_pulse_channels = compute_pulse_statistics(dataframe_raw_pulses)

    results_directory = pathlib.Path('results')
    results_directory.mkdir(parents=True, exist_ok=True)
    dataframe_pulse_channels.to_csv(
        results_directory / 'pulse-channels-classified.csv', index=False
    )

    save_pulse_statistics(
        target_filepath=results_directory / 'pulse-statistics.csv',
        dataframe=dataframe_pulse_channels,
        caption=PULSE_STATISTICS_CAPTION
    )
    dataframe_pulse_summary = save_pulse_statistics(
        target_filepath=results_directory / 'pulse-statistics.tex',
        dataframe=dataframe_pulse_channels,
        caption=PULSE_STATISTICS_CAPTION
    )

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1200)

    print(dataframe_pulse_summary.to_string(index=False))
