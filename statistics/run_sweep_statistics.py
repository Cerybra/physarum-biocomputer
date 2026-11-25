import json
import pathlib

import numpy as np
import pandas as pd

from scipy import ndimage

import matplotlib.pyplot as plt

from matplotlib.transforms import ScaledTranslation

from typing import Dict, List, Optional, Union


SWEEP_STATISTICS_CAPTION = (
    r'\textbf{Pinched hysteresis yield and multi-run reproducibility across'
    r' device architectures.} Proportions are evaluated algorithmically across'
    r' all $N = 61$ characterized devices ($301$ total I-V sweeps), stratified'
    r' into open-air thin-tube bridges ($N = 16$), enclosed receptacle'
    r' channels ($N = 44$), and the Petri dish setup ($N = 1$, $1$ sweep).'
)


def _load_sweep_data(filepath: Union[str, pathlib.Path]) -> Dict:
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)

    return data


def _infer_device_cohort(device: str, trials: List[Dict]) -> str:
    device_index = int(device.split()[-1])
    max_voltage = np.max(np.abs(trials[0]['Voltages']))

    if len(trials) == 1 and np.isclose(max_voltage, 1.0):
        return 'Petri Dish'
    if device_index < 16:
        return 'Open-Air'

    return 'Enclosed'


def filter_outliers(currents: np.ndarray) -> np.ndarray:
    current_values = np.array(currents, dtype=float)
    current_steps = np.abs(np.diff(current_values))

    step_threshold = 3.5 * np.median(current_steps)

    median_filtered = ndimage.median_filter(current_values, size=7)
    outlier_mask = np.zeros_like(current_values, dtype=bool)
    outlier_mask[1:] = current_steps > step_threshold

    cleaned_currents = current_values.copy()
    cleaned_currents[outlier_mask] = median_filtered[outlier_mask]

    return cleaned_currents


def is_sweep_pinched(voltages: np.ndarray, currents: np.ndarray) -> bool:
    voltage_values = np.array(voltages, dtype=float)
    current_values = filter_outliers(currents)

    current_peak_to_peak = np.ptp(current_values)
    max_voltage = np.max(np.abs(voltage_values))
    noise_floor = np.median(np.abs(np.diff(current_values)))

    if (
        current_peak_to_peak < 0.08
        or (current_peak_to_peak / (noise_floor + 1e-12)) < 15.0
    ):
        return False

    voltages_negative = voltage_values[269:180:-1]
    currents_down_negative = current_values[269:180:-1]
    currents_up_negative = current_values[271:360]

    voltages_positive = voltage_values[5:90]
    currents_up_positive = current_values[5:90]
    currents_down_positive = current_values[175:90:-1]

    paired_voltages = np.concatenate([voltages_negative, voltages_positive])
    currents_up = np.concatenate([currents_up_negative, currents_up_positive])
    currents_down = np.concatenate([
        currents_down_negative,
        currents_down_positive
    ])

    valid_mask = (
        ((np.abs(currents_up) < 2.65) | (np.abs(currents_down) < 2.65))
        & (np.abs(paired_voltages) <= 0.90 * max_voltage)
    )
    if np.sum(valid_mask) < 20:
        return False

    valid_voltages = paired_voltages[valid_mask]
    smoothed_branch_diff = ndimage.uniform_filter1d(
        (currents_up - currents_down)[valid_mask], size=5
    )
    abs_branch_diff = np.abs(smoothed_branch_diff)

    number_of_points = len(valid_voltages)
    margin = max(int(0.05 * number_of_points), 2)

    for pinch_index in range(margin, number_of_points - margin):
        if np.abs(valid_voltages[pinch_index]) > 0.82 * max_voltage:
            continue

        neck_value = abs_branch_diff[pinch_index]

        left_branch_diff = smoothed_branch_diff[:pinch_index]
        right_branch_diff = smoothed_branch_diff[pinch_index + 1:]

        left_lobe_max = np.max(np.abs(left_branch_diff))
        right_lobe_max = np.max(np.abs(right_branch_diff))

        max_lobe = max(left_lobe_max, right_lobe_max)
        min_lobe = min(left_lobe_max, right_lobe_max)

        if max_lobe < max(0.025 * current_peak_to_peak, 4.2 * noise_floor):
            continue
        if min_lobe < max(0.0035 * current_peak_to_peak, 2.5 * noise_floor):
            continue

        crosses_zero = (
            np.mean(left_branch_diff) * np.mean(right_branch_diff)
        ) < 0.0
        neck_ratio_max = neck_value / (max_lobe + 1e-12)
        neck_ratio_min = neck_value / (min_lobe + 1e-12)

        if crosses_zero and neck_ratio_max < 0.25 and neck_ratio_min < 0.50:
            return True
        if (
            np.abs(valid_voltages[pinch_index]) <= 0.45 * max_voltage
            and neck_ratio_max < 0.15
            and neck_ratio_min < 0.35
        ):
            return True

    return False


def compute_sweep_statistics(data: Dict) -> pd.DataFrame:
    records = []
    for device, trials in data.items():
        number_of_trials = len(trials)
        cohort = _infer_device_cohort(device, trials)

        pinched_indices = [
            trial_index
            for trial_index, trial in enumerate(trials)
            if is_sweep_pinched(trial['Voltages'], trial['Currents'])
        ]

        number_of_pinched = len(pinched_indices)

        records.append({
            'Device': device,
            'Cohort': cohort,
            'TotalRuns': number_of_trials,
            'PinchedRuns': number_of_pinched,
            'PinchedFraction': round(number_of_pinched / number_of_trials, 3),
            'AtLeastOneRun': number_of_pinched >= 1,
            'MultipleRuns': number_of_pinched >= 2,
            'AtLeastHalfRuns': (number_of_pinched / number_of_trials) >= 0.5,
            'AllRuns': number_of_pinched == number_of_trials,
            'PinchedRunIndices': pinched_indices
        })

    return pd.DataFrame(records)


def format_proportion(
        count: int,
        total: int,
        escape_percent: bool = True
) -> str:
    percent_value = (count / total) * 100.0 if total > 0 else 0.0
    unit = r'\%' if escape_percent else '%'

    return f'{count} / {total} ({percent_value:.2f}{unit})'


def save_sweep_statistics(
        target_filepath: Union[str, pathlib.Path],
        dataframe: pd.DataFrame,
        caption: Optional[str] = ''
) -> pd.DataFrame:
    target_path = pathlib.Path(target_filepath)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    is_tex = target_path.suffix.lower() == '.tex'

    dataframe_open_air = dataframe[dataframe['Cohort'] == 'Open-Air']
    dataframe_enclosed = dataframe[dataframe['Cohort'] == 'Enclosed']
    dataframe_petri = dataframe[dataframe['Cohort'] == 'Petri Dish']

    cohorts = [
        ('Open-Air', dataframe_open_air),
        ('Enclosed', dataframe_enclosed),
        ('Petri Dish', dataframe_petri),
        ('All Devices', dataframe)
    ]

    multiple_label = (
        r'Multiple Pinched Runs ($\ge 2$)'
        if is_tex
        else 'Multiple Pinched Runs (>= 2)'
    )

    records = [
        {
            'Metric': 'Total Devices',
            **{
                name: str(len(sub_dataframe))
                for name, sub_dataframe in cohorts
            }
        },
        {
            'Metric': 'At Least 1 Pinched Run',
            **{
                name: format_proportion(
                    int(sub_dataframe['AtLeastOneRun'].sum()),
                    len(sub_dataframe),
                    is_tex
                )
                for name, sub_dataframe in cohorts
            }
        },
        {
            'Metric': multiple_label,
            **{
                name: format_proportion(
                    int(sub_dataframe['MultipleRuns'].sum()),
                    len(sub_dataframe),
                    is_tex
                )
                for name, sub_dataframe in cohorts
            }
        },
        {
            'Metric': 'At Least Half Have Pinched Runs',
            **{
                name: format_proportion(
                    int(sub_dataframe['AtLeastHalfRuns'].sum()),
                    len(sub_dataframe),
                    is_tex
                )
                for name, sub_dataframe in cohorts
            }
        },
        {
            'Metric': 'All Runs Are Pinched',
            **{
                name: format_proportion(
                    int(sub_dataframe['AllRuns'].sum()),
                    len(sub_dataframe),
                    is_tex
                )
                for name, sub_dataframe in cohorts
            }
        }
    ]

    dataframe_sweep_summary = pd.DataFrame(records)

    if is_tex:
        header_cells = ' & '.join(
            f'\\textbf{{{column}}}'
            for column in dataframe_sweep_summary.columns
        )
        latex_lines = [
            r'\begin{table}[htbp]',
            r'\centering',
            f'\\caption{{{caption}}}' if caption else r'\caption{}',
            r'\label{tab:sweep-statistics}',
            r'\begin{tabular}{lcccc}',
            r'\toprule',
            f'{header_cells} \\\\',
            r'\midrule'
        ]

        for _, row in dataframe_sweep_summary.iterrows():
            row_cells = ' & '.join(str(value) for value in row.values)
            latex_lines.append(f'{row_cells} \\\\')

        latex_lines.extend([
            r'\bottomrule',
            r'\end{tabular}',
            r'\end{table}'
        ])

        target_path.write_text('\n'.join(latex_lines), encoding='utf-8')
    else:
        dataframe_sweep_summary.to_csv(target_path, index=False)

    return dataframe_sweep_summary


def plot_sweep_statistics(
        figure: plt.Figure,
        axis: plt.Axes,
        label: str,
        palette: Dict[str, str],
        dataframe: pd.DataFrame
) -> None:
    axis.set_facecolor('white')

    axis.set_title('Pinched Hysteresis Yield by Type', fontsize=7)

    axis.set_xlabel('Device Type', fontsize=6)
    axis.set_ylabel('Pinched Yield (%)', fontsize=6)

    axis.tick_params(colors='black', labelsize=5, which='both')
    axis.xaxis.set_label_position('bottom')
    axis.xaxis.set_tick_params(
        pad=2,
        labelbottom=True,
        bottom=True,
        labelsize=5,
        labelrotation=0,
        color='black'
    )

    axis.spines[['top', 'right', 'bottom']].set_visible(False)
    axis.spines['left'].set_linewidth(1.0)
    axis.spines['left'].set_color('black')

    cohort_names = ['Open-Air', 'Enclosed', 'All Devices']
    cohort_subsets = [
        dataframe[dataframe['Cohort'] == 'Open-Air'],
        dataframe[dataframe['Cohort'] == 'Enclosed'],
        dataframe
    ]

    at_least_one_percent = [
        (sub_dataframe['AtLeastOneRun'].sum() / len(sub_dataframe)) * 100.0
        for sub_dataframe in cohort_subsets
    ]
    multiple_percent = [
        (sub_dataframe['MultipleRuns'].sum() / len(sub_dataframe)) * 100.0
        for sub_dataframe in cohort_subsets
    ]
    half_percent = [
        (sub_dataframe['AtLeastHalfRuns'].sum() / len(sub_dataframe)) * 100.0
        for sub_dataframe in cohort_subsets
    ]

    x_positions = np.arange(len(cohort_names))
    bar_width = 0.24

    axis.bar(
        x_positions - bar_width,
        at_least_one_percent,
        width=bar_width,
        color=palette['main'],
        label='>= 1 Run'
    )
    axis.bar(
        x_positions,
        multiple_percent,
        width=bar_width,
        color=palette['deep'],
        label='>= 2 Runs'
    )
    axis.bar(
        x_positions + bar_width,
        half_percent,
        width=bar_width,
        color=palette['error'],
        label='>= 50% Runs'
    )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(cohort_names)
    axis.set_ylim(0, 100.0)
    axis.legend(frameon=False, fontsize=5, loc='upper right')

    axis.text(
        0.0,
        1.0,
        label,
        transform=(
            axis.transAxes
            + ScaledTranslation(-10 / 72, 7 / 72, figure.dpi_scale_trans)
        ),
        fontsize=7,
        fontweight='bold',
        va='bottom',
        ha='right'
    )


if __name__ == '__main__':
    sweep_data = _load_sweep_data(
        'data/device-sweep-recordings/device-sweep-recordings.json'
    )

    dataframe_sweep_statistics = compute_sweep_statistics(sweep_data)

    save_sweep_statistics(
        target_filepath='results/sweep-statistics.csv',
        dataframe=dataframe_sweep_statistics,
        caption=SWEEP_STATISTICS_CAPTION
    )
    dataframe_sweep_summary = save_sweep_statistics(
        target_filepath='results/sweep-statistics.tex',
        dataframe=dataframe_sweep_statistics,
        caption=SWEEP_STATISTICS_CAPTION
    )

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1200)

    print(dataframe_sweep_summary.to_string(index=False))
