import pathlib
import re
import warnings

import numpy as np
import pandas as pd

from impedance import preprocessing

from scipy import stats

from typing import Dict, List, Tuple, Union


DEFAULT_ENDURANCE_DIRECTORY = pathlib.Path(
    'data/eis/slime-mould/device-5/before-bias'
)

EIS_ENDURANCE_CAPTION = (
    r'\textbf{Longitudinal EIS statistics across $N = 101$ uninterrupted'
    r' sweeps ($\sim 113$~min, $0$~V DC) on an open-air \textit{P.'
    r' polycephalum} bridge.} Values report mean $\pm$ SD, parametric'
    r' $95\%$ confidence intervals, and linear impedance drift rates'
    r' stratified by sequential run quartiles (Runs~1--25, 26--50, 51--75,'
    r' and 76--101) and across the full $101$-sweep series.'
)


def extract_run_number(filepath: Union[str, pathlib.Path]) -> int:
    match = re.search(r'recordings-(\d+)\.csv$', pathlib.Path(filepath).name)

    return int(match.group(1)) if match else -1


def walk_eis_endurance_data(
        directory_path: Union[str, pathlib.Path] = DEFAULT_ENDURANCE_DIRECTORY
) -> List[Tuple[int, pathlib.Path]]:
    target_directory = pathlib.Path(directory_path)
    if not target_directory.exists():
        raise FileNotFoundError(
            f'EIS endurance directory not found: {target_directory.resolve()}'
        )

    csv_files = [
        filepath
        for filepath in target_directory.glob('recordings-*.csv')
        if filepath.is_file()
    ]
    sorted_files = sorted(csv_files, key=extract_run_number)

    return [
        (extract_run_number(filepath), filepath)
        for filepath in sorted_files
    ]


def prepare_eis_data(
        filepath: Union[str, pathlib.Path],
        trim_inductive: bool = False,
        minimum_frequency: float = 0.0,
        tare_dc_phase: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)

    frequencies = data[:, 0]
    z_real = data[:, 1]
    z_imaginary = data[:, 2]
    z_impedance = z_real + 1j * z_imaginary

    sorted_indices = np.argsort(frequencies)
    frequencies = frequencies[sorted_indices]
    z_impedance = z_impedance[sorted_indices]

    if tare_dc_phase and len(z_impedance) > 0:
        dc_phase_angle = np.angle(z_impedance[0])
        z_impedance = z_impedance * np.exp(-1j * dc_phase_angle)

    if minimum_frequency > 0.0:
        frequency_mask = frequencies >= minimum_frequency
        frequencies = frequencies[frequency_mask]
        z_impedance = z_impedance[frequency_mask]

    if trim_inductive:
        frequencies, z_impedance = preprocessing.ignoreBelowX(
            frequencies, z_impedance
        )

    return frequencies, z_impedance


def load_eis_endurance_dataframe(
        directory_path: Union[str, pathlib.Path] = DEFAULT_ENDURANCE_DIRECTORY,
        trim_inductive: bool = False,
        minimum_frequency: float = 0.0,
        tare_dc_phase: bool = False,
        sweep_duration_seconds: float = 67.6
) -> pd.DataFrame:
    sweep_entries = walk_eis_endurance_data(directory_path)

    records = []
    for run_number, filepath in sweep_entries:
        frequencies, z_impedance = prepare_eis_data(
            filepath=filepath,
            trim_inductive=trim_inductive,
            minimum_frequency=minimum_frequency,
            tare_dc_phase=tare_dc_phase
        )

        z_real_mohm = np.real(z_impedance) / 1e6
        z_imaginary_mohm = np.imag(z_impedance) / 1e6
        z_magnitude_mohm = np.abs(z_impedance) / 1e6
        phase_degrees = np.degrees(np.angle(z_impedance))
        elapsed_minutes = ((run_number - 1) * sweep_duration_seconds) / 60.0

        for frequency, z_real, z_imaginary, z_magnitude, phase in zip(
            frequencies,
            z_real_mohm,
            z_imaginary_mohm,
            z_magnitude_mohm,
            phase_degrees
        ):
            records.append({
                'run': int(run_number),
                'elapsed_min': float(elapsed_minutes),
                'frequency_hz': float(frequency),
                'z_real_mohm': float(z_real),
                'z_imag_mohm': float(z_imaginary),
                'neg_z_imag_mohm': float(-z_imaginary),
                'magnitude_mohm': float(z_magnitude),
                'phase_deg': float(phase)
            })

    return pd.DataFrame(records)


def compute_mean_sd_ci(
        values: np.ndarray,
        confidence: float = 0.95
) -> Dict[str, float]:
    clean_values = np.asarray(values, dtype=float)
    clean_values = clean_values[np.isfinite(clean_values)]

    number_of_samples = len(clean_values)
    if number_of_samples == 0:
        return {
            'n': 0,
            'mean': np.nan,
            'sd': np.nan,
            'sem': np.nan,
            'ci_low': np.nan,
            'ci_high': np.nan,
            'median': np.nan,
            'min': np.nan,
            'max': np.nan
        }

    mean_value = float(np.mean(clean_values))
    if number_of_samples == 1:
        return {
            'n': 1,
            'mean': mean_value,
            'sd': 0.0,
            'sem': 0.0,
            'ci_low': mean_value,
            'ci_high': mean_value,
            'median': mean_value,
            'min': mean_value,
            'max': mean_value
        }

    std_value = float(np.std(clean_values, ddof=1))
    sem_value = std_value / np.sqrt(number_of_samples)
    alpha = 1.0 - confidence
    t_critical = float(
        stats.t.ppf(1.0 - alpha / 2.0, df=number_of_samples - 1)
    )
    margin = t_critical * sem_value

    return {
        'n': int(number_of_samples),
        'mean': mean_value,
        'sd': std_value,
        'sem': float(sem_value),
        'ci_low': float(mean_value - margin),
        'ci_high': float(mean_value + margin),
        'median': float(np.median(clean_values)),
        'min': float(np.min(clean_values)),
        'max': float(np.max(clean_values))
    }


def compute_eis_endurance_statistics(
        dataframe_eis: pd.DataFrame,
        confidence: float = 0.95
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    total_runs = int(dataframe_eis['run'].max())

    intervals = [
        ('Quartile 1 (Runs 1--25)', 1, min(25, total_runs)),
        ('Quartile 2 (Runs 26--50)', 26, min(50, total_runs)),
        ('Quartile 3 (Runs 51--75)', 51, min(75, total_runs)),
        (f'Quartile 4 (Runs 76--{total_runs})', 76, total_runs),
        (f'Full Series (Runs 1--{total_runs})', 1, total_runs)
    ]

    frequency_records = []
    for interval_label, run_start, run_end in intervals:
        sub_interval = dataframe_eis[
            (dataframe_eis['run'] >= run_start)
            & (dataframe_eis['run'] <= run_end)
        ]
        if sub_interval.empty:
            continue

        for frequency in sorted(sub_interval['frequency_hz'].unique()):
            sub_frequency = sub_interval[
                sub_interval['frequency_hz'] == frequency
            ].sort_values('run')

            magnitude_stats = compute_mean_sd_ci(
                sub_frequency['magnitude_mohm'].to_numpy(),
                confidence=confidence
            )
            z_real_stats = compute_mean_sd_ci(
                sub_frequency['z_real_mohm'].to_numpy(),
                confidence=confidence
            )
            z_imaginary_stats = compute_mean_sd_ci(
                sub_frequency['z_imag_mohm'].to_numpy(),
                confidence=confidence
            )
            phase_stats = compute_mean_sd_ci(
                sub_frequency['phase_deg'].to_numpy(),
                confidence=confidence
            )

            if len(sub_frequency) >= 3:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    run_indices = sub_frequency['run'].to_numpy(dtype=float)
                    magnitude_kohm = (
                        sub_frequency['magnitude_mohm'].to_numpy(dtype=float)
                        * 1000.0
                    )
                    slope, _, r_value, p_value, _ = stats.linregress(
                        run_indices, magnitude_kohm
                    )
            else:
                slope, r_value, p_value = np.nan, np.nan, np.nan

            frequency_records.append({
                'interval': interval_label,
                'run_start': int(run_start),
                'run_end': int(run_end),
                'n_sweeps': magnitude_stats['n'],
                'frequency_hz': float(frequency),
                'mag_mean_mohm': round(magnitude_stats['mean'], 4),
                'mag_sd_mohm': round(magnitude_stats['sd'], 4),
                'mag_ci_low_mohm': round(magnitude_stats['ci_low'], 4),
                'mag_ci_high_mohm': round(magnitude_stats['ci_high'], 4),
                'mag_median_mohm': round(magnitude_stats['median'], 4),
                'z_real_mean_mohm': round(z_real_stats['mean'], 4),
                'z_real_sd_mohm': round(z_real_stats['sd'], 4),
                'z_imag_mean_mohm': round(z_imaginary_stats['mean'], 4),
                'z_imag_sd_mohm': round(z_imaginary_stats['sd'], 4),
                'phase_mean_deg': round(phase_stats['mean'], 3),
                'phase_sd_deg': round(phase_stats['sd'], 3),
                'phase_ci_low_deg': round(phase_stats['ci_low'], 3),
                'phase_ci_high_deg': round(phase_stats['ci_high'], 3),
                'drift_kohm_per_run': round(float(slope), 3),
                'drift_r2': round(float(r_value ** 2), 4),
                'drift_p_value': float(p_value)
            })

    dataframe_frequency_summary = pd.DataFrame(frequency_records)

    window_records = []
    for interval_label, run_start, run_end in intervals:
        sub_window = dataframe_eis[
            (dataframe_eis['run'] >= run_start)
            & (dataframe_eis['run'] <= run_end)
        ]
        if sub_window.empty:
            continue

        midband_per_sweep = (
            sub_window[
                sub_window['frequency_hz'].isin([5.0, 10.0, 50.0, 100.0])
            ]
            .groupby('run', as_index=False)['magnitude_mohm']
            .mean()
        )
        midband_stats = compute_mean_sd_ci(
            midband_per_sweep['magnitude_mohm'].to_numpy(),
            confidence=confidence
        )

        sub_01hz = sub_window[sub_window['frequency_hz'] == 0.1]
        magnitude_01hz_stats = compute_mean_sd_ci(
            sub_01hz['magnitude_mohm'].to_numpy(), confidence=confidence
        )
        phase_01hz_stats = compute_mean_sd_ci(
            sub_01hz['phase_deg'].to_numpy(), confidence=confidence
        )

        sub_1hz = sub_window[sub_window['frequency_hz'] == 1.0]
        phase_1hz_stats = compute_mean_sd_ci(
            sub_1hz['phase_deg'].to_numpy(), confidence=confidence
        )

        window_records.append({
            'interval': interval_label,
            'n_sweeps': midband_stats['n'],
            'midband_mag_mean_mohm': round(midband_stats['mean'], 4),
            'midband_mag_sd_mohm': round(midband_stats['sd'], 4),
            'midband_mag_ci_low_mohm': round(midband_stats['ci_low'], 4),
            'midband_mag_ci_high_mohm': round(midband_stats['ci_high'], 4),
            'mag_01hz_mean_mohm': round(magnitude_01hz_stats['mean'], 4),
            'mag_01hz_sd_mohm': round(magnitude_01hz_stats['sd'], 4),
            'mag_01hz_ci_low_mohm': round(magnitude_01hz_stats['ci_low'], 4),
            'mag_01hz_ci_high_mohm': round(magnitude_01hz_stats['ci_high'], 4),
            'phase_01hz_mean_deg': round(phase_01hz_stats['mean'], 2),
            'phase_01hz_sd_deg': round(phase_01hz_stats['sd'], 2),
            'phase_01hz_ci_low_deg': round(phase_01hz_stats['ci_low'], 2),
            'phase_01hz_ci_high_deg': round(phase_01hz_stats['ci_high'], 2),
            'phase_1hz_mean_deg': round(phase_1hz_stats['mean'], 2),
            'phase_1hz_sd_deg': round(phase_1hz_stats['sd'], 2),
            'phase_1hz_ci_low_deg': round(phase_1hz_stats['ci_low'], 2),
            'phase_1hz_ci_high_deg': round(phase_1hz_stats['ci_high'], 2)
        })

    dataframe_window_summary = pd.DataFrame(window_records)

    return dataframe_frequency_summary, dataframe_window_summary


def save_eis_endurance_latex(
        target_filepath: Union[str, pathlib.Path],
        dataframe_frequency_summary: pd.DataFrame,
        caption: str = ''
) -> None:
    target_path = pathlib.Path(target_filepath)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    latex_lines = [
        r'\begin{table*}[htbp]',
        r'\centering',
        rf'\caption{{{caption}}}',
        r'\label{tab:eis-endurance-statistics}',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{lcrccccc}',
        r'\toprule',
        (
            r'\textbf{Run Window} & \textbf{Freq.} & $N$ &'
            r' \textbf{$|Z|$ Mean $\pm$ SD} & \textbf{$|Z|$ 95\% CI} &'
            r' \textbf{Phase Mean $\pm$ SD} & \textbf{Phase 95\% CI} &'
            r' \textbf{Drift Rate} \\'
        ),
        (
            r' & $(\mathrm{Hz})$ & & $(\mathrm{M\Omega})$ &'
            r' $(\mathrm{M\Omega})$ & $(^{\circ})$ & $(^{\circ})$ &'
            r' $(\mathrm{k\Omega/run})$ \\'
        ),
        r'\midrule'
    ]

    previous_interval = None
    for _, row in dataframe_frequency_summary.iterrows():
        interval_label = str(row['interval'])
        if (
            previous_interval is not None
            and interval_label != previous_interval
        ):
            latex_lines.append(r'\midrule')

        display_interval = (
            interval_label if interval_label != previous_interval else ''
        )
        previous_interval = interval_label

        frequency_value = row['frequency_hz']
        frequency_cell = f'{frequency_value:g}'
        sample_count = int(row['n_sweeps'])

        mag_mean = row['mag_mean_mohm']
        mag_sd = row['mag_sd_mohm']
        mag_ci_low = row['mag_ci_low_mohm']
        mag_ci_high = row['mag_ci_high_mohm']
        phase_mean = row['phase_mean_deg']
        phase_sd = row['phase_sd_deg']
        phase_ci_low = row['phase_ci_low_deg']
        phase_ci_high = row['phase_ci_high_deg']
        drift_value = row['drift_kohm_per_run']

        magnitude_mean_sd = f'${mag_mean:.3f} \\pm {mag_sd:.3f}$'
        magnitude_ci = f'$[{mag_ci_low:.3f}, {mag_ci_high:.3f}]$'
        phase_mean_sd = f'${phase_mean:+.2f} \\pm {phase_sd:.2f}$'
        phase_ci = f'$[{phase_ci_low:+.2f}, {phase_ci_high:+.2f}]$'
        drift_cell = f'${drift_value:+.2f}$'

        latex_lines.append(
            f'{display_interval} & {frequency_cell} & {sample_count} &'
            f' {magnitude_mean_sd} & {magnitude_ci} & {phase_mean_sd} &'
            f' {phase_ci} & {drift_cell} \\\\'
        )

    latex_lines.extend([
        r'\bottomrule',
        r'\end{tabular}%',
        r'}',
        r'\end{table*}'
    ])

    target_path.write_text('\n'.join(latex_lines), encoding='utf-8')


if __name__ == '__main__':
    dataframe_raw_eis = load_eis_endurance_dataframe(
        directory_path=DEFAULT_ENDURANCE_DIRECTORY,
        trim_inductive=False,
        minimum_frequency=0.0,
        tare_dc_phase=False
    )

    dataframe_frequency_stats, dataframe_window_stats = (
        compute_eis_endurance_statistics(dataframe_raw_eis, confidence=0.95)
    )

    results_directory = pathlib.Path('results')
    results_directory.mkdir(parents=True, exist_ok=True)

    dataframe_raw_eis.to_csv(
        results_directory / 'eis-endurance-all-sweeps.csv', index=False
    )
    dataframe_frequency_stats.to_csv(
        results_directory / 'eis-endurance-frequency-stats.csv', index=False
    )
    dataframe_window_stats.to_csv(
        results_directory / 'eis-endurance-window-stats.csv', index=False
    )

    save_eis_endurance_latex(
        target_filepath=results_directory / 'eis-endurance-statistics.tex',
        dataframe_frequency_summary=dataframe_frequency_stats,
        caption=EIS_ENDURANCE_CAPTION
    )

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1200)

    print(dataframe_window_stats.to_string(index=False))

    print(dataframe_frequency_stats.to_string(index=False))
