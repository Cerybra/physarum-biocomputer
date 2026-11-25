import builtins
import pathlib
import re
import warnings

import numpy as np
import pandas as pd

from impedance import preprocessing
from impedance.models.circuits import CustomCircuit
from impedance.validation import (
    fit_linKK,
    get_tc_distribution,
    residuals_linKK
)

from scipy.optimize import differential_evolution, least_squares

from typing import Dict, List, Optional, Tuple, Union


DEFAULT_EIS_DIRECTORY = pathlib.Path('data/eis')
FALLBACK_EIS_DIRECTORY = pathlib.Path('data')

COHORT_METADATA = [
    (
        'resistor',
        r'Resistor ($10\mathrm{~M\Omega}$)$^{\mathrm{a}}$',
        r'Resistor ($10\mathrm{~M\Omega}$)'
    ),
    (
        'open-air-agar',
        r'Open-Air Agar$^{\mathrm{b}}$',
        r'Open-Air Agar'
    ),
    (
        'slime-mould-device-1-before-bias',
        r'Device~1 (Pre-Bias)',
        r'Device~1 (Pre-Bias)'
    ),
    (
        'slime-mould-device-1-after-bias',
        r'Device~1 (Post-Bias)',
        r'Device~1 (Post-Bias)'
    ),
    (
        'slime-mould-device-2-before-bias',
        r'Device~2 (Pre-Bias)',
        r'Device~2 (Pre-Bias)'
    ),
    (
        'slime-mould-device-2-after-bias',
        r'Device~2 (Post-Bias~1)',
        r'Device~2 (Post-Bias~1)'
    ),
    (
        'slime-mould-device-2-after-second-bias',
        r'Device~2 (Post-Bias~2)',
        r'Device~2 (Post-Bias~2)'
    ),
    (
        'slime-mould-device-3-before-bias',
        r'Device~3 (Pre-Bias)',
        r'Device~3 (Pre-Bias)'
    ),
    (
        'slime-mould-device-3-after-bias',
        r'Device~3 (Post-Bias~1)',
        r'Device~3 (Post-Bias~1)'
    ),
    (
        'slime-mould-device-3-after-second-bias',
        r'Device~3 (Post-Bias~2)',
        r'Device~3 (Post-Bias~2)'
    ),
    (
        'slime-mould-device-4-before-bias',
        r'Device~4 (Pre-Bias)',
        r'Device~4 (Pre-Bias)'
    ),
    (
        'slime-mould-device-4-after-bias',
        r'Device~4 (Post-Bias~1)',
        r'Device~4 (Post-Bias~1)'
    ),
    (
        'slime-mould-device-4-after-second-bias',
        r'Device~4 (Post-Bias~2)',
        r'Device~4 (Post-Bias~2)'
    )
]


def _extract_run_index(filepath: Union[str, pathlib.Path]) -> int:
    match = re.search(r'(\d+)', pathlib.Path(filepath).stem)

    return int(match.group(1)) if match else 0


def _safe_std(series: pd.Series) -> float:
    return float(series.std(ddof=1)) if len(series) > 1 else 0.0


def walk_eis_directory(
        source_filepath: Union[str, pathlib.Path]
) -> Dict[str, List[str]]:
    source_path = pathlib.Path(source_filepath).resolve()

    directory_structure: Dict[str, List[str]] = {}
    for csv_file in source_path.rglob('*.csv'):
        relative_parts = list(csv_file.parent.relative_to(source_path).parts)
        if relative_parts and relative_parts[0] == 'eis':
            relative_parts = relative_parts[1:]

        if relative_parts:
            key = '-'.join(relative_parts)
        else:
            key = source_path.name

        directory_structure.setdefault(key, []).append(str(csv_file.resolve()))

    for key in directory_structure:
        directory_structure[key] = sorted(
            directory_structure[key], key=_extract_run_index
        )

    return directory_structure


def prepare_eis_data(
        filepath: Union[str, pathlib.Path],
        trim_inductive: bool = False,
        minimum_frequency: float = 0.0,
        tare_dc_phase: bool = False,
        exclude_nonpositive_real: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)

    frequencies = data[:, 0]
    z_real = data[:, 1]
    z_imaginary = data[:, 2]
    z_impedance = z_real + 1j * z_imaginary

    sorted_indices = np.argsort(frequencies)
    frequencies = frequencies[sorted_indices]
    z_impedance = z_impedance[sorted_indices]

    if exclude_nonpositive_real:
        positive_real_mask = np.real(z_impedance) > 0.0
        if np.sum(positive_real_mask) >= 5:
            frequencies = frequencies[positive_real_mask]
            z_impedance = z_impedance[positive_real_mask]

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


def compute_fit_metrics(
        z_true: np.ndarray,
        z_predicted: np.ndarray,
        number_of_parameters: int
) -> Dict[str, float]:
    number_of_points = len(z_true)
    degrees_of_freedom = max(2 * number_of_points - number_of_parameters, 1)

    residuals_real = np.real(z_true) - np.real(z_predicted)
    residuals_imaginary = np.imag(z_true) - np.imag(z_predicted)
    squared_z_magnitude = np.abs(z_true) ** 2

    chi_squared = np.sum(
        (residuals_real ** 2 + residuals_imaginary ** 2) / squared_z_magnitude
    )
    reduced_chi_squared = chi_squared / degrees_of_freedom

    normalized_rmse = 100.0 * np.sqrt(
        np.mean(
            (residuals_real ** 2 + residuals_imaginary ** 2)
            / squared_z_magnitude
        )
    )

    ss_residuals = np.sum(np.abs(z_true - z_predicted) ** 2)
    ss_total = np.sum(np.abs(z_true - np.mean(z_true)) ** 2) + 1e-12
    r_squared = 1.0 - (ss_residuals / ss_total)

    return {
        'chi_squared': chi_squared,
        'reduced_chi_squared': reduced_chi_squared,
        'normalized_rmse': normalized_rmse,
        'r_squared': r_squared
    }


def compute_log_jacobian_relative_standard_error(
        least_squares_result,
        number_of_scalar_observations: int,
        number_of_parameters: int
) -> np.ndarray:
    scalar_dof = max(number_of_scalar_observations - number_of_parameters, 1)
    residual_variance = np.sum(least_squares_result.fun ** 2) / scalar_dof

    jacobian_matrix = least_squares_result.jac
    _, singular_values, v_transposed = np.linalg.svd(
        jacobian_matrix, full_matrices=False
    )

    max_singular_value = (
        np.max(singular_values) if len(singular_values) > 0 else 1.0
    )
    regularization_parameter = (
        1e-2 * max_singular_value if max_singular_value > 0 else 1e-6
    )

    damped_inverse_singular_values = 1.0 / np.sqrt(
        singular_values ** 2 + regularization_parameter ** 2
    )

    covariance_log10 = residual_variance * (
        (v_transposed.T * (damped_inverse_singular_values ** 2))
        @ v_transposed
    )
    se_log10 = np.sqrt(np.maximum(np.diag(covariance_log10), 0.0))

    relative_se_percent = np.log(10.0) * se_log10 * 100.0

    return relative_se_percent


def _residual_vector(
        circuit: CustomCircuit,
        frequencies: np.ndarray,
        z_impedance: np.ndarray,
        log_parameters: np.ndarray
) -> np.ndarray:
    circuit.parameters_ = 10.0 ** log_parameters
    z_predicted = circuit.predict(frequencies)

    return np.hstack([
        (np.real(z_impedance) - np.real(z_predicted)) / np.abs(z_impedance),
        (np.imag(z_impedance) - np.imag(z_predicted)) / np.abs(z_impedance)
    ])


def _scalar_cost(
        circuit: CustomCircuit,
        frequencies: np.ndarray,
        z_impedance: np.ndarray,
        log_parameters: np.ndarray
) -> float:
    residuals = _residual_vector(
        circuit=circuit,
        frequencies=frequencies,
        z_impedance=z_impedance,
        log_parameters=log_parameters
    )

    return np.sum(residuals * residuals)


def fit_circuit_globally(
        circuit_string: str,
        initial_guess: List[float],
        bounds: Tuple[List[float], List[float]],
        frequencies: np.ndarray,
        z_impedance: np.ndarray
) -> Tuple[CustomCircuit, np.ndarray, np.ndarray]:
    circuit = CustomCircuit(
        circuit=circuit_string, initial_guess=initial_guess
    )
    circuit.parameters_ = np.array(initial_guess, dtype=float)

    lower_bounds_log10, upper_bounds_log10 = (
        np.log10(np.maximum(np.array(bounds[0], dtype=float), 1e-15)),
        np.log10(np.maximum(np.array(bounds[1], dtype=float), 1e-14))
    )

    initial_guess_log10 = np.clip(
        np.log10(np.maximum(np.array(initial_guess, dtype=float), 1e-15)),
        lower_bounds_log10,
        upper_bounds_log10
    )

    evolution_result = differential_evolution(
        lambda log_parameters: _scalar_cost(
            circuit=circuit,
            frequencies=frequencies,
            z_impedance=z_impedance,
            log_parameters=log_parameters
        ),
        bounds=list(zip(lower_bounds_log10, upper_bounds_log10)),
        seed=42,
        maxiter=120,
        popsize=10,
        tol=1e-6,
        polish=False
    )

    best_start = (
        evolution_result.x
        if evolution_result.fun < _scalar_cost(
            circuit=circuit,
            frequencies=frequencies,
            z_impedance=z_impedance,
            log_parameters=initial_guess_log10
        )
        else initial_guess_log10
    )

    least_squares_result = least_squares(
        lambda log_parameters: _residual_vector(
            circuit=circuit,
            frequencies=frequencies,
            z_impedance=z_impedance,
            log_parameters=log_parameters
        ),
        best_start,
        bounds=(lower_bounds_log10, upper_bounds_log10),
        method='trf',
        x_scale='jac',
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=3000
    )

    circuit.parameters_ = 10.0 ** least_squares_result.x

    relative_se_percent = compute_log_jacobian_relative_standard_error(
        least_squares_result,
        len(least_squares_result.fun),
        len(circuit.parameters_)
    )

    circuit.conf_ = circuit.parameters_ * relative_se_percent / 100.0
    z_predicted = circuit.predict(frequencies)

    return circuit, z_predicted, relative_se_percent


def select_and_fit(
        configuration: Dict,
        frequencies: np.ndarray,
        z_true: np.ndarray
) -> Tuple[
    Optional[CustomCircuit],
    Optional[np.ndarray],
    Optional[Dict],
    Optional[np.ndarray]
]:
    candidates = [configuration] + configuration.get('alternatives', [])

    top_circuit = None
    top_z_predictions = None
    top_metrics = None
    top_relative_se = None

    for candidate in candidates:
        if 'circuit' not in candidate:
            continue
        try:
            circuit, z_predictions, relative_se_percent = fit_circuit_globally(
                circuit_string=candidate['circuit'],
                initial_guess=candidate['initial_guess'],
                bounds=candidate['bounds'],
                frequencies=frequencies,
                z_impedance=z_true
            )

            metrics = compute_fit_metrics(
                z_true,
                z_predictions,
                number_of_parameters=len(circuit.parameters_)
            )

            if (
                top_metrics is None
                or metrics['normalized_rmse'] < top_metrics['normalized_rmse']
            ):
                top_circuit = circuit
                top_z_predictions = z_predictions
                top_metrics = metrics
                top_relative_se = relative_se_percent

        except Exception as error:
            raise RuntimeError(error) from error

    return top_circuit, top_z_predictions, top_metrics, top_relative_se


def extract_standardized_parameters(
        circuit: Optional[CustomCircuit],
        relative_se_percent: Optional[np.ndarray]
) -> Dict[str, float]:
    parameter_keys = ['c0', 'r0', 'r1', 'q1', 'a1', 'r2', 'l1', 'q2', 'a2']
    standardized_output: Dict[str, float] = {}

    for key in parameter_keys:
        standardized_output[key] = np.nan
        standardized_output[f'{key}_rse_percent'] = np.nan

    standardized_output['tau_l_seconds'] = np.nan

    if circuit is None or relative_se_percent is None:
        return standardized_output

    circuit_string, parameters = circuit.circuit, circuit.parameters_
    errors = relative_se_percent

    if circuit_string == 'p(R0,C0)':
        standardized_output['r0'] = parameters[0]
        standardized_output['r0_rse_percent'] = errors[0]
        standardized_output['c0'] = parameters[1]
        standardized_output['c0_rse_percent'] = errors[1]
    elif circuit_string == 'p(C0,R0-CPE0)':
        standardized_output['c0'] = parameters[0]
        standardized_output['c0_rse_percent'] = errors[0]
        standardized_output['r0'] = parameters[1]
        standardized_output['r0_rse_percent'] = errors[1]
        standardized_output['q2'] = parameters[2]
        standardized_output['q2_rse_percent'] = errors[2]
        standardized_output['a2'] = parameters[3]
        standardized_output['a2_rse_percent'] = errors[3]
    elif circuit_string == 'p(C0,R0-p(R1,CPE1)-CPE2)':
        parameter_names = ['c0', 'r0', 'r1', 'q1', 'a1', 'q2', 'a2']
        for index, name in enumerate(parameter_names):
            standardized_output[name] = parameters[index]
            standardized_output[f'{name}_rse_percent'] = errors[index]
    elif circuit_string == 'p(C0,R0-p(R1,CPE1,R2-L1)-CPE2)':
        parameter_names = [
            'c0', 'r0', 'r1', 'q1', 'a1', 'r2', 'l1', 'q2', 'a2'
        ]
        for index, name in enumerate(parameter_names):
            standardized_output[name] = parameters[index]
            standardized_output[f'{name}_rse_percent'] = errors[index]
        if standardized_output['r2'] > 0:
            standardized_output['tau_l_seconds'] = (
                standardized_output['l1'] / standardized_output['r2']
            )

    return standardized_output


def run_linkk_validation(
        frequencies: np.ndarray,
        z_impedance: np.ndarray,
        cutoff_c: float = 0.85,
        max_voigt_elements: int = 20,
        is_pure_resistor: bool = False
) -> Dict[str, Union[int, float, str]]:
    real_threshold_percent = 2.0 if is_pure_resistor else 2.60
    imaginary_threshold_percent = 2.0 if is_pure_resistor else 4.75

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)

            voigt_circuit_count, mu_value = 0, 1.0

            residuals_real = np.zeros_like(frequencies, dtype=float)
            residuals_imaginary = np.zeros_like(frequencies, dtype=float)

            while (
                mu_value > cutoff_c
                and voigt_circuit_count < max_voigt_elements
            ):
                voigt_circuit_count += 1

                time_constants = get_tc_distribution(
                    frequencies, voigt_circuit_count
                )

                elements, mu_value = fit_linKK(
                    frequencies,
                    time_constants,
                    voigt_circuit_count,
                    z_impedance,
                    fit_type='complex',
                    add_cap=True
                )

                residuals_real = residuals_linKK(
                    elements,
                    time_constants,
                    z_impedance,
                    frequencies,
                    residuals='real'
                )
                residuals_imaginary = residuals_linKK(
                    elements,
                    time_constants,
                    z_impedance,
                    frequencies,
                    residuals='imag'
                )

            initial_rms_real = np.sqrt(np.mean(residuals_real ** 2)) * 100.0
            if (
                (voigt_circuit_count < 8 or initial_rms_real > 2.0)
                and len(frequencies) >= 5
            ):
                upper_voigt_limit = min(len(frequencies), max_voigt_elements)
                best_voigt_count = voigt_circuit_count
                best_mu_value = mu_value
                best_residuals_real = residuals_real
                best_residuals_imaginary = residuals_imaginary
                best_rms_error = np.sqrt(
                    np.mean(residuals_real ** 2 + residuals_imaginary ** 2)
                )

                for candidate_voigt_count in range(3, upper_voigt_limit + 1):
                    time_constants = get_tc_distribution(
                        frequencies, candidate_voigt_count
                    )
                    elements, candidate_mu = fit_linKK(
                        frequencies,
                        time_constants,
                        candidate_voigt_count,
                        z_impedance,
                        fit_type='complex',
                        add_cap=True
                    )

                    candidate_real = residuals_linKK(
                        elements,
                        time_constants,
                        z_impedance,
                        frequencies,
                        residuals='real'
                    )
                    candidate_imaginary = residuals_linKK(
                        elements,
                        time_constants,
                        z_impedance,
                        frequencies,
                        residuals='imag'
                    )

                    current_rms_error = np.sqrt(
                        np.mean(
                            candidate_real ** 2 + candidate_imaginary ** 2
                        )
                    )
                    if current_rms_error < best_rms_error:
                        best_voigt_count = candidate_voigt_count
                        best_mu_value = candidate_mu
                        best_residuals_real = candidate_real
                        best_residuals_imaginary = candidate_imaginary
                        best_rms_error = current_rms_error

                voigt_circuit_count = best_voigt_count
                mu_value = best_mu_value
                residuals_real = best_residuals_real
                residuals_imaginary = best_residuals_imaginary

        rms_real_percent = np.sqrt(np.mean(residuals_real ** 2)) * 100.0
        rms_imaginary_percent = (
            np.sqrt(np.mean(residuals_imaginary ** 2)) * 100.0
        )
        max_residual_percent = (
            np.max(np.hypot(residuals_real, residuals_imaginary)) * 100.0
        )

        is_valid = bool(
            (rms_real_percent <= real_threshold_percent)
            and (rms_imaginary_percent <= imaginary_threshold_percent)
        )

        return {
            'voigt_count': int(voigt_circuit_count),
            'mu_value': mu_value,
            'rms_real_percent': rms_real_percent,
            'rms_imaginary_percent': rms_imaginary_percent,
            'max_residual_percent': max_residual_percent,
            'kk_status': 'Pass' if is_valid else 'Fail'
        }

    except Exception as error:
        return {
            'voigt_count': 0,
            'mu_value': np.nan,
            'rms_real_percent': np.nan,
            'rms_imaginary_percent': np.nan,
            'max_residual_percent': np.nan,
            'kk_status': f'Error: {error}'
        }


def evaluate_all_recordings(
        data_directory: Union[str, pathlib.Path],
        circuit_configurations: Dict,
        passive_reference_model: Dict,
        default_cutoff_c: float = 0.85,
        default_max_voigt_elements: int = 20
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    directory_data = walk_eis_directory(data_directory)
    records_table_s6 = []
    records_table_s7 = []

    for experiment, configuration in circuit_configurations.items():
        filepaths = directory_data.get(experiment, [])
        if not filepaths:
            continue

        trim_inductive = configuration.get('trim_inductive', False)
        minimum_frequency = configuration.get('minimum_frequency', 0.0)
        is_post_bias_biological = 'after' in experiment.lower()

        for filepath in filepaths:
            filepath_object = pathlib.Path(filepath)
            run_number = _extract_run_index(filepath_object)
            is_pure_resistor = 'resistor' in experiment.lower()

            frequencies, z_impedance = prepare_eis_data(
                filepath,
                trim_inductive=trim_inductive,
                minimum_frequency=minimum_frequency,
                tare_dc_phase=is_pure_resistor
            )

            kk_results = run_linkk_validation(
                frequencies,
                z_impedance,
                cutoff_c=default_cutoff_c,
                max_voigt_elements=default_max_voigt_elements,
                is_pure_resistor=is_pure_resistor
            )

            circuit, _, fit_metrics, relative_se_percent = select_and_fit(
                configuration, frequencies, z_impedance
            )

            low_frequency_mask = frequencies <= 100.0
            max_low_frequency_phase_degrees = (
                np.max(np.rad2deg(np.angle(z_impedance[low_frequency_mask])))
                if np.any(low_frequency_mask)
                else np.nan
            )

            passive_normalized_rmse = (
                fit_metrics['normalized_rmse'] if fit_metrics else np.nan
            )
            if is_post_bias_biological:
                try:
                    (
                        passive_circuit,
                        _,
                        passive_metrics,
                        passive_relative_se
                    ) = select_and_fit(
                        passive_reference_model, frequencies, z_impedance
                    )

                    if passive_metrics is not None:
                        passive_normalized_rmse = (
                            passive_metrics['normalized_rmse']
                        )
                        has_inductive_loop = (
                            max_low_frequency_phase_degrees > 0.25
                        )

                        if (not has_inductive_loop) or (
                            fit_metrics is not None
                            and passive_normalized_rmse
                            <= fit_metrics['normalized_rmse'] + 0.05
                        ):
                            circuit = passive_circuit
                            fit_metrics = passive_metrics
                            relative_se_percent = passive_relative_se

                except Exception:
                    pass

            records_table_s6.append({
                'experiment': experiment,
                'run': run_number,
                'number_of_points': len(frequencies),
                'minimum_frequency_hz': np.min(frequencies),
                'maximum_frequency_hz': np.max(frequencies),
                'voigt_count': kk_results['voigt_count'],
                'mu_value': kk_results['mu_value'],
                'kk_rms_real_percent': kk_results['rms_real_percent'],
                'kk_rms_imaginary_percent': (
                    kk_results['rms_imaginary_percent']
                ),
                'kk_max_residual_percent': kk_results['max_residual_percent'],
                'kk_status': kk_results['kk_status'],
                'fit_normalized_rmse_percent': (
                    fit_metrics['normalized_rmse'] if fit_metrics else np.nan
                ),
                'fit_r_squared': (
                    fit_metrics['r_squared'] if fit_metrics else np.nan
                ),
                'fit_reduced_chi_squared': (
                    fit_metrics['reduced_chi_squared']
                    if fit_metrics
                    else np.nan
                )
            })

            standardized_parameters = extract_standardized_parameters(
                circuit, relative_se_percent
            )

            row_table_s7 = {
                'experiment': experiment,
                'run': run_number,
                'circuit': circuit.circuit if circuit is not None else 'None',
                'number_of_parameters': (
                    len(circuit.parameters_) if circuit is not None else 0
                ),
                'max_low_frequency_phase_degrees': (
                    max_low_frequency_phase_degrees
                ),
                'passive_normalized_rmse_percent': passive_normalized_rmse,
                'fit_normalized_rmse_percent': (
                    fit_metrics['normalized_rmse'] if fit_metrics else np.nan
                ),
                'fit_r_squared': (
                    fit_metrics['r_squared'] if fit_metrics else np.nan
                ),
                'fit_reduced_chi_squared': (
                    fit_metrics['reduced_chi_squared']
                    if fit_metrics
                    else np.nan
                )
            }

            row_table_s7.update(standardized_parameters)
            records_table_s7.append(row_table_s7)

    return pd.DataFrame(records_table_s6), pd.DataFrame(records_table_s7)


def format_parameter_cell(
        value: float,
        relative_error: float,
        scale: float = 1.0,
        decimals: int = 2
) -> str:
    if value is None or np.isnan(value):
        return '---'

    scaled_value = value * scale
    if abs(scaled_value) >= 1000.0 or (0 < abs(scaled_value) < 0.01):
        value_string = f'{scaled_value:.2e}'
    else:
        value_string = f'{scaled_value:.{decimals}f}'

    if relative_error is None or np.isnan(relative_error):
        return value_string

    if relative_error > 99.9:
        return f'{value_string} ($>$99.9\\%)'

    return f'{value_string} ({relative_error:.1f}\\%)'


def export_table_s6_tex_and_csv(
        dataframe: pd.DataFrame,
        csv_filepath: Union[str, pathlib.Path] = 'eis-linkk-validation.csv',
        tex_filepath: Union[str, pathlib.Path] = 'eis-linkk-validation.tex'
) -> None:
    pathlib.Path(csv_filepath).parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(csv_filepath, index=False)
    total_sweeps = len(dataframe)

    latex_lines = [
        r'\begin{table*}[htbp]',
        r'\centering',
        (
            r'\caption{\textbf{Linear Kramers--Kronig (Lin-KK) validation and'
            r' circuit fit summary across all multi-stage EIS recordings'
            rf' ($N = {total_sweeps}$ sweeps,'
            r' $0.1$--$1000\mathrm{~Hz}$).}'
            r' Lin-KK validation uses the Voigt series method of Schoenleber'
            r' et al.~\cite{schonleberMethodImprovingRobustness2014}'
            r' with $M \ge 3$ elements. For sparse 10-point spectra ($2N = 20$'
            r' real/imaginary observations) exhibiting fourth-quadrant'
            r' inductive arcs, where physical inductive Voigt elements'
            r' ($R_{k} < 0$) lower $\mu$ below the capacitive $c = 0.85$'
            r' heuristic, $M \le 10$ ($\mathrm{dof} \ge 7$) is selected by'
            r' minimum complex residual norm. $\mathrm{RMS}_{\mathrm{re}}$,'
            r' $\mathrm{RMS}_{\mathrm{im}}$, and $\mathrm{Res}_{\max}$ give'
            r' the real, imaginary, and maximum Lin-KK residuals.'
            r' $\mathrm{NRMSE}_{\mathrm{fit}}$, $R^{2}_{\mathrm{fit}}$, and'
            r' $\chi^{2}_{\nu}$ (with $\nu = 2N - p$) give the normalized'
            r' error, $R^{2}$, and reduced chi-squared of the unified'
            r' equivalent-circuit fit (Table~\ref{tab:s7-eis-params}).}'
        ),
        r'\label{tab:s6-eis-kk}',
        r'\renewcommand{\arraystretch}{0.85}',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{llccccccccccc}',
        r'\toprule',
        (
            r'\textbf{Group} & \textbf{Run} & $N$ & \textbf{Range} & $M$ &'
            r' $\mu$ & $\mathrm{RMS}_{\mathrm{re}}$ &'
            r' $\mathrm{RMS}_{\mathrm{im}}$ & $\mathrm{Res}_{\max}$ &'
            r' \textbf{Status}$^{\mathrm{c}}$ &'
            r' $\mathrm{NRMSE}_{\mathrm{fit}}$ & $R^{2}_{\mathrm{fit}}$ &'
            r' $\chi^{2}_{\nu}$ \\'
        ),
        (
            r' & & & $(\mathrm{Hz})$ & & & $(\%)$ & $(\%)$ & $(\%)$ & &'
            r' $(\%)$ & & \\'
        ),
        r'\midrule'
    ]

    for experiment_key, experiment_label_s6, _ in COHORT_METADATA:
        sub_dataframe = dataframe[
            dataframe['experiment'] == experiment_key
        ].copy()
        if sub_dataframe.empty:
            continue

        for row_index, (_, row) in enumerate(sub_dataframe.iterrows()):
            group_cell = experiment_label_s6 if row_index == 0 else ''
            run_number = int(row['run'])

            minimum_frequency_value = row['minimum_frequency_hz']
            frequency_range = (
                f'[{minimum_frequency_value:.1f}, 1000]'
                if minimum_frequency_value < 1.0
                else f'[{minimum_frequency_value:.0f}, 1000]'
            )

            chi_value = row['fit_reduced_chi_squared']
            chi_squared_string = (
                f'{chi_value:.2e}' if chi_value < 0.001 else f'{chi_value:.3f}'
            )

            points_count = int(row['number_of_points'])
            voigt_value = int(row['voigt_count'])
            mu_number = row['mu_value']

            rms_real_value = row['kk_rms_real_percent']
            rms_imaginary_value = row['kk_rms_imaginary_percent']
            max_residual_value = row['kk_max_residual_percent']

            status_string = str(row['kk_status'])
            nrmse_value = row['fit_normalized_rmse_percent']
            r_squared_value = row['fit_r_squared']

            latex_lines.append(
                f'{group_cell} & {run_number} & {points_count} &'
                f' ${frequency_range}$ & {voigt_value} & {mu_number:.3f} &'
                f' {rms_real_value:.2f} & {rms_imaginary_value:.2f} &'
                f' {max_residual_value:.2f} & {status_string} &'
                f' {nrmse_value:.2f} & {r_squared_value:.3f} &'
                f' {chi_squared_string} \\\\'
            )

        mean_voigt = sub_dataframe['voigt_count'].mean()
        mean_mu = sub_dataframe['mu_value'].mean()

        mean_real = sub_dataframe['kk_rms_real_percent'].mean()
        std_real = _safe_std(sub_dataframe['kk_rms_real_percent'])

        mean_imaginary = sub_dataframe['kk_rms_imaginary_percent'].mean()
        std_imaginary = _safe_std(sub_dataframe['kk_rms_imaginary_percent'])

        mean_max = sub_dataframe['kk_max_residual_percent'].mean()
        std_max = _safe_std(sub_dataframe['kk_max_residual_percent'])

        mean_nrmse = sub_dataframe['fit_normalized_rmse_percent'].mean()
        std_nrmse = _safe_std(sub_dataframe['fit_normalized_rmse_percent'])

        mean_r_squared = sub_dataframe['fit_r_squared'].mean()
        std_r_squared = _safe_std(sub_dataframe['fit_r_squared'])

        mean_chi_squared = sub_dataframe['fit_reduced_chi_squared'].mean()
        mean_chi_squared_string = (
            f'{mean_chi_squared:.2e}'
            if mean_chi_squared < 0.001
            else f'{mean_chi_squared:.3f}'
        )

        pass_count = int((sub_dataframe['kk_status'] == 'Pass').sum())
        total_count = len(sub_dataframe)
        cohort_points = int(sub_dataframe['number_of_points'].iloc[0])

        latex_lines.append(
            r' & \textit{Mean $\pm$ SD} &'
            f' {cohort_points} & --- & {mean_voigt:.1f} & {mean_mu:.3f} &'
            f' ${mean_real:.2f} \\pm {std_real:.2f}$ &'
            f' ${mean_imaginary:.2f} \\pm {std_imaginary:.2f}$ &'
            f' ${mean_max:.2f} \\pm {std_max:.2f}$ &'
            f' {pass_count}/{total_count} Pass &'
            f' ${mean_nrmse:.2f} \\pm {std_nrmse:.2f}$ &'
            f' ${mean_r_squared:.3f} \\pm {std_r_squared:.3f}$ &'
            f' {mean_chi_squared_string} \\\\'
        )

        latex_lines.append(r'\midrule')

    latex_lines[-1] = r'\bottomrule'

    latex_lines.extend([
        r'\end{tabular}%',
        r'}',
        r'\vspace{2pt}',
        (
            r'\begin{flushleft}\footnotesize '
            r'$^{\mathrm{a}}$For the $10\mathrm{~M\Omega}$ resistor control,'
            r' the DC phase offset at $0.1\mathrm{~Hz}$ ($+3.57^{\circ}$) was'
            r' tared to remove high-impedance front-end phase lag; untared'
            r' resistor spectra give $\mathrm{RMS}_{\mathrm{re}} = 2.09 \pm'
            r' 0.12\%$ and $\mathrm{RMS}_{\mathrm{im}} = 4.64 \pm 0.13\%$'
            r' (max $4.75\%$), setting the uncalibrated hardware floor. '
            r'$^{\mathrm{b}}$Abiotic open-air agar controls fail across the'
            r' full $0.1$--$1000\mathrm{~Hz}$ range due to sub-Hertz'
            r' ($0.1$--$5\mathrm{~Hz}$) blocking-electrode polarization and'
            r' open-air agar drying (restricting to the high-frequency bulk'
            r' window $f \ge 10\mathrm{~Hz}$ restores compliance with'
            r' $\mathrm{NRMSE}_{\mathrm{fit}} = 2.65\%$). '
            r'$^{\mathrm{c}}$Pass thresholds are'
            r' $\mathrm{RMS}_{\mathrm{re}}, \mathrm{RMS}_{\mathrm{im}} \le'
            r' 2.00\%$ for the tared resistor and'
            r' $\mathrm{RMS}_{\mathrm{re}} \le 2.60\%,'
            r' \mathrm{RMS}_{\mathrm{im}} \le 4.75\%$ for untared spectra.'
            r'\end{flushleft}'
        ),
        r'\end{table*}'
    ])

    pathlib.Path(tex_filepath).write_text(
        '\n'.join(latex_lines), encoding='utf-8'
    )


def export_table_s7_tex_and_csv(
        dataframe_s7: pd.DataFrame,
        csv_filepath: Union[str, pathlib.Path] = 'eis-circuit-parameters.csv',
        tex_filepath: Union[str, pathlib.Path] = 'eis-circuit-parameters.tex'
) -> None:
    pathlib.Path(csv_filepath).parent.mkdir(parents=True, exist_ok=True)
    dataframe_s7.to_csv(csv_filepath, index=False)
    total_sweeps = len(dataframe_s7)

    latex_lines = [
        r'\begin{table*}[htbp]',
        r'\centering',
        (
            r'\caption{\textbf{Fitted parameters and relative standard errors'
            r' ($\mathrm{RSE}\%$) for the unified equivalent-circuit model'
            rf' across all multi-stage EIS recordings ($N = {total_sweeps}$'
            r' sweeps, $0.1$--$1000\mathrm{~Hz}$).} All colonized'
            r' \textit{P.~polycephalum} sweeps are fitted to the single'
            r' unified circuit $\mathrm{p}(C_{0}, R_{0}\text{--}\mathrm{p}'
            r'(R_{1},\mathrm{CPE}_{1},[R_{2}\text{--}L_{1}])\text{--}'
            r'\mathrm{CPE}_{2})$ (Figure~\ref{fig:equivalent-circuit}), where'
            r' the state-feedback branch is inactive in passive sweeps'
            r' ($\theta \le 0^{\circ}$) and active in fourth-quadrant'
            r' inductive sweeps ($\theta > 0^{\circ}$).'
            r' Abiotic resistor and agar controls use'
            r' $\mathrm{p}(R_{0},C_{0})$ and'
            r' $\mathrm{p}(C_{0},R_{0}\text{--}\mathrm{CPE}_{2})$,'
            r' respectively. Relative standard errors exceeding $99.9\%$ (from'
            r' boundary-constrained or weakly identifiable parameters) are'
            r' denoted as $>99.9\%$. $\mathrm{NRMSE}_{\mathrm{pass}}$ gives'
            r' the reference fit error without the $R_{2}\text{--}L_{1}$'
            r' branch.}'
        ),
        r'\label{tab:s7-eis-params}',
        r'\renewcommand{\arraystretch}{0.88}',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{llcccccccccccc}',
        r'\toprule',
        (
            r'\textbf{Group} & \textbf{Run} & $C_{0}$ & $R_{0}$ & $R_{1}$ &'
            r' $Q_{1}$ & $\alpha_{1}$ & $R_{2}$ & $L_{1}$ & $Q_{2}$ &'
            r' $\alpha_{2}$ & $\mathrm{NRMSE}_{\mathrm{pass}}$ &'
            r' $\mathrm{NRMSE}_{\mathrm{fit}}$ & $R^{2}_{\mathrm{fit}}$ \\'
        ),
        (
            r' & & $(\mathrm{pF})$ & $(\mathrm{M\Omega})$ &'
            r' $(\mathrm{M\Omega})$ & $(\mathrm{nF\,s^{\alpha-1}})$ & &'
            r' $(\mathrm{M\Omega})$ & $(\mathrm{MH})$ &'
            r' $(\mathrm{nF\,s^{\alpha-1}})$ & & $(\%)$ & $(\%)$ & \\'
        ),
        r'\midrule'
    ]

    for experiment_key, _, experiment_label_s7 in COHORT_METADATA:
        sub_dataframe = dataframe_s7[
            dataframe_s7['experiment'] == experiment_key
        ].copy()

        if sub_dataframe.empty:
            continue

        for row_index, (_, row) in enumerate(sub_dataframe.iterrows()):
            group_cell = experiment_label_s7 if row_index == 0 else ''
            run_number = int(row['run'])

            c0_string = format_parameter_cell(
                row['c0'], row['c0_rse_percent'], scale=1e12, decimals=2
            )
            r0_string = format_parameter_cell(
                row['r0'], row['r0_rse_percent'], scale=1e-6, decimals=2
            )
            r1_string = format_parameter_cell(
                row['r1'], row['r1_rse_percent'], scale=1e-6, decimals=2
            )
            q1_string = format_parameter_cell(
                row['q1'], row['q1_rse_percent'], scale=1e9, decimals=2
            )
            a1_string = format_parameter_cell(
                row['a1'], row['a1_rse_percent'], scale=1.0, decimals=2
            )
            r2_string = format_parameter_cell(
                row['r2'], row['r2_rse_percent'], scale=1e-6, decimals=2
            )
            l1_string = format_parameter_cell(
                row['l1'], row['l1_rse_percent'], scale=1e-6, decimals=2
            )
            q2_string = format_parameter_cell(
                row['q2'], row['q2_rse_percent'], scale=1e9, decimals=2
            )
            a2_string = format_parameter_cell(
                row['a2'], row['a2_rse_percent'], scale=1.0, decimals=2
            )

            passive_nrmse_value = row['passive_normalized_rmse_percent']
            passive_nrmse_string = (
                f'{passive_nrmse_value:.2f}'
                if not np.isnan(passive_nrmse_value)
                else '---'
            )

            fit_nrmse_value = row['fit_normalized_rmse_percent']
            fit_nrmse_string = f'{fit_nrmse_value:.2f}'

            fit_r_squared_value = row['fit_r_squared']
            fit_r_squared_string = f'{fit_r_squared_value:.3f}'

            latex_lines.append(
                f'{group_cell} & {run_number} & {c0_string} & {r0_string} &'
                f' {r1_string} & {q1_string} & {a1_string} & {r2_string} &'
                f' {l1_string} & {q2_string} & {a2_string} &'
                f' {passive_nrmse_string} & {fit_nrmse_string} &'
                f' {fit_r_squared_string} \\\\'
            )

        latex_lines.append(r'\midrule')

    latex_lines[-1] = r'\bottomrule'

    latex_lines.extend([
        r'\end{tabular}%',
        r'}',
        r'\end{table*}'
    ])

    pathlib.Path(tex_filepath).write_text(
        '\n'.join(latex_lines), encoding='utf-8'
    )


if __name__ == '__main__':
    if not hasattr(builtins, 'np'):
        builtins.np = np

    abiotic_blocking_cpe_model = {
        'circuit': 'p(C0,R0-CPE0)',
        'initial_guess': [7.5e-11, 1.45e6, 2.0e-6, 0.55],
        'bounds': ([1e-13, 1e4, 1e-10, 0.10], [1e-9, 5e8, 1e-2, 1.00])
    }

    passive_biological_cpe_model = {
        'circuit': 'p(C0,R0-p(R1,CPE1)-CPE2)',
        'initial_guess': [1.5e-11, 5.0e5, 1.5e6, 1.0e-9, 0.75, 5.0e-7, 0.45],
        'bounds': (
            [1e-12, 1e5, 1e5, 5e-12, 0.40, 1e-9, 0.15],
            [2e-10, 1e8, 1.5e8, 1e-4, 1.00, 1e-3, 1.00]
        )
    }

    active_biological_rl_cpe_model = {
        'circuit': 'p(C0,R0-p(R1,CPE1,R2-L1)-CPE2)',
        'initial_guess': [
            1.5e-11, 1.0e6, 2.0e6, 1.0e-9, 0.75, 5.0e6, 5.0e6, 5.0e-7, 0.45
        ],
        'bounds': (
            [1e-12, 1e5, 1e5, 5e-12, 0.40, 5e5, 5e5, 1e-9, 0.15],
            [2e-10, 1e8, 1.5e8, 1e-4, 1.00, 1.5e8, 5e8, 1e-3, 1.00]
        )
    }

    circuit_configurations = {
        'resistor': {
            'circuit': 'p(R0,C0)',
            'initial_guess': [10.87e6, 8.68e-12],
            'bounds': ([1e5, 1e-13], [1e8, 1e-9]),
            'trim_inductive': False,
            'minimum_frequency': 0.0
        },
        'open-air-agar': {
            **abiotic_blocking_cpe_model,
            'trim_inductive': False,
            'minimum_frequency': 0.0
        },
        'slime-mould-device-1-before-bias': {
            **passive_biological_cpe_model,
            'trim_inductive': False,
            'minimum_frequency': 0.0
        },
        'slime-mould-device-1-after-bias': {
            **active_biological_rl_cpe_model,
            'trim_inductive': False,
            'minimum_frequency': 0.0
        },
        'slime-mould-device-2-before-bias': {
            **passive_biological_cpe_model,
            'trim_inductive': False,
            'minimum_frequency': 0.0
        },
        'slime-mould-device-2-after-bias': {
            **active_biological_rl_cpe_model,
            'trim_inductive': False,
            'minimum_frequency': 0.0
        },
        'slime-mould-device-2-after-second-bias': {
            **active_biological_rl_cpe_model,
            'trim_inductive': False,
            'minimum_frequency': 0.0
        },
        'slime-mould-device-3-before-bias': {
            **passive_biological_cpe_model,
            'trim_inductive': False,
            'minimum_frequency': 0.0
        },
        'slime-mould-device-3-after-bias': {
            **active_biological_rl_cpe_model,
            'trim_inductive': False,
            'minimum_frequency': 0.0
        },
        'slime-mould-device-3-after-second-bias': {
            **active_biological_rl_cpe_model,
            'trim_inductive': False,
            'minimum_frequency': 0.0
        },
        'slime-mould-device-4-before-bias': {
            **passive_biological_cpe_model,
            'trim_inductive': False,
            'minimum_frequency': 0.0
        },
        'slime-mould-device-4-after-bias': {
            **active_biological_rl_cpe_model,
            'trim_inductive': False,
            'minimum_frequency': 0.0
        },
        'slime-mould-device-4-after-second-bias': {
            **active_biological_rl_cpe_model,
            'trim_inductive': False,
            'minimum_frequency': 0.0
        }
    }

    source_directory = (
        DEFAULT_EIS_DIRECTORY
        if DEFAULT_EIS_DIRECTORY.exists()
        else FALLBACK_EIS_DIRECTORY
    )

    dataframe_s6, dataframe_s7 = evaluate_all_recordings(
        source_directory,
        circuit_configurations=circuit_configurations,
        passive_reference_model=passive_biological_cpe_model
    )

    results_directory = pathlib.Path('results')
    results_directory.mkdir(parents=True, exist_ok=True)

    export_table_s6_tex_and_csv(
        dataframe_s6,
        csv_filepath=results_directory / 'eis-linkk-validation.csv',
        tex_filepath=results_directory / 'eis-linkk-validation.tex'
    )
    export_table_s7_tex_and_csv(
        dataframe_s7,
        csv_filepath=results_directory / 'eis-circuit-parameters.csv',
        tex_filepath=results_directory / 'eis-circuit-parameters.tex'
    )

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1200)
    pd.set_option(
        'display.float_format',
        lambda value: (
            f'{value:.3f}'
            if abs(value) < 1000 and abs(value) > 0.001
            else f'{value:.2e}'
        )
    )

    print(dataframe_s6.to_string(index=False))

    print(dataframe_s7.to_string(index=False))
