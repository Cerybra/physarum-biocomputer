import pathlib
import re

import numpy as np
import pandas as pd

from scipy import stats

from typing import Dict, List, Tuple, Union


# Excluded due to persistent TIA positive-rail saturation (+2.65 to +2.67 V
# across 89-100% of each epoch) and an exact 10x inter-epoch logging scale
# mismatch starting at row 100.
EXCLUDED_RECORDINGS = frozenset({'regression-with-grad-target-0.19.csv'})

DEFAULT_REGRESSION_DIRECTORY = pathlib.Path(
    'data/device-regression-recordings'
)
FALLBACK_REGRESSION_DIRECTORY = pathlib.Path('device-regression-recordings')

REGRESSION_SUMMARY_CAPTION = (
    r'\textbf{Summary statistics for linear regression and target-matching'
    r' experiments across \textit{P.~polycephalum} devices.} For each'
    r' recording, the active channel is reported alongside the nominal target'
    r' $w_{0}$, terminal weight $w_{\mathrm{tail}}$, terminal prediction'
    r' $\hat{y}_{\mathrm{tail}}$, initial and terminal mean absolute errors'
    r' ($\mathrm{MAE}_{\mathrm{init}}$, $\mathrm{MAE}_{\mathrm{tail}}$),'
    r' terminal root-mean-square error ($\mathrm{RMSE}_{\mathrm{tail}}$), and'
    r' percentage error reduction over the training trajectory.'
)

REGRESSION_CHANNELS_CAPTION = (
    r'\textbf{Complete statistical results for all regression and'
    r' target-matching experiments.}'
)

REGRESSION_HYPOTHESIS_TESTS_CAPTION = (
    r'\textbf{Statistical hypothesis testing of gradient training'
    r' across regression, conductance programming, and target-matching'
    r' experiments.} Per-experiment rows compare terminal versus initial or'
    r' control step windows within a single recording trajectory (note that'
    r' sequential within-run steps are autocorrelated), whereas the pooled row'
    r' tests across all unsaturated gradient-trained channels without outcome'
    r' filtering. Non-parametric Mann--Whitney $U$ ($\mathrm{MWU}$) and'
    r' Wilcoxon signed-rank ($\mathrm{WSR}$) tests are reported alongside'
    ' parametric Welch\'s '
    r'$t$-tests ($\mathrm{WT}$) and paired $t$-tests'
    r' ($\mathrm{PT}$); paired tests ($\mathrm{WSR}$, $\mathrm{PT}$) are'
    r' restricted to within-run/within-channel terminal-versus-initial'
    r' contrasts, while independent cross-run controls use two-sample tests'
    r' ($\mathrm{MWU}$, $\mathrm{WT}$) only. For Exp.~2, untrained baseline'
    r' weights (collected without gradient updates) are evaluated against each'
    r' trained target ($w_{0} = 0.43, 0.72$) to test target proximity.'
    r' Effect sizes report rank-biserial correlation ($r_{\mathrm{rb}}$) for'
    r' $\mathrm{MWU}$ and $\mathrm{WSR}$, and '
    'Cohen\'s '
    r'$d$ for $\mathrm{WT}$'
    r' and $\mathrm{PT}$ ($^{*}p < 0.05$, $^{**}p < 0.01$, $^{***}p < 0.001$,'
    r' $^{****}p < 0.0001$).'
)


def walk_regression_directory(
        source_filepath: Union[str, pathlib.Path]
) -> Dict[str, List[str]]:
    source_path = pathlib.Path(source_filepath).resolve()

    directory_structure: Dict[str, List[str]] = {}
    for csv_file in sorted(source_path.rglob('*.csv')):
        relative_directory = csv_file.parent.relative_to(source_path)
        if relative_directory.parts:
            key = '-'.join(relative_directory.parts)
        else:
            key = source_path.name

        directory_structure.setdefault(key, []).append(str(csv_file.resolve()))

    return directory_structure


def parse_recording_metadata(
        experiment_key: str,
        filepath: Union[str, pathlib.Path],
        columns: List[str]
) -> Dict[str, Union[str, float, bool]]:
    filename = pathlib.Path(filepath).name
    stem = pathlib.Path(filepath).stem.lower()

    has_weight_columns = any(
        column.startswith('Weight') for column in columns
    )
    has_prediction_columns = any(
        column.startswith('Prediction') for column in columns
    )

    if 'matching' in stem:
        task_type = 'Conductance Matching'
        task_objective = 'Target Matching'
    elif 'wandering' in stem or 'untrained' in stem:
        task_type = 'Untrained Baseline'
        task_objective = 'Target Matching'
    elif has_weight_columns and has_prediction_columns:
        task_type = 'Linear Regression'
        task_objective = 'Weight & Target'
    elif has_weight_columns and not has_prediction_columns:
        task_type = 'Conductance Programming'
        task_objective = 'Weight Matching'
    else:
        task_type = 'Trajectory Tracking'
        task_objective = 'Target Tracking'

    if 'without-grad' in stem or 'untrained' in stem or 'wandering' in stem:
        training_mode = 'Without Gradient'
        has_gradient = False
    else:
        training_mode = 'With Gradient'
        has_gradient = True

    target_match = re.search(r'target-([0-9]+(?:\.[0-9]+)?)', stem)
    nominal_target = (
        float(target_match.group(1)) if target_match else np.nan
    )

    return {
        'experiment': experiment_key,
        'filename': filename,
        'task_type': task_type,
        'task_objective': task_objective,
        'training_mode': training_mode,
        'has_gradient': has_gradient,
        'nominal_target': nominal_target
    }


def build_ramp_inputs(
        number_of_steps: int,
        epoch_length: int = 100
) -> np.ndarray:
    if number_of_steps <= epoch_length:
        return np.arange(number_of_steps, dtype=float) / epoch_length

    ramp_values = np.zeros(number_of_steps, dtype=float)
    for index in range(number_of_steps):
        step_in_epoch = index % epoch_length
        ramp_values[index] = step_in_epoch / epoch_length

    return ramp_values


def prepare_regression_data(
        filepath: Union[str, pathlib.Path],
        experiment_key: str = ''
) -> Dict[str, Union[str, float, bool, int, np.ndarray]]:
    dataframe = pd.read_csv(filepath, index_col=False)
    columns = list(dataframe.columns)
    number_of_steps = len(dataframe)

    metadata = parse_recording_metadata(experiment_key, filepath, columns)
    ramp_inputs = build_ramp_inputs(number_of_steps, epoch_length=100)

    if 'Target' in columns:
        targets = dataframe['Target'].to_numpy(dtype=float)
    elif 'Targets' in columns:
        targets = dataframe['Targets'].to_numpy(dtype=float)
    elif not np.isnan(metadata['nominal_target']):
        if str(metadata['task_objective']) == 'Target Matching':
            targets = np.full(
                number_of_steps, metadata['nominal_target'], dtype=float
            )
        else:
            targets = metadata['nominal_target'] * ramp_inputs
    else:
        targets = np.full(number_of_steps, np.nan, dtype=float)

    nominal_target = metadata['nominal_target']
    if np.isnan(nominal_target) and np.any(np.isfinite(targets)):
        if str(metadata['task_objective']) == 'Target Matching':
            nominal_target = np.nanmedian(targets)
        else:
            valid_mask = np.isfinite(targets) & (ramp_inputs > 0.0)
            if np.sum(valid_mask) >= 5:
                slope, _ = np.polyfit(
                    ramp_inputs[valid_mask], targets[valid_mask], 1
                )
                nominal_target = round(slope, 2)

    prediction_columns = sorted(
        [column for column in columns if column.startswith('Prediction')]
    )
    weight_columns = sorted(
        [column for column in columns if column.startswith('Weight')]
    )
    update_columns = sorted(
        [column for column in columns if column.startswith('Update')]
    )

    number_of_channels = max(
        len(prediction_columns), len(weight_columns), 1
    )
    predictions_matrix = np.full(
        (number_of_steps, number_of_channels), np.nan, dtype=float
    )
    weights_matrix = np.full(
        (number_of_steps, number_of_channels), np.nan, dtype=float
    )
    updates_matrix = np.full(
        (number_of_steps, number_of_channels), np.nan, dtype=float
    )

    for channel_index in range(number_of_channels):
        if channel_index < len(prediction_columns):
            predictions_matrix[:, channel_index] = dataframe[
                prediction_columns[channel_index]
            ].to_numpy(dtype=float)

        if channel_index < len(weight_columns):
            weights_matrix[:, channel_index] = dataframe[
                weight_columns[channel_index]
            ].to_numpy(dtype=float)

        if channel_index < len(update_columns):
            updates_matrix[:, channel_index] = dataframe[
                update_columns[channel_index]
            ].to_numpy(dtype=float)

    return {
        'experiment': str(metadata['experiment']),
        'filename': str(metadata['filename']),
        'task_type': str(metadata['task_type']),
        'task_objective': str(metadata['task_objective']),
        'training_mode': str(metadata['training_mode']),
        'has_gradient': bool(metadata['has_gradient']),
        'nominal_target': nominal_target,
        'number_of_steps': number_of_steps,
        'number_of_channels': number_of_channels,
        'ramp_inputs': ramp_inputs,
        'targets': targets,
        'predictions': predictions_matrix,
        'weights': weights_matrix,
        'updates': updates_matrix
    }


def extract_clean_channel_series(
        prepared_data: Dict,
        channel_index: int
) -> Dict[str, np.ndarray]:
    number_of_steps = int(prepared_data['number_of_steps'])
    targets = np.asarray(prepared_data['targets'], dtype=float)
    predictions = np.asarray(
        prepared_data['predictions'][:, channel_index], dtype=float
    )
    weights = np.asarray(
        prepared_data['weights'][:, channel_index], dtype=float
    )

    valid_step_mask = np.ones(number_of_steps, dtype=bool)
    if number_of_steps >= 10:
        valid_step_mask[0] = False

    prediction_mask = (
        valid_step_mask & np.isfinite(predictions) & np.isfinite(targets)
    )
    weight_mask = (
        valid_step_mask
        & np.isfinite(weights)
        & (np.abs(weights) > 1e-9)
        & (np.abs(weights) < 500.0)
    )

    clean_predictions = predictions[prediction_mask]
    clean_prediction_targets = targets[prediction_mask]
    clean_weights = weights[weight_mask]

    return {
        'predictions': clean_predictions,
        'prediction_targets': clean_prediction_targets,
        'weights': clean_weights
    }


def compute_channel_regression_metrics(
        prepared_data: Dict,
        channel_index: int,
        window_size: int = 10,
        saturation_threshold: float = 2.64
) -> Dict[str, Union[str, int, float, bool]]:
    task_type = str(prepared_data['task_type'])
    task_objective = str(prepared_data['task_objective'])
    nominal_target = prepared_data['nominal_target']
    number_of_steps = int(prepared_data['number_of_steps'])

    series = extract_clean_channel_series(prepared_data, channel_index)
    clean_predictions = series['predictions']
    clean_prediction_targets = series['prediction_targets']
    clean_weights = series['weights']

    has_predictions = len(clean_predictions) >= window_size
    has_weights = len(clean_weights) >= window_size

    if has_predictions:
        head_predictions = clean_predictions[:window_size]
        head_targets = clean_prediction_targets[:window_size]
        tail_predictions = clean_predictions[-window_size:]
        tail_targets = clean_prediction_targets[-window_size:]

        initial_prediction_mean = np.mean(head_predictions)
        initial_prediction_std = np.std(head_predictions, ddof=1)
        tail_prediction_mean = np.mean(tail_predictions)
        tail_prediction_std = np.std(tail_predictions, ddof=1)

        prediction_mae_initial = np.mean(
            np.abs(head_predictions - head_targets)
        )
        prediction_mae_tail = np.mean(
            np.abs(tail_predictions - tail_targets)
        )
        prediction_rmse_tail = np.sqrt(
            np.mean((tail_predictions - tail_targets) ** 2)
        )

        rail_saturated = bool(
            np.mean(np.abs(tail_predictions) >= saturation_threshold) >= 0.50
        )
    else:
        initial_prediction_mean = np.nan
        initial_prediction_std = np.nan
        tail_prediction_mean = np.nan
        tail_prediction_std = np.nan
        prediction_mae_initial = np.nan
        prediction_mae_tail = np.nan
        prediction_rmse_tail = np.nan
        rail_saturated = False

    if has_weights and not np.isnan(nominal_target):
        head_weights = clean_weights[:window_size]
        tail_weights = clean_weights[-window_size:]

        initial_weight_mean = np.mean(head_weights)
        initial_weight_std = np.std(head_weights, ddof=1)
        tail_weight_mean = np.mean(tail_weights)
        tail_weight_std = np.std(tail_weights, ddof=1)

        weight_mae_initial = np.mean(np.abs(head_weights - nominal_target))
        weight_mae_tail = np.mean(np.abs(tail_weights - nominal_target))
        weight_rmse_tail = np.sqrt(
            np.mean((tail_weights - nominal_target) ** 2)
        )
    else:
        initial_weight_mean = np.nan
        initial_weight_std = np.nan
        tail_weight_mean = np.nan
        tail_weight_std = np.nan
        weight_mae_initial = np.nan
        weight_mae_tail = np.nan
        weight_rmse_tail = np.nan

    if has_weights and task_objective in (
        'Weight & Target',
        'Weight Matching'
    ):
        primary_mae_initial = weight_mae_initial
        primary_mae_tail = weight_mae_tail
        primary_rmse_tail = weight_rmse_tail
    else:
        primary_mae_initial = prediction_mae_initial
        primary_mae_tail = prediction_mae_tail
        primary_rmse_tail = prediction_rmse_tail

    error_reduction_percent = (
        ((primary_mae_initial - primary_mae_tail) / primary_mae_initial)
        * 100.0
        if (
            not np.isnan(primary_mae_initial)
            and primary_mae_initial > 1e-9
        )
        else np.nan
    )

    return {
        'experiment': str(prepared_data['experiment']),
        'file': str(prepared_data['filename']),
        'task_type': task_type,
        'task_objective': task_objective,
        'training_mode': str(prepared_data['training_mode']),
        'channel': channel_index + 1,
        'steps': number_of_steps,
        'nominal_target': nominal_target,
        'initial_prediction_mean': initial_prediction_mean,
        'initial_prediction_std': initial_prediction_std,
        'tail_prediction_mean': tail_prediction_mean,
        'tail_prediction_std': tail_prediction_std,
        'initial_weight_mean': initial_weight_mean,
        'initial_weight_std': initial_weight_std,
        'tail_weight_mean': tail_weight_mean,
        'tail_weight_std': tail_weight_std,
        'weight_mae_initial': weight_mae_initial,
        'weight_mae_tail': weight_mae_tail,
        'prediction_mae_initial': prediction_mae_initial,
        'prediction_mae_tail': prediction_mae_tail,
        'primary_mae_initial': primary_mae_initial,
        'primary_mae_tail': primary_mae_tail,
        'primary_rmse_tail': primary_rmse_tail,
        'error_reduction_percent': error_reduction_percent,
        'rail_saturated': rail_saturated
    }


def append_hypothesis_test_rows(
        test_records: List[Dict],
        task_label: str,
        comparison_label: str,
        sample_first: np.ndarray,
        sample_second: np.ndarray,
        include_paired: bool = False
) -> None:
    first_clean = np.asarray(sample_first, dtype=float)
    second_clean = np.asarray(sample_second, dtype=float)
    first_clean = first_clean[np.isfinite(first_clean)]
    second_clean = second_clean[np.isfinite(second_clean)]

    count_first = len(first_clean)
    count_second = len(second_clean)
    if count_first < 3 or count_second < 3:
        return

    u_statistic, p_mwu_two_sided = stats.mannwhitneyu(
        first_clean, second_clean, alternative='two-sided'
    )
    _, p_mwu_one_sided = stats.mannwhitneyu(
        first_clean, second_clean, alternative='less'
    )
    rank_biserial = 1.0 - (2.0 * u_statistic) / (count_first * count_second)

    test_records.append({
        'task': task_label,
        'comparison': comparison_label,
        'test': 'MWU',
        'sample': f'n = ({count_first}, {count_second})',
        'statistic': u_statistic,
        'p_two_sided': p_mwu_two_sided,
        'p_one_sided': p_mwu_one_sided,
        'effect_size_type': 'r_rb',
        'effect_size': rank_biserial
    })

    t_welch, p_welch_two_sided = stats.ttest_ind(
        first_clean, second_clean, equal_var=False
    )
    p_welch_one_sided = (
        p_welch_two_sided / 2.0
        if t_welch < 0.0
        else 1.0 - (p_welch_two_sided / 2.0)
    )
    pooled_variance = (
        (count_first - 1) * np.var(first_clean, ddof=1)
        + (count_second - 1) * np.var(second_clean, ddof=1)
    ) / max(count_first + count_second - 2, 1)
    pooled_sd = max(np.sqrt(max(pooled_variance, 0.0)), 1e-12)
    cohen_d_independent = (
        (np.mean(first_clean) - np.mean(second_clean)) / pooled_sd
    )

    test_records.append({
        'task': task_label,
        'comparison': comparison_label,
        'test': 'WT',
        'sample': f'n = ({count_first}, {count_second})',
        'statistic': t_welch,
        'p_two_sided': p_welch_two_sided,
        'p_one_sided': p_welch_one_sided,
        'effect_size_type': 'd',
        'effect_size': cohen_d_independent
    })

    if include_paired and count_first == count_second and count_first >= 5:
        differences = first_clean - second_clean
        if np.any(np.abs(differences) > 1e-12):
            w_statistic, p_wsr_two_sided = stats.wilcoxon(
                first_clean, second_clean, alternative='two-sided'
            )
            w_plus, p_wsr_one_sided = stats.wilcoxon(
                first_clean, second_clean, alternative='less'
            )
            number_of_non_zero = int(np.sum(np.abs(differences) > 1e-12))
            rank_sum_total = (
                number_of_non_zero * (number_of_non_zero + 1) / 2.0
            )
            rank_biserial_wsr = (
                1.0 - (2.0 * w_plus) / rank_sum_total
                if rank_sum_total > 0.0
                else 0.0
            )
            t_paired, p_paired_two_sided = stats.ttest_rel(
                first_clean, second_clean
            )
            p_paired_one_sided = (
                p_paired_two_sided / 2.0
                if t_paired < 0.0
                else 1.0 - (p_paired_two_sided / 2.0)
            )
            difference_sd = max(np.std(differences, ddof=1), 1e-12)
            cohen_d_paired = np.mean(differences) / difference_sd

            test_records.append({
                'task': task_label,
                'comparison': comparison_label,
                'test': 'WSR',
                'sample': f'N = {count_first}',
                'statistic': w_statistic,
                'p_two_sided': p_wsr_two_sided,
                'p_one_sided': p_wsr_one_sided,
                'effect_size_type': 'r_rb',
                'effect_size': rank_biserial_wsr
            })
            test_records.append({
                'task': task_label,
                'comparison': comparison_label,
                'test': 'PT',
                'sample': f'N = {count_first}',
                'statistic': t_paired,
                'p_two_sided': p_paired_two_sided,
                'p_one_sided': p_paired_one_sided,
                'effect_size_type': 'd',
                'effect_size': cohen_d_paired
            })


def compute_regression_hypothesis_tests(
        prepared_by_name: Dict[str, Dict],
        dataframe_channels: pd.DataFrame,
        window_size: int = 10
) -> pd.DataFrame:
    test_records: List[Dict] = []

    if (
        'regression-with-grad-target-0.44.csv' in prepared_by_name
        and 'regression-without-grad-target-0.44.csv' in prepared_by_name
    ):
        prepared_with = prepared_by_name[
            'regression-with-grad-target-0.44.csv'
        ]
        prepared_without = prepared_by_name[
            'regression-without-grad-target-0.44.csv'
        ]
        series_with = extract_clean_channel_series(prepared_with, 0)
        series_without = extract_clean_channel_series(prepared_without, 0)

        weight_error_with = np.abs(series_with['weights'] - 0.44)
        weight_error_without = np.abs(series_without['weights'] - 0.44)
        append_hypothesis_test_rows(
            test_records,
            'Exp. 1 (0.44 Weight)',
            'With vs. No Grad (Tail)',
            weight_error_with[-window_size:],
            weight_error_without[-window_size:],
            include_paired=False
        )
        append_hypothesis_test_rows(
            test_records,
            'Exp. 1 (0.44 Weight)',
            'With vs. No Grad (Full)',
            weight_error_with,
            weight_error_without,
            include_paired=False
        )

        prediction_error_with = np.abs(
            series_with['predictions'][-window_size:]
            - series_with['prediction_targets'][-window_size:]
        )
        prediction_error_without = np.abs(
            series_without['predictions'][-window_size:]
            - series_without['prediction_targets'][-window_size:]
        )
        append_hypothesis_test_rows(
            test_records,
            'Exp. 1 (0.44 Pred.)',
            'With vs. No Grad (Tail)',
            prediction_error_with,
            prediction_error_without,
            include_paired=False
        )

    if 'regression-with-grad-target-0.43.csv' in prepared_by_name:
        prepared_43 = prepared_by_name['regression-with-grad-target-0.43.csv']
        series_43 = extract_clean_channel_series(prepared_43, 0)
        weight_error_43 = np.abs(series_43['weights'] - 0.43)
        append_hypothesis_test_rows(
            test_records,
            'Exp. 2 (0.43 Ch. 1)',
            'Tail vs. Initial Window',
            weight_error_43[-window_size:],
            weight_error_43[:window_size],
            include_paired=True
        )
        if 'regression-without-grad.csv' in prepared_by_name:
            prepared_untrained = prepared_by_name[
                'regression-without-grad.csv'
            ]
            series_untrained = extract_clean_channel_series(
                prepared_untrained, 0
            )
            weight_error_untrained = np.abs(
                series_untrained['weights'][-window_size:] - 0.43
            )
            append_hypothesis_test_rows(
                test_records,
                'Exp. 2 (0.43 Ch. 1)',
                'With vs. No Grad (Tail)',
                weight_error_43[-window_size:],
                weight_error_untrained,
                include_paired=False
            )

    if 'regression-with-grad-target-0.72.csv' in prepared_by_name:
        prepared_72 = prepared_by_name['regression-with-grad-target-0.72.csv']
        series_72 = extract_clean_channel_series(prepared_72, 1)
        weight_error_72 = np.abs(series_72['weights'] - 0.72)
        append_hypothesis_test_rows(
            test_records,
            'Exp. 2 (0.72 Ch. 2)',
            'Tail vs. Initial Window',
            weight_error_72[-window_size:],
            weight_error_72[:window_size],
            include_paired=True
        )
        if 'regression-without-grad.csv' in prepared_by_name:
            prepared_untrained = prepared_by_name[
                'regression-without-grad.csv'
            ]
            series_untrained = extract_clean_channel_series(
                prepared_untrained, 1
            )
            weight_error_untrained = np.abs(
                series_untrained['weights'][-window_size:] - 0.72
            )
            append_hypothesis_test_rows(
                test_records,
                'Exp. 2 (0.72 Ch. 2)',
                'With vs. No Grad (Tail)',
                weight_error_72[-window_size:],
                weight_error_untrained,
                include_paired=False
            )

    if 'matching-with-grad-target-0.88.csv' in prepared_by_name:
        prepared_match_with = prepared_by_name[
            'matching-with-grad-target-0.88.csv'
        ]
        series_match_with = extract_clean_channel_series(
            prepared_match_with, 0
        )
        match_error_with = np.abs(series_match_with['predictions'] - 0.88)

        append_hypothesis_test_rows(
            test_records,
            'Exp. 3 (0.88 Ch. 1)',
            'Tail vs. Initial Window',
            match_error_with[-window_size:],
            match_error_with[:window_size],
            include_paired=True
        )

        if 'matching-without-grad-target-0.88.csv' in prepared_by_name:
            prepared_match_without = prepared_by_name[
                'matching-without-grad-target-0.88.csv'
            ]
            series_match_without = extract_clean_channel_series(
                prepared_match_without, 0
            )
            match_error_without = np.abs(
                series_match_without['predictions'] - 0.88
            )
            append_hypothesis_test_rows(
                test_records,
                'Exp. 3 (0.88 Ch. 1)',
                'With vs. No Grad (Tail)',
                match_error_with[-window_size:],
                match_error_without[-window_size:],
                include_paired=False
            )

        if 'untrained-wandering.csv' in prepared_by_name:
            prepared_wandering = prepared_by_name['untrained-wandering.csv']
            series_wandering = extract_clean_channel_series(
                prepared_wandering, 0
            )
            match_error_wandering = np.abs(
                series_wandering['predictions'] - 0.88
            )
            append_hypothesis_test_rows(
                test_records,
                'Exp. 3 (0.88 Ch. 1)',
                'With Grad vs. Wandering',
                match_error_with[-window_size:],
                match_error_wandering[-window_size:],
                include_paired=False
            )

    trained_unsaturated = dataframe_channels[
        (dataframe_channels['training_mode'] == 'With Gradient')
        & (~dataframe_channels['rail_saturated'])
        & np.isfinite(dataframe_channels['primary_mae_tail'])
        & np.isfinite(dataframe_channels['primary_mae_initial'])
    ]
    if len(trained_unsaturated) >= 5:
        append_hypothesis_test_rows(
            test_records,
            'Pooled Unsaturated Trained',
            'Tail vs. Initial MAE',
            trained_unsaturated['primary_mae_tail'].to_numpy(dtype=float),
            trained_unsaturated['primary_mae_initial'].to_numpy(dtype=float),
            include_paired=True
        )

    return pd.DataFrame(test_records)


def evaluate_all_regression_recordings(
        data_directory: Union[str, pathlib.Path],
        window_size: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_path = pathlib.Path(data_directory)
    if source_path.exists():
        directory_data = walk_regression_directory(source_path)
    else:
        fallback_files = sorted(
            [
                str(path.resolve())
                for path in pathlib.Path('.').glob('*.csv')
                if (
                    path.name.startswith('regression-')
                    or path.name.startswith('matching-')
                    or path.name.startswith('untrained-')
                )
            ]
        )
        directory_data = {'device-regression-recordings': fallback_files}

    channel_records: List[Dict] = []
    summary_records: List[Dict] = []
    prepared_by_name: Dict[str, Dict] = {}

    sequential_experiment_index = 0
    for _, filepaths in directory_data.items():
        valid_filepaths = [
            filepath
            for filepath in filepaths
            if (
                pathlib.Path(filepath).name.startswith('regression-')
                or pathlib.Path(filepath).name.startswith('matching-')
                or pathlib.Path(filepath).name.startswith('untrained-')
            )
            and pathlib.Path(filepath).name not in EXCLUDED_RECORDINGS
        ]
        if not valid_filepaths:
            continue

        sequential_experiment_index += 1
        display_experiment_key = f'experiment-{sequential_experiment_index}'

        for filepath in valid_filepaths:
            prepared = prepare_regression_data(
                filepath, experiment_key=display_experiment_key
            )
            prepared_by_name[str(prepared['filename'])] = prepared
            number_of_channels = int(prepared['number_of_channels'])

            per_file_channels: List[Dict] = []
            for channel_index in range(number_of_channels):
                metrics = compute_channel_regression_metrics(
                    prepared,
                    channel_index=channel_index,
                    window_size=window_size
                )
                channel_records.append(metrics)
                per_file_channels.append(metrics)

            unsaturated_candidates = [
                row
                for row in per_file_channels
                if not bool(row['rail_saturated'])
                and not np.isnan(row['primary_rmse_tail'])
            ]
            candidate_pool = (
                unsaturated_candidates
                if len(unsaturated_candidates) > 0
                else per_file_channels
            )
            best_channel_row = min(
                candidate_pool,
                key=lambda item: (
                    item['primary_rmse_tail']
                    if not np.isnan(item['primary_rmse_tail'])
                    else np.inf
                )
            )

            summary_records.append(best_channel_row.copy())

    dataframe_channels = pd.DataFrame(channel_records)
    dataframe_summary = pd.DataFrame(summary_records)
    dataframe_tests = compute_regression_hypothesis_tests(
        prepared_by_name, dataframe_channels, window_size=window_size
    )

    return dataframe_summary, dataframe_channels, dataframe_tests


def format_numeric_cell(
        value: float,
        decimals: int = 3
) -> str:
    if value is None or np.isnan(value):
        return '---'

    return f'{value:.{decimals}f}'


def format_mean_sd_cell(
        mean_value: float,
        std_value: float,
        decimals: int = 3
) -> str:
    if mean_value is None or np.isnan(mean_value):
        return '---'
    if std_value is None or np.isnan(std_value):
        return f'{mean_value:.{decimals}f}'

    return f'${mean_value:.{decimals}f} \\pm {std_value:.{decimals}f}$'


def format_p_value_with_stars(
        p_value: float,
        include_stars: bool = True
) -> str:
    if p_value is None or np.isnan(p_value):
        return '---'

    stars = ''
    if include_stars:
        if p_value < 0.0001:
            stars = '^{****}'
        elif p_value < 0.001:
            stars = '^{***}'
        elif p_value < 0.01:
            stars = '^{**}'
        elif p_value < 0.05:
            stars = '^{*}'

    if p_value < 1e-3:
        exponent = int(np.floor(np.log10(max(p_value, 1e-300))))
        mantissa = p_value / (10.0 ** exponent)

        if stars:
            return f'${{{mantissa:.2f} \\times 10^{{{exponent}}}}}{stars}$'

        return f'${mantissa:.2f} \\times 10^{{{exponent}}}$'

    return f'${p_value:.4f}{stars}$'


def export_regression_summary_tex_and_csv(
        dataframe_summary: pd.DataFrame,
        csv_filepath: Union[str, pathlib.Path] = (
            'results/regression-summary.csv'
        ),
        tex_filepath: Union[str, pathlib.Path] = (
            'results/regression-summary.tex'
        ),
        caption: str = REGRESSION_SUMMARY_CAPTION
) -> None:
    pathlib.Path(csv_filepath).parent.mkdir(parents=True, exist_ok=True)
    dataframe_summary.to_csv(csv_filepath, index=False)

    latex_lines = [
        r'\begin{table*}[htbp]',
        r'\centering',
        rf'\caption{{{caption}}}',
        r'\label{tab:regression-summary}',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{llllcccccccc}',
        r'\toprule',
        (
            r'\textbf{Exp.} & \textbf{Task} & \textbf{Mode} &'
            r' \textbf{Ch.} & \textbf{Steps} & $w_{0}$ &'
            r' $w_{\mathrm{tail}}$ & $\hat{y}_{\mathrm{tail}}$ &'
            r' $\mathrm{MAE}_{\mathrm{init}}$ &'
            r' $\mathrm{MAE}_{\mathrm{tail}}$ &'
            r' $\mathrm{RMSE}_{\mathrm{tail}}$ &'
            r' $\Delta\mathrm{Err}\ (\%)$ \\'
        ),
        r'\midrule'
    ]

    previous_experiment = None
    for _, row in dataframe_summary.iterrows():
        experiment_name = (
            str(row['experiment'])
            .replace('device-regression-recordings-', '')
            .replace('experiment-', 'Exp.~')
        )
        if (
            previous_experiment is not None
            and experiment_name != previous_experiment
        ):
            latex_lines.append(r'\midrule')

        experiment_cell = (
            experiment_name if experiment_name != previous_experiment else ''
        )
        previous_experiment = experiment_name

        task_cell = str(row['task_type'])
        mode_cell = str(row['training_mode'])
        channel_value = int(row['channel'])
        channel_cell = f'Ch.~{channel_value}'
        steps_cell = str(int(row['steps']))
        target_cell = format_numeric_cell(row['nominal_target'], decimals=2)

        weight_cell = format_mean_sd_cell(
            row['tail_weight_mean'], row['tail_weight_std'], decimals=3
        )
        prediction_cell = format_mean_sd_cell(
            row['tail_prediction_mean'], row['tail_prediction_std'], decimals=3
        )
        initial_mae_cell = format_numeric_cell(
            row['primary_mae_initial'], decimals=3
        )
        tail_mae_cell = format_numeric_cell(
            row['primary_mae_tail'], decimals=3
        )
        tail_rmse_cell = format_numeric_cell(
            row['primary_rmse_tail'], decimals=3
        )
        reduction_cell = format_numeric_cell(
            row['error_reduction_percent'], decimals=1
        )

        latex_lines.append(
            f'{experiment_cell} & {task_cell} & {mode_cell} & {channel_cell}'
            f' & {steps_cell} & {target_cell} & {weight_cell} &'
            f' {prediction_cell} & {initial_mae_cell} & {tail_mae_cell} &'
            f' {tail_rmse_cell} & {reduction_cell} \\\\'
        )

    latex_lines.extend([
        r'\bottomrule',
        r'\end{tabular}%',
        r'}',
        r'\end{table*}'
    ])

    pathlib.Path(tex_filepath).write_text(
        '\n'.join(latex_lines), encoding='utf-8'
    )


def export_regression_channels_tex_and_csv(
        dataframe_channels: pd.DataFrame,
        csv_filepath: Union[str, pathlib.Path] = (
            'results/regression-channels.csv'
        ),
        tex_filepath: Union[str, pathlib.Path] = (
            'results/regression-channels.tex'
        ),
        caption: str = REGRESSION_CHANNELS_CAPTION
) -> None:
    pathlib.Path(csv_filepath).parent.mkdir(parents=True, exist_ok=True)
    dataframe_channels.to_csv(csv_filepath, index=False)

    latex_lines = [
        r'\begin{table*}[htbp]',
        r'\centering',
        rf'\caption{{{caption}}}',
        r'\label{tab:regression-channels}',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{lllcccccccc}',
        r'\toprule',
        (
            r'\textbf{Exp.} & \textbf{Mode} & \textbf{Ch.} & $w_{0}$ &'
            r' $w_{\mathrm{tail}}$ & $\hat{y}_{\mathrm{tail}}$ &'
            r' $\mathrm{MAE}_{\mathrm{init}}$ &'
            r' $\mathrm{MAE}_{\mathrm{tail}}$ &'
            r' $\mathrm{RMSE}_{\mathrm{tail}}$ &'
            r' $\Delta\mathrm{Err}\ (\%)$ & \textbf{Status} \\'
        ),
        r'\midrule'
    ]

    previous_record_key = None
    for _, row in dataframe_channels.iterrows():
        filename = str(row['file']).replace('.csv', '')
        experiment_name = (
            str(row['experiment'])
            .replace('device-regression-recordings-', '')
            .replace('experiment-', 'Exp.~')
        )
        record_key = (experiment_name, filename)

        if (
            previous_record_key is not None
            and record_key != previous_record_key
        ):
            latex_lines.append(r'\midrule')

        experiment_cell = (
            experiment_name if record_key != previous_record_key else ''
        )
        previous_record_key = record_key

        mode_cell = str(row['training_mode'])
        channel_value = int(row['channel'])
        channel_cell = f'Ch.~{channel_value}'
        target_cell = format_numeric_cell(row['nominal_target'], decimals=2)
        weight_cell = format_mean_sd_cell(
            row['tail_weight_mean'], row['tail_weight_std'], decimals=3
        )
        prediction_cell = format_mean_sd_cell(
            row['tail_prediction_mean'], row['tail_prediction_std'], decimals=3
        )
        initial_mae_cell = format_numeric_cell(
            row['primary_mae_initial'], decimals=3
        )
        tail_mae_cell = format_numeric_cell(
            row['primary_mae_tail'], decimals=3
        )
        tail_rmse_cell = format_numeric_cell(
            row['primary_rmse_tail'], decimals=3
        )
        reduction_cell = format_numeric_cell(
            row['error_reduction_percent'], decimals=1
        )
        status_cell = (
            'Rail-Clamped' if bool(row['rail_saturated']) else 'Active'
        )

        latex_lines.append(
            f'{experiment_cell} & {mode_cell} & {channel_cell} &'
            f' {target_cell} & {weight_cell} & {prediction_cell} &'
            f' {initial_mae_cell} & {tail_mae_cell} & {tail_rmse_cell} &'
            f' {reduction_cell} & {status_cell} \\\\'
        )

    latex_lines.extend([
        r'\bottomrule',
        r'\end{tabular}%',
        r'}',
        r'\end{table*}'
    ])

    pathlib.Path(tex_filepath).write_text(
        '\n'.join(latex_lines), encoding='utf-8'
    )


def export_hypothesis_tests_tex_and_csv(
        dataframe_tests: pd.DataFrame,
        csv_filepath: Union[str, pathlib.Path] = (
            'results/regression-hypothesis-tests.csv'
        ),
        tex_filepath: Union[str, pathlib.Path] = (
            'results/regression-hypothesis-tests.tex'
        ),
        caption: str = REGRESSION_HYPOTHESIS_TESTS_CAPTION
) -> None:
    pathlib.Path(csv_filepath).parent.mkdir(parents=True, exist_ok=True)
    dataframe_tests.to_csv(csv_filepath, index=False)

    latex_lines = [
        r'\begin{table*}[htbp]',
        r'\centering',
        rf'\caption{{{caption}}}',
        r'\label{tab:regression-hypothesis-tests}',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{llllcccl}',
        r'\toprule',
        (
            r'\textbf{Task / Cohort} & \textbf{Comparison} & \textbf{Test} &'
            r' \textbf{Sample} & \textbf{Statistic} &'
            r' $p_{\mathrm{two\text{-}sided}}$ &'
            r' $p_{\mathrm{one\text{-}sided}}$ &'
            r' \textbf{Effect Size} \\'
        ),
        r'\midrule'
    ]

    previous_group = None
    for _, row in dataframe_tests.iterrows():
        task_name = str(row['task'])
        comparison_name = str(row['comparison'])
        group_key = f'{task_name}||{comparison_name}'

        if previous_group is not None and group_key != previous_group:
            latex_lines.append(r'\midrule')

        task_cell = task_name if group_key != previous_group else ''
        comparison_cell = (
            comparison_name if group_key != previous_group else ''
        )
        previous_group = group_key

        test_cell = str(row['test'])
        sample_value = str(row['sample'])
        sample_cell = f'${sample_value}$'
        statistic_value = row['statistic']
        if test_cell in ('MWU', 'WSR'):
            statistic_cell = f'${statistic_value:.1f}$'
        else:
            statistic_cell = f'${statistic_value:.2f}$'

        p_two_cell = format_p_value_with_stars(
            row['p_two_sided'], include_stars=True
        )
        p_one_cell = format_p_value_with_stars(
            row['p_one_sided'], include_stars=True
        )

        effect_type = str(row['effect_size_type'])
        effect_value = row['effect_size']
        if effect_type == 'r_rb':
            effect_cell = f'$r_{{\\mathrm{{rb}}}} = {effect_value:.2f}$'
        else:
            effect_cell = f'$d = {effect_value:.2f}$'

        latex_lines.append(
            f'{task_cell} & {comparison_cell} & {test_cell} & {sample_cell}'
            f' & {statistic_cell} & {p_two_cell} & {p_one_cell} &'
            f' {effect_cell} \\\\'
        )

    latex_lines.extend([
        r'\bottomrule',
        r'\end{tabular}%',
        r'}',
        r'\end{table*}'
    ])

    pathlib.Path(tex_filepath).write_text(
        '\n'.join(latex_lines), encoding='utf-8'
    )


if __name__ == '__main__':
    target_directory = (
        DEFAULT_REGRESSION_DIRECTORY
        if DEFAULT_REGRESSION_DIRECTORY.exists()
        else FALLBACK_REGRESSION_DIRECTORY
    )

    dataframe_summary, dataframe_channels, dataframe_tests = (
        evaluate_all_regression_recordings(target_directory)
    )

    results_directory = pathlib.Path('results')
    results_directory.mkdir(parents=True, exist_ok=True)

    export_regression_summary_tex_and_csv(
        dataframe_summary,
        csv_filepath=results_directory / 'regression-summary.csv',
        tex_filepath=results_directory / 'regression-summary.tex'
    )
    export_regression_channels_tex_and_csv(
        dataframe_channels,
        csv_filepath=results_directory / 'regression-channels.csv',
        tex_filepath=results_directory / 'regression-channels.tex'
    )
    export_hypothesis_tests_tex_and_csv(
        dataframe_tests,
        csv_filepath=results_directory / 'regression-hypothesis-tests.csv',
        tex_filepath=results_directory / 'regression-hypothesis-tests.tex'
    )

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1200)
    pd.set_option(
        'display.float_format',
        lambda value: (
            f'{value:.4f}'
            if abs(value) < 1000 and abs(value) > 0.0001
            else f'{value:.2e}'
        )
    )

    print(dataframe_summary.to_string(index=False))

    print(dataframe_channels.to_string(index=False))

    print(dataframe_tests.to_string(index=False))
