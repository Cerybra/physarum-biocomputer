import itertools
import pathlib

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from matplotlib.transforms import ScaledTranslation

from typing import Callable, Dict, List, Optional, Tuple, Union


# Temporal XOR trial accuracies recorded directly from the embedded firmware.
DEFAULT_XOR_BASELINE_RESULTS = np.array([0.5017, 0.5017, 0.66, 0.66, 0.66])
DEFAULT_XOR_RESERVOIR_RESULTS = np.array([1.0, 1.0, 1.0, 0.92, 0.93, 0.90])

DESCRIPTIVE_STATISTICS_CAPTION = (
    r'\textbf{Descriptive performance statistics across biological wetware'
    r' computing benchmarks.} Parametric $95\%$ Student-$t$ confidence'
    r' intervals are clamped to valid metric bounds ($[0, 100]\%$ for'
    r' accuracy; $[0, \infty)$ for MSE and RMSE). Living ODE trials'
    r' ($N = 20$) span two independently grown biological reservoirs ($9$ and'
    r' $11$ trials).'
)


def build_inferential_statistics_caption(baseline_mse_string: str) -> str:
    return (
        r'\textbf{Inferential statistical test results, $p$-values, and'
        r' standardized effect sizes across reservoir computing benchmark'
        r' comparisons.} All contrasts evaluate the computational efficacy of'
        r' living wetware physical reservoirs against substrate baselines and'
        r' controls. For the ODE forecasting benchmark, the baseline condition'
        ' (`Living vs. Baseline\') denotes a readout trained directly on the'
        r' raw input features $[x(t), f(t)]$ without a reservoir substrate.'
        r' The test abbreviations are as follows: MWU, two-sample'
        r' Mann--Whitney $U$ test (non-parametric rank-sum test for'
        ' independent distributions); WT, Welch\'s $t$-test (parametric'
        r' unequal-variances $t$-test, reported with Satterthwaite degrees of'
        r' freedom $\mathrm{df}$); WSR, one-sample Wilcoxon signed-rank test'
        r' against the deterministic raw-feature baseline'
        f' (${baseline_mse_string}$);'
        ' OST, one-sample Student\'s $t$-test against the deterministic'
        f' raw-feature baseline (${baseline_mse_string}$). Standardized effect'
        r' sizes are quantified by the rank-biserial correlation'
        ' ($r_{\\mathrm{rb}}$) for non-parametric contrasts and Cohen\'s $d$'
        r' for parametric contrasts (using pooled standard deviation for'
        r' independent samples and sample standard deviation for one-sample'
        r' contrasts). Sample sizes ($n_{1}, n_{2}$) indicate recording runs'
        r' per condition; $N$ denotes living reservoir evaluation trials (with'
        r' the $N = 20$ ODE trials spanning two independently grown biological'
        r' reservoirs, $9$ and $11$ runs respectively). Both two-sided'
        r' ($p_{\mathrm{2s}}$) and directed one-sided ($p_{\mathrm{1s}}$)'
        r' asymptotic/exact $p$-values are reported, where one-sided tests'
        r' evaluate the directed alternative hypothesis of superior living'
        r' reservoir computational performance (higher accuracy in Temporal'
        r' XOR; reduced MSE in forced ODE forecasting). Significance:'
        r' $^{*}p < 0.05$, $^{**}p < 0.01$, $^{***}p < 0.001$,'
        r' $^{****}p < 0.0001$; values without asterisks are not'
        r' statistically significant ($p \ge 0.05$).'
    )


def run_ode_experiment(
        recordings: np.ndarray,
        model: Callable,
        **kwargs
) -> Tuple[float, float]:
    # Trim the final `horizon` rows (indices STEPS - HORIZON .. STEPS - 1),
    # which lack a valid future target y(t + horizon) and contain uninitialized
    # target values (-0.0 at row 1195 and out-of-bounds memory at 1196..1199).
    horizon = int(kwargs.get('horizon', 5))
    if horizon > 0 and recordings.shape[0] > horizon:
        recordings = recordings[:-horizon]

    valid_row_mask = np.all(np.abs(recordings) < 20.0, axis=1)
    recordings = recordings[valid_row_mask]

    if 'train_size' in kwargs:
        split_index = int(recordings.shape[0] * kwargs['train_size'])
    else:
        train_samples = int(kwargs.get('train_samples', 1000))
        split_index = min(train_samples, recordings.shape[0] - 1)

    train_split = recordings[:split_index]
    test_split = recordings[split_index:]

    x_train_features, x_test_features = (
        train_split[:, [0, 2]],
        test_split[:, [0, 2]]
    )
    x_train_states, x_test_states = train_split[:, 4:], test_split[:, 4:]
    y_train, y_test = train_split[:, 3], test_split[:, 3]

    baseline = model()
    baseline.fit(x_train_features, y_train)

    readout = model()
    readout.fit(np.hstack([x_train_features, x_train_states]), y_train)

    y_pred_baseline = baseline.predict(x_test_features)
    y_pred_readout = readout.predict(
        np.hstack([x_test_features, x_test_states])
    )

    return (
        mean_squared_error(y_true=y_test, y_pred=y_pred_baseline),
        mean_squared_error(y_true=y_test, y_pred=y_pred_readout)
    )


def _summarize_distribution(
        task: str,
        model_condition: str,
        metric_name: str,
        values_array: np.ndarray,
        is_percentage: bool = False
) -> Dict[str, Union[str, int]]:
    values = np.asarray(values_array, dtype=float)

    sample_count = len(values)
    scale = 100.0 if is_percentage else 1.0
    decimal_places = 2 if is_percentage else 4

    mean_value = np.mean(values) * scale
    std_value = (
        np.std(values, ddof=1) if sample_count > 1 else 0.0
    ) * scale
    sem_value = (
        (std_value / np.sqrt(sample_count)) if sample_count > 0 else 0.0
    )

    t_critical = (
        stats.t.ppf(0.975, df=sample_count - 1) if sample_count > 1 else 0.0
    )
    ci_lower = max(0.0, mean_value - t_critical * sem_value)
    ci_upper = mean_value + t_critical * sem_value
    if is_percentage:
        ci_upper = min(100.0, ci_upper)

    median_value = np.median(values) * scale

    return {
        'Task': task,
        'Substrate': model_condition,
        'Metric': metric_name,
        'N': sample_count,
        r'Mean $\pm$ SD': (
            f'{mean_value:.{decimal_places}f} $\\pm$'
            f' {std_value:.{decimal_places}f}'
        ),
        r'95\% CI': (
            f'[{ci_lower:.{decimal_places}f}, {ci_upper:.{decimal_places}f}]'
        ),
        'Median': f'{median_value:.{decimal_places}f}'
    }


def _format_p_value_with_stars(
        p_value: float,
        include_stars: bool = False
) -> str:
    stars = ''
    if include_stars:
        if p_value < 0.0001:
            stars = '****'
        elif p_value < 0.001:
            stars = '***'
        elif p_value < 0.01:
            stars = '**'
        elif p_value < 0.05:
            stars = '*'

    star_suffix = f'^{{{stars}}}' if stars else ''
    if p_value < 1e-4:
        exponent = int(np.floor(np.log10(p_value)))
        mantissa = p_value / (10 ** exponent)
        formatted_number = f'{mantissa:.2f} \\times 10^{{{exponent}}}'

        return (
            f'${{{formatted_number}}}{star_suffix}$'
            if stars
            else f'${formatted_number}$'
        )

    formatted_number = f'{p_value:.4f}'

    return (
        f'${formatted_number}{star_suffix}$'
        if stars
        else f'${formatted_number}$'
    )


def _append_two_sample_test_rows(
        test_records: List[Dict[str, str]],
        task_label: str,
        comparison_label: str,
        sample_first: np.ndarray,
        sample_second: np.ndarray,
        alternative_direction: str
) -> None:
    count_first = len(sample_first)
    count_second = len(sample_second)

    u_statistic, p_mwu_two_sided = stats.mannwhitneyu(
        sample_first, sample_second, alternative='two-sided'
    )
    _, p_mwu_one_sided = stats.mannwhitneyu(
        sample_first, sample_second, alternative=alternative_direction
    )
    if alternative_direction == 'greater':
        rank_biserial = (
            (2.0 * u_statistic) / (count_first * count_second) - 1.0
        )
    else:
        rank_biserial = (
            1.0 - (2.0 * u_statistic) / (count_first * count_second)
        )

    welch_result = stats.ttest_ind(
        sample_first,
        sample_second,
        equal_var=False,
        alternative='two-sided'
    )
    t_welch = welch_result.statistic
    p_welch_two_sided = welch_result.pvalue
    df_welch = welch_result.df
    p_welch_one_sided = stats.ttest_ind(
        sample_first,
        sample_second,
        equal_var=False,
        alternative=alternative_direction
    ).pvalue

    std_first = np.std(sample_first, ddof=1)
    std_second = np.std(sample_second, ddof=1)
    pooled_sd = np.sqrt(
        (
            (count_first - 1) * (std_first ** 2)
            + (count_second - 1) * (std_second ** 2)
        )
        / (count_first + count_second - 2)
    )
    cohen_d_independent = (
        (np.mean(sample_first) - np.mean(sample_second)) / (pooled_sd + 1e-12)
    )

    test_records.append({
        'Task': task_label,
        'Comparison': comparison_label,
        'Test': 'MWU',
        'Sample': f'$n = ({count_first}, {count_second})$',
        'Statistic': f'${u_statistic:.1f}$',
        r'$p_{\mathrm{2s}}$': _format_p_value_with_stars(
            p_mwu_two_sided, include_stars=True
        ),
        r'$p_{\mathrm{1s}}$': _format_p_value_with_stars(p_mwu_one_sided),
        'Effect Size': f'$r_{{\\mathrm{{rb}}}} = {rank_biserial:.2f}$'
    })
    test_records.append({
        'Task': task_label,
        'Comparison': comparison_label,
        'Test': 'WT',
        'Sample': f'$n = ({count_first}, {count_second})$',
        'Statistic': f'$t({df_welch:.1f}) = {t_welch:.2f}$',
        r'$p_{\mathrm{2s}}$': _format_p_value_with_stars(
            p_welch_two_sided, include_stars=True
        ),
        r'$p_{\mathrm{1s}}$': _format_p_value_with_stars(p_welch_one_sided),
        'Effect Size': f'$d = {cohen_d_independent:.2f}$'
    })


def save_reservoir_statistics(
        target_filepath: Union[str, pathlib.Path],
        null_baseline_results: np.ndarray,
        null_readout_results: np.ndarray,
        reservoir_baseline_results: np.ndarray,
        reservoir_readout_results: np.ndarray,
        xor_baseline_results: Optional[np.ndarray] = None,
        xor_reservoir_results: Optional[np.ndarray] = None,
        digit_results: Optional[Dict[str, np.ndarray]] = None,
        caption: str = DESCRIPTIVE_STATISTICS_CAPTION
) -> pd.DataFrame:
    target_path = pathlib.Path(target_filepath)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if xor_baseline_results is None:
        xor_baseline_results = DEFAULT_XOR_BASELINE_RESULTS.copy()
    if xor_reservoir_results is None:
        xor_reservoir_results = DEFAULT_XOR_RESERVOIR_RESULTS.copy()

    records = [
        _summarize_distribution(
            'Temporal XOR',
            'Input Baseline',
            r'Accuracy (\%)',
            xor_baseline_results,
            True
        ),
        _summarize_distribution(
            'Temporal XOR',
            'Living Reservoir',
            r'Accuracy (\%)',
            xor_reservoir_results,
            True
        ),
        _summarize_distribution(
            'ODE',
            'Raw Features $[x(t), f(t)]$',
            'MSE',
            reservoir_baseline_results,
            False
        ),
        _summarize_distribution(
            'ODE',
            'Null Control (Dead)',
            'MSE',
            null_readout_results,
            False
        ),
        _summarize_distribution(
            'ODE',
            'Living Reservoir',
            'MSE',
            reservoir_readout_results,
            False
        ),
        _summarize_distribution(
            'ODE',
            'Raw Features $[x(t), f(t)]$',
            'RMSE',
            np.sqrt(np.maximum(0.0, reservoir_baseline_results)),
            False
        ),
        _summarize_distribution(
            'ODE',
            'Null Control (Dead)',
            'RMSE',
            np.sqrt(np.maximum(0.0, null_readout_results)),
            False
        ),
        _summarize_distribution(
            'ODE',
            'Living Reservoir',
            'RMSE',
            np.sqrt(np.maximum(0.0, reservoir_readout_results)),
            False
        )
    ]

    if digit_results:
        for condition_name, scores in digit_results.items():
            records.append(
                _summarize_distribution(
                    '8x8 Digits',
                    condition_name,
                    r'Accuracy (\%)',
                    scores,
                    True
                )
            )

    dataframe_statistics = pd.DataFrame(records)

    if target_path.suffix.lower() == '.tex':
        mean_sd_key = r'Mean $\pm$ SD'
        ci_key = r'95\% CI'
        latex_lines = [
            r'\begin{table}[htbp]',
            r'\centering',
            rf'\caption{{{caption}}}',
            r'\label{tab:reservoir-stats}',
            r'\resizebox{\textwidth}{!}{%',
            r'\begin{tabular}{lllcrcc}',
            r'\toprule',
            (
                r'\textbf{Task} & \textbf{Substrate} & \textbf{Metric} &'
                r' $N$ & \textbf{Mean $\pm$ SD} & \textbf{95\% CI} &'
                r' \textbf{Median} \\'
            ),
            r'\midrule'
        ]

        for _, row in dataframe_statistics.iterrows():
            task_cell = row['Task']
            substrate_cell = row['Substrate']
            metric_cell = row['Metric']
            sample_cell = row['N']
            mean_sd_cell = row[mean_sd_key]
            ci_cell = row[ci_key]
            median_cell = row['Median']

            latex_lines.append(
                f'{task_cell} & {substrate_cell} & {metric_cell} &'
                f' {sample_cell} & {mean_sd_cell} & {ci_cell} &'
                f' {median_cell} \\\\'
            )

        latex_lines.extend([
            r'\bottomrule',
            r'\end{tabular}%',
            r'}',
            r'\end{table}'
        ])
        target_path.write_text('\n'.join(latex_lines), encoding='utf-8')
    else:
        dataframe_statistics.to_csv(target_path, index=False)

    return dataframe_statistics


def save_statistical_tests(
        target_filepath: Union[str, pathlib.Path],
        null_baseline_results: np.ndarray,
        null_readout_results: np.ndarray,
        reservoir_baseline_results: np.ndarray,
        reservoir_readout_results: np.ndarray,
        xor_baseline_results: Optional[np.ndarray] = None,
        xor_reservoir_results: Optional[np.ndarray] = None,
        caption: str = ''
) -> pd.DataFrame:
    target_path = pathlib.Path(target_filepath)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if xor_baseline_results is None:
        xor_baseline_results = DEFAULT_XOR_BASELINE_RESULTS.copy()
    if xor_reservoir_results is None:
        xor_reservoir_results = DEFAULT_XOR_RESERVOIR_RESULTS.copy()

    test_records: List[Dict[str, str]] = []

    _append_two_sample_test_rows(
        test_records=test_records,
        task_label='Temporal XOR',
        comparison_label='Living vs. Baseline',
        sample_first=xor_reservoir_results,
        sample_second=xor_baseline_results,
        alternative_direction='greater'
    )

    _append_two_sample_test_rows(
        test_records=test_records,
        task_label='ODE',
        comparison_label='Living vs. Null',
        sample_first=reservoir_readout_results,
        sample_second=null_readout_results,
        alternative_direction='less'
    )

    count_reservoir_ode = len(reservoir_readout_results)
    baseline_constant = np.mean(reservoir_baseline_results)
    differences = (
        np.asarray(reservoir_readout_results, dtype=float) - baseline_constant
    )

    if np.any(differences != 0):
        w_statistic, p_wilcoxon_two_sided = stats.wilcoxon(
            differences, alternative='two-sided'
        )
        w_plus, p_wilcoxon_one_sided = stats.wilcoxon(
            differences, alternative='less'
        )
        non_zero_diffs = differences[differences != 0]
        number_of_non_zero = len(non_zero_diffs)
        rank_biserial_wsr = (
            1.0
            - (2.0 * w_plus)
            / (number_of_non_zero * (number_of_non_zero + 1) / 2.0)
            if number_of_non_zero > 0
            else 0.0
        )

        one_sample_result = stats.ttest_1samp(
            reservoir_readout_results,
            popmean=baseline_constant,
            alternative='two-sided'
        )
        t_one_sample = one_sample_result.statistic
        p_one_sample_two_sided = one_sample_result.pvalue
        df_one_sample = one_sample_result.df
        p_one_sample_one_sided = stats.ttest_1samp(
            reservoir_readout_results,
            popmean=baseline_constant,
            alternative='less'
        ).pvalue
        cohen_d_one_sample = (
            np.mean(differences) / (np.std(differences, ddof=1) + 1e-12)
        )

        test_records.append({
            'Task': 'ODE',
            'Comparison': 'Living vs. Baseline',
            'Test': 'WSR',
            'Sample': f'$N = {count_reservoir_ode}$',
            'Statistic': f'${w_statistic:.1f}$',
            r'$p_{\mathrm{2s}}$': _format_p_value_with_stars(
                p_wilcoxon_two_sided, include_stars=True
            ),
            r'$p_{\mathrm{1s}}$': _format_p_value_with_stars(
                p_wilcoxon_one_sided
            ),
            'Effect Size': f'$r_{{\\mathrm{{rb}}}} = {rank_biserial_wsr:.2f}$'
        })
        test_records.append({
            'Task': 'ODE',
            'Comparison': 'Living vs. Baseline',
            'Test': 'OST',
            'Sample': f'$N = {count_reservoir_ode}$',
            'Statistic': f'$t({int(df_one_sample)}) = {t_one_sample:.2f}$',
            r'$p_{\mathrm{2s}}$': _format_p_value_with_stars(
                p_one_sample_two_sided, include_stars=True
            ),
            r'$p_{\mathrm{1s}}$': _format_p_value_with_stars(
                p_one_sample_one_sided
            ),
            'Effect Size': f'$d = {cohen_d_one_sample:.2f}$'
        })

    dataframe_tests = pd.DataFrame(test_records)

    if target_path.suffix.lower() == '.tex':
        p_two_key = r'$p_{\mathrm{2s}}$'
        p_one_key = r'$p_{\mathrm{1s}}$'
        latex_lines = [
            r'\begin{table}[htbp]',
            r'\centering',
            rf'\caption{{{caption}}}',
            r'\label{tab:reservoir-hypothesis-tests}',
            r'\resizebox{\textwidth}{!}{%',
            r'\begin{tabular}{llcccrrc}',
            r'\toprule',
            (
                r'\textbf{Task} & \textbf{Comparison} & \textbf{Test} &'
                r' \textbf{Sample} & \textbf{Statistic} &'
                r' $p_{\mathrm{2s}}$ & $p_{\mathrm{1s}}$ &'
                r' \textbf{Effect Size} \\'
            ),
            r'\midrule'
        ]

        for _, row in dataframe_tests.iterrows():
            task_cell = row['Task']
            comparison_cell = row['Comparison']
            test_cell = row['Test']
            sample_cell = row['Sample']
            statistic_cell = row['Statistic']
            p_two_cell = row[p_two_key]
            p_one_cell = row[p_one_key]
            effect_size_cell = row['Effect Size']

            latex_lines.append(
                f'{task_cell} & {comparison_cell} & {test_cell} &'
                f' {sample_cell} & {statistic_cell} & {p_two_cell} &'
                f' {p_one_cell} & {effect_size_cell} \\\\'
            )

        latex_lines.extend([
            r'\bottomrule',
            r'\end{tabular}%',
            r'}',
            r'\end{table}'
        ])
        target_path.write_text('\n'.join(latex_lines), encoding='utf-8')
    else:
        dataframe_tests.to_csv(target_path, index=False)

    return dataframe_tests


def plot_reservoir_performance_statistics(
        figure: plt.Figure,
        axes: np.ndarray,
        labels: Tuple[str, str],
        palette: Dict[str, str],
        null_baseline_results: np.ndarray,
        null_readout_results: np.ndarray,
        reservoir_readout_results: np.ndarray,
        show_significance: bool = True
) -> None:
    for axis in axes:
        axis.set_facecolor('white')
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

    axis_xor = axes[0]

    axis_xor.set_title('Temporal XOR Performance', fontsize=7)

    axis_xor.set_xlabel('Model', fontsize=6)
    axis_xor.set_ylabel('Accuracy', fontsize=6)

    baseline_results = DEFAULT_XOR_BASELINE_RESULTS.copy()
    reservoir_results = DEFAULT_XOR_RESERVOIR_RESULTS.copy()

    xor_experiments = ['Baseline', 'Living Reservoir Features']
    xor_means = [np.mean(baseline_results), np.mean(reservoir_results)]
    xor_stds = [
        np.std(baseline_results, ddof=1),
        np.std(reservoir_results, ddof=1)
    ]

    axis_xor.bar(
        x=xor_experiments,
        height=xor_means,
        width=0.4,
        yerr=xor_stds,
        color=[palette['error'], palette['main']],
        capsize=2
    )

    axis_xor.set_xticks(xor_experiments)
    axis_xor.set_xticklabels(['Baseline', 'Reservoir'])

    def _format_bracket_p(p_value: float) -> str:
        if p_value < 0.0001:
            stars = '****'
        elif p_value < 0.001:
            stars = '***'
        elif p_value < 0.01:
            stars = '**'
        elif p_value < 0.05:
            stars = '*'
        else:
            stars = 'ns'
        p_text = f'{p_value:.1e}' if p_value < 1e-4 else f'{p_value:.4f}'

        return f'{stars} (p = {p_text})'

    if show_significance:
        p_xor = stats.ttest_ind(
            reservoir_results, baseline_results, equal_var=False
        ).pvalue
        axis_xor.set_ylim(0, 1.25)
        axis_xor.plot(
            [0, 0, 1, 1],
            [1.06, 1.10, 1.10, 1.06],
            color='black',
            lw=0.8
        )
        axis_xor.text(
            0.5,
            1.11,
            _format_bracket_p(p_xor),
            ha='center',
            va='bottom',
            fontsize=5.5
        )

    axis_xor.text(
        0.0,
        1.0,
        labels[0],
        transform=(
            axis_xor.transAxes
            + ScaledTranslation(-10 / 72, 7 / 72, figure.dpi_scale_trans)
        ),
        fontsize=7,
        fontweight='bold',
        va='bottom',
        ha='right'
    )

    axis_ode = axes[1]

    axis_ode.set_title('ODE Reservoir Performance', fontsize=7)

    axis_ode.set_xlabel('Model', fontsize=6)
    axis_ode.set_ylabel('Mean Squared Error', fontsize=6)

    ode_experiments = [
        'Baseline',
        'Null Reservoir Features',
        'Living Reservoir Features'
    ]
    ode_means = [
        np.mean(null_baseline_results),
        np.mean(null_readout_results),
        np.mean(reservoir_readout_results)
    ]
    ode_stds = [
        np.std(null_baseline_results, ddof=1),
        np.std(null_readout_results, ddof=1),
        np.std(reservoir_readout_results, ddof=1)
    ]

    axis_ode.bar(
        x=ode_experiments,
        height=ode_means,
        width=0.4,
        yerr=ode_stds,
        color=[palette['error'], palette['deep'], palette['main']],
        capsize=2
    )

    axis_ode.set_xticks(ode_experiments)
    axis_ode.set_xticklabels(['Baseline', 'Null Reservoir', 'Reservoir'])

    if show_significance:
        p_ode_null = stats.mannwhitneyu(
            reservoir_readout_results,
            null_readout_results,
            alternative='two-sided'
        ).pvalue
        p_ode_baseline = stats.ttest_1samp(
            reservoir_readout_results,
            popmean=float(np.mean(null_baseline_results)),
            alternative='two-sided'
        ).pvalue

        y_max = max(ode_means[1] + ode_stds[1], 1.0)
        axis_ode.set_ylim(0, y_max * 1.25)
        y_bracket_null = y_max * 1.04
        axis_ode.plot(
            [1, 1, 2, 2],
            [
                y_bracket_null,
                y_bracket_null + 0.1,
                y_bracket_null + 0.1,
                y_bracket_null
            ],
            color='black',
            lw=0.8
        )
        axis_ode.text(
            1.5,
            y_bracket_null + 0.11,
            _format_bracket_p(p_ode_null),
            ha='center',
            va='bottom',
            fontsize=5.5
        )

        y_bracket_baseline = y_max * 1.15
        axis_ode.plot(
            [0, 0, 2, 2],
            [
                y_bracket_baseline,
                y_bracket_baseline + 0.1,
                y_bracket_baseline + 0.1,
                y_bracket_baseline
            ],
            color='black',
            lw=0.8
        )
        axis_ode.text(
            1.0,
            y_bracket_baseline + 0.11,
            _format_bracket_p(p_ode_baseline),
            ha='center',
            va='bottom',
            fontsize=5.5
        )

    axis_ode.text(
        0.0,
        1.0,
        labels[1],
        transform=(
            axis_ode.transAxes
            + ScaledTranslation(-10 / 72, 7 / 72, figure.dpi_scale_trans)
        ),
        fontsize=7,
        fontweight='bold',
        va='bottom',
        ha='right'
    )


if __name__ == '__main__':
    matplotlib.use('TkAgg')

    plt.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ytick.labelsize'] = 5
    plt.rcParams['xtick.labelsize'] = 5

    palette = {
        'main': '#FFCB05',
        'deep': '#B6862C',
        'error': 'gray',
        'target': 'black'
    }

    recordings_directory = pathlib.Path('data/reservoir-computing-recordings')
    null_reservoir_recordings = [
        np.load(filepath)['recordings']
        for filepath in sorted((recordings_directory / 'null').glob('*.npz'))
    ]
    reservoir_1_recordings = [
        np.load(filepath)['recordings']
        for filepath in sorted(
            (recordings_directory / 'reservoir-1').glob('*.npz')
        )
    ]
    reservoir_2_recordings = [
        np.load(filepath)['recordings']
        for filepath in sorted(
            (recordings_directory / 'reservoir-2').glob('*.npz')
        )
    ]

    baseline_results, readout_results = [], []
    for reservoir_recordings in itertools.chain(
        null_reservoir_recordings,
        reservoir_1_recordings,
        reservoir_2_recordings
    ):
        baseline_performance, readout_performance = run_ode_experiment(
            reservoir_recordings, LinearRegression
        )

        baseline_results.append(baseline_performance)
        readout_results.append(readout_performance)

    null_count = len(null_reservoir_recordings)

    null_baseline_results = np.array(baseline_results[:null_count])
    null_readout_results = np.array(readout_results[:null_count])

    reservoir_baseline_results = np.array(baseline_results[null_count:])
    reservoir_readout_results = np.array(readout_results[null_count:])

    baseline_mse_string = f'{np.mean(reservoir_baseline_results):.4f}'
    inferential_caption = build_inferential_statistics_caption(
        baseline_mse_string
    )

    results_directory = pathlib.Path('results')
    results_directory.mkdir(parents=True, exist_ok=True)

    plots_directory = pathlib.Path('plots')
    plots_directory.mkdir(parents=True, exist_ok=True)

    dataframe_statistics = save_reservoir_statistics(
        target_filepath=(
            results_directory / 'reservoir-benchmark-statistics.csv'
        ),
        null_baseline_results=null_baseline_results,
        null_readout_results=null_readout_results,
        reservoir_baseline_results=reservoir_baseline_results,
        reservoir_readout_results=reservoir_readout_results
    )
    save_reservoir_statistics(
        target_filepath=(
            results_directory / 'reservoir-benchmark-statistics.tex'
        ),
        null_baseline_results=null_baseline_results,
        null_readout_results=null_readout_results,
        reservoir_baseline_results=reservoir_baseline_results,
        reservoir_readout_results=reservoir_readout_results
    )

    dataframe_tests = save_statistical_tests(
        target_filepath=results_directory / 'reservoir-statistical-tests.csv',
        null_baseline_results=null_baseline_results,
        null_readout_results=null_readout_results,
        reservoir_baseline_results=reservoir_baseline_results,
        reservoir_readout_results=reservoir_readout_results,
        caption=inferential_caption
    )
    save_statistical_tests(
        target_filepath=results_directory / 'reservoir-statistical-tests.tex',
        null_baseline_results=null_baseline_results,
        null_readout_results=null_readout_results,
        reservoir_baseline_results=reservoir_baseline_results,
        reservoir_readout_results=reservoir_readout_results,
        caption=inferential_caption
    )

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1200)

    print(dataframe_statistics.to_string(index=False))

    print(dataframe_tests.to_string(index=False))

    try:
        from xgboost import XGBRegressor

        xgb_baseline_results, xgb_readout_results = [], []
        for reservoir_recordings in itertools.chain(
            null_reservoir_recordings,
            reservoir_1_recordings,
            reservoir_2_recordings
        ):
            xgb_baseline, xgb_readout = run_ode_experiment(
                reservoir_recordings, XGBRegressor
            )
            xgb_baseline_results.append(xgb_baseline)
            xgb_readout_results.append(xgb_readout)

        xgb_null_readout = np.array(xgb_readout_results[:null_count])
        xgb_living_baseline = np.array(xgb_baseline_results[null_count:])
        xgb_living_readout = np.array(xgb_readout_results[null_count:])
        print(
            '\nXGBoost ODE Summary (1,000 train / 195 valid test):'
            f'\n  Raw-Input Baseline MSE: {np.mean(xgb_living_baseline):.4f}'
            f'\n  Living Reservoir MSE  : {np.mean(xgb_living_readout):.4f}'
            f' +/- {np.std(xgb_living_readout, ddof=1):.4f}'
            f' (min = {np.min(xgb_living_readout):.4f})'
            f'\n  Null Control MSE      : {np.mean(xgb_null_readout):.4f}'
            f' +/- {np.std(xgb_null_readout, ddof=1):.4f}'
        )
    except ImportError:
        pass

    figure, axes = plt.subplots(1, 2, figsize=(7.08, 3.2))

    plot_reservoir_performance_statistics(
        figure=figure,
        axes=axes,
        labels=('A', 'B'),
        palette=palette,
        null_baseline_results=reservoir_baseline_results,
        null_readout_results=null_readout_results,
        reservoir_readout_results=reservoir_readout_results,
        show_significance=True
    )

    plt.tight_layout()
    plt.savefig(
        plots_directory / 'reservoir-statistics.pdf',
        dpi=600,
        bbox_inches='tight'
    )
