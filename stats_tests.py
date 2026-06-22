"""
stats_tests.py — Significance testing for GPCN-ViT ablations.

WHY THIS EXISTS
---------------
A reviewer will not accept "GPCN ON = 88.9%, GPCN OFF = 87.1%, therefore GPCN
helps". They want to know whether that gap is real or noise. This module turns a
set of paired per-seed (or per-fold) scores into a defensible claim:

  * mean ± std for each arm,
  * the paired difference (GPCN_on - GPCN_off),
  * a paired two-sided t-test p-value (the arms share seeds/folds, so PAIRED is
    the correct, more powerful test), and
  * a 95% confidence interval on the mean difference.

It is dependency-light: scipy is used if present (exact t/Student CDF); otherwise
it falls back to a normal approximation so the suite still runs on a bare Colab.

Usage:
    from stats_tests import paired_comparison, format_comparison
    rep = paired_comparison(on=[0.889, 0.901, 0.875], off=[0.871, 0.880, 0.860],
                            metric='accuracy', name_a='GPCN ON', name_b='GPCN OFF')
    print(format_comparison(rep))
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

# Two-tailed t critical values at 95% (df -> t*). Mirrors kfold_cv for small n.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 19: 2.093, 24: 2.064, 29: 2.045}


def _t_crit(df: int) -> float:
    if df <= 0:
        return float('nan')
    return _T95.get(df, 1.96)


def _mean_std(xs: Sequence[float]):
    n = len(xs)
    mean = sum(xs) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in xs) / (n - 1)
        return mean, math.sqrt(var)
    return mean, 0.0


def _t_sf(t: float, df: int) -> float:
    """Two-sided p-value for a t statistic. Uses scipy if available, else a
    normal approximation (adequate once df is not tiny; flagged in the report)."""
    t = abs(t)
    try:
        from scipy import stats  # type: ignore
        return float(2.0 * stats.t.sf(t, df))
    except Exception:
        # Normal approximation: 2 * (1 - Phi(t)).
        return float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))))


def paired_comparison(on: Sequence[float], off: Sequence[float], metric: str,
                      name_a: str = 'GPCN ON', name_b: str = 'GPCN OFF'
                      ) -> Dict:
    """Paired two-sided t-test on per-seed/per-fold scores for one metric.

    `on` and `off` must be aligned (same seed/fold at each index) and equal
    length. Returns means, the mean paired difference, a 95% CI on that
    difference, and the p-value.
    """
    if len(on) != len(off):
        raise ValueError(f"paired_comparison needs aligned arms; got {len(on)} vs {len(off)}.")
    n = len(on)
    diffs = [a - b for a, b in zip(on, off)]
    mean_a, std_a = _mean_std(on)
    mean_b, std_b = _mean_std(off)
    mean_d, std_d = _mean_std(diffs)

    df = n - 1
    if n > 1 and std_d > 0:
        se = std_d / math.sqrt(n)
        t_stat = mean_d / se
        p = _t_sf(t_stat, df)
        half = _t_crit(df) * se
    else:
        # Zero variance (or n==1): difference is deterministic. p is 0 if there
        # is a non-zero constant gap, else 1 (no difference at all).
        t_stat = float('inf') if mean_d != 0 else 0.0
        p = 0.0 if mean_d != 0 and n > 1 else float('nan')
        half = 0.0

    scipy_available = True
    try:
        import scipy  # noqa: F401
    except Exception:
        scipy_available = False

    return {
        'metric': metric, 'n': n, 'name_a': name_a, 'name_b': name_b,
        'mean_a': mean_a, 'std_a': std_a, 'values_a': list(on),
        'mean_b': mean_b, 'std_b': std_b, 'values_b': list(off),
        'mean_diff': mean_d, 'std_diff': std_d,
        'ci95_low': mean_d - half, 'ci95_high': mean_d + half,
        't_stat': t_stat, 'p_value': p,
        'significant_05': (p < 0.05) if isinstance(p, float) and not math.isnan(p) else None,
        'p_method': 'scipy-t' if scipy_available else 'normal-approx',
    }


def format_comparison(rep: Dict, as_percent: bool = True) -> str:
    """One-line-per-metric human summary of a paired_comparison report."""
    scale = 100.0 if as_percent else 1.0
    unit = '%' if as_percent else ''

    def f(x):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return '—'
        return f'{x * scale:.2f}{unit}'

    sig = ''
    if rep.get('significant_05') is True:
        sig = '  ✓ significant (p<0.05)'
    elif rep.get('significant_05') is False:
        sig = '  ✗ not significant'
    p = rep['p_value']
    p_str = '—' if (isinstance(p, float) and math.isnan(p)) else f'{p:.4f}'
    return (
        f"[{rep['metric']}] {rep['name_a']}: {f(rep['mean_a'])} ± {f(rep['std_a'])} | "
        f"{rep['name_b']}: {f(rep['mean_b'])} ± {f(rep['std_b'])} | "
        f"Δ={f(rep['mean_diff'])} (95% CI [{f(rep['ci95_low'])}, {f(rep['ci95_high'])}]) | "
        f"paired t-test p={p_str} ({rep['p_method']}, n={rep['n']}){sig}")


def compare_runs(records_a: List[Dict], records_b: List[Dict],
                 metrics: Sequence[str], name_a: str = 'GPCN ON',
                 name_b: str = 'GPCN OFF') -> Dict[str, Dict]:
    """Given two lists of result records (each a dict with a 'metrics' sub-dict),
    one per aligned seed/fold, run a paired comparison per requested metric.

    Records are paired by order, so callers must build the two lists with the
    same seed/fold ordering.
    """
    if len(records_a) != len(records_b):
        raise ValueError("compare_runs needs the same number of paired records per arm.")
    out: Dict[str, Dict] = {}
    for m in metrics:
        on = [r['metrics'][m] for r in records_a if m in r.get('metrics', {})]
        off = [r['metrics'][m] for r in records_b if m in r.get('metrics', {})]
        if len(on) != len(off) or not on:
            continue
        out[m] = paired_comparison(on, off, m, name_a, name_b)
    return out


def format_comparison_table(reports: Dict[str, Dict]) -> str:
    """Markdown table summarising several per-metric paired comparisons."""
    lines = [f'| Metric | {next(iter(reports.values()))["name_a"]} | '
             f'{next(iter(reports.values()))["name_b"]} | Δ (95% CI) | p-value | Sig. |',
             '|---|---|---|---|---|---|'] if reports else []

    def f(x, pct=True):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return '—'
        return f'{x * 100:.2f}' if pct else f'{x:.4f}'

    for m, r in reports.items():
        pct = m not in ('auc_roc', 'auc_macro_ovr', 'auc_weighted_ovr', 'mcc',
                        'ece', 'mce', 'brier')
        a = f'{f(r["mean_a"], pct)} ± {f(r["std_a"], pct)}'
        b = f'{f(r["mean_b"], pct)} ± {f(r["std_b"], pct)}'
        d = f'{f(r["mean_diff"], pct)} [{f(r["ci95_low"], pct)}, {f(r["ci95_high"], pct)}]'
        p = r['p_value']
        p_str = '—' if (isinstance(p, float) and math.isnan(p)) else f'{p:.4f}'
        sig = '✓' if r.get('significant_05') else ('✗' if r.get('significant_05') is False else '—')
        lines.append(f'| {m} | {a} | {b} | {d} | {p_str} | {sig} |')
    return '\n'.join(lines)
