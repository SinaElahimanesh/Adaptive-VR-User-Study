from __future__ import annotations

from typing import Dict, Tuple, List
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.metrics import cohen_kappa_score

from .config import ANCHORS, TASKS, CONDITIONS


def compute_descriptive_stats(ratings_long: pd.DataFrame, metric: str = "easy_to_use") -> Dict[str, pd.DataFrame]:
    df = ratings_long[ratings_long["metric"] == metric].copy()
    overall = (
        df.groupby(["task", "anchor"])['rating']
        .agg(['count', 'mean', 'median', 'std'])
        .reset_index()
    )
    by_cond = (
        df.groupby(["condition", "task", "anchor"])['rating']
        .agg(['count', 'mean', 'median', 'std'])
        .reset_index()
    )
    return {"overall": overall, "by_condition": by_cond}


def compute_preference_distribution(rankings: pd.DataFrame) -> pd.DataFrame:
    counts = (
        rankings
        .dropna(subset=["planned_top"])  # remove rows without top
        .groupby(["task", "condition", "planned_top"])  # condition present per trial
        .size()
        .reset_index(name="count")
    )
    counts["proportion"] = counts.groupby(["task", "condition"])['count'].transform(lambda s: s / s.sum())
    return counts


def run_friedman_by_task(ratings_long: pd.DataFrame, metric: str = "easy_to_use") -> pd.DataFrame:
    results = []
    for cond in sorted(ratings_long['condition'].dropna().unique()):
        for task in TASKS:
            df = ratings_long[(ratings_long["metric"] == metric) & (ratings_long["task"] == task) & (ratings_long['condition'] == cond)]
            if df.empty:
                continue
            pivot = df.pivot_table(index="participant_id", columns="anchor", values="rating")
            anchors_present = [a for a in ANCHORS if a in pivot.columns]
            pivot = pivot[anchors_present]
            pivot = pivot.dropna()
            if pivot.shape[0] < 3 or pivot.shape[1] < 2:
                continue
            try:
                stat, p = stats.friedmanchisquare(*[pivot[a] for a in pivot.columns])
                k = pivot.shape[1]
                n = pivot.shape[0]
                w = stat / (n * (k - 1)) if k > 1 and n > 0 else np.nan
                results.append({
                    "condition": cond,
                    "task": task,
                    "anchors": list(pivot.columns),
                    "friedman_chi2": stat,
                    "kendalls_w": w,
                    "p_value": p,
                    "n": int(n),
                })
            except Exception:
                continue
    return pd.DataFrame(results)


def compute_condition_effects(ratings_long: pd.DataFrame, anchor: str | None = None, task: str | None = None, metric: str = "easy_to_use") -> pd.DataFrame:
    subset = ratings_long[ratings_long["metric"] == metric]
    anchors = [anchor] if anchor else sorted(subset["anchor"].dropna().unique())
    tasks = [task] if task else sorted(subset["task"].dropna().unique())
    rows = []
    for t in tasks:
        for a in anchors:
            df = subset[(subset["task"] == t) & (subset["anchor"] == a)]
            groups = [g['rating'].dropna().values for _, g in df.groupby("condition")]
            labels = [k for k, _ in df.groupby("condition")]
            if len(groups) < 2 or min(map(len, groups)) < 3:
                continue
            stat, p = stats.kruskal(*groups)
            rows.append({"task": t, "anchor": a, "test": "kruskal", "h_stat": stat, "p_value": p, "conditions": labels})
    return pd.DataFrame(rows)


def compute_correlations(participants: pd.DataFrame, ratings_long: pd.DataFrame, rankings: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    easy = ratings_long[ratings_long["metric"] == "easy_to_use"].copy()
    avg_by_participant = (
        easy.groupby(["participant_id"])['rating'].mean().rename("avg_ease")
    )
    demo = (participants.drop_duplicates(subset=['participant_id'])
            .set_index("participant_id")[['age', 'xr_experience', 'game_experience', 'condition', 'gender']])
    corr_df = pd.concat([avg_by_participant, demo[['age', 'xr_experience', 'game_experience']]], axis=1)

    corr_rows = []
    for x in ['age', 'xr_experience', 'game_experience']:
        s1 = corr_df['avg_ease']
        s2 = corr_df[x]
        valid = (~s1.isna()) & (~s2.isna())
        if valid.sum() >= 5:
            rho, p = stats.spearmanr(s1[valid], s2[valid])
            corr_rows.append({"x": x, "y": "avg_ease", "spearman_rho": rho, "p_value": p, "n": int(valid.sum())})
    corr_numeric = pd.DataFrame(corr_rows)

    pref_trials = rankings.dropna(subset=['planned_top'])
    chi_rows = []
    ct = pd.crosstab(pref_trials['condition'], pref_trials['planned_top'])
    if ct.size and ct.values.sum() > 0 and ct.shape[0] > 1 and ct.shape[1] > 1:
        chi2, p, dof, exp = stats.chi2_contingency(ct)
        chi_rows.append({"variable": "condition", "chi2": chi2, "dof": dof, "p_value": p})
    top_by_participant = (pref_trials.groupby(['participant_id', 'task']).first().reset_index()
                          .merge(participants.drop_duplicates('participant_id')[['participant_id', 'gender']], on='participant_id', how='left'))
    ct_g = pd.crosstab(top_by_participant['gender'], top_by_participant['planned_top'])
    if ct_g.size and ct_g.values.sum() > 0 and ct_g.shape[0] > 1 and ct_g.shape[1] > 1:
        chi2, p, dof, exp = stats.chi2_contingency(ct_g)
        chi_rows.append({"variable": "gender", "chi2": chi2, "dof": dof, "p_value": p})
    chi = pd.DataFrame(chi_rows)

    return {"spearman": corr_numeric, "chi_square": chi}


def compute_consistency_agreement(ratings_long: pd.DataFrame, rankings: pd.DataFrame, metric: str = "easy_to_use") -> pd.DataFrame:
    df = ratings_long[ratings_long["metric"] == metric].copy()
    top_rated = (
        df.groupby(["participant_id", "condition", "task", "anchor"])['rating'].mean().reset_index()
        .sort_values(["participant_id", "condition", "task", "rating"], ascending=[True, True, True, False])
        .groupby(["participant_id", "condition", "task"]).first().reset_index().rename(columns={"anchor": "rated_top"})
    )

    planned = rankings[['participant_id', 'condition', 'task', 'planned_top']]
    merged = planned.merge(top_rated[['participant_id', 'condition', 'task', 'rated_top']], on=["participant_id", "condition", "task"], how="left")
    merged["agree"] = (merged['planned_top'] == merged['rated_top']).astype(int)

    # Cohen's kappa by condition×task (robust to chance agreement)
    kappas = []
    for (cond, task), sub in merged.groupby(['condition', 'task']):
        if sub['planned_top'].notna().sum() >= 3 and sub['rated_top'].notna().sum() >= 3:
            try:
                kappa = cohen_kappa_score(sub['planned_top'], sub['rated_top'])
            except Exception:
                kappa = np.nan
        else:
            kappa = np.nan
        kappas.append({"condition": cond, "task": task, "cohens_kappa": kappa, "n": int(len(sub))})
    kappa_df = pd.DataFrame(kappas)

    summary = (merged.groupby(['condition', 'task'])['agree']
               .agg(['mean', 'count']).reset_index()
               .rename(columns={'mean': 'percent_agreement'}))
    return summary.merge(kappa_df, on=['condition', 'task'], how='left')


# Post-hoc helpers

def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta effect size for two independent samples."""
    a = a.flatten()
    b = b.flatten()
    gt = 0
    lt = 0
    for x in a:
        gt += np.sum(x > b)
        lt += np.sum(x < b)
    n1 = len(a)
    n2 = len(b)
    if n1 == 0 or n2 == 0:
        return np.nan
    delta = (gt - lt) / (n1 * n2)
    return float(delta)


def _rank_biserial_from_wilcoxon(x: np.ndarray, y: np.ndarray) -> float:
    """Matched rank-biserial effect size for paired samples.
    r_rb = (W_pos - W_neg) / (n*(n+1)/2), using signed rank sums.
    """
    diff = x - y
    diff = diff[diff != 0]
    n = len(diff)
    if n == 0:
        return np.nan
    ranks = stats.rankdata(np.abs(diff))
    W_pos = np.sum(ranks[diff > 0])
    W_neg = np.sum(ranks[diff < 0])
    denom = n * (n + 1) / 2.0
    return float((W_pos - W_neg) / denom)


def posthoc_condition_effects_pairwise(
    ratings_long: pd.DataFrame,
    task: str,
    anchor: str,
    metric: str = "easy_to_use",
) -> pd.DataFrame:
    subset = ratings_long[(ratings_long['metric'] == metric) & (ratings_long['task'] == task) & (ratings_long['anchor'] == anchor)]
    cond_groups = {cond: grp['rating'].dropna().values for cond, grp in subset.groupby('condition') if grp['rating'].notna().sum() >= 3}
    pairs = list(itertools.combinations(sorted(cond_groups.keys()), 2))
    rows: List[dict] = []
    pvals = []
    for a, b in pairs:
        u, p = stats.mannwhitneyu(cond_groups[a], cond_groups[b], alternative='two-sided')
        delta = _cliffs_delta(cond_groups[a], cond_groups[b])
        rows.append({"task": task, "anchor": anchor, "cond_a": a, "cond_b": b, "u_stat": u, "p_raw": p, "cliffs_delta": delta})
        pvals.append(p)
    if not rows:
        return pd.DataFrame()
    _, p_adj, _, _ = multipletests(pvals, method='holm')
    for r, pa in zip(rows, p_adj):
        r['p_holm'] = pa
    return pd.DataFrame(rows)


def posthoc_friedman_within_condition(
    ratings_long: pd.DataFrame,
    condition: str,
    task: str,
    metric: str = "easy_to_use",
) -> pd.DataFrame:
    df = ratings_long[(ratings_long['metric'] == metric) & (ratings_long['task'] == task) & (ratings_long['condition'] == condition)]
    pivot = df.pivot_table(index='participant_id', columns='anchor', values='rating')
    anchors_present = [a for a in ANCHORS if a in pivot.columns]
    if len(anchors_present) < 2:
        return pd.DataFrame()
    pivot = pivot[anchors_present].dropna()
    pairs = list(itertools.combinations(anchors_present, 2))
    rows: List[dict] = []
    pvals = []
    for a, b in pairs:
        if (pivot[a] == pivot[b]).all():
            continue
        try:
            stat, p = stats.wilcoxon(pivot[a], pivot[b], zero_method='wilcox', alternative='two-sided')
            r_rb = _rank_biserial_from_wilcoxon(pivot[a].values, pivot[b].values)
        except ValueError:
            continue
        rows.append({"condition": condition, "task": task, "anchor_a": a, "anchor_b": b, "w_stat": stat, "p_raw": p, "rank_biserial": r_rb, "n": int(pivot.shape[0])})
        pvals.append(p)
    if not rows:
        return pd.DataFrame()
    _, p_adj, _, _ = multipletests(pvals, method='holm')
    for r, pa in zip(rows, p_adj):
        r['p_holm'] = pa
    return pd.DataFrame(rows)


# Advanced models

def fit_mixedlm_ratings(ratings_long: pd.DataFrame, metric: str = 'easy_to_use') -> pd.DataFrame:
    """Linear mixed-effects model treating rating as continuous.
    Fixed: condition * task * anchor; Random: participant intercept.
    Returns fixed effect coefficients.
    """
    df = ratings_long[ratings_long['metric'] == metric].dropna(subset=['rating', 'condition', 'task', 'anchor', 'participant_id']).copy()
    # Cast to categories for design matrices
    df['condition'] = df['condition'].astype('category')
    df['task'] = df['task'].astype('category')
    df['anchor'] = df['anchor'].astype('category')
    try:
        model = smf.mixedlm("rating ~ C(condition)*C(task)*C(anchor)", data=df, groups=df['participant_id'])
        res = model.fit(method='lbfgs', maxiter=1000, disp=False)
        summ = pd.DataFrame({
            'term': res.params.index,
            'coef': res.params.values,
            'std_err': res.bse.values,
            'z': res.tvalues.values,
            'p_value': res.pvalues.values,
        })
        return summ
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})


# Rank-based analyses on planned orders

def compute_mean_ranks(rankings: pd.DataFrame) -> pd.DataFrame:
    """Compute mean rank (1 best) per condition×task×anchor from planned lists."""
    rows = []
    for _, r in rankings.iterrows():
        order = r.get('planned_order', [])
        if not isinstance(order, list) or len(order) == 0:
            continue
        rank_map = {anchor: i + 1 for i, anchor in enumerate(order)}
        for a in ANCHORS:
            if a in rank_map:
                rows.append({
                    'participant_id': r['participant_id'],
                    'condition': r['condition'],
                    'task': r['task'],
                    'anchor': a,
                    'rank': rank_map[a],
                })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    mean_ranks = df.groupby(['condition', 'task', 'anchor'])['rank'].mean().reset_index().rename(columns={'rank': 'mean_rank'})
    return mean_ranks


def run_friedman_on_planned_ranks(rankings: pd.DataFrame) -> pd.DataFrame:
    """Friedman tests on ranks per condition×task across anchors, with Kendall's W."""
    rows = []
    for (cond, task), sub in rankings.groupby(['condition', 'task']):
        # build pivot participants x anchors with rank
        parts = []
        for pid, g in sub.groupby('participant_id'):
            order = g['planned_order'].iloc[0]
            if not isinstance(order, list) or len(order) == 0:
                continue
            rank_map = {a: i + 1 for i, a in enumerate(order)}
            row = {a: rank_map.get(a, np.nan) for a in ANCHORS}
            row['participant_id'] = pid
            parts.append(row)
        if not parts:
            continue
        pv = pd.DataFrame(parts).set_index('participant_id')
        pv = pv.dropna(axis=0)
        anchors_present = [a for a in ANCHORS if a in pv.columns]
        if pv.shape[0] < 3 or len(anchors_present) < 2:
            continue
        try:
            stat, p = stats.friedmanchisquare(*[pv[a] for a in anchors_present])
            k = len(anchors_present)
            n = pv.shape[0]
            w = stat / (n * (k - 1)) if k > 1 and n > 0 else np.nan
            rows.append({'condition': cond, 'task': task, 'friedman_chi2': stat, 'kendalls_w': w, 'p_value': p, 'n': int(n)})
        except Exception:
            continue
    return pd.DataFrame(rows)


# Multinomial choice model for planned top

def run_multinomial_choice_model(rankings: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    """Multinomial logit for planned_top with predictors: condition, task, age, xr_experience, game_experience, gender.
    Baseline anchor will be chosen automatically by MNLogit.
    Returns coefficients and p-values per outcome category.
    """
    df = rankings.dropna(subset=['planned_top']).copy()
    demo = participants.drop_duplicates('participant_id')[['participant_id', 'age', 'xr_experience', 'game_experience', 'gender']]
    df = df.merge(demo, on='participant_id', how='left')
    # build design matrix
    exog = pd.get_dummies(df[['condition', 'task', 'gender']], drop_first=True)
    exog = pd.concat([exog, df[['age', 'xr_experience', 'game_experience']]], axis=1)
    exog = sm.add_constant(exog, has_constant='add')
    y = df['planned_top']
    try:
        model = sm.MNLogit(y, exog)
        res = model.fit(method='newton', maxiter=200, disp=False)
        out = []
        for cat, params in res.params.iterrows():
            for var in params.index:
                out.append({'outcome': cat, 'predictor': var, 'coef': params[var], 'p_value': res.pvalues.loc[cat, var]})
        return pd.DataFrame(out)
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})


# Cross-condition adaptability per task

def compute_cross_condition_adaptability(rankings: pd.DataFrame, ratings_long: pd.DataFrame, metric: str = 'easy_to_use') -> Dict[str, pd.DataFrame]:
    """Count how often participants change top anchor across conditions per task.
    Computes for planned_top and rated_top separately.
    """
    # Planned
    plan = rankings.dropna(subset=['planned_top'])[['participant_id', 'condition', 'task', 'planned_top']]
    rows_plan = []
    for (pid, task), sub in plan.groupby(['participant_id', 'task']):
        order = sub.sort_values('condition')  # lexicographic; ensure consistent ordering
        anchors = order['planned_top'].tolist()
        changes = sum(1 for i in range(1, len(anchors)) if anchors[i] != anchors[i-1])
        rows_plan.append({'participant_id': pid, 'task': task, 'changes_planned_across_conditions': changes})
    df_plan = pd.DataFrame(rows_plan)

    # Rated
    top_rated = (
        ratings_long[ratings_long['metric'] == metric]
        .groupby(['participant_id', 'condition', 'task', 'anchor'])['rating'].mean().reset_index()
        .sort_values(['participant_id', 'condition', 'task', 'rating'], ascending=[True, True, True, False])
        .groupby(['participant_id', 'condition', 'task']).first().reset_index()
        .rename(columns={'anchor': 'rated_top'})
    )
    rows_rate = []
    for (pid, task), sub in top_rated.groupby(['participant_id', 'task']):
        order = sub.sort_values('condition')
        anchors = order['rated_top'].tolist()
        changes = sum(1 for i in range(1, len(anchors)) if anchors[i] != anchors[i-1])
        rows_rate.append({'participant_id': pid, 'task': task, 'changes_rated_across_conditions': changes})
    df_rate = pd.DataFrame(rows_rate)

    return {'planned': df_plan, 'rated': df_rate}
