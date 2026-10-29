"""Customer 360: reproducible 90-day repurchase experiment on synthetic 2025 data.

Run: python customer360_ml.py
No source files are modified. All outputs are written below --output-dir.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
DATA_START = pd.Timestamp('2025-01-01')
DATA_END = pd.Timestamp('2025-12-31')
SPLITS = {'train': '2025-03-31', 'validation': '2025-06-30', 'test': '2025-09-30'}
FEATURES = [
    'recency_capped_91', 'orders_90d', 'orders_30d', 'log_revenue_90d',
    'log_revenue_30d', 'log_aov_90d', 'log_max_order_90d', 'channels_90d',
    'active_30d_blocks',
]
FEATURE_DESCRIPTIONS = {
    'recency_capped_91': 'Số ngày từ lần mua Completed cuối, chặn tại 91 ngày.',
    'orders_90d': 'Số đơn Completed trong 90 ngày tính cả ngày chốt.',
    'orders_30d': 'Số đơn Completed trong 30 ngày tính cả ngày chốt.',
    'log_revenue_90d': 'log1p(tổng giá trị đơn VND trong 90 ngày).',
    'log_revenue_30d': 'log1p(tổng giá trị đơn VND trong 30 ngày).',
    'log_aov_90d': 'log1p(giá trị đơn trung bình 90 ngày); 0 nếu không mua.',
    'log_max_order_90d': 'log1p(giá trị đơn cao nhất 90 ngày); 0 nếu không mua.',
    'channels_90d': 'Số kênh có đơn Completed trong 90 ngày.',
    'active_30d_blocks': 'Số khoảng 30 ngày có mua trong 90 ngày, từ 0 đến 3.',
}


def load_data(data_dir: Path):
    customers = pd.read_csv(data_dir / 'dim_customer.csv', parse_dates=['join_date'])
    orders = pd.read_csv(data_dir / 'fact_order.csv', parse_dates=['order_date'])
    for frame, key in [(customers, 'customer_id'), (orders, 'order_id')]:
        if frame[key].isna().any() or not frame[key].is_unique:
            raise ValueError(f'Missing or duplicate primary key: {key}')
    if not orders.customer_id.isin(customers.customer_id).all():
        raise ValueError('Order with unknown customer.')
    if not orders.order_date.between(DATA_START, DATA_END).all():
        raise ValueError('This experiment expects the supplied 2025 order window.')
    if orders[['order_date', 'order_total_amount', 'order_status', 'sales_channel']].isna().any().any():
        raise ValueError('Missing required transaction fields.')
    if not orders.order_status.isin(['Completed', 'Cancelled']).all():
        raise ValueError('Unexpected order status.')
    if (orders.order_total_amount < 0).any():
        raise ValueError('Negative order value.')
    joined = orders.merge(customers[['customer_id', 'join_date']], on='customer_id', validate='many_to_one')
    if customers.join_date.isna().any() or (joined.order_date < joined.join_date).any():
        raise ValueError('Missing join date or transaction before customer joined.')
    return customers, orders


def build_snapshot(customers, orders, cutoff, *, labeled=True, coverage_end=DATA_END):
    """One row per previously observed buyer; predictors only use dates <= cutoff.

    Observation window: [cutoff - 89 days, cutoff].
    Label window: (cutoff, cutoff + 90 days]. Unobserved labels are never zero-filled.
    Assumption: supplied final Completed status was available on order_date.
    """
    cutoff = pd.Timestamp(cutoff)
    end = cutoff + pd.Timedelta(days=90)
    if cutoff - pd.Timedelta(days=89) < DATA_START:
        raise ValueError('Not enough history for a full 90-day observation window.')
    if cutoff > pd.Timestamp(coverage_end):
        raise ValueError('Cutoff exceeds documented coverage.')
    if labeled and end > pd.Timestamp(coverage_end):
        raise ValueError('The 90-day outcome window is not fully observed.')
    sales = orders.loc[orders.order_status.eq('Completed')]
    past = sales.loc[sales.order_date.le(cutoff)]
    ids = sorted(past.customer_id.unique())
    base = pd.DataFrame(index=pd.Index(ids, name='customer_id'))
    registered = customers.set_index('customer_id').join_date.reindex(base.index)
    if not registered.le(cutoff).all():
        raise ValueError('Buyer was not yet registered at cutoff.')
    last = past.groupby('customer_id').order_date.max().reindex(base.index)
    base['recency_capped_91'] = (cutoff - last).dt.days.clip(upper=91)
    recent = past.loc[past.order_date.ge(cutoff - pd.Timedelta(days=89))].copy()
    for days in (90, 30):
        window = recent.loc[recent.order_date.ge(cutoff - pd.Timedelta(days=days - 1))]
        grp = window.groupby('customer_id')
        base[f'orders_{days}d'] = grp.size().reindex(base.index, fill_value=0)
        base[f'log_revenue_{days}d'] = np.log1p(grp.order_total_amount.sum().reindex(base.index, fill_value=0))
    grp = recent.groupby('customer_id')
    base['log_aov_90d'] = np.log1p(grp.order_total_amount.mean().reindex(base.index, fill_value=0))
    base['log_max_order_90d'] = np.log1p(grp.order_total_amount.max().reindex(base.index, fill_value=0))
    base['channels_90d'] = grp.sales_channel.nunique().reindex(base.index, fill_value=0)
    recent['block'] = (cutoff - recent.order_date).dt.days // 30
    base['active_30d_blocks'] = recent.groupby('customer_id').block.nunique().reindex(base.index, fill_value=0)
    base['cutoff'] = cutoff
    base['label_end'] = end
    if labeled:
        future_buyers = sales.loc[sales.order_date.gt(cutoff) & sales.order_date.le(end), 'customer_id']
        base['repurchase_90d'] = base.index.isin(future_buyers).astype(int)
    if base[FEATURES].isna().any().any() or not np.isfinite(base[FEATURES].to_numpy()).all():
        raise ValueError('Invalid feature matrix.')
    return base.reset_index()


def model_candidates():
    return {
        'logistic_regression': make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)),
        'random_forest': RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=15, random_state=SEED, n_jobs=1),
    }


def baseline_scores(frame):
    # Scores are ranks, not probabilities; no threshold / Brier score for these.
    return {
        'recency_rule': -frame.recency_capped_91.to_numpy(dtype=float),
        'frequency_rule': frame.orders_90d.to_numpy(dtype=float),
    }


def ranked_indices(scores, customer_ids):
    # Pseudorandom fixed tie-breaks avoid ordering by synthetic customer sequence.
    # IDs are never model inputs. No labels are used for ties.
    tie = np.array([hashlib.sha256(f'{SEED}:{x}'.encode()).hexdigest() for x in customer_ids])
    return np.lexsort((tie, -np.asarray(scores)))


def evaluate(name, frame, scores, *, probability=False):
    y = frame.repurchase_90d.to_numpy()
    k = max(1, math.ceil(0.2 * len(y)))
    top = ranked_indices(scores, frame.customer_id)[:k]
    precision_top = float(y[top].mean())
    prevalence = float(y.mean())
    row = dict(model=name, n=len(y), positives=int(y.sum()), prevalence=prevalence,
               average_precision=float(average_precision_score(y, scores)),
               roc_auc=float(roc_auc_score(y, scores)), top20_n=k,
               precision_at_20pct=precision_top, recall_at_20pct=float(y[top].sum()/y.sum()),
               lift_at_20pct=precision_top/prevalence)
    if name == 'constant_prior':
        # With all scores tied there is no ranking; show expected random capture.
        row.update(precision_at_20pct=prevalence, recall_at_20pct=k/len(y), lift_at_20pct=1.0)
    if probability:
        predicted = np.asarray(scores) >= 0.5
        tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        row.update(brier=float(brier_score_loss(y, scores)), accuracy_at_05=float(accuracy_score(y, predicted)),
                   precision_at_05=float(precision_score(y, predicted, zero_division=0)),
                   recall_at_05=float(recall_score(y, predicted, zero_division=0)),
                   tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    return row


def bootstrap_ap_difference(frame, selected_scores, baseline, n=1000):
    """Paired bootstrap on independent customer rows of the single test snapshot.

    Conditional on this date and the fitted model; not a CI for future periods.
    """
    rng = np.random.default_rng(SEED)
    y = frame.repurchase_90d.to_numpy()
    diffs = []
    for _ in range(n):
        ix = rng.integers(0, len(y), len(y))
        if np.unique(y[ix]).size == 2:
            diffs.append(average_precision_score(y[ix], selected_scores[ix]) - average_precision_score(y[ix], baseline[ix]))
    lo, hi = np.quantile(diffs, [.025, .975])
    return dict(comparison='selected_ml_minus_validation_selected_rule', statistic='average_precision_difference',
                estimate=float(average_precision_score(y, selected_scores)-average_precision_score(y, baseline)),
                ci95_low=float(lo), ci95_high=float(hi), resamples=len(diffs))


def plot_results(frame, scores, selected, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    y = frame.repurchase_90d.to_numpy()
    colors = ['#146C94', '#D17A22', '#6956A5', '#279B80']
    for color, (name, score) in zip(colors, scores.items()):
        if name == 'constant_prior':
            continue
        p, r, _ = precision_recall_curve(y, score)
        axes[0].step(r, p, where='post', label=name, color=color, linewidth=1.8)
        order = ranked_indices(score, frame.customer_id)
        xs = np.arange(1, len(y)+1)/len(y)
        axes[1].plot(xs, np.cumsum(y[order])/y.sum(), label=name, color=color, linewidth=1.8)
    axes[0].axhline(y.mean(), color='#777777', linestyle='--', label='test positive rate')
    axes[1].plot([0, 1], [0, 1], color='#777777', linestyle='--', label='random selection')
    axes[1].axvline(.2, color='#AAAAAA', linestyle=':')
    axes[0].set(xlabel='Recall', ylabel='Precision', title='90-day repurchase: test PR curves', xlim=(0, 1), ylim=(0, 1.03))
    axes[1].set(xlabel='Fraction of customers selected', ylabel='Fraction of future buyers captured', title='Cumulative gains on Sep 30 snapshot', xlim=(0, 1), ylim=(0, 1.03))
    for ax in axes:
        ax.legend(fontsize=8, loc='lower left')
        ax.grid(alpha=.15)
    fig.savefig(output_dir/'evaluation.png', dpi=170)
    plt.close(fig)


def run_experiment(data_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    customers, orders = load_data(data_dir)
    snapshots = {name: build_snapshot(customers, orders, date) for name, date in SPLITS.items()}
    train, val, test = [snapshots[name] for name in SPLITS]
    assert train.label_end.max() <= val.cutoff.min()
    assert val.label_end.max() <= test.cutoff.min()
    all_rows = pd.concat([f.assign(split=name) for name, f in snapshots.items()], ignore_index=True)
    all_rows.to_csv(output_dir/'snapshots_labeled.csv', index=False, encoding='utf-8-sig')
    split_summary = pd.DataFrame([
        dict(split=name, cutoff=str(frame.cutoff.iloc[0].date()),
             observation_start=str((frame.cutoff.iloc[0]-pd.Timedelta(days=89)).date()),
             label_start=str((frame.cutoff.iloc[0]+pd.Timedelta(days=1)).date()),
             label_end=str(frame.label_end.iloc[0].date()), n=len(frame),
             positives=int(frame.repurchase_90d.sum()), prevalence=float(frame.repurchase_90d.mean()))
        for name, frame in snapshots.items()
    ])
    split_summary.to_csv(output_dir/'split_summary.csv', index=False)
    candidates = model_candidates()
    val_rows = []
    for name, estimator in candidates.items():
        estimator.fit(train[FEATURES], train.repurchase_90d)
        val_rows.append(evaluate(name, val, estimator.predict_proba(val[FEATURES])[:, 1], probability=True))
    for name, scores in baseline_scores(val).items():
        val_rows.append(evaluate(name, val, scores))
    dummy = DummyClassifier(strategy='prior').fit(train[FEATURES], train.repurchase_90d)
    val_rows.append(evaluate('constant_prior', val, dummy.predict_proba(val[FEATURES])[:, 1], probability=True))
    validation = pd.DataFrame(val_rows)
    validation.to_csv(output_dir/'validation_metrics.csv', index=False)
    # Lock choices BEFORE inspecting test labels. No test-driven tuning or feature search.
    selected = validation.loc[validation.model.isin(candidates)].sort_values('average_precision', ascending=False, kind='stable').iloc[0].model
    selected_rule = validation.loc[validation.model.isin(baseline_scores(val))].sort_values('average_precision', ascending=False, kind='stable').iloc[0].model
    development = pd.concat([train, val], ignore_index=True)
    fitted = clone(candidates[selected]).fit(development[FEATURES], development.repurchase_90d)
    dummy.fit(development[FEATURES], development.repurchase_90d)
    ml_scores = fitted.predict_proba(test[FEATURES])[:, 1]
    test_scores = {selected: ml_scores, **baseline_scores(test),
                   'constant_prior': dummy.predict_proba(test[FEATURES])[:, 1]}
    test_metrics = pd.DataFrame([evaluate(name, test, score, probability=name in (selected, 'constant_prior')) for name, score in test_scores.items()])
    test_metrics.to_csv(output_dir/'test_metrics.csv', index=False)
    prediction = test[['customer_id', 'cutoff', 'label_end', 'repurchase_90d']].copy()
    for name, score in test_scores.items():
        prediction[name] = score
    prediction.to_csv(output_dir/'test_predictions.csv', index=False, encoding='utf-8-sig')
    ci = bootstrap_ap_difference(test, ml_scores, test_scores[selected_rule])
    pd.DataFrame([ci]).to_csv(output_dir/'bootstrap_ap_difference.csv', index=False)
    # Diagnostic only: this does not trigger model/feature changes.
    importance = permutation_importance(fitted, test[FEATURES], test.repurchase_90d, scoring='average_precision', n_repeats=15, random_state=SEED, n_jobs=1)
    pd.DataFrame({'feature': FEATURES, 'ap_decrease_mean': importance.importances_mean,
                  'ap_decrease_std': importance.importances_std}).sort_values('ap_decrease_mean', ascending=False).to_csv(output_dir/'permutation_importance.csv', index=False)
    plot_results(test, test_scores, selected, output_dir)
    # Separate demonstration artifact. December predictions have NO observed labels.
    latest = build_snapshot(customers, orders, DATA_END, labeled=False)
    final_model = clone(candidates[selected]).fit(all_rows[FEATURES], all_rows.repurchase_90d)
    latest['estimated_repurchase_probability'] = final_model.predict_proba(latest[FEATURES])[:, 1]
    latest['model'] = selected
    latest['outcome_observed'] = False
    latest.sort_values(['estimated_repurchase_probability', 'customer_id'], ascending=[False, True]).to_csv(output_dir/'demo_scores_2025_12_31.csv', index=False, encoding='utf-8-sig')
    # Export transparent LR parameters for readers. No serialized executable model needed.
    if selected == 'logistic_regression':
        pd.DataFrame({'feature': FEATURES, 'standardized_coefficient': fitted.steps[-1][1].coef_[0]}).to_csv(output_dir/'logistic_coefficients.csv', index=False)
    manifest = dict(
        python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__, sklearn=sklearn.__version__,
        seed=SEED, selected_ml=selected, selected_rule=selected_rule,
        selection_metric='validation average_precision', threshold=0.5, ranking_budget_fraction=0.2,
        source_rows={'customers': len(customers), 'orders': len(orders), 'completed': int(orders.order_status.eq('Completed').sum())},
        source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data_dir.glob('*.csv'))},
        split_summary=split_summary.to_dict(orient='records'), bootstrap=ci,
        december_demo_rows=len(latest), december_outcomes_observed=False,
        evaluation_scope='Future purchases of known observed buyers, on synthetic data; not unseen-customer generalization.',
        historical_availability_assumption='Final Completed status is treated as known on order_date; no completion timestamp supplied.',
    )
    (output_dir/'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame([{'feature': k, 'description_vi': v} for k, v in FEATURE_DESCRIPTIONS.items()]).to_csv(output_dir/'feature_dictionary.csv', index=False, encoding='utf-8-sig')
    return dict(manifest=manifest, splits=split_summary, validation=validation, test=test_metrics, snapshots=snapshots)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent/'data')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent/'results')
    args = parser.parse_args()
    result = run_experiment(args.data_dir, args.output_dir)
    print('Selected ML:', result['manifest']['selected_ml'])
    print(result['splits'].to_string(index=False))
    print(result['test'].to_string(index=False))
