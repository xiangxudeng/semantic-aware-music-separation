"""第四周正式全量LoRA与同机条件分离统计；历史原始结果保持只读。"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_stats import STEMS, paired_test, holm


def validate_tracks(frame, index=None):
    if len(frame) != 50 or not frame.index.is_unique or not np.isfinite(frame[STEMS]).all().all():
        raise ValueError("必须包含50首唯一曲目和有限的四轨SI-SDR")
    if index is not None and set(frame.index) != set(index):
        raise ValueError("测试曲目未配对")
    return frame if index is None else frame.loc[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora', type=Path)
    parser.add_argument('--conditioned', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        good = pd.DataFrame(np.ones((50, 4)), columns=STEMS, index=[f't{i}' for i in range(50)])
        assert validate_tracks(good.iloc[::-1], good.index).equals(good)
        for bad in [good.iloc[:-1], good.set_axis(['x'] * 50), good.assign(vocals=np.nan)]:
            try:
                validate_tracks(bad)
                raise AssertionError('无效结果未拒绝')
            except ValueError:
                pass
        assert paired_test([2, 2, 2], [1, 1, 1])['p_value'] == 0.25
        print('FINAL_STATISTICS_SELF_CHECK_PASS')
        return
    if not all([args.lora, args.conditioned, args.output]):
        parser.error('需要 --lora、--conditioned、--output')
    args.output.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.lora / 'formal_multiseed_validation.json').read_text())
    assert source['metrics_recomputed_match'] and source['finite'] and source['seeds'] == [1, 2, 3]
    lora = pd.DataFrame(source['rows']).set_index('seed')
    baselines = {'map': source['baseline_map'], 'tail_recall': 0.1408}
    seed_tests = []
    for metric, baseline in baselines.items():
        values = lora[metric].to_numpy()
        seed_tests.append({'metric': metric, 'baseline': baseline,
                           'unit': 'training seed vs fixed baseline', 'training_seeds': 3,
                           'mean': float(values.mean()), 'sample_std': float(values.std(ddof=1)),
                           'relative_gain': float(values.mean() / baseline - 1),
                           **paired_test(values, np.full(3, baseline))})
    seed_tests = pd.DataFrame(seed_tests)
    seed_tests['p_holm'] = holm(seed_tests.p_value)
    seed_tests.to_csv(args.output / 'formal_seed_tests.csv', index=False)
    plan = json.loads((args.conditioned / 'conditioned_analysis_plan.json').read_text())
    frames = {}
    for group in plan['groups']:
        frame = pd.read_csv(args.conditioned / f'cond_eval_w4_p40_C_{group}.csv').set_index('track')
        frames[group] = validate_tracks(frame, frames['none'].index if 'none' in frames else None)
    summaries = []
    for name, frame in frames.items():
        summaries.append({'group': name, 'test_tracks': 50,
                          'four_stem_mean': float(frame[STEMS].mean(axis=1).mean()),
                          **{f'{stem}_median': float(frame[stem].median()) for stem in STEMS}})
    pd.DataFrame(summaries).to_csv(args.output / 'conditioned_summary.csv', index=False)
    comparisons = []
    for a, b in plan['primary_comparisons']:
        for metric in ['four_stem_mean'] + STEMS:
            left, right = (frames[a][STEMS].mean(axis=1), frames[b][STEMS].mean(axis=1)) if metric == 'four_stem_mean' else (frames[a][metric], frames[b][metric])
            comparisons.append({'comparison': f'{a}-{b}', 'metric': metric,
                                'unit': 'test track', 'training_seed': 0, 'training_seeds': 1,
                                **paired_test(left, right)})
    tests = pd.DataFrame(comparisons)
    tests['p_holm'] = np.nan
    for mask in [tests.metric == 'four_stem_mean', tests.metric != 'four_stem_mean']:
        tests.loc[mask, 'p_holm'] = holm(tests.loc[mask, 'p_value'])
    tests.to_csv(args.output / 'conditioned_source_tests.csv', index=False)
    metadata = {'analysis_plan': plan, 'lora_baselines': baselines,
                'seed_holm_family_size': 2,
                'limitations': ['3 fixed training seeds; exact two-sided p cannot be below 0.25',
                                'sample standard deviation is not a confidence interval',
                                'seed bootstrap intervals are conditional on 3 observed seeds',
                                '50 test-track bootstrap pairs do not include training variance',
                                'four stems are correlated; sample size is 50, not 200',
                                'fixed generic text is not target-instrument conditioning']}
    (args.output / 'final_statistics_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    metrics = {'lora': source, 'seed_tests': seed_tests.to_dict(orient='records'),
               'conditioned_summary': summaries,
               'conditioned_primary_tests': tests[tests.metric == 'four_stem_mean'].to_dict(orient='records')}
    (args.output / 'final_metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(seed_tests.to_string(index=False))
    print(tests[tests.metric == 'four_stem_mean'].to_string(index=False))


if __name__ == '__main__':
    main()
