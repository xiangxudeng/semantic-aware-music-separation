"""对真实样例的四轨输出做客观指标评审；不做任何人工试听结论。

使用原始混音核对四轨求和与混音的差异（重建误差）；
本次评审没有使用配对参考分轨，不报告真值分离 SDR/SI-SDR。
"""
import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np

STEMS = ['vocals', 'drums', 'bass', 'other']
SR = 44100


def load(path):
    audio, _ = librosa.load(str(path), sr=SR, mono=True)
    return audio


def db(value):
    return float(20 * np.log10(max(value, 1e-12)))


def stem_metrics(signal, others):
    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak = float(np.max(np.abs(signal)))
    spectrum = np.abs(librosa.stft(signal, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
    energy = spectrum ** 2
    total = energy.sum()
    high = energy[freqs >= 6000].sum()
    centroid = float(np.sum(freqs[:, None] * energy) / max(total, 1e-12))
    flatness = float(np.mean(librosa.feature.spectral_flatness(S=energy)))
    frames = librosa.feature.rms(S=np.sqrt(energy)).ravel()
    active = frames > frames.max() * 0.05
    cross = {name: float(np.corrcoef(signal, other)[0, 1]) for name, other in others.items()}
    return {
        'rms_dbfs': round(db(rms), 2),
        'peak_dbfs': round(db(peak), 2),
        'crest_factor_db': round(db(peak) - db(rms), 2),
        'clipped_samples': int(np.sum(np.abs(signal) >= 0.999)),
        'spectral_centroid_hz': round(centroid, 1),
        'high_band_6k_energy_ratio': round(float(high / max(total, 1e-12)), 4),
        'spectral_flatness': round(flatness, 4),
        'active_frame_ratio': round(float(active.mean()), 3),
        'max_abs_correlation_with_other_stems': round(max(abs(value) for value in cross.values()), 3),
        'correlation_with_other_stems': {
            name: round(value, 3) for name, value in cross.items()},
    }


def review(label, mixture_path, stem_dir):
    mixture = load(mixture_path)
    stems = {name: load(Path(stem_dir) / f'{name}.wav') for name in STEMS}
    length = min([len(mixture)] + [len(v) for v in stems.values()])
    mixture = mixture[:length]
    for name in stems:
        stems[name] = stems[name][:length]
    summed = sum(stems.values())
    residual = mixture - summed
    report = {
        'sample': label,
        'mixture': str(mixture_path),
        'stem_dir': str(stem_dir),
        'seconds': round(length / SR, 3),
        'reconstruction': {
            'sum_vs_mixture_sdr_db': round(db(np.sqrt(np.mean(summed ** 2)) / np.sqrt(np.mean(residual ** 2))), 2),
            'residual_rms_dbfs': round(db(np.sqrt(np.mean(residual ** 2))), 2),
            'sum_vs_mixture_correlation': round(float(np.corrcoef(mixture, summed)[0, 1]), 4),
        },
        'mixture_metrics': {
            'rms_dbfs': round(db(np.sqrt(np.mean(mixture ** 2))), 2),
            'peak_dbfs': round(db(np.max(np.abs(mixture))), 2),
        },
        'stems': {},
        'limits': [
            '本次评审未使用配对参考分轨，不报告真值分离 SDR/SI-SDR。',
            '相关系数与频谱统计只是客观代理，不能替代人工试听判断音质。',
            '单条样例的客观统计，不能代表整体性能。',
        ],
    }
    for name in STEMS:
        report['stems'][name] = stem_metrics(stems[name], {k: v for k, v in stems.items() if k != name})
    return report


def main():
    parser = argparse.ArgumentParser(description='对真实样例的四轨输出做客观指标评审')
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1],
                        help='交付目录，包含系统示例；默认取工程目录的上一级')
    parser.add_argument('--mixture', type=Path, help='原始混音，默认使用随包12秒样例')
    parser.add_argument('--stems', type=Path, help='四轨WAV目录，默认使用随包界面四轨')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parent / 'outputs' / '客观指标评审')
    args = parser.parse_args()
    if args.self_check:
        t = np.arange(4096) * 2 * np.pi / 64
        metrics = stem_metrics(np.sin(t), {'bass': np.sin(t) * .5, 'other': np.cos(t)})
        assert metrics['correlation_with_other_stems'] == {'bass': 1.0, 'other': 0.0}
        assert metrics['max_abs_correlation_with_other_stems'] == 1.0
        import tempfile
        import soundfile as sf

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            tracks = [np.sin(t * (i + 1)) * .05 for i in range(4)]
            for name, signal in zip(STEMS, tracks):
                sf.write(folder / f'{name}.wav', signal, SR)
            mixture = folder / 'mixture.wav'
            sf.write(mixture, sum(tracks) + np.cos(t * 6) * .001, SR)
            result = review('self-check', mixture, folder)
            assert result['stem_dir'] == str(folder)
            assert set(result['stems']) == set(STEMS)
        print('OBJECTIVE_REVIEW_SELF_CHECK_PASS')
        return
    # 默认只读取随包样例；其他音频通过显式路径传入，不依赖原工作区。
    cases = [
        review('四轨样例', args.mixture or args.root / '系统示例/sample12.wav',
               args.stems or args.root / '系统示例/界面四轨'),
    ]
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / '客观指标评审.json').write_text(
        json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf-8')
    for case in cases:
        print('==', case['sample'], case['seconds'], 's')
        print('   重建:', case['reconstruction'])
        for name, metric in case['stems'].items():
            print(f"   {name:7s} rms={metric['rms_dbfs']:>7} peak={metric['peak_dbfs']:>7} "
                  f"centroid={metric['spectral_centroid_hz']:>8} flat={metric['spectral_flatness']:>6} "
                  f"maxcorr={metric['max_abs_correlation_with_other_stems']:>6}")
    print('written', out / '客观指标评审.json')
    assert all(np.isfinite(case['reconstruction']['sum_vs_mixture_sdr_db']) for case in cases)


if __name__ == '__main__':
    sys.exit(main())
