# Semantic-Aware Multi-Track Music Source Separation and Open-Vocabulary Music Understanding

A research project on connecting **audio source separation** with **cross-modal language understanding**:
separate a full song into four stems (vocals / drums / bass / other) and then *understand* what the music is —
genre, instruments, mood and vocal attributes — instead of treating every track identically.

## Highlights

| Area | Result |
|---|---|
| Separation baseline (Demucs v4, MUSDB18-HQ test, 50 tracks) | **8.86 dB** mean SDR over four stems (paper reports ≈ 9.0 dB) |
| Four-stem vs six-stem protocol study | Six-stem scored 8.38 dB even after merging guitar+piano into "other" — a **labelling-protocol artefact** was identified and quantified |
| Zero-shot music tagging (MagnaTagATune, 50 tags) | mAP **0.2744**, mean AUC **0.828** after correcting an implementation mismatch |
| Long-tail tags (14 rarest) | recall **+50.3%** over zero-shot, via semantic prompt expansion and few-shot prototype calibration |
| Cross-modal fine-tuning | **38,157** instruction samples; LoRA + 4-bit quantization, **7.04 GB** peak VRAM |

### One finding worth highlighting

The zero-shot baseline was initially **57% lower** than expected. Rather than assuming a model limitation,
the ported implementation (`transformers`) and the reference implementation were compared on identical data:
their embeddings were **not in the same space** (cosine similarity 0.29 for audio, 0.33 for text).
Switching to the reference implementation raised mAP from 0.175 to 0.2744.

Negative results are documented too, with attribution — fine-tuning did not improve separation
(the pretrained model already covered the training split), and semantic conditioning of the separator
was shown to be uninformative because the conditioning signal was derived from the same audio the model already sees.

## Repository layout

```
src/
  data/            unified dataset interface + 5 waveform augmentations
  semantic/        CLAP semantic encoder (track-level and clip-level 512-d), prompt building, tagging metrics
  separation/      FiLM conditioning layer for the separation network
scripts/           numbered pipeline: 00 environment → 10 baselines → 20 fine-tuning → 40 long-tail study
configs/           augmentation configuration
docs/              weekly experiment records
```

Every script is self-documenting and prints its own verification checks; the pipeline is designed so that
**each number in the reports can be traced back to a specific script and command**.

## Setup

```bash
pip install -r requirements.txt
```

Datasets are **not** included in this repository (all are public academic datasets and must be downloaded
by the user, subject to their own licences):

| Dataset | Used for |
|---|---|
| MUSDB18-HQ | source separation training / evaluation (150 tracks) |
| MagnaTagATune | music tagging (25,863 clips, 188 tags; a frozen 50-tag subset is used) |
| LP-MusicCaps-MTT | music captioning (3,300 clips, 4 descriptions each) |

## Pipeline overview

1. **Data** — `scripts/01`–`05`: download, extract, build the frozen tag list, fetch audio.
2. **Baselines** — `scripts/10`–`12`: Demucs separation SDR (museval, BSS-Eval v4 protocol), CLAP zero-shot tagging, quantization smoke test.
3. **Fine-tuning** — `scripts/20`–`22`: Demucs fine-tuning with per-epoch SI-SDR validation, multi-process museval evaluation, reference CLAP implementation.
4. **Long-tail study** — `scripts/23`–`46`: linear probe upper bound, unsupervised post-processing, semantic prompt expansion, few-shot prototypes, contrastive alignment with held-out-tag validation, and a unified recall metric.
5. **Cross-modal** — `scripts/50`–`65`: MUSDB semantic vectors, FiLM-conditioned separation with ablation, instruction dataset construction, CLAP→projector→LLM forward pass, LoRA fine-tuning and evaluation.

## Method notes

- **Metric protocol.** Separation is evaluated with `museval` (the official MUSDB protocol, BSS-Eval v4),
  while training uses SI-SDR as a cheap proxy. Tagging is reported as AP, AUC *and* recall —
  because AP alone penalises rare tags structurally and can be misread across papers.
  Recall is computed with a fixed decision fraction taken from the **training** split, so that the
  definition never drifts with the evaluation subset.
- **Conditioning layer.** The FiLM layer is zero-initialised so that a conditioned model starts
  *element-wise identical* to its baseline; any change can therefore be attributed to the conditioning
  signal rather than to initialisation noise.
- **Reproducibility.** Each run records its command line, configuration, environment versions and
  artefacts, which is how two silent failure modes were caught (a stem-order mismatch between model
  output and reference targets, and a group definition that drifted with the sampled subset).

## Licence / data notice

Code is released for academic and portfolio purposes. No audio data, model weights or third-party
resources are redistributed here; please obtain the datasets from their official sources.
