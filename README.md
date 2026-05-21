# bowdet

**Audio-based bow-change and note-boundary detection for string instruments.**

`bowdet` detects bow changes (down-bow ↔ up-bow transitions) in bowed string instrument recordings using a two-stage mel-spectrogram boundary approach.

Version `0.2.1` also includes **BowDet-NB**, an experimental unsupervised note-boundary detector for bowed string recordings.

## Install

```bash
pip install bowdet
```

## Bow-change detection

```python
from bowdet import detect

times = detect("recording.wav")
print(times)  # array of bow-change times in seconds
```

### Options

```python
times = detect(
    "recording.wav",
    threshold=0.40,
    min_dist_sec=0.12,
)
```

## BowDet-NB: note-boundary detection

BowDet-NB detects note boundaries using an unsupervised V3 multi-scale mel boundary-strength method. It uses only the boundary proposal stage, without the supervised Stage-2 classifier.

```python
from bowdet import detect_nb

times = detect_nb("recording.wav")
print(times)  # array of note-boundary times in seconds
```

The full descriptive API is also available:

```python
from bowdet import detect_note_boundaries

times = detect_note_boundaries("recording.wav")
```

## Method

The bow-change detector uses a two-stage pipeline:

1. **Stage 1 — Boundary Proposal**: Computes log-mel spectrogram boundary strength and proposes candidate positions at spectral peaks.
2. **Stage 2 — Candidate Classification**: A compact CNN fused with 15 acoustic boundary features classifies each candidate as a bow change or not.

BowDet-NB uses a standalone unsupervised version of Stage 1 for note-boundary detection, based on multi-scale left-right mel-spectrogram differences.

**Key insight**: Bow changes and bowed-string note boundaries often produce broadband spectral transitions visible in the log-mel spectrogram, spanning approximately 100–300 ms across frequency bands.

## Performance

### Bow-change detection

Evaluated under leave-one-performer-out (LOPO) on 6 viola performers (9 recordings, ~17.7 min, 1,020 annotated bow changes):

| Metric | Score |
|---|---|
| IoU@0.1 F1 | 0.645 ± 0.012 |
| Point F1 @100ms | 0.635 |

### BowDet-NB note-boundary detection

Evaluated on 9 bowed-string recordings spanning viola, violin, and cello:

| Metric | Tonal mean F1 |
|---|---|
| Point F1 @50ms | 0.701 |
| Point F1 @100ms | 0.774 |
| Point F1 @150ms | 0.793 |
| Pseudo-region IoU@0.1 F1 | 0.812 |
| Pseudo-region IoU@0.3 F1 | 0.803 |

## Citation

If you use `bowdet` in your research, please cite this repository:

```bibtex
@software{yuan2026bowdet,
  author = {Haotian Yuan},
  title  = {bowdet: Audio-based bow-change and note-boundary detection for string instruments},
  year   = {2026},
  url    = {https://github.com/Yuan-618/bowdet}
}
```

## License

MIT