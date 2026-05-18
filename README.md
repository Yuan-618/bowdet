# bowdet

**Audio-based bow-change detection for string instruments.**

`bowdet` detects bow changes (down-bow ↔ up-bow transitions) in bowed string instrument recordings using a two-stage mel-spectrogram boundary approach.

## Install

```bash
pip install bowdet
```

## Usage

```python
from bowdet import detect

times = detect("recording.wav")
print(times)  # array of bow-change times in seconds
```

### Options

```python
times = detect(
    "recording.wav",
    threshold=0.40,      # classification threshold (default: 0.40)
    min_dist_sec=0.12,   # minimum distance between events in seconds
)
```

## Method

The system uses a two-stage pipeline:

1. **Stage 1 — Boundary Proposal**: Computes log-mel spectrogram boundary strength (L2 norm of frame differences), and proposes candidate bow-change positions at spectral peaks.

2. **Stage 2 — Candidate Classification**: A compact CNN fused with 15 acoustic boundary features classifies each candidate as a bow change or not.

**Key insight**: Bow changes produce a broadband energy transition visible in the log-mel spectrogram, spanning approximately 100–300 ms across all frequency bands.

## Performance

Evaluated under leave-one-performer-out (LOPO) on 6 viola performers (9 recordings, ~17.7 min, 1,020 annotated bow changes):

| Metric | Score |
|--------|-------|
| IoU@0.1 F1 | 0.645 ± 0.012 |
| Point F1 @100ms | 0.635 |

## Citation

If you use bowdet in your research, please cite this repository:

```bibtex
@software{yuan2026bowdet,
  author = {Haotian Yuan},
  title  = {bowdet: Audio-based bow-change detection for string instruments},
  year   = {2026},
  url    = {https://github.com/Yuan-618/bowdet}
}
```

## License

MIT