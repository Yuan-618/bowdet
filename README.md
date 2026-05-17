# bowdet

**Bow change detection for bowed string instruments**

bowdet detects bow changes in audio recordings of bowed string instruments (viola, violin, cello) using deep learning.

## Installation

```bash
pip install bowdet
```

## Quick Start

```python
from bowdet import detect

# Detect bow changes (returns list of timestamps in seconds)
bow_changes = detect("recording.wav")
print(bow_changes)  # [1.23, 2.45, 3.67, ...]
```

## Models

| Model | IoU@0.1 F1 | Speed | Size |
|-------|-----------|-------|------|
| MERT (default) | 0.616 | ~2 min/min audio | 378 MB |
| CNN | 0.554 | ~30 sec/min audio | 5 MB |

## Parameters

```python
detect(
    audio_path,           # path to wav file
    model="mert",         # "mert" or "cnn"
    threshold=0.5,        # peak detection threshold (0-1)
    min_dist=0.35,        # minimum distance between bow changes (seconds)
                          # decrease for fast passages (e.g. spiccato): min_dist=0.2
                          # increase for slow passages (e.g. long bows): min_dist=0.5
)
```

## Weights

Weights are downloaded automatically on first use (~380 MB for MERT).
Cached at `~/.bowdet/weights/`

## Limitations

- Trained on 9 performers across diverse repertoire (Bach, Biber, Penderecki, Hindemith, etc.)
- Performance may degrade on playing styles significantly different from training data
- Evaluated on viola recordings; expected to generalize to violin and cello
- Designed for solo string instrument recordings; accompaniment or ensemble recordings may reduce accuracy

## Citation

[paper pending]

## License

MIT
