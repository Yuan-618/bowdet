# bowdet

**Audio-based bow-change and note-boundary detection for bowed string instruments.**

`bowdet` is an open-source Python toolkit for detecting fine-grained temporal boundaries in bowed-string performance directly from audio.

Its primary model, **BowDET**, detects audible boundaries between successive bowing actions using a two-stage log-Mel spectrogram-based framework. Version `0.2.1` also includes **BowDet-NB**, an experimental unsupervised method for detecting note boundaries in violin, viola, and cello recordings.

## Install

```bash
pip install bowdet
```

## Bow-change detection

```python
from bowdet import detect

times = detect("recording.wav")
print(times)  # array of predicted bow-change times in seconds
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

**BowDet-NB** detects note boundaries using an unsupervised multi-scale Mel-spectrogram boundary-strength method.

Unlike the bow-change detector, BowDet-NB uses only the spectral boundary-proposal stage and does not rely on the supervised Stage-2 classifier.

```python
from bowdet import detect_nb

times = detect_nb("recording.wav")
print(times)  # array of predicted note-boundary times in seconds
```

The full descriptive API is also available:

```python
from bowdet import detect_note_boundaries

times = detect_note_boundaries("recording.wav")
```

## Method

### Bow-change detection

BowDET uses a two-stage temporal boundary-detection pipeline:

1. **Stage 1 — Boundary Proposal**
   Computes local left-right differences in the log-Mel spectrogram and proposes candidate boundary positions at spectral change peaks.

2. **Stage 2 — Candidate Classification**
   A compact CNN, combined with 15 hand-crafted acoustic boundary features, classifies each candidate as a bow-change boundary or non-boundary.

The system is designed specifically for fine-grained analysis of sound-producing actions in bowed-string performance.

### BowDet-NB

BowDet-NB uses an unsupervised multi-scale version of the Stage-1 spectral boundary representation. Candidate note boundaries are estimated directly from local changes in the Mel spectrogram, without labelled training data or Stage-2 classification.

### Motivation

Bow changes and note transitions in bowed-string performance can produce short, broadband changes in spectral structure. BowDET models these local acoustic changes as temporal boundary events rather than treating performance as a sequence of fixed analysis frames.

## Research corpus

The current BowDET research corpus contains:

* **21 recordings**
* **12 performer groups**
* **Violin, viola, and cello**
* Approximately **72 minutes of performance**
* **3,138 manually annotated bow-change events**

Bow-change annotations were created using a Mel-spectrogram-assisted temporal annotation protocol. A subset of the annotations was independently checked against synchronised video and across annotators to assess the reliability of audio-based boundary labelling.

The corpus spans multiple performers and repertoire excerpts in order to evaluate generalisation across performance conditions rather than within individual recordings alone.

## Performance

### Bow-change detection

The current BowDET model was evaluated using performer-independent cross-validation across the research corpus.

| Metric           |             Score |
| ---------------- | ----------------: |
| Point F1 @100 ms | **0.655 ± 0.008** |
| IoU@0.1 F1       | **0.657 ± 0.008** |

Evaluation uses leave-one-performer-out-style separation to reduce performer leakage between training and testing.

### BowDet-NB note-boundary detection

BowDet-NB was separately evaluated on **9 bowed-string recordings** spanning violin, viola, and cello.

| Metric                   | Tonal mean F1 |
| ------------------------ | ------------: |
| Point F1 @50 ms          |     **0.701** |
| Point F1 @100 ms         |     **0.774** |
| Point F1 @150 ms         |     **0.793** |
| Pseudo-region IoU@0.1 F1 |     **0.812** |
| Pseudo-region IoU@0.3 F1 |     **0.803** |

The BowDet-NB benchmark is separate from the full BowDET bow-change research corpus described above.

## Research outputs

BowDET is being developed as both a research system and an accessible analysis toolkit for instrumental-performance research.

Current research outputs include work on:

* supervised audio-based bow-change boundary detection;
* temporal annotation and evaluation of fine-grained bowing actions;
* unsupervised note-boundary detection for bowed strings; and
* performer-independent evaluation of musical boundary-detection systems.

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
