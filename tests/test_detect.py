"""
Test suite for bowdet's public API (detect, detect_nb, detect_note_boundaries).

Uses short synthetic audio signals generated on the fly (no fixture audio files
required), so the suite is self-contained and runs anywhere `pytest` runs.
"""

import numpy as np
import pytest
import soundfile as sf

from bowdet import detect, detect_nb, detect_note_boundaries

SR = 22050


def _write_tone_with_transient(path, duration=3.0, sr=SR):
    """Sine tone with a phase-flip discontinuity partway through, simulating
    a single sharp spectral transition (e.g. a bow change)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    audio[int(sr * duration / 2):] *= -1.0
    sf.write(path, audio.astype("float32"), sr)
    return path


def _write_silence(path, duration=2.0, sr=SR):
    sf.write(path, np.zeros(int(sr * duration), dtype="float32"), sr)
    return path


@pytest.fixture
def transient_wav(tmp_path):
    return str(_write_tone_with_transient(tmp_path / "transient.wav"))


@pytest.fixture
def silence_wav(tmp_path):
    return str(_write_silence(tmp_path / "silence.wav"))


class TestDetectOutputContract:
    """Basic output-shape / type contract for detect()."""

    def test_returns_array(self, transient_wav):
        times = detect(transient_wav)
        assert isinstance(times, np.ndarray)

    def test_results_within_audio_duration(self, transient_wav):
        times = detect(transient_wav)
        assert all(0.0 <= t <= 3.0 for t in times)

    def test_results_sorted_ascending(self, transient_wav):
        times = detect(transient_wav)
        assert list(times) == sorted(times)


class TestDetectBehaviour:
    """Functional behaviour of detect() against known signal properties."""

    def test_silence_produces_no_detections(self, silence_wav):
        times = detect(silence_wav)
        assert len(times) == 0

    def test_higher_threshold_does_not_increase_detections(self, transient_wav):
        """Raising the classification threshold should never surface more
        candidates than the default threshold."""
        default_times = detect(transient_wav)
        strict_times = detect(transient_wav, threshold=0.9)
        assert len(strict_times) <= len(default_times)

    def test_short_clip_does_not_raise(self, tmp_path):
        """Sub-frame-length audio should degrade gracefully, not crash."""
        short_path = tmp_path / "short.wav"
        t = np.linspace(0, 0.05, int(SR * 0.05), endpoint=False)
        sf.write(short_path, (0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32"), SR)
        times = detect(str(short_path))
        assert isinstance(times, np.ndarray)


class TestDetectErrorHandling:
    """Failure modes that calling code needs to be able to rely on."""

    def test_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            detect("this_file_does_not_exist_123.wav")


class TestNoteBoundaryDetection:
    """detect_nb / detect_note_boundaries share the same underlying method;
    check their output contract and that both entry points agree."""

    def test_returns_sorted_array(self, transient_wav):
        times = detect_nb(transient_wav)
        assert isinstance(times, np.ndarray)
        assert list(times) == sorted(times)

    def test_respects_min_dist_sec_approximately(self, transient_wav):
        """Consecutive detections should be spaced at roughly min_dist_sec
        apart. Frame quantisation means this isn't exact, so we allow a
        small tolerance rather than asserting a hard floor."""
        min_dist = 0.12
        times = detect_nb(transient_wav, min_dist_sec=min_dist)
        gaps = np.diff(times)
        if len(gaps):
            assert gaps.min() >= min_dist * 0.9

    def test_detect_note_boundaries_matches_detect_nb(self, transient_wav):
        """detect_note_boundaries is documented as the descriptive alias for
        detect_nb; both should return identical results for the same input."""
        a = detect_nb(transient_wav)
        b = detect_note_boundaries(transient_wav)
        np.testing.assert_array_equal(a, b)
