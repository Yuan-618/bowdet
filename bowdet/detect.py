"""
bowdet - Audio-based bow-change detection for string instruments
Two-stage mel-spectrogram boundary approach
"""

import json
import numpy as np
from pathlib import Path

_MODEL      = None
_FEAT_MEAN  = None
_FEAT_STD   = None
_CONFIG     = None
_ASSETS_DIR = Path(__file__).parent / "assets"


def _load():
    global _MODEL, _FEAT_MEAN, _FEAT_STD, _CONFIG
    if _MODEL is not None:
        return

    import tensorflow as tf

    with open(_ASSETS_DIR / "config.json") as f:
        _CONFIG = json.load(f)

    _MODEL     = tf.keras.models.load_model(str(_ASSETS_DIR / "model.keras"))
    _FEAT_MEAN = np.load(str(_ASSETS_DIR / "feat_mean.npy"))
    _FEAT_STD  = np.load(str(_ASSETS_DIR / "feat_std.npy"))


def _smooth_1d(x, win_size=5):
    kernel = np.ones(win_size) / win_size
    return np.convolve(x, kernel, mode="same")


def _compute_logmel(y, sr, cfg):
    import librosa
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        n_mels=cfg["n_mels"],
        power=2.0
    )
    logmel = librosa.power_to_db(mel, ref=np.max).T.astype(np.float32)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-8)
    frame_times = np.arange(logmel.shape[0]) * (cfg["hop_length"] / sr)
    return logmel, frame_times


def _boundary_strength(logmel):
    diff     = np.diff(logmel, axis=0)
    strength = np.sqrt(np.sum(diff ** 2, axis=1))
    strength = np.concatenate([[0.0], strength])
    strength = _smooth_1d(strength, win_size=5)
    min_v, max_v = strength.min(), strength.max()
    if max_v > min_v:
        strength = (strength - min_v) / (max_v - min_v)
    else:
        strength = np.zeros_like(strength)
    return strength


def _pick_candidates(frame_times, strength, cfg):
    frame_sec       = cfg["hop_length"] / cfg["sr"]
    threshold       = np.percentile(strength, cfg["peak_percentile"])
    min_dist_frames = max(1, int(cfg["min_peak_distance_sec"] / frame_sec))
    peaks = []
    for i in range(1, len(strength) - 1):
        if strength[i] < threshold:
            continue
        if strength[i] >= strength[i-1] and strength[i] >= strength[i+1]:
            if not peaks:
                peaks.append(i)
            else:
                last = peaks[-1]
                if i - last >= min_dist_frames:
                    peaks.append(i)
                elif strength[i] > strength[last]:
                    peaks[-1] = i
    return frame_times[peaks], np.array(peaks, dtype=int)


def _safe_slice(arr, start, end):
    start = max(0, start)
    end   = min(len(arr), end)
    if end <= start:
        return arr[0:0]
    return arr[start:end]


def _extract_features(logmel, strength, candidate_indices, cfg):
    feats        = []
    n_mels       = cfg["n_mels"]
    frame_sec    = cfg["hop_length"] / cfg["sr"]
    local_radius = 5
    side_frames  = max(2, int(0.25 / frame_sec))
    raw_diff     = np.diff(logmel, axis=0)
    flux         = np.sqrt(np.sum(raw_diff ** 2, axis=1))
    flux         = np.concatenate([[0.0], flux])

    for idx in candidate_indices:
        idx      = int(idx)
        local_s  = _safe_slice(strength, idx - local_radius, idx + local_radius + 1)
        s_center = float(strength[idx]) if 0 <= idx < len(strength) else 0.0
        s_mean   = float(np.mean(local_s)) if len(local_s) else 0.0
        s_max    = float(np.max(local_s))  if len(local_s) else 0.0
        s_std    = float(np.std(local_s))  if len(local_s) else 0.0

        pre  = _safe_slice(logmel, idx - side_frames, idx)
        post = _safe_slice(logmel, idx + 1, idx + 1 + side_frames)

        if len(pre) == 0 or len(post) == 0:
            mel_diff_mean = mel_diff_l2 = low_diff = mid_diff = 0.0
            high_diff = std_diff = energy_diff = 0.0
        else:
            pre_mean  = np.mean(pre,  axis=0)
            post_mean = np.mean(post, axis=0)
            dv            = post_mean - pre_mean
            mel_diff_mean = float(np.mean(np.abs(dv)))
            mel_diff_l2   = float(np.sqrt(np.sum(dv ** 2)))
            low_diff      = float(np.mean(np.abs(dv[:n_mels // 3])))
            mid_diff      = float(np.mean(np.abs(dv[n_mels // 3: 2 * n_mels // 3])))
            high_diff     = float(np.mean(np.abs(dv[2 * n_mels // 3:])))
            std_diff      = float(abs(np.std(post) - np.std(pre)))
            energy_diff   = float(np.mean(post) - np.mean(pre))

        local_flux  = _safe_slice(flux, idx - local_radius, idx + local_radius + 1)
        flux_center = float(flux[idx]) if 0 <= idx < len(flux) else 0.0
        flux_mean   = float(np.mean(local_flux)) if len(local_flux) else 0.0
        flux_max    = float(np.max(local_flux))  if len(local_flux) else 0.0
        flux_std    = float(np.std(local_flux))  if len(local_flux) else 0.0

        feats.append([
            s_center, s_mean, s_max, s_std,
            mel_diff_mean, mel_diff_l2,
            low_diff, mid_diff, high_diff,
            std_diff, energy_diff,
            flux_center, flux_mean, flux_max, flux_std
        ])

    return np.array(feats, dtype=np.float32)


def _extract_windows(logmel, candidate_indices, cfg):
    half_win = cfg["win_frames"] // 2
    X, kept  = [], []
    n_frames = logmel.shape[0]
    for k, center in enumerate(candidate_indices):
        start = center - half_win
        end   = center + half_win
        if start < 0 or end > n_frames:
            continue
        window = logmel[start:end]
        if window.shape[0] != cfg["win_frames"]:
            continue
        X.append(window[..., np.newaxis])
        kept.append(k)
    return np.array(X, dtype=np.float32), np.array(kept, dtype=int)


def detect(audio_path, threshold=None, min_dist_sec=None):
    """
    Detect bow changes in a string instrument recording.

    Parameters
    ----------
    audio_path : str or Path
        Path to audio file (wav, mp3, flac, m4a, etc.)
    threshold : float, optional
        Classification threshold (default: 0.40)
    min_dist_sec : float, optional
        Minimum distance between detected events in seconds (default: 0.12)

    Returns
    -------
    times : np.ndarray
        Array of detected bow-change times in seconds.

    Example
    -------
    >>> from bowdet import detect
    >>> times = detect("recording.wav")
    >>> print(times)
    """
    import librosa

    _load()
    cfg = _CONFIG.copy()
    cfg["sr"] = cfg.get("sr", 22050)
    if threshold    is not None: cfg["classify_threshold"]    = threshold
    if min_dist_sec is not None: cfg["min_peak_distance_sec"] = min_dist_sec

    y, sr = librosa.load(str(audio_path), sr=cfg["sr"], mono=True)

    logmel, frame_times  = _compute_logmel(y, sr, cfg)
    strength             = _boundary_strength(logmel)
    cand_times, cand_idx = _pick_candidates(frame_times, strength, cfg)

    if len(cand_idx) == 0:
        return np.array([])

    X_img, kept  = _extract_windows(logmel, cand_idx, cfg)
    X_feat       = _extract_features(logmel, strength, cand_idx, cfg)
    X_feat       = X_feat[kept]
    cand_times   = cand_times[kept]

    if len(X_img) == 0:
        return np.array([])

    X_feat_std = (X_feat - _FEAT_MEAN) / _FEAT_STD
    probs      = _MODEL.predict([X_img, X_feat_std], verbose=0).reshape(-1)
    pred_mask  = probs >= cfg["classify_threshold"]
    pred_times = cand_times[pred_mask]
    pred_probs = probs[pred_mask]

    if len(pred_times) == 0:
        return np.array([])

    # NMS
    order      = np.argsort(pred_times)
    pred_times = pred_times[order]
    pred_probs = pred_probs[order]
    selected   = [0]
    for i in range(1, len(pred_times)):
        if pred_times[i] - pred_times[selected[-1]] >= cfg["min_peak_distance_sec"]:
            selected.append(i)
        elif pred_probs[i] > pred_probs[selected[-1]]:
            selected[-1] = i

    return pred_times[selected]
