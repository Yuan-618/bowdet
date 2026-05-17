"""
bowdet - Bow change detection for bowed string instruments
Main detection function
"""

import os
from pathlib import Path

import numpy as np
import torch
import librosa
import soundfile as sf
from scipy.signal import find_peaks
from huggingface_hub import hf_hub_download


# ========== Weight Download ==========

HF_REPO_ID = "Haotian-Yuan/bowdet"


def _get_weights_dir():
    weights_dir = Path.home() / ".bowdet" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    return weights_dir


def _download_weight(filename):
    weights_dir = _get_weights_dir()
    weight_path = weights_dir / filename
    if not weight_path.exists():
        print(f"Downloading {filename} from Hugging Face Hub...")
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            local_dir=str(weights_dir),
        )
        print(f"Weights saved to {weight_path}")
    return weight_path


# ========== CNN Inference (parameters match training exactly) ==========

CNN_SR         = 22050
CNN_N_MELS     = 80
CNN_HOP        = 220
CNN_WIN        = 512
CNN_N_FFT      = 512
CNN_WIN_SEC    = 1.0
CNN_STRIDE_SEC = 0.1


def _infer_cnn(audio, model, threshold, min_dist):
    win_samples    = int(CNN_WIN_SEC * CNN_SR)
    stride_samples = int(CNN_STRIDE_SEC * CNN_SR)
    times, probs   = [], []

    model.eval()
    with torch.no_grad():
        start = 0
        while start + win_samples <= len(audio):
            chunk = audio[start: start + win_samples].astype(np.float32)

            # Mel spectrogram — identical to training
            mel = librosa.feature.melspectrogram(
                y=chunk, sr=CNN_SR,
                n_mels=CNN_N_MELS, n_fft=CNN_N_FFT,
                hop_length=CNN_HOP, win_length=CNN_WIN
            )
            mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
            mel = (mel - mel.mean()) / (mel.std() + 1e-9)

            x = torch.tensor(mel).unsqueeze(0).unsqueeze(0)  # [1, 1, 80, T]
            prob = torch.sigmoid(model(x)).item()
            times.append((start + win_samples / 2) / CNN_SR)
            probs.append(prob)
            start += stride_samples

    times  = np.array(times)
    probs  = np.array(probs)
    min_dist_frames = max(1, int(min_dist / CNN_STRIDE_SEC))
    peaks, _ = find_peaks(probs, height=threshold, distance=min_dist_frames)
    return times[peaks].tolist()


# ========== MERT Inference (parameters match training exactly) ==========

MERT_SR         = 24000
MERT_WIN_SEC    = 1.0
MERT_STRIDE_SEC = 0.1


def _infer_mert(audio, model, processor, threshold, min_dist):
    win_samples    = int(MERT_WIN_SEC * MERT_SR)
    stride_samples = int(MERT_STRIDE_SEC * MERT_SR)
    times, probs   = [], []

    model.eval()
    with torch.no_grad():
        start = 0
        while start + win_samples <= len(audio):
            chunk  = audio[start: start + win_samples].astype(np.float32)
            inputs = processor(chunk, sampling_rate=MERT_SR, return_tensors="pt")
            logit  = model(inputs["input_values"])
            probs.append(torch.sigmoid(logit).item())
            times.append((start + win_samples / 2) / MERT_SR)
            start += stride_samples

    times  = np.array(times)
    probs  = np.array(probs)
    min_dist_frames = max(1, int(min_dist / MERT_STRIDE_SEC))
    peaks, _ = find_peaks(probs, height=threshold, distance=min_dist_frames)
    return times[peaks].tolist()


# ========== Public API ==========

def detect(audio_path, model="mert", threshold=0.5, min_dist=0.35):
    """
    Detect bow changes in a bowed string instrument recording.

    Parameters
    ----------
    audio_path : str
        Path to audio file (wav recommended).
    model : str, default="mert"
        Model to use: "mert" (more accurate, IoU@0.1 F1=0.616) or
        "cnn" (faster, IoU@0.1 F1=0.554).
    threshold : float, default=0.5
        Peak detection threshold (0-1).
        Lower values detect more bow changes but may increase false positives.
    min_dist : float, default=0.35
        Minimum time between bow changes in seconds.
        Decrease for fast passages (e.g. spiccato): min_dist=0.2
        Increase for slow passages (e.g. long bows): min_dist=0.5

    Returns
    -------
    list of float
        Timestamps of detected bow changes in seconds.

    Examples
    --------
    >>> from bowdet import detect
    >>> bow_changes = detect("recording.wav")
    >>> print(bow_changes)
    [1.23, 2.45, 3.67, ...]

    >>> # Use CNN model (faster, less accurate)
    >>> bow_changes = detect("recording.wav", model="cnn")

    >>> # Custom parameters for fast spiccato passages
    >>> bow_changes = detect("recording.wav", threshold=0.4, min_dist=0.2)
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if model == "cnn":
        from .model_cnn import BowCNN

        weight_path = _download_weight("BowDET-C.pth")
        cnn = BowCNN()
        ckpt = torch.load(str(weight_path), map_location="cpu", weights_only=False)
        cnn.load_state_dict(ckpt["model_state_dict"])
        cnn.eval()

        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != CNN_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=CNN_SR)

        return _infer_cnn(audio, cnn, threshold, min_dist)

    elif model == "mert":
        from .model_mert import MERTClassifier
        from transformers import AutoProcessor

        weight_path = _download_weight("BowDET-M.pth")
        processor = AutoProcessor.from_pretrained(
            "m-a-p/MERT-v1-95M", trust_remote_code=True
        )
        mert = MERTClassifier()
        ckpt = torch.load(str(weight_path), map_location="cpu", weights_only=False)
        mert.load_state_dict(ckpt["model_state_dict"])
        mert.eval()

        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != MERT_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=MERT_SR)

        return _infer_mert(audio, mert, processor, threshold, min_dist)

    else:
        raise ValueError(f"Unknown model '{model}'. Choose 'mert' or 'cnn'.")