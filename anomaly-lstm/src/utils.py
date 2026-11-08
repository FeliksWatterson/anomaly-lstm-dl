from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

NAB_COMMIT = "ea702d75cc2258d9d7dd35ca8e5e2539d71f3140"
NORMAL = "artificialNoAnomaly/art_daily_small_noise.csv"
TEST = "artificialWithAnomaly/art_daily_jumpsup.csv"
FILES = {
    "normal.csv": f"data/{NORMAL}",
    "test.csv": f"data/{TEST}",
    "combined_windows.json": "labels/combined_windows.json",
    "combined_labels.json": "labels/combined_labels.json",
    "NAB_README.md": "README.md",
    "NAB_LICENSE.txt": "LICENSE.txt",
}


def project_root():
    """Find project from either root or notebooks/, without changing cwd."""
    for path in (Path.cwd(), *Path.cwd().parents):
        if (path / "src" / "utils.py").is_file():
            return path
    raise FileNotFoundError("Run inside anomaly-lstm")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download_nab(root):
    """Pinned upstream files; reuse and verify cache; never replace unknown files."""
    raw = Path(root) / "data" / "raw" / "nab"
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = raw / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["commit"] != NAB_COMMIT:
            raise ValueError("Cached commit differs")
        for name in FILES:
            record = manifest["files"][name]
            if sha256(raw / name) != record["sha256"]:
                raise ValueError(f"Corrupt cached file: {name}")
        return raw, manifest
    conflicts = [name for name in FILES if (raw / name).exists()]
    if conflicts:
        raise FileExistsError(f"Untracked cached files; inspect manually: {conflicts}")
    # Fetch all first so network errors leave no partial dataset.
    fetched = {}
    for name, remote in FILES.items():
        url = f"https://raw.githubusercontent.com/numenta/NAB/{NAB_COMMIT}/{remote}"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        fetched[name] = (url, response.content)
    manifest = {"repository": "https://github.com/numenta/NAB", "commit": NAB_COMMIT,
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "files": {}}
    for name, (url, content) in fetched.items():
        (raw / name).write_bytes(content)
        manifest["files"][name] = {"url": url, "bytes": len(content),
                                    "sha256": hashlib.sha256(content).hexdigest()}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return raw, manifest


def load_series(raw, filename, source_id):
    """Preserve original order and row index. Labels are inclusive NAB intervals."""
    frame = pd.read_csv(Path(raw) / filename)
    if list(frame.columns) != ["timestamp", "value"]:
        raise ValueError("Unexpected schema")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame["value"] = pd.to_numeric(frame["value"], errors="raise")
    intervals = json.loads((Path(raw) / "combined_windows.json").read_text())[source_id]
    frame["label"] = np.zeros(len(frame), dtype=np.int8)
    for start, end in intervals:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if start > end or start < frame.timestamp.min() or end > frame.timestamp.max():
            raise ValueError("Label interval outside source")
        mask = frame.timestamp.between(start, end, inclusive="both")
        if not mask.any():
            raise ValueError("Empty label interval")
        frame.loc[mask, "label"] = 1
    frame["source_id"] = source_id
    frame["source_row"] = np.arange(len(frame))
    return frame


def audit_series(frame):
    times = frame.timestamp
    gaps = times.diff().dropna()
    return {
        "rows": len(frame), "features": 1,
        "normal": int((frame.label == 0).sum()), "anomaly": int((frame.label == 1).sum()),
        "missing_cells": int(frame[["timestamp", "value", "label"]].isna().sum().sum()),
        "nonfinite_values": int((~np.isfinite(frame.value)).sum()),
        "duplicate_rows": int(frame.duplicated(["timestamp", "value"]).sum()),
        "duplicate_timestamps": int(times.duplicated().sum()),
        "repeated_values": int(frame.value.duplicated().sum()),
        "ordered": bool(times.is_monotonic_increasing),
        "non_5min_gaps": int((gaps != pd.Timedelta(minutes=5)).sum()),
        "start": str(times.min()), "end": str(times.max()),
    }


def chronological_splits(normal, test):
    """Custom split by whole days: 8 train, 3 validation, 3 calibration."""
    if len(normal) != 14 * 288:
        raise ValueError("Review split policy for changed dataset length")
    parts = {"train": normal.iloc[:2304].copy(),
             "validation": normal.iloc[2304:3168].copy(),
             "calibration": normal.iloc[3168:].copy(),
             "test": test.copy()}
    for name, part in parts.items():
        report = audit_series(part)
        if any(report[k] for k in ["missing_cells", "nonfinite_values", "duplicate_timestamps", "non_5min_gaps"]):
            raise ValueError(f"Invalid series: {name}")
        if not report["ordered"] or part.source_id.nunique() != 1:
            raise ValueError("Each split must contain one ordered independent series")
        if name != "test" and part.label.ne(0).any():
            raise ValueError(f"Anomaly in normal-only split: {name}")
        part["split"] = name
    return parts


def make_windows(frame, values, length=288, stride=1):
    """Single split/series only. End is inclusive; source rows survive slicing."""
    values = np.asarray(values)
    if length < 1 or stride < 1 or values.ndim != 2 or len(values) != len(frame):
        raise ValueError("Invalid window arguments")
    if not np.isfinite(values).all() or len(frame) < length:
        raise ValueError("Insufficient or nonfinite data")
    if frame.source_id.nunique() != 1 or frame.split.nunique() != 1:
        raise ValueError("Do not mix splits or independent series")
    if not frame.timestamp.diff().dropna().eq(pd.Timedelta(minutes=5)).all():
        raise ValueError("Non-contiguous timestamps")
    if not frame.source_row.diff().dropna().eq(1).all():
        raise ValueError("Non-contiguous source rows")
    starts = np.arange(0, len(frame) - length + 1, stride)
    ends = starts + length - 1
    windows = np.stack([values[i:i + length] for i in starts]).astype(np.float32)
    mapping = pd.DataFrame({
        "window_id": np.arange(len(starts)), "split": frame.split.iloc[0],
        "source_id": frame.source_id.iloc[0], "local_start": starts, "local_end": ends,
        "source_start": frame.source_row.iloc[starts].to_numpy(),
        "source_end": frame.source_row.iloc[ends].to_numpy(),
        "timestamp_start": frame.timestamp.iloc[starts].to_numpy(),
        "timestamp_end": frame.timestamp.iloc[ends].to_numpy(),
        "label_endpoint": frame.label.iloc[ends].to_numpy(),
        "label_any": [int(frame.label.iloc[i:i + length].max()) for i in starts],
    })
    return windows, mapping


def endpoint_mse(observed, reconstructed):
    """Causal endpoint score, one score per mapped window end (no future overlap)."""
    observed, reconstructed = np.asarray(observed), np.asarray(reconstructed)
    if observed.shape != reconstructed.shape or observed.ndim != 3:
        raise ValueError("Expected equal (windows, time, features) shapes")
    if not np.isfinite(observed).all() or not np.isfinite(reconstructed).all():
        raise ValueError("Nonfinite reconstruction")
    return np.mean((observed[:, -1, :] - reconstructed[:, -1, :]) ** 2, axis=-1)


def normal_threshold(calibration_scores, quantile=0.99):
    """Only normal calibration scores; caller must enforce provenance."""
    scores = np.asarray(calibration_scores)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Expected finite, nonempty scores")
    if not 0 < quantile < 1:
        raise ValueError("Quantile must be between zero and one")
    return float(np.quantile(scores, quantile, method="higher"))
