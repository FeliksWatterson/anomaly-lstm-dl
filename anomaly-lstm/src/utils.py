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


# Pilot helpers. Importing this module does not initialize a CUDA context.
import os
import random
import time
import torch
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def seed_pilot(seed=42):
    """Fixed RNGs and deterministic FP32 backend settings; call before model init."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def timestamp_slots(timestamps):
    """Original wall-clock 5-minute slot, 0..287; never use window index as slot."""
    times = pd.DatetimeIndex(timestamps)
    if times.hasnans or not (times == times.floor("5min")).all():
        raise ValueError("Expected timestamps aligned to 5-minute slots")
    return np.asarray(times.hour * 12 + times.minute // 5, dtype=np.int64)


def fit_pilot_baselines(train_rows, scaler_state):
    """Fit both templates on unique original train rows, not overlapping windows."""
    if not train_rows.split.eq("train").all() or not train_rows.label.eq(0).all():
        raise ValueError("Baseline fit requires normal train rows only")
    if train_rows.duplicated(["source_id", "source_row"]).any():
        raise ValueError("Baseline fit rows must be unique")
    values = (train_rows.value.to_numpy() - scaler_state["mean"][0]) / scaler_state["scale"][0]
    slots = timestamp_slots(train_rows.timestamp)
    template = pd.Series(values).groupby(slots).median().reindex(range(288))
    counts = pd.Series(slots).value_counts().sort_index()
    if template.isna().any() or not counts.eq(8).all() or len(train_rows) != 2304:
        raise ValueError("Expected eight train observations per original daily slot")
    return {"mean_scaled": float(values.mean()), "median_scaled_by_slot": template.tolist(),
            "fit_rows": len(train_rows), "fit_split": "train", "slot_counts": counts.tolist(),
            "fit_source_rows": [int(train_rows.source_row.min()), int(train_rows.source_row.max())]}


def baseline_reconstruct(mapping, baseline_state, method, length=288):
    """Construct full-window reconstructions on the same mapped windows as the AE."""
    if method == "train_mean":
        return np.full((len(mapping), length, 1), baseline_state["mean_scaled"], dtype=np.float32)
    if method != "seasonal_median":
        raise ValueError("Unknown baseline")
    start_slots = timestamp_slots(mapping.timestamp_start)
    slots = (start_slots[:, None] + np.arange(length)[None, :]) % 288
    np.testing.assert_array_equal(slots[:, -1], timestamp_slots(mapping.timestamp_end))
    template = np.asarray(baseline_state["median_scaled_by_slot"], dtype=np.float32)
    return template[slots, None]


class LSTMAutoencoder(torch.nn.Module):
    """32-unit bottleneck, repeated latent -> independent decoder -> linear output.

    Decoder only sees repeated encoder hidden vector, with zero initial state.
    There is no input shortcut, teacher forcing or per-timestep encoder output.
    """
    def __init__(self, length=288, features=1, hidden=32):
        super().__init__()
        self.length, self.features, self.hidden = length, features, hidden
        self.encoder = torch.nn.LSTM(features, hidden, batch_first=True)
        self.decoder = torch.nn.LSTM(hidden, hidden, batch_first=True)
        self.output = torch.nn.Linear(hidden, features)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.length, self.features):
            raise ValueError("Unexpected autoencoder input shape")
        _, (hidden, _) = self.encoder(x)
        repeated = hidden[-1].unsqueeze(1).repeat(1, self.length, 1)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded)


def reconstruct_lstm(model, windows, device, batch_size=32):
    """Order-preserving, deterministic inference, returning full reconstruction."""
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(windows[start:start+batch_size])).to(device)
            chunks.append(model(batch).cpu().numpy())
    output = np.concatenate(chunks)
    if output.shape != windows.shape or not np.isfinite(output).all():
        raise ValueError("Invalid reconstruction")
    return output


def train_pilot_lstm(train_windows, validation_windows, config, device, results):
    """Only train/validation enter this function. Strict improvement, patience=5.

    Saves a state_dict checkpoint on validation improvement and history every epoch.
    Never overwrites a previous pilot. Checkpoint is restored before return.
    """
    from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
    results = Path(results)
    checkpoint = results / "pilot_best.pt"
    history_path = results / "pilot_history.csv"
    if checkpoint.exists() or history_path.exists():
        raise FileExistsError("Pilot already exists; review it before starting another training run")
    seed_pilot(config["seed"])
    model = LSTMAutoencoder(config["length"], config["features"], config["hidden"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.MSELoss(reduction="mean")
    loader = DataLoader(TensorDataset(torch.from_numpy(train_windows)),
                        batch_size=config["batch_size"], shuffle=False, num_workers=0)
    assert isinstance(loader.sampler, SequentialSampler)
    validation = DataLoader(TensorDataset(torch.from_numpy(validation_windows)),
                            batch_size=config["batch_size"], shuffle=False, num_workers=0)
    best, bad_epochs, best_epoch = float("inf"), 0, None
    history = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for epoch in range(1, config["max_epochs"] + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_sum, train_count = 0., 0
        for (batch_cpu,) in loader:
            batch = batch_cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            train_sum += loss.item() * batch.numel()
            train_count += batch.numel()
        model.eval()
        val_sum, val_count = 0., 0
        with torch.inference_mode():
            for (batch_cpu,) in validation:
                batch = batch_cpu.to(device)
                loss = criterion(model(batch), batch)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite validation loss")
                val_sum += loss.item() * batch.numel()
                val_count += batch.numel()
        train_loss, val_loss = train_sum / train_count, val_sum / val_count
        improved = val_loss < best  # min_delta = 0; validation only
        if improved:
            best, best_epoch, bad_epochs = val_loss, epoch, 0
            torch.save({"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "epoch": epoch, "validation_mse": best, "config": config}, checkpoint)
        else:
            bad_epochs += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        history.append({"epoch": epoch, "train_mse": train_loss, "validation_mse": val_loss,
                        "improved": improved, "bad_epochs": bad_epochs,
                        "seconds": time.perf_counter() - epoch_start})
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(f"Epoch {epoch:02d}: train={train_loss:.8f}, val={val_loss:.8f}, best={best_epoch}, stale={bad_epochs}", flush=True)
        if bad_epochs >= config["patience"]:
            break
    elapsed = time.perf_counter() - started
    restored = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(restored["state_dict"])
    model.eval().requires_grad_(False)
    summary = {"best_epoch": best_epoch, "best_validation_mse": best, "epochs_run": len(history),
               "training_seconds": elapsed, "stopped_early": len(history) < config["max_epochs"],
               "parameter_count": sum(p.numel() for p in model.parameters())}
    return model, pd.DataFrame(history), summary


def endpoint_metrics(labels, scores, threshold):
    """Metrics on original endpoint labels. AP is average_precision_score, not trapezoidal PR area."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    if labels.shape != scores.shape or labels.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Misaligned or nonfinite scores")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(threshold):
        raise ValueError("Invalid labels/threshold")
    predicted = (scores > threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {"Precision": float(precision_score(labels, predicted, zero_division=0)),
            "Recall": float(recall_score(labels, predicted, zero_division=0)),
            "F1": float(f1_score(labels, predicted, zero_division=0)),
            "Average Precision (AP)": float(average_precision_score(labels, scores)) if labels.any() else None,
            "ROC-AUC": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "threshold": float(threshold), "n_evaluated": len(labels), "n_positive": int(labels.sum()),
            "predicted_positive": int(predicted.sum()), "predicted_positive_rate": float(predicted.mean())}
