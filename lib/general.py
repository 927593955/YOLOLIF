from pathlib import Path
from typing import Any, Dict
import torch

def print_metrics(title: str, metrics: Any) -> None:
    print(f"\n=== {title} ===")
    if metrics is None:
        print("No metrics returned.")
        return

    if hasattr(metrics, "results_dict"):
        metrics = metrics.results_dict

    if isinstance(metrics, dict):
        for key in sorted(metrics.keys()):
            value = metrics[key]
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")
        return

    print(metrics)


def resolve_device(device_arg: str) -> str:
    """Resolve device with CUDA-first behavior for torch environments."""
    if not device_arg or device_arg == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    return device_arg


def resolve_local_weights(weights_arg: str) -> str:
    if not weights_arg:
        raise ValueError("Please provide local pretrained weights with --weights.")

    weights_path = Path(weights_arg).expanduser().resolve()
    if not weights_path.exists() or not weights_path.is_file():
        raise FileNotFoundError(f"Local pretrained weights not found: {weights_path}")

    if weights_path.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError(f"Expected .pt/.pth weights, got: {weights_path}")

    return str(weights_path)

def _parse_phrase_types(raw: str) -> list[str]:
    values = [x.strip().upper() for x in str(raw or "").split(",") if x.strip()]
    return values or ["NP", "PP", "ADJP"]


def _parse_phrase_weight_string(raw: str) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for item in str(raw or "").split(","):
        part = item.strip()
        if not part or ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip().upper()
        value = value.strip()
        if not key:
            continue
        try:
            score = float(value)
        except ValueError:
            continue
        if score > 0:
            weights[key] = score
    return weights