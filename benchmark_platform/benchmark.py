from __future__ import annotations

import hashlib
import json
import math
import platform
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Any, Protocol


class BenchmarkError(ValueError):
    """Invalid case data or predictions; reports are never silently partial."""


class Predictor(Protocol):
    def predict(self, text: str) -> str: ...


_CASES_FILE = Path(__file__).with_name("cases.json")


def load_cases() -> dict[str, Any]:
    data = json.loads(_CASES_FILE.read_text(encoding="utf-8"))
    ids = [case["case_id"] for case in data["cases"]]
    if len(ids) != len(set(ids)):
        raise BenchmarkError("benchmark contains duplicate case IDs")
    if any(case["expected"] not in data["labels"] for case in data["cases"]):
        raise BenchmarkError("benchmark contains an unknown expected label")
    return data


def _latency(value: Any, case_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkError(f"{case_id}: latency_ms must be a non-negative number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise BenchmarkError(f"{case_id}: latency_ms must be finite and non-negative")
    return value


def summarize(rows: list[dict[str, Any]], source: str, *,
              resource_peak_bytes: int | None = None) -> dict[str, Any]:
    if not rows:
        raise BenchmarkError("cannot summarize an empty benchmark")
    correct = sum(row["correct"] for row in rows)
    latencies = [row["latency_ms"] for row in rows if row.get("latency_ms") is not None]
    report: dict[str, Any] = {
        "benchmark_version": load_cases()["benchmark_version"],
        "task": load_cases()["task"],
        "source": source,
        "python_version": platform.python_version(),
        "case_count": len(rows),
        "correct_count": correct,
        "accuracy": correct / len(rows),
        "latency_measurement_count": len(latencies),
        "latency_ms": None if not latencies else {
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "min": min(latencies),
            "max": max(latencies),
        },
        "cases": rows,
    }
    if resource_peak_bytes is not None:
        report["python_tracemalloc_peak_bytes"] = resource_peak_bytes
    return report


def _score(predictions: dict[str, Any], source: str,
           latencies: dict[str, float] | None = None, *,
           resource_peak_bytes: int | None = None) -> dict[str, Any]:
    data = load_cases()
    expected_ids = {case["case_id"] for case in data["cases"]}
    supplied_ids = set(predictions)
    missing, extra = expected_ids - supplied_ids, supplied_ids - expected_ids
    if missing or extra:
        details = []
        if missing:
            details.append("missing: " + ", ".join(sorted(missing)))
        if extra:
            details.append("unknown: " + ", ".join(sorted(extra)))
        raise BenchmarkError("prediction IDs do not match benchmark (" + "; ".join(details) + ")")
    rows = []
    for case in data["cases"]:
        case_id, prediction = case["case_id"], predictions[case["case_id"]]
        if not isinstance(prediction, str) or not prediction.strip():
            raise BenchmarkError(f"{case_id}: prediction must be a non-empty string")
        normalized = prediction.strip().casefold()
        rows.append({
            "case_id": case_id,
            "expected": case["expected"],
            "prediction": normalized,
            "correct": normalized == case["expected"],
            "latency_ms": None if latencies is None else latencies.get(case_id),
        })
    return summarize(rows, source, resource_peak_bytes=resource_peak_bytes)


def run_prediction_file(path: str | Path) -> dict[str, Any]:
    prediction_path = Path(path)
    raw = prediction_path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"invalid prediction JSON: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("source"), str) \
            or not payload["source"].strip() or not isinstance(payload.get("predictions"), list):
        raise BenchmarkError("expected object with non-empty 'source' and a 'predictions' list")
    predictions: dict[str, str] = {}
    latencies: dict[str, float] = {}
    for record in payload["predictions"]:
        if not isinstance(record, dict) or not isinstance(record.get("case_id"), str):
            raise BenchmarkError("every prediction must be an object with a string case_id")
        case_id = record["case_id"]
        if case_id in predictions:
            raise BenchmarkError(f"duplicate prediction for {case_id}")
        if "prediction" not in record:
            raise BenchmarkError(f"{case_id}: missing prediction")
        predictions[case_id] = record["prediction"]
        if "latency_ms" in record:
            latencies[case_id] = _latency(record["latency_ms"], case_id)
    report = _score(predictions, payload["source"].strip(), latencies)
    report["prediction_file_sha256"] = hashlib.sha256(raw).hexdigest()
    return report


def run_adapter(adapter: Predictor, *, source: str, measure_resources: bool = False) -> dict[str, Any]:
    """Execute a trusted caller-provided local adapter and measure its predictions."""
    if not isinstance(source, str) or not source.strip():
        raise BenchmarkError("source must describe the adapter/model and setup")
    data = load_cases()
    predictions: dict[str, str] = {}
    latencies: dict[str, float] = {}
    if measure_resources:
        tracemalloc.start()
    try:
        for case in data["cases"]:
            start = time.perf_counter()
            prediction = adapter.predict(case["text"])
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            predictions[case["case_id"]] = prediction
            latencies[case["case_id"]] = elapsed_ms
        peak_bytes = tracemalloc.get_traced_memory()[1] if measure_resources else None
    finally:
        if measure_resources:
            tracemalloc.stop()
    return _score(predictions, source.strip(), latencies, resource_peak_bytes=peak_bytes)
