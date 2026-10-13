# AI Model Benchmarking Platform

A reproducible, offline harness for a fixed, small sentiment-classification benchmark. It scores predictions supplied by a local model adapter or a prediction file; it does not ship a model, generate predictions, call external APIs, or claim quality that was not measured. The included cases are short, synthetic examples with transparent reference labels, not a comprehensive or representative evaluation.

## Prediction-file workflow

1. Run your own local model and save one prediction for **each** benchmark `case_id` using the JSON shape below. Optional `latency_ms` values must be measured by your inference process.
2. Run `python -m benchmark_platform predictions.json --output results.json`.
3. Inspect the report's source, case count, exact accuracy, and latency coverage/summary. Missing or duplicate case IDs fail closed.

```json
{
  "source": "local model name, version, and setup description",
  "predictions": [
    {"case_id": "sentiment-01", "prediction": "positive", "latency_ms": 1.8}
  ]
}
```

Labels are `positive`, `negative`, and `neutral`. See `benchmark_platform/cases.json` for the full case set and references. Treat this file as a local integration artifact, not a model-quality claim.

## Adapter workflow

The library's `run_adapter(adapter, source=...)` accepts an object with a synchronous `predict(text)` method. This deliberately executes the caller-supplied adapter in the current process; use only trusted, local adapter code and disable any networking in its own configuration. Per-case latency is measured around each call with `perf_counter`. `measure_resources=True` reports Python `tracemalloc` peak allocations for the adapter run (not total process, GPU, or system memory).

```python
from benchmark_platform import run_adapter

class MyLocalAdapter:
    def predict(self, text):
        return my_local_model.predict(text)

report = run_adapter(MyLocalAdapter(), source="local checkpoint sha ...")
```

No credentials or external services are needed. Each report includes the benchmark version, source description, Python version, per-case correctness, and computed summary. Predictions and latency values are not invented by the harness.

## Run and test

```powershell
python -m benchmark_platform predictions.json
python -m unittest discover -s tests -v
```
