import json
import unittest
from pathlib import Path

from benchmark_platform import BenchmarkError, load_cases, run_adapter, run_prediction_file


class BenchmarkTests(unittest.TestCase):
    def complete_payload(self):
        cases = load_cases()["cases"]
        return {
            "source": "unit-test local fake predictor; not a quality claim",
            "predictions": [
                {"case_id": case["case_id"], "prediction": case["expected"], "latency_ms": 2.0}
                for case in cases
            ],
        }

    def write_payload(self, payload):
        path = Path(__file__).with_name("_predictions_test.json")
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.addCleanup(path.unlink, missing_ok=True)
        return path

    def test_prediction_file_reports_exact_computed_metrics_and_provenance(self):
        report = run_prediction_file(self.write_payload(self.complete_payload()))
        self.assertEqual(report["case_count"], 12)
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["latency_measurement_count"], 12)
        self.assertEqual(report["latency_ms"]["mean"], 2.0)
        self.assertEqual(len(report["prediction_file_sha256"]), 64)
        self.assertIn("unit-test local", report["source"])

    def test_missing_duplicate_and_extra_cases_are_rejected(self):
        payload = self.complete_payload()
        payload["predictions"].pop()
        with self.assertRaisesRegex(BenchmarkError, "missing"):
            run_prediction_file(self.write_payload(payload))
        payload = self.complete_payload()
        payload["predictions"].append(payload["predictions"][0])
        with self.assertRaisesRegex(BenchmarkError, "duplicate"):
            run_prediction_file(self.write_payload(payload))
        payload = self.complete_payload()
        payload["predictions"][-1]["case_id"] = "not-in-benchmark"
        with self.assertRaisesRegex(BenchmarkError, "unknown"):
            run_prediction_file(self.write_payload(payload))

    def test_invalid_latency_fails_without_clamping(self):
        payload = self.complete_payload()
        payload["predictions"][0]["latency_ms"] = -1
        with self.assertRaisesRegex(BenchmarkError, "non-negative"):
            run_prediction_file(self.write_payload(payload))

    def test_adapter_is_measured_and_resource_metric_is_explicitly_python_only(self):
        class Adapter:
            def predict(self, text):
                return "positive" if "love" in text.lower() or "wonderful" in text.lower() \
                    or "brilliant" in text.lower() or "fantastic" in text.lower() else \
                    ("negative" if any(word in text.lower() for word in ("hate", "terrible", "dreadful", "broken")) else "neutral")

        report = run_adapter(Adapter(), source="in-process test adapter", measure_resources=True)
        self.assertEqual(report["case_count"], 12)
        self.assertEqual(report["latency_measurement_count"], 12)
        self.assertIn("python_tracemalloc_peak_bytes", report)
        self.assertNotIn("gpu_memory_bytes", report)


if __name__ == "__main__":
    unittest.main()
