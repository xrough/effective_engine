import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import walk_forward_pipeline as wf


def write_panel(path: Path, dates: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_utc"])
        for d in dates:
            writer.writerow([f"{d}T13:31:00+00:00"])
            writer.writerow([f"{d}T13:32:00+00:00"])


def write_valid_artifacts(path: Path) -> None:
    (path / "checkpoints").mkdir(parents=True, exist_ok=True)
    (path / "neural_bsde.onnx").write_bytes(b"onnx")
    (path / "Y0_init.json").write_text('{"Y0": 1.0, "state_dim": 7}\n')
    (path / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
    norm = {
        "feature_order": ["tau", "log_moneyness", "V_t", "U1", "U2", "U3", "U4"],
        "mean": [0.0] * 7,
        "std": [1.0] * 7,
        "state_dim": 7,
        "m": 4,
        "K": 637.0,
        "T": 0.05,
        "r": 0.05,
        "model_params": {"theta": 0.02},
    }
    (path / "normalization.json").write_text(json.dumps(norm))


def passing_retrain(**kwargs):
    artifacts_dir = Path(kwargs["artifacts_dir"])
    write_valid_artifacts(artifacts_dir)
    return {
        "train": {
            "delta_sanity": {
                "model_delta": 0.51,
                "bs_delta": 0.50,
                "error": 0.01,
            }
        },
        "export": {
            "validated": True,
            "validation_passed": True,
            "max_err_Y": 1e-7,
            "max_err_Z": 1e-7,
        },
    }


def passing_replay(**kwargs):
    results_csv = Path(kwargs["results_csv"])
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    pnl = 100.0 if kwargs["hedger"] == "neural" else 90.0
    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "total_pnl", "n_fills"])
        writer.writerow([kwargs["deploy_date"], pnl, 2])
    return {
        "returncode": 0,
        "command": ["fake"],
        "metrics": wf.parse_replay_csv(results_csv),
        "log_path": str(kwargs["log_path"]),
    }


class WalkForwardPipelineTests(unittest.TestCase):
    def test_daily_expanding_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "panel.csv"
            write_panel(csv_path, ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06"])

            schedule = wf.build_walk_forward_schedule(
                csv_path,
                start_date="2025-01-03",
                end_date="2025-01-06",
                min_train_days=2,
            )

            self.assertEqual([w.deploy_date for w in schedule], ["2025-01-03", "2025-01-06"])
            self.assertEqual(schedule[0].train_dates, ["2025-01-01", "2025-01-02"])
            self.assertEqual(schedule[0].train_rows, 4)
            self.assertEqual(schedule[0].deploy_rows, 2)

    def test_artifact_gate_requires_validation_and_delta_sanity(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            write_valid_artifacts(artifacts)
            metrics = passing_retrain(artifacts_dir=artifacts)

            gate = wf.evaluate_artifact_gates(artifacts, metrics, delta_tolerance=0.15)

            self.assertTrue(gate["passed"])

            bad_metrics = passing_retrain(artifacts_dir=artifacts)
            bad_metrics["train"]["delta_sanity"]["error"] = 0.25
            bad_gate = wf.evaluate_artifact_gates(artifacts, bad_metrics, delta_tolerance=0.15)
            self.assertFalse(bad_gate["passed"])

    def test_promotion_archives_existing_live_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            live = root / "live"
            write_valid_artifacts(source)
            live.mkdir()
            (live / "neural_bsde.onnx").write_bytes(b"old")
            (live / "normalization.json").write_text("{}")
            (live / "Y0_init.json").write_text("{}")

            result = wf.promote_artifacts(source, live, {"source_run_id": "r1"})

            self.assertTrue(result["promoted"])
            self.assertEqual((live / "neural_bsde.onnx").read_bytes(), b"onnx")
            self.assertTrue((live / "manifest.json").exists())
            archive_dir = Path(result["archive_dir"])
            self.assertEqual((archive_dir / "neural_bsde.onnx").read_bytes(), b"old")

    def test_run_pipeline_writes_manifest_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "panel.csv"
            write_panel(csv_path, ["2025-01-01", "2025-01-02", "2025-01-03"])
            config = wf.PipelineConfig(
                csv_path=csv_path,
                run_id="unit",
                run_root=root / "runs",
                start_date=None,
                end_date=None,
                min_train_days=2,
                profile="smoke",
                epochs=1,
                n_syn=8,
                n_steps=3,
                seed=1,
                promote=False,
                build_dir=root / "build",
                live_artifacts=root / "live",
            )

            result = wf.run_pipeline(config, retrain_fn=passing_retrain, replay_fn=passing_replay)

            run_dir = Path(result["run_dir"])
            self.assertTrue((run_dir / "manifest.json").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "summary.md").exists())
            self.assertEqual(result["summary"]["n_windows"], 1)
            self.assertEqual(result["summary"]["n_passed"], 1)


if __name__ == "__main__":
    unittest.main()
