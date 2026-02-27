from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from updated_sample.pipeline import run_pipeline


def test_pipeline_outputs_exist() -> None:
    run_pipeline(ROOT)
    out = ROOT / "reports" / "results"
    assert (out / "model_comparison.csv").exists()
    assert (out / "pairwise_stat_tests.csv").exists()
    assert (out / "hypothesis_summary.json").exists()

    data = json.loads((out / "hypothesis_summary.json").read_text(encoding="utf-8"))
    assert "hypothesis_tests" in data
