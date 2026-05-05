"""Demo entrypoint for the ProjectCerebro agent graph."""
from __future__ import annotations

import numpy as np

from agents.graph import run_epoch, run_hpo, run_report


def main() -> None:
    """Run one synthetic EEG epoch through the graph."""
    fake_features = np.random.randn(2560).astype(float).tolist()
    result = run_epoch(features=fake_features, subject_id="A01", session_id="session_001")

    print(f"Signal quality: {result['signal_quality']} (score={result['quality_score']:.2f})")
    print(f"Prediction:     {result['label_name']} (confidence={result['confidence']})")
    print(f"Explanation:    {result['explanation']}")
    print(f"Alerts:         {result['alerts']}")
    print(f"Severity:       {result['final_severity']}")

    run_report("session_001", "A01")
    run_hpo("session_001", "A01")


if __name__ == "__main__":
    main()

