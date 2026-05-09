#!/bin/bash
# ProjectCerebro — Full Pipeline
# Waits for PhysioNet download to finish then runs Stage 1 + Stage 2
# Usage: nohup bash scripts/run_full_pipeline.sh > pipeline.log 2>&1 &

set -e
cd /teamspace/studios/this_studio/projectcerebro

PHYSIONET_DIR="data/physionet_mne/MNE-eegbci-data/files/eegmmidb/1.0.0"
EXPECTED_SUBJECTS=104

echo "=== ProjectCerebro Full Pipeline ==="
echo "Started: $(date)"

# ---- Wait for all PhysioNet subjects to download ----
echo ""
echo "Waiting for PhysioNet download to complete..."
while true; do
    N=$(ls "$PHYSIONET_DIR" 2>/dev/null | wc -l || echo 0)
    echo "  $(date +%H:%M:%S) — $N/$EXPECTED_SUBJECTS subjects downloaded"
    if [ "$N" -ge "$EXPECTED_SUBJECTS" ]; then
        echo "  Download complete!"
        break
    fi
    sleep 60
done

echo ""
echo "=== Stage 1 + Stage 2 Pipeline ==="
echo "Started: $(date)"

# Run full pipeline: PhysioNet (4 parallel workers) + BCI IV-2a via MOABB
python scripts/run_pipeline.py --n-workers 4

echo ""
echo "=== Pipeline complete: $(date) ==="
echo "Output files:"
ls -lh parquet_output/*.parquet 2>/dev/null || echo "  No parquet files found"
