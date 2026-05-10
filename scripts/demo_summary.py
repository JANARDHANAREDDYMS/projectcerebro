"""
ProjectCerebro Demo Summary
Shows what happened across all storage layers after a demo run.

Usage:
    python scripts/demo_summary.py
    python scripts/demo_summary.py --session demo_20260509_162334
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import psycopg2
from pymongo import MongoClient

MONGO_URI   = "mongodb://cerebro:cerebro123@localhost:27017/"
MONGO_DB    = "projectcerebro"
PG_CONN_STR = (
    "host=localhost port=5433 dbname=projectcerebro "
    "user=cerebro password=cerebro123"
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default=None,
        help="Filter by session ID. Shows all if not provided.")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  ProjectCerebro Demo Summary")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*55}\n")

    # ── MongoDB ──────────────────────────────────────────
    mongo = MongoClient(MONGO_URI)
    db    = mongo[MONGO_DB]

    query = {}
    if args.session:
        query["session_id"] = args.session

    n_pred   = db.predictions.count_documents(query)
    n_flags  = db.trial_quality_flags.count_documents(query)
    n_alerts = db.alerts.count_documents(query)
    n_recs   = db.hpo_recommendations.count_documents({})
    n_reports= db.session_reports.count_documents({})

    print(f"MongoDB (db=cerebro):")
    print(f"  predictions:         {n_pred}")
    print(f"  trial_quality_flags: {n_flags}")
    print(f"  alerts:              {n_alerts}")
    print(f"  hpo_recommendations: {n_recs}")
    print(f"  session_reports:     {n_reports}")

    # Prediction breakdown
    if n_pred > 0:
        pipeline = [
            {"$match": query},
            {"$group": {
                "_id": "$label_name",
                "count": {"$sum": 1},
                "avg_conf": {"$avg": "$confidence"},
            }}
        ]
        by_label = list(db.predictions.aggregate(pipeline))
        print(f"\n  Predictions by class:")
        for item in sorted(by_label, key=lambda x: x["_id"] or ""):
            print(f"    {item['_id']:<8} "
                  f"count={item['count']:<5} "
                  f"avg_conf={item['avg_conf']:.3f}")

        # Model usage
        pipeline2 = [
            {"$match": query},
            {"$group": {"_id": "$model_used", "count": {"$sum": 1}}}
        ]
        by_model = list(db.predictions.aggregate(pipeline2))
        print(f"\n  Predictions by model:")
        for item in sorted(by_model, key=lambda x: x["_id"] or ""):
            print(f"    {item['_id']:<20} count={item['count']}")

        # Calibration evidence
        personalized = db.predictions.count_documents(
            {**query, "model_used": "personalized"}
        )
        ensemble = db.predictions.count_documents(
            {**query, "model_used": "ensemble"}
        )
        print(f"\n  Transfer learning evidence:")
        print(f"    Zero-shot (ensemble):    {ensemble} epochs")
        print(f"    Personalized (few-shot): {personalized} epochs")
        if personalized > 0:
            print(f"    Calibration: COMPLETED ✓")
        else:
            print(f"    Calibration: not reached yet")

    # Latest HPO recommendation
    if n_recs > 0:
        latest_rec = db.hpo_recommendations.find_one(
            {}, sort=[("timestamp", -1)]
        )
        print(f"\n  Latest HPO recommendation:")
        print(f"    {latest_rec.get('rationale', 'N/A')}")
        cfg = latest_rec.get("suggested_config", {})
        print(f"    Suggested: {cfg}")

    mongo.close()

    # ── pgvector ─────────────────────────────────────────
    print(f"\npgvector (eeg_embeddings):")
    try:
        conn = psycopg2.connect(PG_CONN_STR)
        cur  = conn.cursor()

        cur.execute("""
            SELECT
                COUNT(*) as total,
                COUNT(DISTINCT subject_id) as subjects,
                COUNT(CASE WHEN label_name='left'  THEN 1 END) as left_n,
                COUNT(CASE WHEN label_name='right' THEN 1 END) as right_n,
                COUNT(CASE WHEN label_name='rest'  THEN 1 END) as rest_n,
                ROUND(AVG(confidence)::numeric, 3) as avg_conf
            FROM eeg_embeddings
        """)
        row = cur.fetchone()
        print(f"  total embeddings: {row[0]}")
        print(f"  subjects:         {row[1]}")
        print(f"  left:  {row[2]}  right: {row[3]}  rest: {row[4]}")
        print(f"  avg confidence:   {row[5]}")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"  Error: {e}")

    # ── Reports ──────────────────────────────────────────
    reports_dir = Path("artifacts/reports")
    if reports_dir.exists():
        reports = sorted(reports_dir.glob("*.txt"))
        print(f"\nSession reports: {len(reports)} files")
        if reports:
            latest = reports[-1]
            print(f"  Latest: {latest.name}")
            print(f"  Content preview:")
            with open(latest) as f:
                for line in f.readlines()[:15]:
                    print(f"    {line.rstrip()}")

    print(f"\n{'='*55}")
    print(f"  MLflow UI: http://localhost:5001")
    print(f"  FastAPI:   http://localhost:8001/docs")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    main()
