#!/bin/bash
echo "Watching MongoDB cerebro database..."
echo "Updates every 3 seconds. Ctrl+C to stop."
echo ""

cd /Users/janardhanareddyms/Documents/Tandon/courses/bigdata/projectcerebro

while true; do
  cerebro_env/bin/python - << 'PY'
from pymongo import MongoClient
from datetime import datetime

try:
    mongo = MongoClient(
        "mongodb://cerebro:cerebro123@localhost:27017/",
        serverSelectionTimeoutMS=2000
    )
    db = mongo["projectcerebro"]

    pred   = db.predictions.count_documents({})
    flags  = db.trial_quality_flags.count_documents({})
    alerts = db.alerts.count_documents({})
    recs   = db.hpo_recommendations.count_documents({})

    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{ts} | predictions={pred} flags={flags} alerts={alerts} hpo={recs}")

    latest = db.predictions.find_one(
        {},
        {"label_name":1,"confidence":1,"model_used":1,"signal_quality":1,"_id":0},
        sort=[("_id", -1)]
    )
    if latest:
        conf = latest.get("confidence") or 0
        print(f"         latest: label={latest.get('label_name','?'):<6} "
              f"conf={conf:.3f} "
              f"model={latest.get('model_used','?')} "
              f"quality={latest.get('signal_quality','?')}")
    mongo.close()
except Exception as e:
    print(f"MongoDB error: {e}")
PY
  sleep 3
done
