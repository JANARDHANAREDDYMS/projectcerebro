#!/bin/bash
# Watch pgvector embeddings table grow in real-time

echo "Watching pgvector eeg_embeddings table..."
echo "Updates every 5 seconds. Ctrl+C to stop."
echo ""

while true; do
  docker exec cerebro_postgres \
    psql -U cerebro -d cerebro -t -c "
      SELECT
        TO_CHAR(NOW(), 'HH24:MI:SS') as time,
        COUNT(*) as total_embeddings,
        COUNT(DISTINCT subject_id) as subjects,
        COUNT(CASE WHEN label_name='left'  THEN 1 END) as left_count,
        COUNT(CASE WHEN label_name='right' THEN 1 END) as right_count,
        COUNT(CASE WHEN label_name='rest'  THEN 1 END) as rest_count
      FROM eeg_embeddings;
    " 2>/dev/null | grep -v "^$" | \
    awk '{print "pgvector | total="$2" subjects="$3" left="$4" right="$5" rest="$6}'
  sleep 5
done
