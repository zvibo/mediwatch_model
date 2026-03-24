#!/usr/bin/env bash
# scripts/run_acceptance_test.sh
# ---------------------------------------------------------------------------
# Brings up the Airflow stack, runs the acceptance test DAG, and reports
# pass/fail.  Tears down nothing — leave the stack running for inspection.
#
# Usage:
#   chmod +x scripts/run_acceptance_test.sh
#   ./scripts/run_acceptance_test.sh
#
#   # To also tear down after test:
#   ./scripts/run_acceptance_test.sh --down
# ---------------------------------------------------------------------------

set -euo pipefail

COMPOSE_FILE="$(dirname "$0")/../docker-compose.yml"
SCRIPT_DIR="$(dirname "$0")"
DOWN_AFTER=${1:-""}

echo "============================================================"
echo " Mediwatch — Airflow Acceptance Test"
echo "============================================================"

# ── 1. Init and start the stack ────────────────────────────────────────────
echo ""
echo "► Starting Airflow stack..."
docker compose -f "$COMPOSE_FILE" up -d --wait airflow-init
docker compose -f "$COMPOSE_FILE" up -d --wait webserver scheduler triggerer

echo ""
echo "► Stack running. Waiting for webserver health check..."

# Extra grace period — Airflow's /health endpoint can lag behind 'healthy'
WAIT=0
MAX=90
until curl -sf http://localhost:8080/health | grep -q '"healthy"'; do
  if [ $WAIT -ge $MAX ]; then
    echo "  Timed out waiting for Airflow webserver."
    echo "  Run: docker compose logs webserver"
    exit 1
  fi
  sleep 5
  WAIT=$((WAIT + 5))
  echo "  ...${WAIT}s"
done
echo "  Webserver healthy."

# ── 2. Trigger acceptance test via REST API ─────────────────────────────────
echo ""
echo "► Triggering acceptance test DAG..."

python3 "$SCRIPT_DIR/trigger_windows.py" \
  --dag-id mediwatch_acceptance_test \
  --single \
  --poll-secs 10 \
  --timeout 300

EXIT_CODE=$?

# ── 3. Report ───────────────────────────────────────────────────────────────
echo ""
if [ $EXIT_CODE -eq 0 ]; then
  echo "============================================================"
  echo " ACCEPTANCE TEST PASSED"
  echo " Airflow UI: http://localhost:8080  (admin / admin)"
  echo "============================================================"
else
  echo "============================================================"
  echo " ACCEPTANCE TEST FAILED"
  echo " Check logs: docker compose logs scheduler"
  echo " Airflow UI: http://localhost:8080  (admin / admin)"
  echo "============================================================"
fi

# ── 4. Optional teardown ────────────────────────────────────────────────────
if [ "$DOWN_AFTER" = "--down" ]; then
  echo ""
  echo "► Tearing down stack..."
  docker compose -f "$COMPOSE_FILE" down -v
fi

exit $EXIT_CODE
