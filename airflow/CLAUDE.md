# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Airflow orchestration layer for the Mediwatch ML pipeline. This subdirectory contains the Docker Compose stack, DAGs, and trigger scripts. The parent project (`../`) holds the actual ML code (`src/`, `runner.py`) which gets volume-mounted into the Airflow containers at `/opt/mediwatch`.

## Common Commands

```bash
# Build custom image and start the Airflow stack (standalone — acceptance test only)
docker compose up -d --build

# Start with pipeline support (mounts parent project into containers)
docker compose -f docker-compose.yml -f docker-compose.pipeline.yml up -d --build

# Run the full acceptance test (starts stack, triggers DAG, reports pass/fail)
./scripts/run_acceptance_test.sh          # leaves stack running
./scripts/run_acceptance_test.sh --down   # tears down after

# Trigger acceptance test manually via REST API
python scripts/trigger_windows.py --dag-id mediwatch_acceptance_test --single

# Trigger pipeline DAG for every window date
python scripts/trigger_windows.py --dag-id mediwatch_pipeline --poll-secs 30 --timeout 600

# Dry run (print what would be triggered, no API calls)
python scripts/trigger_windows.py --dag-id mediwatch_pipeline --dry-run

# Tear down stack and remove DB volume
docker compose down -v

# Airflow UI
open http://localhost:8080   # admin / admin
```

## Architecture

- **Dockerfile**: Custom image based on `apache/airflow:2.9.3-python3.11` that adds `uv`.
- **docker-compose.yml**: Standalone Airflow stack (LocalExecutor + Postgres). No project code mounted — sufficient for the acceptance test DAG.
- **docker-compose.pipeline.yml**: Override that adds the `..:/opt/mediwatch` volume mount and `PYTHONPATH` to all Airflow services. Required for the pipeline DAG.
- **dags/acceptance_test_dag.py**: Standalone stack validation DAG (`mediwatch_acceptance_test`) — 4 tasks: `env_check → python_check → [uv_check, mlflow_check]`. No project imports.
- **dags/pipeline_dag.py**: Champion/Challenger pipeline DAG (`mediwatch_pipeline`) — 6 tasks per run: `detect_window → drift_report → train_challenger → evaluate_models → promote_decision → log_summary`. Each run processes one window date from conf. Cross-run state (current champion) is stored in the MLflow model registry via the `@champion` alias.
- **scripts/trigger_windows.py**: REST API client that triggers DAGs and polls until completion. Supports single-shot mode (`--single` for acceptance) or one-run-per-window-date mode. Requires the `requests` package.
- **scripts/run_acceptance_test.sh**: End-to-end wrapper that starts the stack, waits for health, triggers the acceptance DAG, and reports results.

## Key Details

- **Two compose modes**: standalone (`docker-compose.yml` only) for acceptance testing, and pipeline mode (with `docker-compose.pipeline.yml` override) for running the ML pipeline.
- **`uv` installs deps from `pyproject.toml`** in the Dockerfile. The pipeline override sets the build context to the parent directory so `pyproject.toml` is available. `requires-python` is `>=3.11` (not pinned to 3.13) because the container image uses Python 3.11.
- **Cross-run state via MLflow registry**: Airflow DAG runs are isolated — there's no shared memory between the run that processes 2004 and the run that processes 2005. The pipeline uses the MLflow model registry's `@champion` alias as the bridge: `detect_window` looks up the alias to find the incumbent champion, and `promote_decision` updates it. This is the production pattern.
- **Single MLflow run per window**: `detect_window` creates the run, and all downstream tasks reopen it by `run_id` (passed via XCom).
- **Shared constants**: `EXPERIMENT_NAME`, `REGISTERED_MODEL`, `CHAMPION_ALIAS`, and `PROMOTION_THRESHOLD` live in `src/config.py` and are imported by both `runner.py` and `pipeline_dag.py`.
- The trigger script uses Airflow's REST API v1 with basic auth (default: admin/admin).
- **Pipeline mode writes MLflow data to the host** via `MLFLOW_TRACKING_URI=sqlite:////opt/mediwatch/mlflow.db` set in the pipeline override. Run `mlflow ui` from the project root to view results.
