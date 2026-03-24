# Mediwatch — Airflow Orchestration

## Files

```
docker-compose.yml                  ← Airflow stack (webserver, scheduler, triggerer, postgres)
dags/
  acceptance_test_dag.py            ← 5-task acceptance DAG (env, import, data, uv, mlflow)
scripts/
  trigger_windows.py                ← REST API trigger script — acceptance test or full pipeline
  run_acceptance_test.sh            ← One-shot: start stack → trigger DAG → report pass/fail
```

## Quick start

```bash
# 1. Start the stack
docker compose up -d

# 2. Run the acceptance test (waits for healthy, triggers DAG, polls to completion)
chmod +x scripts/run_acceptance_test.sh
./scripts/run_acceptance_test.sh

# 3. Open the UI
open http://localhost:8080          # admin / admin
```

## Trigger the acceptance DAG manually

```bash
python scripts/trigger_windows.py \
  --dag-id mediwatch_acceptance_test \
  --single
```

## Trigger the pipeline DAG for every window

```bash
python scripts/trigger_windows.py \
  --dag-id mediwatch_pipeline \
  --poll-secs 30 \
  --timeout 600
```

## What the acceptance DAG checks

| Task | What it proves |
|---|---|
| `env_check` | Python version, PYTHONPATH, required env vars present |
| `project_import` | `src.config` importable from mounted project root |
| `data_check` | `WINDOW_DATES` is a non-empty list of strings |
| `uv_check` | `uv` binary available in container PATH |
| `mlflow_check` | MLflow tracking URI reachable (local FS or remote server) |

## Tear down

```bash
docker compose down -v              # stops containers and removes DB volume
```

## Adding uv to the container

The standard `apache/airflow` image does not include `uv`.
To make `uv_check` pass, add this to docker-compose.yml under the
`airflow-init` command, or build a custom image:

```dockerfile
FROM apache/airflow:2.9.3-python3.11
RUN pip install uv
```

Or mount the uv binary from your host:

```yaml
volumes:
  - ~/.local/bin/uv:/usr/local/bin/uv:ro
```
