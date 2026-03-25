FROM apache/airflow:2.9.3-python3.11

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ARG INSTALL_DEPS=false
COPY --chown=airflow:root pyproject.toml* /tmp/
USER airflow
RUN if [ "$INSTALL_DEPS" = "true" ] && [ -f /tmp/pyproject.toml ]; then \
      uv pip install -r /tmp/pyproject.toml; \
    fi
