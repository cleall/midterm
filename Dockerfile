FROM python:3.10.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
WORKDIR /code

ENV PATH="/code/.venv/bin:$PATH"

COPY "pyproject.toml" "uv.lock" ".python-version" ./
RUN uv sync --locked

COPY "predict.py" "cpu_tf_rfpl_v1.bin" ./
COPY "response/cpu.py" "response/predict_response.py" ./response/
COPY "static/styles.css" ./static/
COPY "web/index.html" ./web/

COPY "entrypoint.sh" /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 10000

ENTRYPOINT ["/entrypoint.sh"]