FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

EXPOSE 8080
WORKDIR /app

COPY . ./

RUN uv sync

ENTRYPOINT ["uv", "run", "main.py", "--host", "0.0.0.0", "--port", "8080"]