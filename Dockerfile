FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

EXPOSE 8080
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . ./

# Create a script to load .env file if it exists
RUN echo '#!/bin/bash\n\
if [ -f /app/.env ]; then\n\
    export $(cat /app/.env | grep -v "^#" | xargs)\n\
fi\n\
exec uv run python main.py --host 0.0.0.0 --port 8080' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]