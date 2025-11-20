FROM python:3.11-slim

# 1. Setup Env
WORKDIR /app
ENV PYTHONUNBUFFERED=1

# 2. Install System Deps
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# 3. Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# 4. Copy Deps
COPY pyproject.toml poetry.lock ./
# Disable virtualenv creation for Docker (install globally in container)
RUN poetry config virtualenvs.create false && poetry install --no-root --no-interaction

# 5. Copy Code
COPY . .

# 6. Expose Port
EXPOSE 8000

# 7. Run Command (Start API)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]