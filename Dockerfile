# Используем официальный slim-образ Python 3.12
FROM python:3.12-slim

# Устанавливаем системные зависимости для OCR (Docling/RapidOCR)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Копируем бинарник uv из официального образа
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Копируем файлы конфигурации зависимостей
COPY pyproject.toml uv.lock ./

# Устанавливаем зависимости проекта (без установки самого проекта)
# Флаг --frozen гарантирует, что будет использован uv.lock
RUN uv sync --frozen --no-cache

# Копируем весь остальной код проекта
COPY . .

CMD ["uv", "run", "python", "-m", "src.main"]