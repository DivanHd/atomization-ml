# Базовый образ с Python 3.10
FROM python:3.10-slim

# Рабочая директория внутри контейнера
WORKDIR /app

# Установка системных зависимостей (нужны для сборки некоторых библиотек)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 1. Копируем файл зависимостей
COPY pyproject.toml ./

# 2. Копируем исходный код и скрипты СРАЗУ (чтобы uv sync их видел)
COPY src/ ./src/
COPY scripts/ ./scripts/

# 3. Устанавливаем uv
RUN pip install --no-cache-dir uv

# 4. Устанавливаем зависимости проекта
RUN uv sync --no-dev

# 5. Делаем скрипт исполняемым
RUN chmod +x scripts/run.sh

# 6. Создаём папки для результатов и данных
RUN mkdir -p data models results

# Точка входа — bash-скрипт
ENTRYPOINT ["bash", "scripts/run.sh"]
