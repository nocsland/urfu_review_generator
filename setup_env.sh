#!/bin/bash

# Выход при ошибке
set -e

# Переход в директорию скрипта
cd "$(dirname "$0")"

# Проверка, активировано ли виртуальное окружение
if [[ "$VIRTUAL_ENV" != "" ]]; then
  echo ">>> Виртуальное окружение уже активировано: $VIRTUAL_ENV"
else
  # Создаём виртуальное окружение, если его нет
  if [[ ! -d "venv" ]]; then
    echo ">>> Виртуальное окружение не найдено. Создаём новое..."
    python3 -m venv venv
  fi
fi

# Настройка PYTHONPATH
PROJECT_DIR=$(pwd)
export PYTHONPATH=$PROJECT_DIR
echo ">>> PYTHONPATH настроен: $PROJECT_DIR"

# Активация окружения
echo ">>> Активация виртуального окружения..."
source venv/bin/activate

# Установка зависимостей, если файл requirements.txt существует
if [[ -f requirements.txt ]]; then
  echo ">>> Установка зависимостей..."
  pip install --upgrade pip
  pip install -r requirements.txt
else
  echo ">>> Файл requirements.txt не найден. Пропускаем установку зависимостей."
fi

echo "========================================"
echo ">>> Всё готово! Виртуальное окружение активировано."
echo "========================================"