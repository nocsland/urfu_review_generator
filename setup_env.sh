#!/bin/bash

# Выход при ошибке
set -e

# Переход в директорию скрипта
cd "$(dirname "$0")"

# Настройка переменных
PROJECT_DIR=$(pwd)
ACTIVATE_FILE="venv/bin/activate"

# Создание виртуального окружения, если его нет
[[ -d "venv" ]] || {
  echo ">>> Виртуальное окружение не найдено. Создаём новое..."
  python3 -m venv venv
}

# Проверка, активировано ли виртуальное окружение
if [[ -z "$VIRTUAL_ENV" ]]; then
  # Активация виртуального окружения
  source "$ACTIVATE_FILE"
  echo ">>> Виртуальное окружение активировано."
fi

# Добавление PYTHONPATH в activate, если строки нет
if ! grep -Fxq "export PYTHONPATH=$PROJECT_DIR" "$ACTIVATE_FILE"; then
  echo "export PYTHONPATH=$PROJECT_DIR" >> "$ACTIVATE_FILE"
  echo ">>> PYTHONPATH добавлен в $ACTIVATE_FILE"
fi

# Настройка PYTHONPATH для текущей сессии, если окружение активировано
if [[ -n "$VIRTUAL_ENV" && "$PYTHONPATH" != "$PROJECT_DIR" ]]; then
  export PYTHONPATH=$PROJECT_DIR
  echo ">>> PYTHONPATH настроен для текущей сессии"
else
  echo ">>> PYTHONPATH уже настроен: $PYTHONPATH"
fi

# Установка зависимостей, если есть requirements.txt
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
