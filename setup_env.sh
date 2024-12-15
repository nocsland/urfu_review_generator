#!/bin/bash

# Выход при ошибке
set -e

# Переход в директорию скрипта
cd "$(dirname "$0")"

# Настройка переменных
PROJECT_DIR=$(pwd)
ACTIVATE_FILE="venv/bin/activate"

# Создание виртуального окружения, если его нет
if [[ ! -d "venv" ]]; then
  echo ">>> Виртуальное окружение не найдено. Создаём новое..."
  python3 -m venv venv
fi

# Проверка, активировано ли виртуальное окружение
if [[ -z "$VIRTUAL_ENV" ]]; then
  # Если окружение не активно, активируем его
  echo ">>> Виртуальное окружение не активно. Активируем..."
  # shellcheck disable=SC1090
  source "$ACTIVATE_FILE"
  echo ">>> Виртуальное окружение активировано."
fi

# Установка зависимостей, если есть requirements.txt
if [[ -f requirements.txt ]]; then
  echo ">>> Установка зависимостей..."
  pip install --upgrade pip
  pip install -r requirements.txt
else
  echo ">>> Файл requirements.txt не найден. Пропускаем установку зависимостей."
fi

# Добавление PYTHONPATH в activate, если строки нет
if ! grep -Fxq "export PYTHONPATH=$PROJECT_DIR" "$ACTIVATE_FILE"; then
  echo "export PYTHONPATH=$PROJECT_DIR" >> "$ACTIVATE_FILE"
  echo ">>> PYTHONPATH добавлен в $ACTIVATE_FILE"
fi

# Настройка PYTHONPATH для текущей сессии
if [[ -n "$VIRTUAL_ENV" && "$PYTHONPATH" != "$PROJECT_DIR" ]]; then
  export PYTHONPATH=$PROJECT_DIR
  echo ">>> PYTHONPATH настроен для текущей сессии"
else
  echo ">>> PYTHONPATH уже настроен: $PYTHONPATH"
  echo ">>> Пожалуйста, активируйте виртуальное окружение вручную с помощью команды: source venv/bin/activate"
fi

# Уведомление о необходимости активировать окружение вручную, если оно не было активировано в начале
if [[ -z "$VIRTUAL_ENV" ]]; then
  echo ">>> Пожалуйста, активируйте виртуальное окружение вручную с помощью команды: source venv/bin/activate"
fi

echo "========================================"
echo ">>> Всё готово! Виртуальное окружение настроено."
echo "========================================"
