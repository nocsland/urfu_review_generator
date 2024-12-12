#!/bin/bash

# Выход при ошибке
set -e

# Проверка, активировано ли виртуальное окружение
if [[ "$VIRTUAL_ENV" != "" ]]; then
  echo ">>> Виртуальное окружение уже активировано. Ожидаемое окружение: $VIRTUAL_ENV"

  # Экспортируем PYTHONPATH в текущую сессию оболочки
  PROJECT_DIR=$(pwd)
  export PYTHONPATH=$PROJECT_DIR
  echo "export PYTHONPATH=$PROJECT_DIR" >> venv/bin/activate
  echo ">>> PYTHONPATH настроен: $PROJECT_DIR"

else
  echo ">>> Виртуальное окружение не активировано. Создаём окружение..."

  # Создание виртуального окружения
  python3 -m venv venv

  echo ">>> Активация виртуального окружения..."
  source venv/bin/activate

  echo ">>> Установка зависимостей..."
  pip install --upgrade pip
  pip install -r requirements.txt

  echo ">>> Настройка PYTHONPATH..."
  PROJECT_DIR=$(pwd)
  echo "export PYTHONPATH=$PROJECT_DIR" >> venv/bin/activate
  echo ">>> PYTHONPATH настроен: $PROJECT_DIR"
fi

echo ">>> Всё готово! Если виртуальное окружение не было активировано, активируйте его с помощью: source venv/bin/activate"