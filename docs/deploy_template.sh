#!/bin/bash

# Этот скрипт является шаблоном для автоматического развертывания приложения
# Настройте его в соответствии с вашими требованиями перед использованием.


# Переменные
APP_DIR="/opt/myapp"
SERVICE_FILE="/etc/systemd/system/myapp.service"
USER="myuser"
GROUP="mygroup"
PYTHON_VERSION="python3.9"
REPO_URL="https://github.com/yourusername/yourrepository.git"  # Замените на URL вашего репозитория

echo "Начало развертывания приложения..."

# 1. Установка необходимых пакетов
echo "Установка необходимых пакетов..."
sudo apt update
sudo apt install -y $PYTHON_VERSION $PYTHON_VERSION-venv $PYTHON_VERSION-pip git

# 2. Клонирование репозитория
echo "Клонирование репозитория..."
sudo git clone $REPO_URL $APP_DIR
sudo chown -R $USER:$GROUP $APP_DIR

# 3. Создание виртуального окружения
echo "Создание виртуального окружения..."
sudo -u $USER bash -c "cd $APP_DIR && $PYTHON_VERSION -m venv venv && source venv/bin/activate && pip install -r requirements.txt"

# 4. Создание службы systemd
echo "Создание службы systemd..."
sudo bash -c "cat > $SERVICE_FILE" <<EOL
[Unit]
Description=My Python Application
After=network.target

[Service]
User=$USER
Group=$GROUP
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin"
ExecStart=$APP_DIR/venv/bin/python app.py

[Install]
WantedBy=multi-user.target
EOL

# 5. Перезагрузка systemd и запуск службы
echo "Перезагрузка systemd и запуск службы..."
sudo systemctl daemon-reload
sudo systemctl enable myapp
sudo systemctl start myapp

echo "Развертывание завершено! Приложение работает."

