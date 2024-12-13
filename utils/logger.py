import logging
import os


def setup_logger(log_file, log_dir='logs'):
    """
    Настраивает логирование с заданным файлом и форматом.

    :param log_file: Имя файла для логов (обязательно)
    :param log_dir: Директория для логов (по умолчанию 'logs')
    :return: Конфигурированный объект logging
    """
    if not log_file:
        raise ValueError("log_file is a required argument.")

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_path
    )
    return logging
