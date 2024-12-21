import json
import logging
import os

from config.settings import SOURCE_DATA_PATH, RAW_DATA_PATH
from utils.logger import setup_logger

# Настройка логирования: папка и файл для логов
logger = setup_logger(log_file='review_parser.log')


def parse_tskv_file(file_path: str):
    """
    Парсит TSKV-файл с отзывами, преобразуя строки в словари.
    :param file_path: Путь к файлу формата TSKV.
    :return: Список словарей, содержащих разобранные данные из файла.
    """
    # Список для хранения обработанных записей
    reviews = []
    logging.info(f"Начало обработки файла: {file_path}")

    # Проверка существования файла
    if not os.path.exists(file_path):
        logging.error(f"Файл {file_path} не найден.")
        raise FileNotFoundError(f"Файл {file_path} не найден.")

    # Чтение и разбор строк из TSKV-файла
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()  # Удаление пробелов и переводов строк

            if line:  # Пропуск пустых строк
                fields = line.split('\t')  # Разделение строки на поля
                review_data = {}  # Словарь для текущей записи
                for field in fields:
                    if '=' in field:  # Проверка наличия ключа и значения
                        key, value = field.split('=', 1)
                        key = key.strip()  # Очистка ключа от лишних пробелов
                        value = value.strip()  # Очистка значения от лишних пробелов

                        # Обработка рейтинга: преобразование в число
                        if key == "rating":
                            try:
                                value = float(value)
                            except ValueError:
                                logging.warning(f"Ошибка при обработке рейтинга: {value}")
                                value = None
                        review_data[key] = value  # Добавление ключа и значения в словарь

                reviews.append(review_data)  # Добавление записи в список

    logging.info(f"Обработано записей: {len(reviews)}")
    return reviews


def save_to_json(data, output_path):
    """
    Сохраняет данные в формате JSON в указанный файл.
    :param data: Данные для сохранения.
    :param output_path: Путь для сохранения JSON-файла.
    """
    logging.info(f"Начало сохранения данных в {output_path}")

    # Предупреждение о перезаписи существующего файла
    if os.path.exists(output_path):
        logging.warning(f"Файл {output_path} уже существует и будет перезаписан.")

    # Сохранение данных в формате JSON
    try:
        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        logging.info(f"Данные успешно сохранены в {output_path}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении данных в файл {output_path}: {e}")
        raise


if __name__ == "__main__":
    # Путь к исходному TSKV-файлу и выходному JSON-файлу
    data_file_path = SOURCE_DATA_PATH
    output_file_path = RAW_DATA_PATH

    try:
        # Парсинг TSKV-файла и сохранение данных в формате JSON
        parsed_reviews = parse_tskv_file(data_file_path)
        save_to_json(parsed_reviews, output_file_path)
        logging.info(f"Данные успешно обработаны и сохранены в {output_file_path}")
    except Exception as e:
        logging.error(f"Ошибка: {e}")
