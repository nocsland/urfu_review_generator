import json
import logging
import os

# Настройка логгирования
log_dir = 'logs'
log_file = 'review_parser.log'
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, log_file)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_path
)


def parse_tskv_file(file_path: str):
    """
    Парсер TSKV-файла с данными об отзывах.
    :param file_path: Путь к файлу формата TSKV.
    :return: Список словарей, содержащих разобранные данные.
    """
    reviews = []
    logging.info(f"Начало обработки файла: {file_path}")

    if not os.path.exists(file_path):
        logging.error(f"Файл {file_path} не найден.")
        raise FileNotFoundError(f"Файл {file_path} не найден.")

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            if line:
                fields = line.split('\t')
                review_data = {}
                for field in fields:
                    if '=' in field:
                        key, value = field.split('=', 1)
                        key = key.strip()
                        value = value.strip()

                        if key == "rating":
                            try:
                                value = float(value)
                            except ValueError:
                                logging.warning(f"Ошибка при обработке рейтинга: {value}")
                                value = None
                        review_data[key] = value

                reviews.append(review_data)

    logging.info(f"Обработано записей: {len(reviews)}")
    return reviews


def save_to_json(data, output_path):
    """
    Сохраняет данные в формате JSON.
    :param data: Данные для сохранения.
    :param output_path: Путь для сохранения JSON-файла.
    """
    logging.info(f"Начало сохранения данных в {output_path}")
    if os.path.exists(output_path):
        logging.warning(f"Файл {output_path} уже существует и будет перезаписан.")
    try:
        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        logging.info(f"Данные успешно сохранены в {output_path}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении данных в файл {output_path}: {e}")
        raise


if __name__ == "__main__":
    data_file_path = "data/geo-reviews-dataset-2023.tskv"
    output_file_path = "data/geo_reviews_raw.json"

    try:
        parsed_reviews = parse_tskv_file(data_file_path)
        save_to_json(parsed_reviews, output_file_path)
        logging.info(f"Данные успешно обработаны и сохранены в {output_file_path}")
    except Exception as e:
        logging.error(f"Ошибка: {e}")
