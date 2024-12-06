import json
import logging
import os

logging.basicConfig(level=logging.INFO)


def parse_tskv_file(file_path: str):
    """
    Парсер TSKV-файла с данными об отзывах.
    :param file_path: Путь к файлу формата TSKV.
    :return: Список словарей, содержащих разобранные данные.
    """
    reviews = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден.")

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            if line:  # Сохраняем только непустые строки
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
                                value = None  # Можно оставить значение как None или пустое
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
    if os.path.exists(output_path):
        logging.warning(f"Файл {output_path} уже существует и будет перезаписан.")
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    data_file_path = "../data/geo-reviews-dataset-2023.tskv"
    output_file_path = "../data/geo_reviews_raw.json"

    try:
        parsed_reviews = parse_tskv_file(data_file_path)
        save_to_json(parsed_reviews, output_file_path)
        logging.info(f"Данные успешно обработаны и сохранены в {output_file_path}")
    except Exception as e:
        logging.error(f"Ошибка: {e}")
