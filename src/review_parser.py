import os
import json


def parse_tskv_file(file_path: str):
    """
    Парсер TSKV-файла с данными об отзывах.
    :param file_path: Путь к файлу формата TSKV.
    :return: Список словарей, содержащих разобранные данные.
    """
    reviews = []

    # Проверяем, существует ли файл
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден.")

    # Читаем файл построчно
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Убираем лишние пробелы и пропускаем пустые строки
            line = line.strip()
            if not line:
                continue

            # Разбираем строку
            fields = line.split('\t')
            review_data = {}
            for field in fields:
                if '=' in field:
                    key, value = field.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    # Преобразуем значение, если это числовой тип
                    if key == "rating":
                        value = float(value)

                    review_data[key] = value

            reviews.append(review_data)

    return reviews


def save_to_json(data, output_path):
    """
    Сохраняет данные в формате JSON.
    :param data: Данные для сохранения.
    :param output_path: Путь для сохранения JSON-файла.
    """
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


# Основной блок
if __name__ == "__main__":
    data_file_path = "../data/geo-reviews-dataset-2023.tskv"
    output_file_path = "../data/geo_reviews_raw.json"

    try:
        # Парсим файл
        parsed_reviews = parse_tskv_file(data_file_path)

        # Сохраняем результат в JSON
        save_to_json(parsed_reviews, output_file_path)
        print(f"Данные успешно обработаны и сохранены в {output_file_path}")
    except Exception as e:
        print(f"Ошибка: {e}")
