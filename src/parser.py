import os


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


# Пример использования
data_file_path = "../data/geo-reviews-dataset-2023.tskv"
parsed_reviews = parse_tskv_file(data_file_path)

# Выводим первые 5 записей
for i, review in enumerate(parsed_reviews[:5]):
    print(f"Отзыв {i + 1}: {review}")
