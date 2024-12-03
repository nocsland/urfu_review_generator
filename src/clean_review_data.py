import json
import os
import re
from typing import Dict


def clean_review_data(review: Dict[str, str]) -> Dict[str, str]:
    """
    Очищает текстовые поля записи отзыва.
    :param review: Словарь с данными отзыва.
    :return: Очищенный словарь.
    """
    cleaned_review = {}

    # Очистка текстового поля `text`
    if "text" in review:
        text = review["text"]
        text = text.lower()  # Приводим к нижнему регистру
        text = re.sub(r"<[^>]+>", "", text)  # Удаляем HTML-теги
        text = re.sub(r"[^\w\s,.!?]", "", text)  # Удаляем спецсимволы, кроме пунктуации
        text = re.sub(r"\s+", " ", text).strip()  # Убираем лишние пробелы
        cleaned_review["text"] = text

    # Очистка поля `rubrics`
    if "rubrics" in review:
        rubrics = review["rubrics"]
        rubrics = rubrics.lower().strip()
        cleaned_review["rubrics"] = rubrics

    # Очистка поля `name_ru`
    if "name_ru" in review:
        name = review["name_ru"]
        name = name.strip()
        cleaned_review["name_ru"] = name

    # Минимальная очистка адреса
    if "address" in review:
        address = review["address"]
        address = re.sub(r"\s+", " ", address).strip()
        cleaned_review["address"] = address

    # Преобразование оценки в float
    if "rating" in review:
        try:
            cleaned_review["rating"] = float(review["rating"])
        except ValueError:
            cleaned_review["rating"] = None  # Обрабатываем некорректные значения

    return cleaned_review


def clean_dataset(input_path: str, output_path: str):
    """
    Читает необработанные данные, очищает их и сохраняет в новый файл.
    :param input_path: Путь к исходному JSON-файлу с сырыми данными.
    :param output_path: Путь для сохранения очищенных данных.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Файл {input_path} не найден.")

    with open(input_path, "r", encoding="utf-8") as infile:
        raw_data = json.load(infile)

    cleaned_data = [clean_review_data(review) for review in raw_data]

    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(cleaned_data, outfile, ensure_ascii=False, indent=4)

    print(f"Очищенные данные сохранены в {output_path}")


# Основной блок
if __name__ == "__main__":
    input_file = "../data/geo_reviews_raw.json"
    output_file = "../data/geo_reviews_cleaned.json"

    try:
        clean_dataset(input_file, output_file)
    except Exception as e:
        print(f"Ошибка: {e}")
