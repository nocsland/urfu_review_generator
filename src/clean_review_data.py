import json
import os
import re
from typing import Dict


def clean_review_data(review: Dict[str, str]) -> Dict[str, str]:
    """
    Очищает текстовые поля записи отзыва.
    :param review: Словарь с данными отзыва.
    :return: Очищенный словарь или None, если данные некорректны.
    """
    # Проверка на обязательные столбцы
    if not review.get("text") or not review.get("name_ru") or review.get("rating") is None:
        return None  # Возвращаем None, если ключевые столбцы пустые или отсутствуют

    cleaned_review = {}

    # Очистка текстового поля `text`
    text = review["text"]
    text = text.lower()  # Приводим к нижнему регистру
    text = re.sub(r"<[^>]+>", "", text)  # Удаляем HTML-теги
    text = re.sub(r"[^\w\s,.!?()]+", "", text)  # Удаляем спецсимволы, кроме пунктуации и скобок
    text = re.sub(r"\)\)+", " ", text)  # Заменяем смайлики вида "))" на пробел
    text = re.sub(r"\s+", " ", text).strip()  # Убираем лишние пробелы
    text = re.sub(r"[\n\r]+", " ", text)  # Заменяем переносы строк на пробелы
    cleaned_review["text"] = text

    # Очистка поля `rubrics`
    rubrics = review.get("rubrics", "").lower().strip()
    cleaned_review["rubrics"] = rubrics

    # Очистка поля `name_ru`
    name = review.get("name_ru", "").strip()
    cleaned_review["name_ru"] = name

    # Минимальная очистка адреса
    address = review.get("address", "").strip()
    address = re.sub(r"\s+", " ", address)
    cleaned_review["address"] = address

    # Преобразование оценки в float
    try:
        cleaned_review["rating"] = float(review["rating"])
    except (ValueError, TypeError):
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

    # Фильтруем данные, удаляя записи с пустыми обязательными полями
    cleaned_data = [clean_review_data(review) for review in raw_data]
    cleaned_data = [review for review in cleaned_data if review is not None]  # Убираем None

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