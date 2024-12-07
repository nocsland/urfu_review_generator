import json
import logging
import os
import re
from typing import Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Глобальная переменная для подсчета некорректных рейтингов
invalid_rating_count = 0


def clean_review_data(review: Dict[str, str]) -> Dict[str, str]:
    """
    Очищает текстовые поля записи отзыва.
    :param review: Словарь с данными отзыва.
    :return: Очищенный словарь или None, если данные некорректны.
    """
    # Проверка на обязательные столбцы
    if not review.get("text") or not review.get("name_ru") or review.get("rating") is None:
        return None  # Возвращаем None, если ключевые столбцы пустые или отсутствуют

    # Дополнительные проверки данных
    if len(review["text"].strip()) < 10:  # Например, если текст отзыва слишком короткий
        return None

    cleaned_review = {}

    # Очистка текстового поля `text`
    text = review["text"]
    text = text.lower()  # Приводим к нижнему регистру
    text = re.sub(r"<[^>]+>", "", text)  # Удаляем HTML-теги
    text = re.sub(r"[^\w\s,.!?()]+", "", text)  # Удаляем спецсимволы, кроме пунктуации и скобок
    text = re.sub(r"\)\)+", " ", text)  # Заменяем смайлики вида "))" на пробел
    text = re.sub(r"[\n\r]+", " ", text)  # Заменяем переносы строк на пробелы
    text = re.sub(r"n", " ", text)  # Заменяем 'n' на пробел
    text = re.sub(r"\s+", " ", text).strip()  # Убираем лишние пробелы
    cleaned_review["text"] = text

    # Очистка поля `rubrics`
    rubrics = review.get("rubrics", "").lower().strip()
    cleaned_review["rubrics"] = rubrics

    # Очистка поля `name_ru`
    name = review.get("name_ru", "").strip()
    cleaned_review["name_ru"] = name

    # Минимальная очистка адреса (замена слешей на пробелы, если рядом нет чисел)
    address = review.get("address", "").strip()
    address = re.sub(r"(?<!\d)/|(?!\d)/", " ", address)  # Заменяем слеши на пробелы, если рядом нет чисел
    address = re.sub(r"\s+", " ", address)  # Убираем лишние пробелы
    cleaned_review["address"] = address

    # Преобразование рейтинга (без нормализации)
    try:
        rating = float(review["rating"])
        if rating < 1 or rating > 5:  # Если рейтинг вне диапазона, возвращаем None
            global invalid_rating_count
            invalid_rating_count += 1  # Увеличиваем счетчик некорректных рейтингов
            return None  # Удаляем записи с рейтингом, выходящим за пределы диапазона
        # Оставляем рейтинг без изменений
        cleaned_review["rating"] = rating
    except (ValueError, TypeError) as e:
        logging.warning(f"Ошибка при обработке рейтинга: {review['rating']} - {e}")
        return None  # Удаляем записи с некорректным значением рейтинга

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
    total_reviews = len(raw_data)
    cleaned_data = [clean_review_data(review) for review in raw_data]
    cleaned_data = [review for review in cleaned_data if review is not None]  # Убираем None

    # Подсчёт дубликатов
    before_deduplication = len(cleaned_data)
    unique_reviews = {review["text"]: review for review in cleaned_data}.values()
    after_deduplication = len(unique_reviews)

    # Логи
    logging.info(f"Очищенные данные сохранены в {output_path}")
    logging.info(f"Обработано записей: {total_reviews}")
    logging.info(f"Удалено записей с пустыми обязательными полями: {total_reviews - before_deduplication}")
    logging.info(f"Удалено дубликатов: {before_deduplication - after_deduplication}")
    logging.info(f"Удалено записей с некорректным рейтингом: {invalid_rating_count}")
    logging.info(f"Сохранено уникальных записей: {after_deduplication}")

    # Сохранение очищенных данных
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(list(unique_reviews), outfile, ensure_ascii=False, indent=4)


# Основной блок
if __name__ == "__main__":
    input_file = "../data/geo_reviews_raw.json"
    output_file = "../data/geo_reviews_cleaned.json"

    try:
        clean_dataset(input_file, output_file)
    except Exception as e:
        logging.error(f"Ошибка: {e}")
