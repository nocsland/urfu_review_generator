import json
import logging
import os
import re
from typing import Dict

# Настройка логгирования
log_dir = 'logs'
log_file = 'review_cleaner.log'
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, log_file)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_path
)

# Глобальная переменная для подсчета записей с некорректными рейтингами
invalid_rating_count = 0


def clean_review_data(review: Dict[str, str]) -> Dict[str, str]:
    """
    Очищает данные отзыва, удаляя ненужные символы и проверяя корректность полей.
    :param review: Словарь с данными отзыва.
    :return: Очищенный словарь или None, если данные некорректны.
    """
    logging.debug(f"Начало обработки отзыва: {review}")

    # Проверяем наличие обязательных полей: text, name_ru, rating
    if not review.get("text") or not review.get("name_ru") or review.get("rating") is None:
        logging.warning(f"Пропущены обязательные поля в отзыве: {review}")
        return None

    # Проверяем длину текста отзыва (минимум 10 символов)
    if len(review["text"].strip()) < 10:
        logging.warning(f"Слишком короткий текст отзыва: {review['text']}")
        return None

    cleaned_review = {}

    try:
        # Очистка текста от HTML-тегов, спецсимволов и лишних пробелов
        text = review["text"]
        # Приведение к нижнему регистру
        text = text.lower()
        # Удаление HTML-тегов
        text = re.sub(r"<[^>]+>", "", text)
        # Удаление спецсимволов, кроме указанных
        text = re.sub(r"[^\w\s,.!?()]+", "", text)
        # Удаление лишних скобок
        text = re.sub(r"\)\)+", " ", text)
        # Замена переносов строк пробелами
        text = re.sub(r"[\n\r]+", " ", text)
        # Удаление одиночных символов 'n', которые могут оставаться после других преобразований
        text = re.sub(r"n", " ", text)
        # Удаление лишних пробелов
        text = re.sub(r"\s+", " ", text).strip()
        cleaned_review["text"] = text

        # Очистка рубрик (rubrics)
        rubrics = review.get("rubrics", "").lower().strip()
        cleaned_review["rubrics"] = rubrics

        # Очистка названия (name_ru)
        name = review.get("name_ru", "").strip()
        cleaned_review["name_ru"] = name

        # Минимальная очистка адреса (address)
        address = review.get("address", "").strip()
        # Замена слешей, не связанных с цифрами, на пробелы
        address = re.sub(r"(?<!\d)/|(?!\d)/", " ", address)
        # Удаление лишних пробелов
        address = re.sub(r"\s+", " ", address)
        cleaned_review["address"] = address

        # Проверка и преобразование рейтинга (rating)
        rating = float(review["rating"])
        if rating < 1 or rating > 5:
            global invalid_rating_count
            invalid_rating_count += 1
            logging.warning(f"Некорректный рейтинг: {rating} в отзыве {review}")
            return None
        cleaned_review["rating"] = rating

    except (ValueError, TypeError) as e:
        logging.error(f"Ошибка при обработке записи: {review} - {e}")
        return None

    logging.debug(f"Обработанный отзыв: {cleaned_review}")
    return cleaned_review


def clean_dataset(input_path: str, output_path: str):
    """
    Обрабатывает данные из входного файла, очищает их и сохраняет в новый файл.
    :param input_path: Путь к исходному JSON-файлу с сырыми данными.
    :param output_path: Путь для сохранения очищенных данных.
    """
    if not os.path.exists(input_path):
        logging.error(f"Файл {input_path} не найден.")
        raise FileNotFoundError(f"Файл {input_path} не найден.")

    logging.info(f"Начало обработки файла: {input_path}")

    with open(input_path, "r", encoding="utf-8") as infile:
        raw_data = json.load(infile)

    # Логируем общее количество записей
    total_reviews = len(raw_data)
    logging.info(f"Всего записей в файле: {total_reviews}")

    # Очищаем каждую запись
    cleaned_data = [clean_review_data(review) for review in raw_data]
    cleaned_data = [review for review in cleaned_data if review is not None]

    # Удаляем дубликаты записей на основе текстового поля
    before_deduplication = len(cleaned_data)
    unique_reviews = {review["text"]: review for review in cleaned_data}.values()
    after_deduplication = len(unique_reviews)

    # Логируем результаты очистки
    logging.info(f"Удалено записей с пустыми обязательными полями: {total_reviews - before_deduplication}")
    logging.info(f"Удалено дубликатов: {before_deduplication - after_deduplication}")
    logging.info(f"Удалено записей с некорректным рейтингом: {invalid_rating_count}")
    logging.info(f"Сохранено уникальных записей: {after_deduplication}")

    # Сохраняем очищенные данные
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(list(unique_reviews), outfile, ensure_ascii=False, indent=4)

    logging.info(f"Очищенные данные сохранены в {output_path}")


# Основной блок программы
if __name__ == "__main__":
    input_file = "data/dataset/geo_reviews_raw.json"
    output_file = "data/dataset/geo_reviews_cleaned.json"

    try:
        clean_dataset(input_file, output_file)
    except Exception as e:
        logging.error(f"Ошибка: {e}")
