import json
import logging
import os
import re
from typing import Dict

# Настройка логгирования
log_dir = 'logs'
log_file = 'clean_review.log'
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, log_file)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_path
)

# Глобальная переменная для подсчета некорректных рейтингов
invalid_rating_count = 0


def clean_review_data(review: Dict[str, str]) -> Dict[str, str]:
    """
    Очищает текстовые поля записи отзыва.
    :param review: Словарь с данными отзыва.
    :return: Очищенный словарь или None, если данные некорректны.
    """
    logging.debug(f"Начало обработки отзыва: {review}")

    # Проверка на обязательные столбцы
    if not review.get("text") or not review.get("name_ru") or review.get("rating") is None:
        logging.warning(f"Пропущены обязательные поля в отзыве: {review}")
        return None

    # Дополнительные проверки данных
    if len(review["text"].strip()) < 10:  # Например, если текст отзыва слишком короткий
        logging.warning(f"Слишком короткий текст отзыва: {review['text']}")
        return None

    cleaned_review = {}

    try:
        # Очистка текстового поля `text`
        text = review["text"]
        text = text.lower()
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"[^\w\s,.!?()]+", "", text)
        text = re.sub(r"\)\)+", " ", text)
        text = re.sub(r"[\n\r]+", " ", text)
        text = re.sub(r"n", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        cleaned_review["text"] = text

        # Очистка поля `rubrics`
        rubrics = review.get("rubrics", "").lower().strip()
        cleaned_review["rubrics"] = rubrics

        # Очистка поля `name_ru`
        name = review.get("name_ru", "").strip()
        cleaned_review["name_ru"] = name

        # Минимальная очистка адреса
        address = review.get("address", "").strip()
        address = re.sub(r"(?<!\d)/|(?!\d)/", " ", address)
        address = re.sub(r"\s+", " ", address)
        cleaned_review["address"] = address

        # Преобразование рейтинга
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
    Читает необработанные данные, очищает их и сохраняет в новый файл.
    :param input_path: Путь к исходному JSON-файлу с сырыми данными.
    :param output_path: Путь для сохранения очищенных данных.
    """
    if not os.path.exists(input_path):
        logging.error(f"Файл {input_path} не найден.")
        raise FileNotFoundError(f"Файл {input_path} не найден.")

    logging.info(f"Начало обработки файла: {input_path}")

    with open(input_path, "r", encoding="utf-8") as infile:
        raw_data = json.load(infile)

    # Фильтруем данные, удаляя записи с пустыми обязательными полями
    total_reviews = len(raw_data)
    logging.info(f"Всего записей в файле: {total_reviews}")

    cleaned_data = [clean_review_data(review) for review in raw_data]
    cleaned_data = [review for review in cleaned_data if review is not None]  # Убираем None

    # Подсчёт дубликатов
    before_deduplication = len(cleaned_data)
    unique_reviews = {review["text"]: review for review in cleaned_data}.values()
    after_deduplication = len(unique_reviews)

    # Логи
    logging.info(f"Удалено записей с пустыми обязательными полями: {total_reviews - before_deduplication}")
    logging.info(f"Удалено дубликатов: {before_deduplication - after_deduplication}")
    logging.info(f"Удалено записей с некорректным рейтингом: {invalid_rating_count}")
    logging.info(f"Сохранено уникальных записей: {after_deduplication}")

    # Сохранение очищенных данных
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(list(unique_reviews), outfile, ensure_ascii=False, indent=4)

    logging.info(f"Очищенные данные сохранены в {output_path}")


# Основной блок
if __name__ == "__main__":
    input_file = "data/geo_reviews_raw.json"
    output_file = "data/geo_reviews_cleaned.json"

    try:
        clean_dataset(input_file, output_file)
    except Exception as e:
        logging.error(f"Ошибка: {e}")
