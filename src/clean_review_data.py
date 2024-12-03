import json
import os
import re
import logging
from typing import Dict, Optional


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def clean_text(text: str) -> str:
    """
    Очищает текстовое поле от нежелательных символов.

    Args:
        text: Исходный текст

    Returns:
        Очищенный текст
    """
    if not text:
        return ""

    # Заменяем переносы строк на пробелы
    text = re.sub(r'\\n', ' ', text)

    # Приводим к нижнему регистру
    text = text.lower()

    # Удаляем HTML-теги
    text = re.sub(r'<[^>]+>', '', text)

    # Удаляем спецсимволы, сохраняя базовую пунктуацию
    text = re.sub(r'[^\w\s,.!?]', '', text)

    # Убираем множественные пробелы
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def clean_review_data(review: Dict[str, str]) -> Optional[Dict[str, str]]:
    """
    Очищает текстовые поля записи отзыва.

    Args:
        review: Словарь с данными отзыва

    Returns:
        Очищенный словарь или None, если данные некорректны
    """
    if not isinstance(review, dict):
        logging.error(f"Некорректный формат отзыва: {type(review)}")
        return None

    cleaned_review = {}

    try:
        # Очистка текстового поля `text`
        if "text" in review:
            cleaned_review["text"] = clean_text(review["text"])
        else:
            logging.warning("Отсутствует поле 'text'")
            return None

        # Очистка поля `rubrics`
        if "rubrics" in review:
            rubrics = review["rubrics"]
            # Разделяем рубрики, очищаем каждую и соединяем обратно
            rubrics = [r.strip().lower() for r in rubrics.split(';')]
            cleaned_review["rubrics"] = ';'.join(filter(None, rubrics))
        else:
            logging.warning("Отсутствует поле 'rubrics'")
            return None

        # Очистка поля `name_ru`
        if "name_ru" in review:
            name = review["name_ru"]
            cleaned_review["name_ru"] = name.strip()
        else:
            logging.warning("Отсутствует поле 'name_ru'")
            return None

        # Минимальная очистка адреса
        if "address" in review:
            address = review["address"]
            cleaned_review["address"] = re.sub(r'\s+', ' ', address).strip()
        else:
            logging.warning("Отсутствует поле 'address'")
            return None

        # Преобразование оценки в float
        if "rating" in review:
            try:
                rating = float(review["rating"].rstrip('.'))
                if 1 <= rating <= 5:
                    cleaned_review["rating"] = rating
                else:
                    logging.warning(
                        f"Некорректное значение рейтинга: {rating}"
                    )
                    return None
            except (ValueError, AttributeError) as e:
                logging.warning(f"Ошибка преобразования рейтинга: {e}")
                return None
        else:
            logging.warning("Отсутствует поле 'rating'")
            return None

        return cleaned_review

    except Exception as e:
        logging.error(f"Ошибка при очистке отзыва: {e}")
        return None


def clean_dataset(input_path: str, output_path: str) -> None:
    """
    Читает необработанные данные, очищает их и сохраняет в новый файл.

    Args:
        input_path: Путь к исходному JSON-файлу с сырыми данными
        output_path: Путь для сохранения очищенных данных
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Файл {input_path} не найден")

    try:
        with open(input_path, "r", encoding="utf-8") as infile:
            raw_data = json.load(infile)
    except json.JSONDecodeError as e:
        logging.error(f"Ошибка чтения JSON файла: {e}")
        raise

    if not isinstance(raw_data, list):
        raise ValueError("Входные данные должны быть списком отзывов")

    # Очищаем данные и фильтруем None значения
    cleaned_data = list(filter(None, [
        clean_review_data(review) for review in raw_data
    ]))

    logging.info(f"Обработано отзывов: {len(raw_data)}")
    logging.info(f"Успешно очищено: {len(cleaned_data)}")

    # Создаем директорию для выходного файла, если её нет
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as outfile:
            json.dump(cleaned_data, outfile, ensure_ascii=False, indent=4)
        logging.info(f"Очищенные данные сохранены в {os.path.normpath(output_path)}")
    except IOError as e:
        logging.error(f"Ошибка сохранения файла: {e}")
        raise


if __name__ == "__main__":
    # Используем абсолютные пути
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.normpath(os.path.join(current_dir, "..", "data", "geo_reviews_raw.json"))
    output_file = os.path.normpath(os.path.join(
        current_dir, "..", "data", "geo_reviews_cleaned.json"
    ))

    try:
        clean_dataset(input_file, output_file)
    except Exception as e:
        logging.error(f"Критическая ошибка: {e}")
        raise
