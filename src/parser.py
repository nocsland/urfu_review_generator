import os
import json
import logging
from typing import List, Dict, Any


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# Константы
REQUIRED_FIELDS = {'text', 'rating', 'rubrics', 'name_ru', 'address'}
RATING_MIN = 1.0
RATING_MAX = 5.0


class ParsingError(Exception):
    """Исключение при ошибках парсинга данных."""
    pass


def validate_review(review_data: Dict[str, Any]) -> None:
    """
    Проверяет корректность данных отзыва.

    Args:
        review_data: Словарь с данными отзыва

    Raises:
        ParsingError: Если данные не проходят валидацию
    """
    # Проверка обязательных полей
    missing_fields = REQUIRED_FIELDS - set(review_data.keys())
    if missing_fields:
        raise ParsingError(f"Отсутствуют обязательные поля: {missing_fields}")

    # Проверка типов и значений
    try:
        rating = float(review_data['rating'])
        if not (RATING_MIN <= rating <= RATING_MAX):
            raise ParsingError(
                f"Рейтинг должен быть от {RATING_MIN} до {RATING_MAX}"
            )
    except (ValueError, TypeError):
        raise ParsingError("Некорректное значение рейтинга")

    if not (
        isinstance(review_data.get('text', ''), str)
        and review_data['text'].strip()
    ):
        raise ParsingError("Текст отзыва не может быть пустым")

    if not (
        isinstance(review_data.get('name_ru', ''), str)
        and review_data['name_ru'].strip()
    ):
        raise ParsingError("Название организации не может быть пустым")


def parse_tskv_file(file_path: str, max_reviews: int = None) -> List[Dict[str, Any]]:
    """
    Парсер TSKV-файла с данными об отзывах.

    Args:
        file_path: Путь к файлу формата TSKV
        max_reviews: Максимальное количество отзывов для обработки

    Returns:
        Список словарей с данными отзывов

    Raises:
        FileNotFoundError: Если файл не найден
        ParsingError: При ошибках парсинга
    """
    reviews = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден")

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            review_count = 0
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    # Разбор строки TSKV
                    review_data = {}
                    fields = line.split('\t')
                    for field in fields:
                        if '=' not in field:
                            logging.warning(
                                f"Пропущено поле без разделителя '=' в строке {line_num}"
                            )
                            continue

                        key, value = field.split('=', 1)
                        review_data[key.strip()] = value.strip()

                    # Валидация отзыва
                    validate_review(review_data)
                    reviews.append(review_data)
                    review_count += 1

                    if review_count % 100 == 0:
                        logging.info(f"Обработано отзывов: {review_count}")

                    if max_reviews and review_count >= max_reviews:
                        logging.info(f"Достигнут лимит в {max_reviews} отзывов")
                        break

                except ParsingError as e:
                    logging.warning(f"Ошибка в строке {line_num}: {str(e)}")
                    continue

    except Exception as e:
        logging.error(f"Ошибка при чтении файла: {str(e)}")
        raise

    logging.info(f"Всего обработано отзывов: {len(reviews)}")
    return reviews


def save_to_json(data: List[Dict[str, Any]], output_path: str) -> None:
    """
    Сохраняет данные в формате JSON.

    Args:
        data: Данные для сохранения
        output_path: Путь для сохранения JSON-файла

    Raises:
        OSError: При ошибках создания директории или записи файла
    """
    try:
        # Создаем директорию, если её нет
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        logging.info(f"Данные успешно сохранены в {os.path.normpath(output_path)}")

    except Exception as e:
        logging.error(f"Ошибка при сохранении файла: {str(e)}")
        raise


if __name__ == "__main__":
    # Используем абсолютные пути
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.normpath(os.path.join(
        current_dir, "..", "data", "geo-reviews-dataset-2023.tskv"
    ))
    output_file_path = os.path.normpath(os.path.join(
        current_dir, "..", "data", "geo_reviews_raw.json"
    ))
    max_reviews = 1000  # Ограничиваем количество отзывов до 1000

    try:
        # Парсим файл
        parsed_reviews = parse_tskv_file(data_file_path, max_reviews)

        # Сохраняем результат в JSON
        save_to_json(parsed_reviews, output_file_path)

    except Exception as e:
        logging.error(f"Критическая ошибка: {str(e)}")
        raise
