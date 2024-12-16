import json
import logging
import os
import string
from typing import Dict

import nltk
import pymorphy2
from nltk.corpus import stopwords
from tqdm import tqdm  # Импортируем tqdm для прогресс-бара

# Настройка логирования
log_dir = 'logs'
log_file = 'review_normalizer.log'
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, log_file)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_path
)

# Загрузка необходимых ресурсов NLTK (если не установлены, нужно выполнить: nltk.download('stopwords'))
nltk.download('stopwords')

# Инициализация лемматизатора и стоп-слов
morph = pymorphy2.MorphAnalyzer()
stop_words = set(stopwords.words("russian"))


def normalize_text(text: str) -> str:
    """
    Нормализует текст: удаляет стоп-слова, лемматизирует слова и удаляет знаки препинания.
    :param text: Исходный текст.
    :return: Нормализованный текст.
    """
    # Разделяем текст на слова
    words = text.split()

    # Лемматизация и удаление стоп-слов
    normalized_words = [
        # Лемматизация
        morph.parse(word)[0].normal_form
        for word in words
        # Удаление стоп-слов и знаков препинания
        if word not in stop_words and word not in string.punctuation
    ]

    # Собираем текст обратно из нормализованных слов
    return " ".join(normalized_words)


def normalize_review_data(review: Dict[str, str]) -> Dict[str, str]:
    """
    Нормализует данные отзыва.
    :param review: Словарь с очищенными данными отзыва.
    :return: Нормализованный словарь с данными отзыва.
    """
    review["text"] = normalize_text(review["text"])  # Нормализуем текст отзыва
    # Применяем нормализацию к другим полям, если это нужно
    review["rubrics"] = normalize_text(review.get("rubrics", ""))  # Нормализация рубрик
    review["name_ru"] = review.get("name_ru", "").strip()  # Очистка названия, если необходимо
    review["address"] = review.get("address", "").strip()  # Очистка адреса, если необходимо
    return review


# Основной процесс нормализации для набора данных
def normalize_dataset(input_path: str, output_path: str):
    """
    Обрабатывает данные из файла, нормализует их и сохраняет в новый файл.
    :param input_path: Путь к исходному JSON-файлу с очищенными данными.
    :param output_path: Путь для сохранения нормализованных данных.
    """
    if not os.path.exists(input_path):
        logging.error(f"Файл {input_path} не найден.")
        raise FileNotFoundError(f"Файл {input_path} не найден.")

    logging.info(f"Начало нормализации файла: {input_path}")

    with open(input_path, "r", encoding="utf-8") as infile:
        raw_data = json.load(infile)

    # Нормализуем каждую запись с прогресс-баром
    normalized_data = []
    for review in tqdm(raw_data, desc="Нормализация отзывов", unit="отзыв"):
        normalized_data.append(normalize_review_data(review))

    # Сохраняем нормализованные данные
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(normalized_data, outfile, ensure_ascii=False, indent=4)

    logging.info(f"Нормализованные данные сохранены в {output_path}")


# Основной блок программы для нормализации
if __name__ == "__main__":
    input_file = "data/dataset/geo_reviews_cleaned.json"
    output_file = "data/dataset/geo_reviews_normalized.json"

    try:
        normalize_dataset(input_file, output_file)
    except Exception as e:
        logging.error(f"Ошибка: {e}")
