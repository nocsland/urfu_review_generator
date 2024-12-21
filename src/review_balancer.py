import json
import logging
import random

from ruwordnet import RuWordNet
from tqdm import tqdm  # Импортируем tqdm для прогресс-бара

from config.settings import CLEANED_DATA_PATH, BALANCED_DATA_PATH
from utils.logger import setup_logger

# Инициализация RuWordNet
ruwn = RuWordNet()

# Настройка логирования: создание директории и файла для логов
logger = setup_logger(log_file='review_balancer.log')

# Функция для получения синонимов слова
synonym_cache = {}


def get_synonyms(word: str) -> list:
    """
    Получает список синонимов для заданного лемматизированного слова.

    Если синонимы для слова уже были получены, возвращает их из кэша.
    Иначе, получает синонимы через RuWordNet и сохраняет их в кэш.

    :param word: Лемматизированное слово.
    :return: Список синонимов для слова.
    """
    if word in synonym_cache:
        return synonym_cache[word]

    logging.debug(f"Получение синонимов для слова: {word}")
    synonyms = set()
    try:
        synsets = ruwn.get_synsets(word)
        for synset in synsets:
            for sense in synset.senses:
                synonyms.add(sense.lemma)
    except Exception as e:
        logging.error(f"Ошибка при получении синонимов для слова '{word}': {e}")

    synonyms.discard(word)
    synonym_cache[word] = list(synonyms)
    logging.debug(f"Найдены синонимы: {synonyms}")
    return synonym_cache[word]


# Функция для случайной аугментации текста с помощью синонимов
def augment_text(text: str) -> str:
    """
    Выполняет замену слов в тексте случайными синонимами, приводя текст к нижнему регистру, но сохраняя
    первую букву заглавной, если слово начинается с неё.
    :param text: Лемматизированный текст.
    :return: Текст с заменёнными словами в нижнем регистре, с сохранением первой буквы заглавной.
    """
    logging.info(f"Аугментация текста: {text}")
    words = text.split()
    augmented_words = []

    for word in words:
        # Приводим слово к нижнему регистру
        lower_word = word.lower()

        # Если слово начинается с большой буквы, сохраняем эту информацию
        is_capitalized = word[0].isupper()

        # Получаем синонимы для слова
        synonyms = get_synonyms(lower_word)

        # Если есть синонимы, случайным образом выбираем один
        if synonyms:
            new_word = random.choice(synonyms).lower()  # Преобразуем синоним в нижний регистр
            # Если исходное слово начиналось с заглавной буквы, делаем первую букву заглавной
            if is_capitalized:
                new_word = new_word.capitalize()
            augmented_words.append(new_word)
            logging.debug(f"Слово '{word}' заменено на '{new_word}'")
        else:
            augmented_words.append(word)  # если синонимов нет, оставляем слово без изменений
            logging.debug(f"Слово '{word}' оставлено без изменений")

    augmented_text = " ".join(augmented_words)
    logging.info(f"Текст после аугментации: {augmented_text}")
    return augmented_text


# Функция для балансировки отзывов
def balance_reviews_by_rating(reviews: list) -> list:
    """
    Балансирует отзывы по количеству записей для каждого рейтинга.
    :param reviews: Список отзывов, где каждый отзыв содержит ключи 'rating' и 'text'.
    :return: Сбалансированный список отзывов.
    """
    logging.info("Начинаем балансировку отзывов...")

    # Шаг 1: Подсчитать количество отзывов для каждого рейтинга
    rating_counts = {}
    for review in reviews:
        rating = review['rating']
        if rating not in rating_counts:
            rating_counts[rating] = 0
        rating_counts[rating] += 1

    logging.info(f"Подсчитано количество отзывов по рейтингам: {rating_counts}")

    # Шаг 2: Найти максимальное количество записей среди всех рейтингов
    max_reviews_count = max(rating_counts.values())
    logging.info(f"Максимальное количество записей среди рейтингов: {max_reviews_count}")

    # Шаг 3: Для каждого рейтинга, если записей меньше максимального, добавляем недостающие уникальные записи
    balanced_reviews = []
    for rating in tqdm(rating_counts, desc="Обработка рейтингов"):  # Прогресс-бар для обработки рейтингов
        logging.info(f"Обрабатываем рейтинг: {rating}")

        # Список отзывов для текущего рейтинга
        reviews_with_current_rating = [review for review in reviews if review['rating'] == rating]

        # Получаем текущее количество отзывов для данного рейтинга
        count_for_rating = len(reviews_with_current_rating)
        logging.info(f"Количество отзывов с рейтингом {rating}: {count_for_rating}")

        # Если количество записей с данным рейтингом меньше максимального, создаем новые уникальные записи
        if count_for_rating < max_reviews_count:
            num_to_add = max_reviews_count - count_for_rating
            logging.info(f"Не хватает {num_to_add} записей для рейтинга {rating}")

            for _ in tqdm(range(num_to_add), desc=f"Аугментация для рейтинга {rating}"):  # Прогресс-бар для аугментации
                # Выбираем случайный отзыв с текущим рейтингом для аугментации
                review_to_augment = random.choice(reviews_with_current_rating)
                logging.debug(f"Выбранный отзыв для аугментации: {review_to_augment}")

                # Создаем уникальный текст с помощью аугментации
                new_review = review_to_augment.copy()
                new_review['text'] = augment_text(new_review['text'])  # Аугментация текста

                balanced_reviews.append(new_review)
        else:
            balanced_reviews.extend(reviews_with_current_rating)  # Добавляем все отзывы с этим рейтингом

    logging.info("Балансировка завершена.")
    return balanced_reviews


# Чтение данных из файла и запись результата
def main(input_file: str, output_file: str):
    """
    Основная функция для чтения данных, балансировки и сохранения результата.
    :param input_file: Путь к входному файлу.
    :param output_file: Путь к выходному файлу.
    """
    try:
        # Шаг 1: Чтение данных из файла
        logging.info(f"Чтение данных из файла: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as infile:
            reviews = json.load(infile)
        logging.info(f"Успешно загружено {len(reviews)} отзывов.")

        # Шаг 2: Балансировка отзывов
        balanced_reviews = balance_reviews_by_rating(reviews)

        # Шаг 3: Сохранение результата в файл
        logging.info(f"Сохранение сбалансированных отзывов в файл: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(balanced_reviews, outfile, ensure_ascii=False, indent=4)
        logging.info(f"Результаты успешно сохранены. Обработано {len(balanced_reviews)} отзывов.")
    except Exception as e:
        logging.error(f"Ошибка при обработке данных: {e}")


# Параметры входного и выходного файла
input_file = CLEANED_DATA_PATH
output_file = BALANCED_DATA_PATH  # Файл для сохранения

if __name__ == "__main__":
    main(input_file, output_file)
