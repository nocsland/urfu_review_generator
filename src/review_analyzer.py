"""Модуль для анализа отзывов и извлечения ключевых характеристик."""

import json
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import logging
import os
from tqdm import tqdm
import pandas as pd


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ReviewAnalyzer:
    """Класс для анализа отзывов и извлечения ключевых характеристик."""

    def __init__(self, input_file: str):
        """Инициализация анализатора отзывов.

        Args:
            input_file (str): Путь к файлу с отзывами.
        """
        # Загружаем необходимые ресурсы NLTK
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)

        self.input_file = input_file
        self.reviews = []
        self.stats = defaultdict(int)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set()
        self._load_stop_words()

        # Создаем директории для графиков
        project_root = os.path.dirname(os.path.dirname(input_file))
        data_dir = os.path.join(project_root, "data")
        self.plots_dir = os.path.normpath(os.path.join(data_dir, "plots"))
        self.keywords_dir = os.path.normpath(os.path.join(data_dir, "keywords"))
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.keywords_dir, exist_ok=True)

    def _load_stop_words(self):
        """Загрузка стоп-слов с предварительной обработкой."""
        # Базовые стоп-слова
        self.stop_words = set(stopwords.words("russian"))

        # Добавляем специфичные стоп-слова
        custom_stops = {
            "это",
            "весь",
            "свой",
            "который",
            "также",
            "более",
            "менее",
            "очень",
            "просто",
            "только",
            "такой",
            "самый",
            "ещё",
            "где",
            "когда",
            "почему",
            "как",
            "что",
            "чем",
            "зачем",
            "сейчас",
            "тоже",
            "уже",
            "еще",
            "вот",
            "быть",
            "мой",
            "наш",
            "ваш",
            "их",
            "его",
            "её",
            "там",
            "здесь",
            "туда",
            "сюда",
            "куда",
            "где",
            "везде",
            "нигде",
            "потом",
            "теперь",
            "всегда",
            "никогда",
        }
        self.stop_words.update(custom_stops)

        # Используем NLTK для лемматизации стоп-слов
        lemmatized_stops = set()
        for word in self.stop_words:
            lemmatized_stops.add(self.lemmatizer.lemmatize(word))
        self.stop_words = lemmatized_stops

    def load_reviews(self) -> None:
        """Загрузка отзывов из JSON файла"""
        try:
            # Проверяем размер файла
            file_size = os.path.getsize(self.input_file)
            max_size = 100 * 1024 * 1024  # 100 MB
            if file_size > max_size:
                msg = f"Файл слишком большой: {file_size} байт"
                logging.error(msg)
                raise ValueError(msg)

            with open(self.input_file, "r", encoding="utf-8") as f:
                content = f.read()
                if not content.strip():
                    logging.error("Файл пуст")
                    raise ValueError("Пустой файл")
                self.reviews = json.loads(content)

            if not isinstance(self.reviews, list):
                msg = "Некорректный формат данных: ожидается список отзывов"
                logging.error(msg)
                raise ValueError(msg)

            # Проверка структуры каждого отзыва
            required_fields = {"text", "rating", "rubrics"}
            valid_reviews = []

            for i, review in enumerate(self.reviews):
                try:
                    # Проверяем наличие всех полей
                    missing_fields = required_fields - set(review.keys())
                    if missing_fields:
                        msg = f"Отзыв #{i}: отсутствуют поля {missing_fields}"
                        logging.warning(msg)
                        continue

                    # Валидация рейтинга
                    try:
                        review["rating"] = float(review["rating"])
                        if not 1 <= review["rating"] <= 5:
                            msg = f"Отзыв #{i}: некорректный рейтинг {review['rating']}"
                            logging.warning(msg)
                            continue
                    except (ValueError, TypeError):
                        msg = f"Отзыв #{i}: рейтинг не является числом"
                        logging.warning(msg)
                        continue

                    # Валидация текста
                    text = review["text"].strip()
                    if not text:
                        msg = f"Отзыв #{i}: пустой текст"
                        logging.warning(msg)
                        continue

                    # Валидация рубрик
                    rubrics = review["rubrics"].split(";")
                    if not any(r.strip() for r in rubrics):
                        msg = f"Отзыв #{i}: нет валидных рубрик"
                        logging.warning(msg)
                        continue

                    valid_reviews.append(review)

                except Exception as e:
                    msg = f"Отзыв #{i}: ошибка обработки: {str(e)}"
                    logging.warning(msg)
                    continue

            self.reviews = valid_reviews
            if not self.reviews:
                raise ValueError("Нет валидных отзывов после проверки")

            path = os.path.normpath(self.input_file)
            msg = f"Загружено {len(self.reviews)} валидных отзывов из {path}"
            logging.info(msg)

            self.reviews_df = pd.DataFrame(self.reviews)

        except FileNotFoundError:
            path = os.path.normpath(self.input_file)
            msg = f"Файл не найден: {path}"
            logging.error(msg)
            raise
        except json.JSONDecodeError as e:
            path = os.path.normpath(self.input_file)
            msg = f"Ошибка JSON в файле {path}: {str(e)}"
            logging.error(msg)
            raise
        except Exception as e:
            msg = f"Непредвиденная ошибка: {str(e)}"
            logging.error(msg)
            raise

    def calculate_basic_stats(self) -> Dict[str, float]:
        """Расчет базовых статистик для отзывов.

        Returns:
            Dict[str, float]: Словарь с базовыми статистиками
        """
        try:
            # Подсчет количества слов в каждом отзыве
            word_counts = [len(review["text"].split()) for review in self.reviews]

            # Расчет статистик
            stats = {
                "avg_words": np.mean(word_counts),
                "std_words": np.std(word_counts),
                "min_words": np.min(word_counts),
                "max_words": np.max(word_counts),
                "median_words": np.median(word_counts),
            }

            return stats

        except Exception:
            return {}

    def get_top_rubrics(self, n: int) -> List[str]:
        """Получение топ-N рубрик по количеству отзывов"""
        rubrics = self.reviews_df["rubrics"].apply(lambda x: x.split(";")).explode()
        rubrics = rubrics.apply(lambda x: x.strip())
        top_rubrics = rubrics.value_counts().head(n).index.tolist()
        return top_rubrics

    def plot_ratings_distribution(self):
        """Построение графика распределения рейтингов по рубрикам"""
        plt.figure(figsize=(15, 10))

        # Получаем топ-10 рубрик по количеству отзывов
        top_rubrics = self.get_top_rubrics(10)

        # Создаем DataFrame для построения графика
        ratings_data = []
        for rubric in top_rubrics:
            rubric_reviews = self.reviews_df[
                self.reviews_df["rubrics"].apply(lambda x: rubric in x)
            ]
            ratings = rubric_reviews["rating"].value_counts().sort_index()
            for rating, count in ratings.items():
                ratings_data.append(
                    {"rubric": rubric, "rating": rating, "count": count}
                )

        df = pd.DataFrame(ratings_data)

        # Создаем график
        plt.style.use("ggplot")
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_rubrics)))

        bottom = np.zeros(len(df["rating"].unique()))
        bars = []  # Сохраняем объекты баров для добавления подписей

        # Добавляем количество отзывов в легенду
        legend_labels = []
        for rubric, color in zip(top_rubrics, colors):
            total_reviews = len(
                self.reviews_df[self.reviews_df["rubrics"].apply(lambda x: rubric in x)]
            )
            legend_labels.append(f"{rubric} ({total_reviews} отзывов)")

            mask = df["rubric"] == rubric
            bar = plt.bar(
                df[mask]["rating"],
                df[mask]["count"],
                bottom=bottom[df[mask]["rating"].astype(int) - 1],
                color=color,
                label=legend_labels[-1],
                alpha=0.8,
            )
            bars.append(bar)

            # Добавляем подписи к каждому сегменту стека
            for i, rect in enumerate(bar):
                height = rect.get_height()
                if height > 0:  # Только если есть значение
                    plt.text(
                        rect.get_x() + rect.get_width() / 2.0,
                        rect.get_y() + height / 2.0,
                        f"{int(height)}",
                        ha="center",
                        va="center",
                        color="black",
                        fontweight="bold",
                        fontsize=8,
                    )

            bottom = bottom + np.array(
                [
                    (
                        df[(df["rubric"] == rubric) & (df["rating"] == i)][
                            "count"
                        ].iloc[0]
                        if len(df[(df["rubric"] == rubric) & (df["rating"] == i)]) > 0
                        else 0
                    )
                    for i in range(1, 6)
                ]
            )

        plt.grid(True, alpha=0.3)
        plt.title("Распределение рейтингов по топ-10 рубрикам", fontsize=14, pad=20)
        plt.xlabel("Рейтинг", fontsize=12)
        plt.ylabel("Количество отзывов", fontsize=12)

        # Настраиваем оси
        plt.xticks(range(1, 6))
        plt.xlim(0.5, 5.5)

        # Настраиваем легенду
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

        plt.tight_layout()

        # Сохраняем график
        plot_path = os.path.join(self.plots_dir, "ratings_distribution.png")
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        logging.info(f"График сохранен в {plot_path}")
        plt.close()

    def analyze_length_distribution(self) -> Dict[str, float]:
        """Анализ распределения длин отзывов.

        Returns:
            Dict[str, float]: Словарь со статистикой длин
        """
        lengths = [len(review["text"].split()) for review in self.reviews]

        stats = {
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "std": float(np.std(lengths)),
            "min": float(np.min(lengths)),
            "max": float(np.max(lengths)),
            "percentile_25": float(np.percentile(lengths, 25)),
            "percentile_75": float(np.percentile(lengths, 75)),
        }

        # Построение гистограммы
        plt.figure(figsize=(12, 8))
        plt.hist(lengths, bins=50, color="skyblue", edgecolor="black")
        plt.title("Распределение длин отзывов", fontsize=14, pad=20)
        plt.xlabel("Количество слов", fontsize=12)
        plt.ylabel("Частота", fontsize=12)
        plt.grid(True, alpha=0.3)

        # Добавляем статистику на график
        plt.axvline(stats["mean"], color="red", linestyle="--", label="Среднее")
        plt.axvline(stats["median"], color="green", linestyle="--", label="Медиана")
        plt.legend()

        plot_path = os.path.join(self.plots_dir, "length_distribution.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"График сохранен в {os.path.normpath(plot_path)}")
        return stats

    def analyze_rating_distribution(self) -> Dict[str, Dict[int, int]]:
        """Анализ распределения рейтингов по рубрикам.

        Returns:
            Dict[str, Dict[int, int]]: Распределение рейтингов
        """
        try:
            # Группировка отзывов по рубрикам
            reviews_by_rubric = defaultdict(list)
            for review in self.reviews:
                rubrics = review["rubrics"].split(";")
                for rubric in rubrics:
                    if rubric.strip():
                        reviews_by_rubric[rubric.strip()].append(review)

            # Подсчет распределения рейтингов для каждой рубрики
            rating_stats = {}
            for rubric, ratings in reviews_by_rubric.items():
                rating_counts = Counter(ratings)
                rating_stats[rubric] = dict(rating_counts)

            return rating_stats

        except Exception as e:
            msg = "Ошибка при анализе распределения рейтингов: " f"{str(e)}"
            logging.error(msg)
            return {}

    def _preprocess_text(self, text: str) -> str:
        """Предварительная обработка текста.

        Args:
            text: исходный текст

        Returns:
            обработанный текст
        """
        # Приводим к нижнему регистру и удаляем пунктуацию
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)

        # Используем NLTK для лемматизации
        lemmatized = [self.lemmatizer.lemmatize(word) for word in text.split()]

        # Фильтруем стоп-слова, пробелы и короткие слова
        lemmatized = [
            word.strip()
            for word in lemmatized
            if (
                word.strip()
                and word.strip() not in self.stop_words
                and len(word.strip()) > 2  # убираем слова короче 3 символов
                and not word.strip().isdigit()  # убираем числа
            )
        ]
        return " ".join(lemmatized)

    def extract_keywords(self) -> Dict[str, List[Tuple[str, float]]]:
        """Извлечение ключевых слов из отзывов по рубрикам.

        Returns:
            Dict[str, List[Tuple[str, float]]]: Ключевые слова по рубрикам
        """
        # Путь к файлу с сохраненными ключевыми словами
        keywords_file = os.path.join(os.path.dirname(self.input_file), "keywords.json")

        # Проверяем наличие данных
        if not self.reviews:
            return {}

        # Группировка отзывов по рубрикам
        reviews_by_rubric = self._group_reviews_by_rubric()
        if not reviews_by_rubric:
            return {}

        keywords = {}
        vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.95,
            stop_words=list(self.stop_words),
            ngram_range=(1, 2),
            preprocessor=self._preprocess_text,
        )

        # Обработка каждой рубрики
        for i, (rubric, reviews) in enumerate(
            tqdm(reviews_by_rubric.items(), desc="Извлечение ключевых слов", ncols=100)
        ):
            try:
                # Проверяем отзывы
                if not reviews:
                    continue

                # Фильтруем и подготавливаем тексты
                texts = []
                for review in reviews:
                    text = review.get("text", "").strip()
                    if text and len(text) >= 10:  # минимальная длина текста
                        texts.append(text)

                n_reviews = len(texts)
                if n_reviews < 2:  # Минимум 2 отзыва
                    continue

                # Предварительная обработка текстов
                processed_texts = [self._preprocess_text(text) for text in texts]
                processed_texts = [text for text in processed_texts if text.strip()]

                if len(processed_texts) < 2:  # Минимум 2 отзыва после обработки
                    continue

                try:
                    # Адаптивные параметры в зависимости от количества отзывов
                    min_df = 1
                    max_features = None
                    if n_reviews >= 10:
                        min_df = 2
                        max_features = 1000
                    elif n_reviews >= 5:
                        min_df = 1
                        max_features = 500
                    else:
                        min_df = 1
                        max_features = 200

                    vectorizer = TfidfVectorizer(
                        max_features=max_features,
                        min_df=min_df,
                        max_df=0.95,
                        stop_words=list(self.stop_words),
                        ngram_range=(1, 2),
                        preprocessor=self._preprocess_text,
                    )

                    # Токенизация и построение словаря
                    vectorizer.fit(processed_texts)

                    # Преобразование текстов
                    tfidf_matrix = vectorizer.transform(processed_texts)

                    feature_names = vectorizer.get_feature_names_out()

                    if len(feature_names) == 0:
                        continue

                    # Получаем сырые частоты слов (без IDF)
                    raw_frequencies = np.array(
                        tfidf_matrix.tocsr().sum(axis=0)
                    ).flatten()

                    # Создаем список слов с весами на основе частот
                    word_scores = []
                    for word, freq in zip(feature_names, raw_frequencies):
                        if freq > 0:
                            # Для биграмм увеличиваем вес
                            is_bigram = len(word.split()) > 1
                            # Используем частоту как основу веса
                            score = float(freq)
                            if is_bigram:
                                # Увеличиваем вес биграмм
                                score *= 1.5
                            word_scores.append((word, score))

                    # Нормализуем веса относительно максимума
                    if word_scores:
                        max_score = max(score for _, score in word_scores)
                        word_scores = [
                            (word, score / max_score) for word, score in word_scores
                        ]

                    # Сортируем по убыванию веса
                    word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)

                    # Ограничиваем до топ-10 ключевых слов
                    word_scores = word_scores[:10]

                    if not word_scores:
                        continue

                    keywords[rubric] = word_scores

                    # Визуализация ключевых слов
                    self.plot_keywords_cloud(rubric, word_scores)

                except Exception:
                    continue

            except Exception:
                continue

        if keywords:
            try:
                with open(keywords_file, "w", encoding="utf-8") as f:
                    keywords_data = {
                        rubric: [[word, float(score)] for word, score in words]
                        for rubric, words in keywords.items()
                    }
                    json.dump(keywords_data, f, ensure_ascii=False, indent=2)
                logging.info(
                    f"Извлечение ключевых слов завершено для {len(keywords)} рубрик"
                )
            except Exception:
                pass

        return keywords

    def plot_keywords_cloud(self, rubric: str, keywords: List[Tuple[str, float]]):
        """Построение облака слов для ключевых слов рубрики.

        Args:
            rubric: название рубрики
            keywords: список кортежей (слово, значимость)
        """
        plt.figure(figsize=(12, 8))

        # Ограничиваем количество ключевых слов до 10
        keywords = keywords[:10]
        words = [word for word, _ in keywords]
        scores = [score for _, score in keywords]

        # Создание горизонтального бар-плота
        y_pos = np.arange(len(words))
        plt.barh(y_pos, scores, align="center")
        plt.yticks(y_pos, words)

        plt.title(f'Топ-{len(words)} ключевых слов для рубрики "{rubric}"')
        plt.xlabel("TF-IDF значимость")

        # Сохраняем график
        plt.tight_layout()
        plot_path = os.path.join(
            self.keywords_dir, f'keywords_{rubric.replace("/", "_")}.png'
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _group_reviews_by_rubric(self):
        reviews_by_rubric = defaultdict(list)
        for review in self.reviews:
            rubrics = review["rubrics"].split(";")
            for rubric in rubrics:
                if rubric.strip():
                    reviews_by_rubric[rubric.strip()].append(review)
        return reviews_by_rubric


def download_nltk_resources():
    """Download required NLTK resources if not already present."""
    # Download basic resources
    resources = ["stopwords"]

    for resource in resources:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource)


def main():
    """Основная функция для запуска анализа отзывов."""
    download_nltk_resources()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.normpath(
        os.path.join(current_dir, "..", "data", "geo_reviews_cleaned.json")
    )

    analyzer = ReviewAnalyzer(input_file)
    analyzer.load_reviews()

    # Расчет базовой статистики
    analyzer.calculate_basic_stats()

    # Анализ длин отзывов
    analyzer.analyze_length_distribution()

    # Анализ рейтингов
    analyzer.plot_ratings_distribution()

    # Извлечение ключевых слов
    analyzer.extract_keywords()


if __name__ == "__main__":
    main()
