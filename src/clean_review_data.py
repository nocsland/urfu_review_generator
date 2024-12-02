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


# Пример вызова
example_review = {
    "address": "Екатеринбург, ул. Московская / ул. Волгоградская / ул. Печатников  ",
    "name_ru": " Московский квартал ",
    "rating": "3.",
    "rubrics": "Жилой комплекс ",
    "text": "Московский квартал 2.\nШумно : летом <b>по ночам</b> дикие гонки!"
}

cleaned_review = clean_review_data(example_review)
print(cleaned_review)
