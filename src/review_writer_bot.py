import hashlib
import logging
import os
import re

import torch
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackQueryHandler, ContextTypes
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from config.settings import MODEL_PATH, CACHE_PATH
from utils.logger import setup_logger

# Загрузка переменных окружения из .env
load_dotenv()

# Инициализация логера
logger = setup_logger(log_file='review_writer_bot.log')

# Получение значений
api_key = os.getenv("BOT_KEY")

# Пути для модели и кэша
model_path = MODEL_PATH
cache_path = CACHE_PATH

# Загрузка модели и токенизатора
model_name = 'fine_tuned_geo_reviews_model'
tokenizer = GPT2Tokenizer.from_pretrained(
    f'{model_path}{model_name}', cache_dir=cache_path)
model = GPT2LMHeadModel.from_pretrained(
    f'{model_path}{model_name}', cache_dir=cache_path)

# Настройка устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Функция подготовки входного текста
def prepare_input_text(name_ru, rubrics, rating, address=None):
    address_part = f"<address> {address} " if address else ""
    return f"<name_ru> {name_ru} <rubrics> {rubrics} {address_part}<rating> {rating} {tokenizer.eos_token}"


# Функция инференса
def generate_text(input_text, max_length=135, temperature=0.9, top_k=50, top_p=0.95, no_repeat_ngram_size=2):
    model.eval()
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Функция очистки текста
def clean_text(text):
    # Замены "ещ" на "еще" и "вс" на "все", если после них не идет другая буква
    text = re.sub(r'\bещ\b(?!\w)', 'еще', text, flags=re.IGNORECASE)
    text = re.sub(r'\bвс\b(?!\w)', 'все', text, flags=re.IGNORECASE)

    # Удаление латиницы
    text = re.sub(r'[A-Za-z]', '', text)

    # Удаление паттернов "ыы", "ы я"
    text = re.sub(r'\bыы+\b', '', text)
    text = re.sub(r'\bы я\b', '', text, flags=re.IGNORECASE)

    # Замена повторяющихся знаков препинания на один
    text = re.sub(r'[!]{2,}', '!', text)  # Замена !! на !
    text = re.sub(r'[?]{2,}', '?', text)  # Замена ?? на ?
    text = re.sub(r'[.]{2,}', '.', text)  # Замена .... на .
    text = re.sub(r'[,]{2,}', ',', text)  # Замена ,,,, на ,

    # Обрезка после последней точки, восклицательного знака или вопросительного знака
    text = re.sub(r"([.!?])([^.!?]*)$", r"\1", text)

    # Очистка странных паттернов в зависимости от наличия восклицательного знака
    contains_exclamation = '!' in text
    if contains_exclamation:
        text = re.sub(r'[!?.,]{2,}', '!', text)  # Заменить странные паттерны на "!" если есть !
    else:
        text = re.sub(r'[!?.,]{2,}', '.', text)  # Заменить странные паттерны на "." если нет !

    # Удаление лишних символов
    text = re.sub(r'[^А-Яа-яёЁ0-9\s,\.!?()""\'\\/-]', '', text)
    # Замена смайлов на пробелы
    text = re.sub(r'!+(\))', '!', text)
    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()

    # Преобразуем первое слово в тексте в заглавную букву
    if text:
        text = text[0].upper() + text[1:]

    # Преобразуем первую букву после каждого знака препинания в заглавную
    text = re.sub(r'([.!?]\s*)([a-zа-я])', lambda m: m.group(1) + m.group(2).upper(), text)

    return text


# Обработчик команды /start с кнопкой
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("Начать", callback_data="start")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "Нажмите 'Начать' и введите данные об объекте. Я сгенерирую отзыв.",
        reply_markup=reply_markup
    )


# Обработчик нажатия кнопки
async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    await query.edit_message_text(
        "Введите описание объекта в формате:\n"
        "Название | Рубрики | Рейтинг | Адрес (опционально)"
    )


# Обработчик текстовых сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text = update.message.text

        # Хэшируем ID пользователя
        user_id_hashed = hashlib.sha256(str(update.message.from_user.id).encode()).hexdigest()[:8]

        logging.info(f"Получено сообщение от {user_id_hashed}: {text}")

        # Обработка сообщения и генерация ответа
        parts = text.split("|")
        if len(parts) < 3:
            await update.message.reply_text(
                "Ошибка: Убедитесь, что вы ввели данные в формате:\n"
                "Название | Рубрики | Рейтинг | Адрес (опционально)"
            )
            return

        name_ru = parts[0].strip()
        rubrics = parts[1].strip()
        rating = parts[2].strip()
        address = parts[3].strip() if len(parts) > 3 else None

        input_text = prepare_input_text(name_ru, rubrics, rating, address)
        generated_text = generate_text(input_text)

        # Удаление мета-тегов и параметров
        parameters = [
            f"<name_ru> {name_ru}",
            f"<rubrics> {rubrics}",
            f"<rating> {rating}",
        ]
        if address:
            parameters.append(f"<address> {address}")

        for param in parameters:
            generated_text = generated_text.replace(param, "")

        filtered_text = re.sub(r"<.*?>", "", generated_text).strip()
        filtered_text = clean_text(filtered_text)

        await update.message.reply_text(filtered_text)
        logging.info(f"Отправлен сгенерированный текст пользователю {user_id_hashed}: {filtered_text}")


    except Exception:
        logging.exception(f"Произошла ошибка при обработке сообщения")
        await update.message.reply_text(f"Произошла ошибка при обработке сообщения")


# Основной код запуска бота
if __name__ == "__main__":
    BOT_TOKEN = api_key

    application = ApplicationBuilder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(button))

    print("Бот запущен. Нажмите Ctrl+C для остановки.")
    application.run_polling()
