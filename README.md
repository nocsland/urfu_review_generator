# Проект по созданию нейронной сети для генерации текстовых отзывов

## Описание проекта

Целью проекта является разработка нейронной сети, способной генерировать текстовые отзывы о различных местах (
ресторанах, отелях, туристических объектах и т.д.) на основе заданных входных параметров, таких как рейтинг и
категория.  
Взаимодействие с моделью осуществляется через чат-бот, который позволяет пользователю вводить данные и получать
результаты.

---

## Данные

| **Источник**      | Датасет `geo-reviews-dataset-2023` от Яндекса, содержащий 500 000 уникальных отзывов об организациях в России.                                                               |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Период**        | Январь – июль 2023 года                                                                                                                                                      |
| **Фильтрация**    | Удалены короткие односложные отзывы, тексты очищены от персональных данных                                                                                                   |
| **Поля датасета** | - `address`: Адрес организации <br> - `name_ru`: Название организации <br> - `rubrics`: Список рубрик <br> - `rating`: Оценка пользователя (1-5) <br> - `text`: Текст отзыва |

После очистки данных был проведен разведочный анализ (EDA), результаты которого можно посмотреть в
файле [EDA.md](EDA/EDA.md). Использованный для анализа ноутбук находится в файле [EDA.ipynb](EDA/EDA.ipynb).

---

## Модель

| **Название**    | `ai-forever/rugpt3small_based_on_gpt2`                                                                                                   |
|-----------------|------------------------------------------------------------------------------------------------------------------------------------------|
| **Описание**    | Модель семейства RuGPT, разработанная для работы с текстами на русском языке. Базируется на архитектуре GPT-2, облегченной версии GPT-3. |
| **Особенности** | Адаптирована для узких задач с учетом особенностей русского языка. Разработана в рамках проекта AI-Forever.                              |

---

## Обучение модели

| **Модель**                | `ai-forever/rugpt3small_based_on_gpt2`, дообученная на датасете `geo-reviews-dataset-2023`.   |
|---------------------------|-----------------------------------------------------------------------------------------------|
| **Длительность обучения** | 10 эпох                                                                                       |
| **Результат**             | Метрика Perplexity: **7.563** — низкий уровень неопределенности, высокая связность текстов.   |
| **Среда обучения**        | 2 × NVIDIA RTX 3060 12Gb, библиотеки **PyTorch**, **Hugging Face Transformers**, **Datasets** |
| **Доля выборок**          | Обучение/валидация/тестирование — 80/10/10                                                    |
| **Гиперпараметры**        | - Размер батча: 16 <br> - Learning Rate: $\(5 \times 10^{-5}\)$ <br> - Weight Decay: 0.01     |

На этапе обучения модели была выполнена интеграция с **Clear ML** для удобства фиксации метрик и мониторинга процесса
обучения. Для настройки интеграции в этап обучения, после клонирования репозитория и установки зависимостей необходимо
выполнить команду `clearml-init` и ввести `App Credentials`, полученные на сайте
разработчика [clear.ml](https://clear.ml/)

---

## Стек технологий

| **Язык**            | Python                                                                           |
|---------------------|----------------------------------------------------------------------------------|
| **Фреймворк**       | PyTorch                                                                          |
| **API и интерфейс** | Telegram API для взаимодействия, Hugging Face Transformers для обработки текстов |
| **Среда**           | Серверное окружение с поддержкой GPU                                             |

---

## План реализации проекта

1. **Анализ**: Изучение задачи, выбор подходящей модели, анализ данных.
2. **Подготовка данных**: Очистка текстов, удаление дубликатов, проведение EDA, настройка гиперпараметров, обучение
   модели.
3. **Разработка бота**: Создание логики взаимодействия, настройка Telegram API.
4. **Интеграция**: Объединение модели и бота, тестирование взаимодействия.
5. **Развертывание**: Настройка серверов, создание скриптов автоматизации.

Первоначальную концепцию реализации можно посмотреть в файле [сonception.md](docs/сonception.md)

---

## Как запустить проект

1. **Клонировать репозиторий**: Выполните команду `git clone <ссылка>`, затем перейдите в директорию проекта с
   помощью: `cd <папка>`.
2. **Запустить настройку окружения**:
    - В процессе настройки будет создано виртуальное окружение и установлены необходимые зависимости
    - Для настройки окружения выполните команду:
      ```
      bash setup_env.sh
      ```
2. **Активировать виртуальное окружение**:
    - Для активации окружения выполните команду:
      ```
      source venv/bin/activate
      ```
3. **Установить зависимости**: Установите все необходимые библиотеки, выполнив
   команду: `pip install -r requirements.txt`. (Можно пропустить, если выполнен п. 2)
4. **Настроить токен**: Создайте файл `.env` в корне проекта и добавьте в него ключ Telegram API в
   формате: `BOT_KEY=<ваш токен>`.
5. **Скачать модель**: Выполните команду `dvc pull`, чтобы скачать модель с DVC-хранилища.
    - Для использования DVC необходимо иметь сервисный аккаунт Google с настроенными правами.
    - Файл ключа сервисного аккаунта в формате JSON необходимо разместить в каталог `gdrive` в корне проекта,
      предварительно создав этот каталог.
6. **Запустить бота**: Для запуска бота используйте команду: `python src/review_writer_bot.py`, и начните взаимодействие
   с ним.

Шаблон скрипта для автоматического развертывания приложения можно посмотреть в
файле [deploy_template.md](docs/deploy_template.sh)

---

## Структура проекта

- Подробнее о структуре можно посмотреть в файле [structure.md](docs/structure.md)

---

## Команда проекта

| **Имя**          | **Роль**                         |
|------------------|----------------------------------|
| Надежда Сопилова | Data Scientist, работа с данными |
| Аркадий Дубовик  | Machine Learning Engineer        |
| Максим Фролов    | NLP-специалист, настройка модели |
| Юрий Запатоцкий  | Backend-разработчик              |
| Евгений Тимофеев | DevOps-инженер                   |
| Антон Власов     | Team Lead, управление проектом   |
