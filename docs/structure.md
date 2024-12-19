```sh
.
├── config                                  # Конфигурационные файлы проекта
│   ├── __init__.py                         # Инициализация пакета конфигурации
│   └── settings.py                         # Основные настройки проекта
├── data                                    # Данные проекта
│   ├── dataset                             # Исходные и обработанные данные
│   │   ├── geo_reviews_cleaned.json        # Очищенные отзывы
│   │   ├── geo-reviews-dataset-2023.tskv   # Исходный датасет отзывов
│   │   └── geo_reviews_raw.json            # Сырые отзывы
│   └── model                               # Модели и связанные артефакты
│       ├── fine_tuned_geo_reviews_model    # Обученная модель
│       ├── model_cache                     # Кэш для загрузки модели
│       └── sample                          # Примеры результатов модели
├── data.dvc                                # Конфигурация DVC для управления данными
├── docs                                    # Документация проекта
│   ├── сonception.md                       # Описание концепции проекта
│   ├── deploy_template.sh                  # Шаблон скрипта для развертывания
│   ├── structure.md                        # Описание структуры проекта
│   └── task.md                             # Постановка задач
├── .dvc                                    # Системные файлы DVC
│   ├── cache                               # Кэш DVC
│   │   └── files                           # Файлы, закэшированные DVC
│   ├── config                              # Конфигурация DVC
│   └── tmp                                 # Временные файлы DVC
│       ├── btime                           # Лог временных операций
│       ├── lock                            # Блокировка операций
│       ├── rwlock                          # Блокировка чтения/записи
│       └── rwlock.lock                     # Файл блокировки
├── .dvcignore                              # Список файлов и папок, игнорируемых DVC
├── EDA                                     # Анализ данных (EDA)
│   ├── EDA.ipynb                           # Jupyter Notebook с анализом данных
│   ├── EDA.md                              # Описание EDA в Markdown
│   └── img                                 # Визуализации для EDA
│       ├── combined_wordcloud.png          # Облака слов
│       ├── length_distribution.png         # Распределение длины отзывов
│       ├── length_distribution_by_top_rubrics.png # Распределение длинны отзывов по рубрикам
│       ├── ratings_distribution.png        # Диаграмма распределение оценок
│       └── rubrics_pie_chart.png           # Диаграмма рубрик
├── .env                                    # Файл переменных окружения
├── gdrive                                  # Доступ к Google Drive
│   └── service_account.json                # Учетные данные для Google Drive
├── LICENSE                                 # Лицензия проекта
├── logs                                    # Логи работы компонентов
│   ├── model_trainer.log                   # Лог обучения модели
│   ├── review_cleaner.log                  # Лог очистки отзывов
│   ├── review_parser.log                   # Лог парсинга отзывов
│   └── review_writer_bot.log               # Лог работы бота для записи отзывов
├── README.md                               # Основное описание проекта
├── requirements.txt                        # Список зависимостей Python
├── setup_env.sh                            # Скрипт для настройки окружения
├── src                                     # Исходный код проекта
│   ├── model_trainer.py                    # Скрипт обучения модели
│   ├── review_cleaner.py                   # Скрипт очистки отзывов
│   ├── review_parser.py                    # Скрипт парсинга отзывов
│   └── review_writer_bot.py                # Скрипт бота для работы с отзывами
└── utils                                   # Утилиты проекта
    ├── __init__.py                         # Инициализация пакета утилит
    └── logger.py                           # Логирование
```