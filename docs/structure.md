```sh
.
├── data                                   # Данные проекта
│   ├── dataset                            # Исходные и обработанные данные
│   │   ├── geo_reviews_cleaned.json       # Очищенные отзывы
│   │   ├── geo-reviews-dataset-2023.tskv  # Исходный датасет отзывов
│   │   └── geo_reviews_raw.json           # Сырые отзывы
│   └── model                              # Модели и связанные артефакты
│       ├── fine_tuned_geo_reviews_model   # Обученная модель
│       ├── model_cache                    # Кэш для загрузки модели
│       └── sample                         # Примеры результатов модели
├── data.dvc                               # Конфигурация DVC для управления данными
├── docs                                   # Документация проекта
│   ├── Conception.md                      # Описание концепции проекта
│   ├── deploy_template.sh                 # Шаблон скрипта для развертывания
│   ├── structure.md                       # Описание структуры проекта
│   └── Task.md                            # Постановка задач
├── .dvc                                   # Системные файлы DVC
│   ├── cache                              # Кэш DVC
│   │   └── files                          # Файлы, закэшированные DVC
│   ├── config                             # Конфигурация DVC
│   └── tmp                                # Временные файлы DVC
│       ├── btime                          # Лог временных операций
│       ├── lock                           # Блокировка операций
│       ├── rwlock                         # Блокировка чтения/записи
│       └── rwlock.lock                    # Файл блокировки
├── .dvcignore                             # Список файлов и папок, игнорируемых DVC
├── .env                                   # Файл переменных окружения
├── gdrive                                 # Доступ к Google Drive
│   └── service_account.json               # Файл с учетными данными для Google Drive
├── LICENSE                                # Лицензия проекта
├── logs                                   # Логи работы различных компонентов
│   ├── model_trainer.log                  # Лог обучения модели
│   ├── review_cleaner.log                 # Лог очистки отзывов
│   ├── review_normalizer.log              # Лог нормализации данных
│   ├── review_parser.log                  # Лог парсинга отзывов
│   └── review_writer_bot.log              # Лог работы бота для записи отзывов
├── README.md                              # Основное описание проекта
├── requirements.txt                       # Список зависимостей Python
└── src                                    # Исходный код проекта
    ├── model_trainer.py                   # Скрипт обучения модели
    ├── review_cleaner.py                  # Скрипт очистки отзывов
    ├── review_parser.py                   # Скрипт парсинга отзывов
    └── review_writer_bot.py               # Скрипт бота для работы с отзывами
```