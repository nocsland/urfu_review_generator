import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from clearml import Task
from sklearn.model_selection import train_test_split
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
)

from config.settings import MODEL_NAME, CACHE_PATH, DATASETS_PATH, CLEANED_DATA_PATH, MODEL_PATH, OUTPUT_MODEL_NAME
from utils.logger import setup_logger

# Настройка логирования
logger = setup_logger(log_file='model_trainer.log')

# Отключение всех предупреждений для упрощения вывода
warnings.filterwarnings("ignore")

# Интеграция с ClearML
task = Task.init(project_name="review_generator", task_name="Fine-tuning model")


# Функция для вычисления перплексии модели
def compute_perplexity(logits, labels):
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    cross_entropy = torch.nn.functional.cross_entropy(
        log_probs.view(-1, log_probs.size(-1)), labels.view(-1), reduction='none'
    )
    perplexity = torch.exp(cross_entropy.mean())
    return perplexity


# Класс для кастомного колбэка в процессе обучения
class TrainingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "eval_loss" in logs:
                eval_loss = logs["eval_loss"]
                task.get_logger().report_scalar("Evaluation", "eval_loss", iteration=state.global_step, value=eval_loss)
                perplexity = np.exp(eval_loss)
                logs["eval_perplexity"] = perplexity
                task.get_logger().report_scalar("Evaluation", "Perplexity", iteration=state.global_step,
                                                value=perplexity)

            # Логирование всех метрик, доступных в logs
            logging.info(f"Logs: {logs}")


# Основной класс для обучения модели
class FineTuner:
    def __init__(self, model_name=MODEL_NAME, cache_dir=CACHE_PATH, model_path=MODEL_PATH, data_path=DATASETS_PATH):
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)

        # Создание директории для кэша
        self.cache_dir = Path(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=str(self.cache_dir))
        self.model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=str(self.cache_dir))

    def prepare_data(self, df):
        """Подготовка данных для обучения."""
        df['input'] = df.apply(
            lambda
                row: f"<name_ru> {row['name_ru']} <rubrics> {row['rubrics']} <rating> {row['rating']} <address> {row['address']} {self.tokenizer.eos_token}",
            axis=1,
        )
        df['output'] = df.apply(lambda row: f"<text> {row['text']} {self.tokenizer.eos_token}", axis=1)

        dataset_path = self.data_path / 'full_dataset.txt'
        # Сохранение объединенных текстов в файл
        with dataset_path.open('w', encoding='utf-8') as file:
            for input_text, target_text in zip(df['input'], df['output']):
                file.write(input_text + ' ' + target_text + '\n')

        logging.info(f"Data prepared and saved at {dataset_path}")
        return dataset_path

    def compute_class_weights(self, df):
        """Вычисление весов классов с увеличением влияния редких классов."""
        rating_counts = df['rating'].value_counts()

        # Рассчитываем веса как обратное количество отзывов
        weights = {rating: 1 / count for rating, count in rating_counts.items()}

        # Нормализуем веса так, чтобы их сумма равнялась 1
        weight_sum = sum(weights.values())
        normalized_weights = {rating: weight / weight_sum for rating, weight in weights.items()}

        logging.info(f"Normalized class weights: {normalized_weights}")
        return normalized_weights

    def split_dataset(self, input_file, train_file, val_file, test_file, train_size=0.8, val_size=0.1, test_size=0.1):
        """Разделение данных на тренировочную, валидационную и тестовую выборки."""
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Разделение на тренировочный набор и временный набор для валидации и теста
        train_lines, temp_lines = train_test_split(lines, train_size=train_size, random_state=42)
        val_lines, test_lines = train_test_split(temp_lines, train_size=val_size / (val_size + test_size),
                                                 random_state=42)

        # Сохранение выборок в отдельные файлы
        with open(train_file, 'w', encoding='utf-8') as f:
            f.writelines(train_lines)
        with open(val_file, 'w', encoding='utf-8') as f:
            f.writelines(val_lines)
        with open(test_file, 'w', encoding='utf-8') as f:
            f.writelines(test_lines)

        logging.info(
            f"Dataset split: {len(train_lines)} train lines, {len(val_lines)} validation lines, {len(test_lines)} test lines."
        )

    def fine_tune(self, dataset_path, df, output_name=OUTPUT_MODEL_NAME, num_train_epochs=5,
                  per_device_train_batch_size=16, learning_rate=5e-5, save_steps=10_000):
        """Процесс дообучения модели на кастомных данных."""
        logging.info("Starting fine-tuning process.")
        full_dataset_path = dataset_path
        train_dataset_path = self.data_path / 'train_dataset.txt'
        val_dataset_path = self.data_path / 'val_dataset.txt'
        test_dataset_path = self.data_path / 'test_dataset.txt'

        # Разделение датасета на части
        self.split_dataset(full_dataset_path, train_dataset_path, val_dataset_path, test_dataset_path)

        # Подготовка датасетов
        train_dataset = TextDataset(tokenizer=self.tokenizer, file_path=str(full_dataset_path), block_size=256)
        eval_dataset = TextDataset(tokenizer=self.tokenizer, file_path=str(val_dataset_path), block_size=256)

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        # Вычисление и применение нормализованных весов
        class_weights = self.compute_class_weights(df)
        sample_weights = torch.tensor([class_weights[int(row['rating'])] for _, row in df.iterrows()])
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

        training_args = TrainingArguments(
            output_dir=str(self.model_path / output_name),
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            save_steps=save_steps,
            learning_rate=learning_rate,
            save_total_limit=2,
            logging_dir=str(self.model_path / 'logs'),
            logging_steps=1000,
            eval_steps=1000,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            no_cuda=not torch.cuda.is_available(),
            fp16=True,
            warmup_steps=5000,
            lr_scheduler_type='linear',
            metric_for_best_model="eval_loss",
            weight_decay=0.01,
        )

        # Настройка Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            train_sampler=sampler,
            callbacks=[TrainingCallback(), EarlyStoppingCallback(early_stopping_patience=3)],
        )

        logging.info("Training started.")
        trainer.train()

        logging.info("Evaluating the model on the test dataset...")
        test_dataset = TextDataset(tokenizer=self.tokenizer, file_path=str(test_dataset_path), block_size=256)
        test_results = trainer.evaluate(test_dataset)
        logging.info(f"Test Results: {test_results}")
        task.get_logger().report_scalar("Test Metrics", "Eval Loss", iteration=0, value=test_results["eval_loss"])
        task.get_logger().report_scalar("Test Metrics", "Perplexity", iteration=0,
                                        value=np.exp(test_results["eval_loss"]))

        logging.info("Saving the fine-tuned model...")
        self.model.save_pretrained(str(self.model_path / output_name))
        self.tokenizer.save_pretrained(str(self.model_path / output_name))


if __name__ == "__main__":
    DATA_PATH = DATASETS_PATH

    # Инициализация FineTuner
    fine_tuner = FineTuner(model_path=MODEL_PATH)
    cleaned_data_path = CLEANED_DATA_PATH

    # Загрузка и подготовка данных
    df = pd.read_json(cleaned_data_path)
    dataset_path = fine_tuner.prepare_data(df)

    # Запуск обучения модели
    fine_tuner.fine_tune(
        dataset_path=dataset_path,
        df=df,
        output_name=OUTPUT_MODEL_NAME,
        num_train_epochs=15,
        per_device_train_batch_size=16,
        learning_rate=5e-5,
    )
