import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

# Настройка логирования
log_dir = 'logs'
log_file = 'fine_tuning.log'
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, log_file)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_path
)

# Отключить все предупреждения
warnings.filterwarnings("ignore")


# Функция для вычисления перплексии
def compute_perplexity(logits, labels):
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    cross_entropy = torch.nn.functional.cross_entropy(log_probs.view(-1, log_probs.size(-1)), labels.view(-1),
                                                      reduction='none')
    perplexity = torch.exp(cross_entropy.mean())
    return perplexity


class TrainingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logging.info(
                f"Loss: {logs.get('loss')}, Grad Norm: {logs.get('grad_norm')}, Learning Rate: {logs.get('learning_rate')}, Epoch: {logs.get('epoch')}")


class PerplexityCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if "eval_loss" in logs:
            eval_loss = logs["eval_loss"]
            perplexity = np.exp(eval_loss)
            logs["eval_perplexity"] = perplexity
            logging.info(f"Perplexity: {perplexity}")


class FineTuner:
    def __init__(self, model_name='ai-forever/rugpt3small_based_on_gpt2', cache_dir='model_cache', data_path='data'):
        self.data_path = Path(data_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=str(self.data_path / cache_dir))
        self.model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=str(self.data_path / cache_dir))

    def prepare_data(self, df):
        """Подготовка данных и сохранение их в файл."""
        df['input'] = df.apply(
            lambda
                row: f"<name_ru> {row['name_ru']} <rubrics> {row['rubrics']} <rating> {row['rating']} <address> {row['address']} {self.tokenizer.eos_token}",
            axis=1
        )
        df['output'] = df.apply(lambda row: f"<text> {row['text']} {self.tokenizer.eos_token}", axis=1)

        dataset_path = self.data_path / 'full_dataset.txt'
        with dataset_path.open('w', encoding='utf-8') as file:
            for input_text, target_text in zip(df['input'], df['output']):
                file.write(input_text + ' ' + target_text + '\n')

        logging.info(f"Data prepared and saved at {dataset_path}")
        return dataset_path

    def split_dataset(self, input_file, train_file, val_file, test_file, train_size=0.8, val_size=0.1, test_size=0.1):
        """Разделяет датасет на тренировочную, валидационную и тестовую выборки."""
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        train_lines, temp_lines = train_test_split(lines, train_size=train_size, random_state=42)
        val_lines, test_lines = train_test_split(temp_lines, train_size=val_size / (val_size + test_size),
                                                 random_state=42)

        with open(train_file, 'w', encoding='utf-8') as f:
            f.writelines(train_lines)

        with open(val_file, 'w', encoding='utf-8') as f:
            f.writelines(val_lines)

        with open(test_file, 'w', encoding='utf-8') as f:
            f.writelines(test_lines)

        logging.info(
            f"Dataset split: {len(train_lines)} train lines, {len(val_lines)} validation lines, {len(test_lines)} test lines.")

    def fine_tune(self, dataset_path, output_name='fine_tuned_geo_reviews_model', num_train_epochs=5,
                  per_device_train_batch_size=16, learning_rate=5e-5, save_steps=10_000):
        """Тренировка модели на полном датасете, но с использованием разделенных данных для оценки."""
        logging.info("Starting fine-tuning process.")
        full_dataset_path = dataset_path
        train_dataset_path = self.data_path / 'train_dataset.txt'
        val_dataset_path = self.data_path / 'val_dataset.txt'
        test_dataset_path = self.data_path / 'test_dataset.txt'

        # Разделение датасета на тренировочную, валидационную и тестовую выборки
        self.split_dataset(full_dataset_path, train_dataset_path, val_dataset_path, test_dataset_path, train_size=0.8,
                           val_size=0.1, test_size=0.1)

        # Загрузка данных
        train_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=str(full_dataset_path),  # Полный датасет используется для тренировки
            block_size=256
        )
        eval_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=str(val_dataset_path),
            block_size=256
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        training_args = TrainingArguments(
            output_dir=str(self.data_path / output_name),
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            save_steps=save_steps,
            learning_rate=learning_rate,
            save_total_limit=2,
            logging_dir=str(self.data_path / 'logs'),
            logging_steps=1000,
            eval_steps=1000,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            no_cuda=not torch.cuda.is_available(),
            fp16=True,
            warmup_steps=3000,
            lr_scheduler_type='linear',
            metric_for_best_model="perplexity",
        )

        # Добавление callback для перплексии
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,  # Используем полный датасет для тренировки
            eval_dataset=eval_dataset,
            callbacks=[PerplexityCallback, TrainingCallback]
        )

        logging.info("Training started.")
        trainer.train()

        # Оценка модели на тестовом наборе
        logging.info("Evaluating the model on the test dataset...")
        test_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=str(test_dataset_path),
            block_size=256
        )
        test_results = trainer.evaluate(test_dataset)

        logging.info(f"Test Results: {test_results}")

        # Сохранение обученной модели и токенизатора
        logging.info("Saving the fine-tuned model...")
        self.model.save_pretrained(str(self.data_path / output_name))
        self.tokenizer.save_pretrained(str(self.data_path / output_name))


if __name__ == "__main__":
    DATA_PATH = 'data'
    CLEANED_DATA_FILE = 'geo_reviews_cleaned.json'

    fine_tuner = FineTuner(data_path=DATA_PATH)
    cleaned_data_path = Path(DATA_PATH) / CLEANED_DATA_FILE
    df = pd.read_json(cleaned_data_path)

    # Подготовка данных
    dataset_path = fine_tuner.prepare_data(df)

    # Тренировка и оценка модели
    fine_tuner.fine_tune(
        dataset_path=dataset_path,
        output_name="fine_tuned_geo_reviews_model",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        learning_rate=5e-5
    )
