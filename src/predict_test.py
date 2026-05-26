import csv
import os
import numpy as np
from src.preprocess import load_titanic_data, normalize


def generate_submission(model, test_filepath, output_filepath):
    """
    Зчитує тестові дані, обробляє їх, робить реальне передбачення навченою моделлю
    і зберігає у форматі для Kaggle.
    """
    # 1. Завантажуємо та обробляємо тестові дані (використовуємо наш preprocess)
    if not os.path.exists(test_filepath):
        print(f"Помилка: Файл {test_filepath} не знайдено.")
        return

    # Завантажуємо фічі. y_test буде пустим масивом, бо в test.csv немає колонки Survived
    X_test_raw, _ = load_titanic_data(test_filepath)
    X_test = normalize(X_test_raw)

    # 2. Робимо реальний прогноз моделлю
    predictions = model.predict(X_test)

    # 3. Зчитуємо PassengerId з оригінального файлу, щоб зберегти структуру
    submission_data = []
    with open(test_filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            submission_data.append({
                'PassengerId': row['PassengerId'],
                'Survived': int(predictions[i])
            })

    # 4. Записуємо результати у файл submission.csv
    with open(output_filepath, mode='w', newline='', encoding='utf-8') as outfile:
        fieldnames = ['PassengerId', 'Survived']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(submission_data)

    print(f"Файл {output_filepath} успішно згенеровано! Кількість записів: {len(submission_data)}")