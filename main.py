import os
import numpy as np

from src.preprocess import load_titanic_data, normalize
from src.model import LogisticRegression
from src.metrics import calculate_metrics
from src.predict_test import generate_submission


def main():
    base_path = os.path.dirname(__file__)
    train_path = os.path.join(base_path, 'data', 'train.csv')
    test_path = os.path.join(base_path, 'data', 'test.csv')
    submission_path = os.path.join(base_path, 'submission.csv')

    # 1. Підготовка тренувальних даних
    print("Завантаження даних...")
    X_raw, y = load_titanic_data(train_path)
    X = normalize(X_raw)

    # 2. Тренування моделі
    model = LogisticRegression(learning_rate=0.1, iterations=5000)
    print("Починаємо навчання моделі...")
    model.fit(X, y)

    # 3. Результати на тренувальних даних
    predictions = model.predict(X)

    # 4. Розрахунок кастомних метрик
    acc, prec, rec = calculate_metrics(y, predictions)

    print("\n=== ФІНАЛЬНІ РЕЗУЛЬТАТИ МОДЕЛІ ===")
    print(f"Accuracy  (Точність) : {acc * 100:.2f}%")
    print(f"Precision (Влучність): {prec * 100:.2f}%")
    print(f"Recall    (Повнота)  : {rec * 100:.2f}%")
    print("==================================\n")

    # 5. Генерація прогнозу для тестових даних (Kaggle Submission)
    print("Генерація файлу передбачень для тестів...")
    generate_submission(model, test_path, submission_path)


if __name__ == "__main__":
    main()