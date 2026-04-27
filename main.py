import os
import numpy as np
from src.preprocess import load_titanic_data, normalize
from src.model import LogisticRegression

if __name__ == "__main__":
    base_path = os.path.dirname(__file__)
    path = os.path.join(base_path, 'data', 'train.csv')
    X_raw, y = load_titanic_data(path)
    X = normalize(X_raw)
    model = LogisticRegression(learning_rate=0.1, iterations=5000)
    print("Починаємо навчання моделі...")
    model.fit(X, y)
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y) * 100
    print(f"Готово! Точність на тренувальних даних: {accuracy:.2f}%")