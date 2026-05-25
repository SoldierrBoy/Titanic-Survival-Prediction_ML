import os
import numpy as np

from src.preprocess import load_titanic_data, normalize
from src.model import LogisticRegression

# from src.metrics import calculate_metrics

def main():
    # шляхи до файлів
    base_path = os.path.dirname(__file__)
    path = os.path.join(base_path, 'data', 'train.csv')
    
    # підготовка даних
    X_raw, y = load_titanic_data(path)
    X = normalize(X_raw)
    
    # тренування моделі
    model = LogisticRegression(learning_rate=0.1, iterations=5000)
    print("Починаємо навчання моделі...")
    model.fit(X, y)
    
    # результати
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y) * 100
    
    print(f"Готово! Точність на тренувальних даних: {accuracy:.2f}%")
    
    # кастомні метрики (поки вимкнено)
    # acc, prec, rec = calculate_metrics(y.tolist(), predictions.tolist())
    # print(f"Precision: {prec:.4f} | Recall: {rec:.4f}")

if __name__ == "__main__":
    main()