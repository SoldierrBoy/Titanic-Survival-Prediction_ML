import os
import sys


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import train_and_save_model
from src.predict_test import generate_submission

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(base_path, 'data', 'test.csv')
    model_path = os.path.join(base_path, 'src', 'titanic_model.pkl')
    submission_path = os.path.join(base_path, 'submission.csv')

    print("====================================================")
    print(" ЕТАП 1: Навчання моделі та пошук гіперпараметрів ")
    print("====================================================")
    train_and_save_model()

    print("\n====================================================")
    print(" ЕТАП 2: Генерація фінальних передбачень для Kaggle ")
    print("====================================================")
    generate_submission(model_path, test_path, submission_path)
    print("\n[УСПІХ] Повний цикл виконання завершено!")

if __name__ == "__main__":
    main()