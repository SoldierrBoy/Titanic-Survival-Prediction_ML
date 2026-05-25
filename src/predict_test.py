import csv

# Тобі потрібно буде імпортувати функцію передбачення, яку написав твій друг
# Наприклад, якщо вона в файлі src/model.py і називається predict:
# from src.model import predict

def generate_submission(test_filepath, output_filepath):
    """
    Зчитує тестові дані, робить передбачення та зберігає їх у форматі для Kaggle.
    """
    predictions = []

    # 1. Зчитуємо дані з test.csv
    try:
        with open(test_filepath, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            
            for row in reader:
                passenger_id = row['PassengerId']
                
                # ТУТ МАЄ БУТИ ПІДГОТОВКА ФІЧ (ОЗНАК) ДЛЯ МОДЕЛІ
                # Тобі треба витягнути дані з row так само, як це робилося для тренувальних даних.
                # Наприклад (це лише приклад, запитай друга, які саме фічі він використовує):
                # pclass = int(row['Pclass'])
                # sex = 1 if row['Sex'] == 'female' else 0
                # features = [pclass, sex]
                
                # 2. Робимо передбачення
                # prediction = predict(features)
                
                # ТИМЧАСОВА ЗАГЛУШКА: поки ти не підключиш реальну модель, 
                # ставимо 0 (ніби ніхто не вижив), щоб перевірити генерацію файлу.
                prediction = 0 
                
                # Додаємо результат у список
                predictions.append({
                    'PassengerId': passenger_id,
                    'Survived': prediction
                })
    except FileNotFoundError:
        print(f"Помилка: Файл {test_filepath} не знайдено.")
        return

    # 3. Записуємо результати у submission.csv
    with open(output_filepath, mode='w', newline='', encoding='utf-8') as outfile:
        fieldnames = ['PassengerId', 'Survived']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(predictions)
        
    print(f"Файл {output_filepath} успішно згенеровано! Кількість записів: {len(predictions)}")

if __name__ == "__main__":
    # Вкажи правильні шляхи до файлів залежно від структури вашого проєкту
    # Зазвичай тестовий файл лежить у папці data/
    TEST_FILE = 'data/test.csv'
    SUBMISSION_FILE = 'submission.csv'
    
    generate_submission(TEST_FILE, SUBMISSION_FILE)