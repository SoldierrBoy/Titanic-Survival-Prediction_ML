import csv
import os
import pandas as pd
import joblib


def generate_submission(model_path, test_filepath, output_filepath):
    """
    Зчитує тестові дані, робить Feature Engineering, передбачення новою моделлю
    і зберігає у форматі для Kaggle.
    """
    if not os.path.exists(test_filepath):
        print(f"Помилка: Файл {test_filepath} не знайдено.")
        return

    if not os.path.exists(model_path):
        print(f"Помилка: Модель {model_path} не знайдено. Спочатку навчіть її!")
        return

    df_test = pd.read_csv(test_filepath)
    df = df_test.copy()
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping).fillna(0)

    features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'Title']
    X_test = df[features]
    model = joblib.load(model_path)
    predictions = model.predict(X_test)
    submission_data = []
    for i, row in df_test.iterrows():
        submission_data.append({
            'PassengerId': row['PassengerId'],
            'Survived': int(predictions[i])
        })

    with open(output_filepath, mode='w', newline='', encoding='utf-8') as outfile:
        fieldnames = ['PassengerId', 'Survived']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(submission_data)

    print(f"Файл {output_filepath} успішно згенеровано для Kaggle! Кількість записів: {len(submission_data)}")