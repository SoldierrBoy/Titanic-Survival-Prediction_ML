import numpy as np
import csv


def load_titanic_data(file_path):
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)


    X = []
    y = []
    ages = [float(row['Age']) for row in data if row['Age']]
    mean_age = sum(ages) / len(ages)

    for row in data:
        gender = 1 if row['Sex'] == 'female' else 0
        age = float(row['Age']) if row['Age'] else mean_age
        pclass = float(row['Pclass'])
        fare = float(row['Fare']) if row['Fare'] else 0.0

        X.append([pclass, gender, age, fare])

        if 'Survived' in row:
            y.append(float(row['Survived']))

    return np.array(X), np.array(y)


def normalize(X):

    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    return (X - x_min) / (x_max - x_min)