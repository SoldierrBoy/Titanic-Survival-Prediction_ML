import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


def train_and_save_model():
    base_path = os.path.dirname(os.path.dirname(__file__))
    train_path = os.path.join(base_path, 'data', 'train.csv')

    df = pd.read_csv(train_path)
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

    features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone']
    X = df[features]
    y = df['Survived']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    print("Навчаємо покращену модель Random Forest (з Feature Engineering)...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    acc = accuracy_score(y_val, predictions) * 100
    prec = precision_score(y_val, predictions) * 100
    rec = recall_score(y_val, predictions) * 100

    print("\n=== РЕЗУЛЬТАТИ ПОКРАЩЕНОЇ МОДЕЛІ ===")
    print(f"Accuracy  (Точність) : {acc:.2f}%")
    print(f"Precision (Влучність): {prec:.2f}%")
    print(f"Recall    (Повнота)  : {rec:.2f}%")
    print("====================================\n")

    model_path = os.path.join(base_path, 'src', 'titanic_model.pkl')
    joblib.dump(model, model_path)
    print(f"Нову модель збережено у файл: {model_path}")


if __name__ == "__main__":
    train_and_save_model()