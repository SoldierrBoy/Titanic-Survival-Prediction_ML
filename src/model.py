import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
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
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(0)
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'Title']
    X = df[features]
    y = df['Survived']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    base_model = RandomForestClassifier(class_weight='balanced', random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [3, 5, 7, 10]
    }

    print("Запускаємо GridSearchCV для пошуку найкращих гіперпараметрів...")
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print("\n=== НАЙКРАЩІ ПАРАМЕТРИ ЗНАЙДЕНІ GRIDSEARCH ===")
    print(grid_search.best_params_)
    print(f"Найкраща точність на крос-валідації під час пошуку: {grid_search.best_score_ * 100:.2f}%")
    print("=============================================\n")

    predictions = best_model.predict(X_val)
    acc = accuracy_score(y_val, predictions) * 100
    prec = precision_score(y_val, predictions) * 100
    rec = recall_score(y_val, predictions) * 100

    print("=== РЕЗУЛЬТАТИ ОПТИМІЗОВАНОЇ МОДЕЛІ ===")
    print(f"Accuracy  (Точність) : {acc:.2f}%")
    print(f"Precision (Влучність): {prec:.2f}%")
    print(f"Recall    (Повнота)  : {rec:.2f}%")
    print("=======================================\n")

    print("=== ОНОВЛЕНА ВАЖЛИВІСТЬ ОЗНАК ===")
    importances = best_model.feature_importances_
    for name, importance in zip(features, importances):
        print(f"Ознака '{name}': {importance * 100:.2f}%")
    print("=================================\n")

    # СТВОРЕННЯ ГРАФІКА ВАЖЛИВОСТІ ОЗНАК
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        static_dir = os.path.join(base_path, 'static')
        os.makedirs(static_dir, exist_ok=True)

        plt.figure(figsize=(8, 5))
        feat_importances = pd.Series(importances, index=features).sort_values(ascending=True)
        feat_importances.plot(kind='barh', color='#2a5298')
        plt.title('Важливість ознак у моделі Random Forest (%)', fontsize=12, fontweight='bold', pad=15)
        plt.xlabel('Вплив на прогноз', fontsize=10)
        plt.tight_layout()

        plt.savefig(os.path.join(static_dir, 'importance.png'), dpi=150)
        plt.close()
        print("Графік важливості ознак успішно збережено у static/importance.png")
    except Exception as e:
        print(f"Помилка генерації графіка: {e}")

    model_path = os.path.join(base_path, 'src', 'titanic_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"Оптимальну модель успішно збережено у файл: {model_path}")


if __name__ == "__main__":
    train_and_save_model()