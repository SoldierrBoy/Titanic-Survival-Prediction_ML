import os
import sqlite3
import logging
from datetime import datetime
import joblib
import numpy as np
from flask import Flask, request, render_template
import pandas as pd
app = Flask(__name__, template_folder='templates', static_folder='static')

# ==========================================
# 1. НАЛАШТУВАННЯ ЛОГУВАННЯ (LOGGING)
# ==========================================
base_path = os.path.dirname(__file__)
log_file_path = os.path.join(base_path, 'app.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s]: %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.info("Сервер запускається. Налаштування логування успішне.")

# ==========================================
# 2. ЗАВАНТАЖЕННЯ ML-МОДЕЛІ
# ==========================================
model_path = os.path.join(base_path, 'src', 'titanic_model.pkl')
try:
    model = joblib.load(model_path)
    logging.info("ML-модель Random Forest успішно завантажена з файлу .pkl")
except Exception as e:
    logging.error(f"Критична помилка при завантаженні моделі: {str(e)}")
    raise e

# ==========================================
# 3. НАЛАШТУВАННЯ БАЗИ ДАНИХ (SQLite)
# ==========================================
db_path = os.path.join(base_path, 'database.db')


def init_db():
    """Створює базу даних та таблицю, якщо вони ще не існують."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pclass INTEGER,
                sex INTEGER,
                age REAL,
                fare REAL,
                prediction_result INTEGER,
                probability REAL,
                created_at TEXT
            )
        ''')
        conn.commit()
        conn.close()
        logging.info("База даних SQLite успішно ініціалізована.")
    except Exception as e:
        logging.error(f"Помилка ініціалізації бази даних: {str(e)}")


init_db()


# ==========================================
# 4. МАРШРУТИ (ROUTES) FLASK
# ==========================================

@app.route('/')
def home():
    logging.info("Користувач зайшов на головну сторінку.")

    # ДІСТАЄМО З БАЗИ 5 ОСТАННІХ ЗАПИСІВ
    history = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT pclass, sex, age, fare, prediction_result, probability 
            FROM predictions 
            ORDER BY id DESC LIMIT 5
        ''')
        history = cursor.fetchall()
        conn.close()
    except Exception as e:
        logging.error(f"Помилка читання історії з БД: {str(e)}")

    # Передаємо історію у фронтенд
    return render_template('index.html', history=history)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        pclass = int(request.form['pclass'])
        sex = int(request.form['sex'])
        age = float(request.form['age'])
        fare = float(request.form['fare'])

        logging.info(f"Отримано запит на прогноз: Pclass={pclass}, Sex={sex}, Age={age}, Fare={fare}")

        # Визначення титулу
        if sex == 1:
            title = 3
        else:
            title = 4 if age < 18 else 1

        family_size = 1
        is_alone = 1

        features = pd.DataFrame([{
            'Pclass': pclass, 'Sex': sex, 'Age': age, 'Fare': fare,
            'FamilySize': family_size, 'IsAlone': is_alone, 'Title': title
        }])

        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1] * 100)

        # Збереження в БД
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (pclass, sex, age, fare, prediction_result, probability, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (pclass, sex, age, fare, prediction, probability, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()

        # ОДРАЗУ ОНОВЛЮЄМО ІСТОРІЮ ДЛЯ СТОРІНКИ ВІДПОВІДІ
        cursor.execute('''
            SELECT pclass, sex, age, fare, prediction_result, probability 
            FROM predictions 
            ORDER BY id DESC LIMIT 5
        ''')
        history = cursor.fetchall()
        conn.close()

        logging.info(f"Прогноз успішно збережено в БД. Результат={prediction}, Ймовірність={probability:.2f}%")
        result_text = "Вижив(ла) 🎉" if prediction == 1 else "Загинув(ла) 💀"
        return render_template('index.html',
                               prediction_text=f"Вердикт: {result_text}",
                               probability_text=f"Ймовірність виживання: {probability:.2f}%",
                               pclass=pclass, sex=sex, age=age, fare=fare,
                               history=history)  # Передаємо оновлену історію

    except Exception as e:
        logging.error(f"Помилка під час обробки прогнозу: {str(e)}")
        return f"Помилка сервера: {str(e)}"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)