import os
import sys
import unittest
import joblib
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app import app


class TitanicAppTestCase(unittest.TestCase):

    def setUp(self):
        """Налаштування перед кожним тестом: запускаємо Flask у тестовому режимі."""
        app.config['TESTING'] = True
        app.config['DEBUG'] = False
        self.client = app.test_client()
        base_path = os.path.dirname(os.path.dirname(__file__))
        self.model_path = os.path.join(base_path, 'src', 'titanic_model.pkl')

    def test_model_file_exists(self):
        """Тест: Перевірка, що файл навченої моделі .pkl існує."""
        self.assertTrue(os.path.exists(self.model_path), "Файл моделі .pkl не знайдено!")

    def test_model_prediction_logic(self):
        """Тест: Перевірка працездатності самої ML-моделі."""
        model = joblib.load(self.model_path)
        test_features = pd.DataFrame([{
            'Pclass': 1, 'Sex': 1, 'Age': 22.0, 'Fare': 50.0, 'FamilySize': 1, 'IsAlone': 1
        }])

        prediction = model.predict(test_features)[0]
        self.assertIn(prediction, [0, 1], "Модель повернула некоректне значення класу!")

    def test_home_page(self):
        """Тест: Перевірка, що головна сторінка сайту відкривається успішно (Status 200)."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        # Перевіряємо наявність реального тексту форми (в UTF-8 кодуванні)
        self.assertIn('Прогноз'.encode('utf-8'), response.data)

    def test_prediction_endpoint(self):
        """Тест: Перевірка відправки даних форми на бекенд."""
        form_data = {
            'pclass': '3',
            'sex': '0',
            'age': '22',
            'fare': '7.25'
        }
        response = self.client.post('/predict', data=form_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('Вердикт'.encode('utf-8'), response.data)


if __name__ == '__main__':
    unittest.main()