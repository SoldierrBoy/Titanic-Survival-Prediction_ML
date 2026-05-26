# 🚢 Titanic Survival Prediction (ML from Scratch)

Реалізація моделі машинного навчання для прогнозування виживання пасажирів на "Титаніку" без використання спеціалізованих бібліотек (`scikit-learn`, `pandas` тощо). Проєкт виконано в межах дисципліни **"Машинне навчання"** студентами групи **ІПЗ-32**.

---

## 📌 Огляд проєкту
Мета роботи — побудувати та навчити модель **Логістичної регресії (Logistic Regression)** з нуля, використовуючи лише базовий Python та бібліотеку `NumPy` для векторних обчислень.

### Основні етапи реалізації:
* **Data Preprocessing:** Завантаження даних за допомогою вбудованого модуля `csv`, очищення від пропущених значень (імпутація середнім віком) та бінаризація категоріальних ознак (`Sex`).
* **Feature Scaling:** Нормалізація ознак за допомогою MinMax Scaling для стабілізації градієнтного спуску.
* **Math Core:** Реалізація функції активації Sigmoid та алгоритму градієнтного спуску (Gradient Descent) для оптимізації ваг моделі.
* **Evaluation:** Розрахунок кастомних метрик оцінки якості класифікації (Accuracy, Precision, Recall).
* **Kaggle Submission:** Генерація фінального файлу передбачень для тестової вибірки.

---

## 📐 Математична модель

Модель базується на лінійній комбінації ознак, яка пропускається через логістичну функцію (сигмоїду):

$$z = X \cdot w + b$$
$$y_{\text{pred}} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Де $w$ — вектор ваг, $b$ — зсув (bias).

### Градієнтний спуск (Gradient Descent)
Для мінімізації функції втрат (Binary Cross-Entropy) на кожній ітерації обчислюються градієнти:

$$\frac{\partial L}{\partial w} = \frac{1}{n} X^T (y_{\text{pred}} - y)$$
$$\frac{\partial L}{\partial b} = \frac{1}{n} \sum (y_{\text{pred}} - y)$$

Оновлення параметрів відбувається за формулою:
$$w \leftarrow w - \alpha \cdot \frac{\partial L}{\partial w}$$
$$b \leftarrow b - \alpha \cdot \frac{\partial L}{\partial b}$$
*(де $\alpha$ — learning rate).*

---

## 📊 Результати навчання

Після $5000$ ітерацій навчання з коефіцієнтом швидкості навчання $\alpha = 0.1$, модель продемонструвала такі результати на тренувальних даних:

| Метрика | Значення | Опис |
| :--- | :--- | :--- |
| **Accuracy** | **79.01%** | Загальний відсоток правильно вгаданих відповідей |
| **Precision** | **73.85%** | Точність визначення виживших пасажирів |
| **Recall** | **70.18%** | Повнота покриття реальних виживших пасажирів |

Успішно згенеровано файл передбачень `submission.csv` для тестової вибірки Kaggle (кількість записів: 418).

---

## 📂 Структура проєкту

```text
Titanic-Survival-Prediction_ML/
├── data/
│   ├── train.csv         # Тренувальний датасет з Kaggle
│   └── test.csv          # Тестовий датасет для перевірки
├── src/
│   ├── preprocess.py     # Завантаження та MinMax нормалізація даних
│   ├── model.py          # Клас LogisticRegression (fit, predict, sigmoid)
│   ├── metrics.py        # Розрахунок Accuracy, Precision, Recall з нуля
│   └── predict_test.py   # Генерація submission.csv для тестів
├── main.py               # Головна точка входу (pipeline)
├── submission.csv        # Згенерований файл передбачень
└── README.md             # Документація проєкту
```
## 🚀 Запуск проєкту
### 1. Клонування репозиторію:
git clone [https://github.com/SoldierrBoy/Titanic-Survival-Prediction_ML.git](https://github.com/SoldierrBoy/Titanic-Survival-Prediction_ML.git)
cd Titanic-Survival-Prediction_ML
### 2. Встановлення залежностей:
pip install numpy
### 3. Запуск конвеєра навчання та прогнозування:
python main.py