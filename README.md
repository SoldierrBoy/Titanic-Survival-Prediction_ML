# 🚢 Titanic Survival Prediction & Analytics Dashboard

An industrial-grade, full-stack Machine Learning web application designed to predict passenger survival rates using the Titanic dataset. This project encompasses the entire lifecycle of a classical Data Science product—from exploratory data analysis (EDA) and automated hyperparameter optimization to database storage pipelines and containerized deployment.

Developed as a team architecture project within the **"Machine Learning"** curriculum by 3rd-year Software Engineering students (**Group ІПЗ-32**).

---

## 📊 Performance Matrix & ML Innovations
By migrating from a legacy baseline model to an advanced ensemble architecture and implementing deep feature engineering, the prediction framework achieved a significant performance leap:

* **Automated Hyperparameter Tuning:** Replaced manual tuning with `scikit-learn`'s `GridSearchCV` running a 5-fold cross-validation mesh across 16 combinations.
* **Optimal Architecture Selected:** `RandomForestClassifier(max_depth=7, n_estimators=100, class_weight='balanced')`.
* **Advanced Feature Engineering:** Extracted hidden socioeconomic features from passenger names via regex parsing (`r' ([A-Za-z]+)\.'`) to cluster titles (`Mr`, `Miss`, `Mrs`, `Master`, `Rare`). Introduced dynamic runtime computation for `FamilySize` and `IsAlone` parameters.

### Final Validation Metrics:
| Evaluation Metric | Baseline (Logistic Regression) | Optimized Production Model (Random Forest) |
| :--- | :---: | :---: |
| **Accuracy (Overall Correctness)** | 79.01% | **85.47%** |
| **Precision (Inference Exactness)** | 73.85% | **82.43%** |
| **Recall (Inference Completeness)** | 70.18% | **82.43%** |

---

## 🏗️ Technical Architecture & Features
* **Production Core API:** Lightweight asynchronous backend engine implemented using **Flask** and serialized model deserialization using `joblib`.
* **Advanced Analytics UI:** A responsive, two-column analytical dashboard built with **HTML5/CSS3** and **Jinja2 templates**, rendering live evaluation states and dynamic text-parsing states.
* **Data Persistence Layer:** Isolated **SQLite** database pipeline integrated to persist every user parameters input matrix alongside the inference probability vector with strict ISO timestamps.
* **Data Visualization:** Automated generation of `Feature Importance` horizontal plots via `matplotlib` at training runtime, proving that the engineered `Title` attribute accounts for **26.73%** of the model's total decision-making weight.
* **Robust Logging Subsystem:** System runtime tracking configured via the Python `logging` module to output system operational telemetry directly to an automated rolling `app.log` file.
* **Quality Assurance (QA):** Comprehensive unit-testing suite running via the `unittest` framework to execute automated integrity assertions on HTTP request/response loops.

---

## 📂 Project Directory Structure
```text
Titanic-Survival-Prediction_ML/
├── data/
│   ├── train.csv             # Training dataset from Kaggle
│   └── test.csv              # Test dataset for submission parsing
├── src/
│   ├── model.py              # GridSearchCV training logic & feature importance viz
│   ├── predict_test.py       # Legacy fallback synchronization module
│   └── titanic_model.pkl     # Optimized serialized Random Forest binary
├── static/
│   └── importance.png        # Production runtime generated analysis chart
├── templates/
│   └── index.html            # Refactored CSS Grid dashboard with DB history table
├── tests/
│   └── test_app.py           # Automated execution endpoint unit tests
├── app.log                   # Real-time industrial system operations log
├── app.py                    # Core Flask production application server
├── database.db               # Relational SQLite transaction data storage
├── Dockerfile                # Production multi-stage deployment manifest
├── main.py                   # Automated end-to-end pipeline execution orchestrator
├── requirements.txt          # Explicit locked project dependencies manifest
├── research_notebook.ipynb   # Complete 5-stage exploratory Jupyter Notebook
├── submission.csv            # Automated Kaggle-compliant testing output
└── README.md                 # Production architectural documentation
```
# 🚀 Deployment & Local Execution Guide
# Option 1: Standard Virtual Environment Installation
git clone [https://github.com/SoldierrBoy/Titanic-Survival-Prediction_ML.git](https://github.com/SoldierrBoy/Titanic-Survival-Prediction_ML.git)
cd Titanic-Survival-Prediction_ML
## Initialize Environment & Install Dependencies:
python -m venv .venv
### Windows Activation:
.venv\Scripts\activate
### Linux/macOS Activation:
source .venv/bin/activate

pip install -r requirements.txt
## Execute End-to-End Pipeline (Train & Submit):
python main.py
## Boot the Web API Server:
python app.py
#### Navigate to http://127.0.0.1:5000 in your web browser.
# Option 2: Isolated Containerized Deployment (Docker)
#### Ensure Docker Desktop is running on your machine, then execute:
### Build the production container image
docker build -t titanic-dashboard .
### Launch the container instance mapping port 5000
docker run -p 5000:5000 titanic-dashboard
# Option 3: Launching the Exploratory Jupyter Notebook
#### To view or re-run the exploratory analysis pipeline, trigger the local notebook server:
jupyter notebook research_notebook.ipynb