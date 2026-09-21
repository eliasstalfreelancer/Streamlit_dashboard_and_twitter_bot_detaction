# README.md

# Streamlit Bot Detection Project

This project explores how Streamlit can be used to build an interactive Data Science application in Python.

The application uses a Random Forest model trained on the Cresci-2017 Twitter bot dataset.

---

## How to run the project

### 1. Create a virtual environment

python -m venv .venv

### 2. Activate the virtual environment

Windows:

.venv\Scripts\activate

### 3. Install dependencies

pip install -r requirements.txt

### 4. Run the backend pipeline

Before starting the Streamlit application, run:

python backend.py

The backend pipeline:

- loads the raw datasets
- preprocesses the data
- creates EDA artifacts
- balances the account dataset
- aggregates tweet behaviour
- trains the machine learning model
- saves the trained model and model results

Processed datasets are stored as checkpoints, so completed processing steps do not need to be repeated every time the pipeline is executed.

### 5. Start the Streamlit application

After the backend pipeline has completed, run:

streamlit run app.py

Streamlit will start a local web server.

The application can normally be opened at:

http://localhost:8501

---

## Project structure

Fördjupning_python/
│
├── app.py
├── main.py
│
├── screens/
│   ├── starting_screen.py
│   ├── eda_screen.py
│   ├── model_screen.py
│   └── model_result_screen.py
│
├── src/
│   ├── data_processing.py
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── visualizations.py
│   ├── logger.py
│   └── utils.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── viz/
│
├── artifacts/
│   └── model/
│       ├── bot_model.joblib
│       ├── results.json
│       ├── confusion_matrix.png
│       └── feature_importance.png
│
├── requirements.txt
└── README.md

---

## Application

The Streamlit application contains three main parts.

### Explore Data

Displays:

- dataset summaries
- missing values
- correlation heatmaps
- preprocessing results

### Bot Detection

Allows the user to manually enter account and posting behaviour features and run them through the trained model.

### Model Results

Displays:

- accuracy
- precision
- recall
- F1 score
- cross-validation results
- confusion matrix
- feature importance

---

## Dataset

The project uses the Cresci-2017 Twitter bot dataset.

After preprocessing and balancing, the final dataset contains:

- 1,982 accounts
- 991 human accounts
- 991 bot accounts
- 21 model features

The machine learning model is intended as a demonstration for the Streamlit application and should not be interpreted as a production-ready detector for modern X/Twitter accounts.

---

## Main technologies

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- PyArrow / Parquet

---

## Author

Elias Stålhjärta


# Streamlit_dashboard_and_twitter_bot_detaction
