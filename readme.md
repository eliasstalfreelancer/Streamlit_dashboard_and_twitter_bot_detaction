# README.md

# Streamlit Bot Detection Project

This project explores how Streamlit can be used to build an interactive Data Science application in Python.

The application uses a Random Forest model trained on the Cresci-2017 Twitter bot dataset.

---

## Dataset Setup

The dataset is not included in this GitHub repository because the raw
Cresci-2017 data is too large to store directly in the repository.

This project uses the **Cresci-2017 Twitter Bot Dataset**.

### 1. Download the dataset

Download Cresci-2017 from the Keggle:

https://www.kaggle.com/datasets/hashemalsalmi/cresci-2017/data

Press Zip download and create an account if need be.

### 2. Extract the dataset

After downloading the archive, extract it.

This project only uses the following two subsets:

- `genuine_accounts`
- `social_spambots_1`

Each subset contains:

- `users.csv`
- `tweets.csv`

### 3. Place the files in the project

Create the following folder structure:

```text
Fördjupning_python/
│
└── data/
    └── raw/
        ├── genuine_accounts/
        │   ├── users.csv
        │   └── tweets.csv
        │
        └── social_spambots_1/
            ├── users.csv
            └── tweets.csv
```
The final paths should therefore be:

data/raw/genuine_accounts/users.csv
data/raw/genuine_accounts/tweets.csv

data/raw/social_spambots_1/users.csv
data/raw/social_spambots_1/tweets.csv

Only these four CSV files are required for this project.

---

## How to Run the Project

### 1. Create a virtual environment

Windows:

python -m venv .venv

### 2. Activate the virtual environment

.venv\Scripts\activate

### 3. Install the dependencies

pip install -r requirements.txt

### 4. Download and prepare the dataset

Download the Cresci-2017 dataset using the instructions above and make
sure the following files exist:

data/raw/genuine_accounts/users.csv
data/raw/genuine_accounts/tweets.csv
data/raw/social_spambots_1/users.csv
data/raw/social_spambots_1/tweets.csv

### 5. Run the backend pipeline

python main.py

The backend pipeline will:

- load the raw Cresci-2017 data
- preprocess the user and tweet datasets
- handle missing values
- balance the account classes
- aggregate tweet behaviour per account
- create visualisation artifacts
- create the final model dataset
- train the Random Forest model
- save the trained model and evaluation results

Processed data is stored locally as checkpoints so that completed
processing steps do not need to be repeated unnecessarily.

### 6. Start the Streamlit application

After the backend pipeline has finished successfully, run:

streamlit run app.py

Streamlit will normally start the application at:

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
