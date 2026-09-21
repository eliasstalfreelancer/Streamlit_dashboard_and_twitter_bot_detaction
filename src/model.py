from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.logger import get_logger
import streamlit as st


logger = get_logger(__name__)


# ============================================================
# PATHS
# ============================================================

MODEL_DATASET_PATH = Path(
    "data/proccessed/model_dataset.parquet"
)

MODEL_PATH = Path(
    "artifacts/model/bot_model.joblib"
)

MODEL_RESULTS_PATH = Path(
    "artifacts/model/results.json"
)

CONFUSION_MATRIX_PATH = Path(
    "artifacts/model/confusion_matrix.png"
)

FEATURE_IMPORTANCE_PATH = Path(
    "artifacts/model/feature_importance.png"
)




@st.cache_resource
def load_model():
    return joblib.load(
        "artifacts/model/bot_model.joblib"
    )
# ============================================================
#    run account on model
# ============================================================
def run_account_on_model(
    model,
    account_data: dict
) -> dict:
    """
    Run one account through the trained model.

    Returns:
        prediction:
            0 = Human
            1 = Bot

        bot_probability:
            Probability between 0.0 and 1.0
    """

    # Get the exact features the model was trained with
    expected_features = list(
        model.feature_names_in_
    )

    # Check for missing features
    missing_features = [
        feature
        for feature in expected_features
        if feature not in account_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    # Create one-row dataframe
    input_df = pd.DataFrame(
        [account_data]
    )

    # Make sure columns are in the same order
    input_df = input_df[
        expected_features
    ]

    # Prediction
    prediction = int(
        model.predict(input_df)[0]
    )

    # Probability that the account is a bot
    bot_probability = float(
        model.predict_proba(input_df)[0][1]
    )

    logger.info(
        "Prediction completed | "
        "Prediction: %s | "
        "Bot probability: %.4f",
        prediction,
        bot_probability
    )

    return {
        "prediction": prediction,
        "label": (
            "Bot"
            if prediction == 1
            else "Human"
        ),
        "bot_probability": bot_probability
    }
# ============================================================
# TWEET AGGREGATION
# ============================================================

def aggregate_tweet_features(
    df_tweets: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate tweet-level behaviour into one row per account.

    The project focuses on account classification, so millions
    of tweet rows are reduced to behavioural statistics for
    each user.
    """

    logger.info(
        "Aggregating tweet features by user_id."
    )

    tweet_features = (
        df_tweets
        .groupby("user_id")
        .agg(
            tweet_count=("id", "count"),

            avg_retweet_count=(
                "retweet_count",
                "mean"
            ),

            avg_favorite_count=(
                "favorite_count",
                "mean"
            ),

            avg_hashtags=(
                "num_hashtags",
                "mean"
            ),

            avg_urls=(
                "num_urls",
                "mean"
            ),

            avg_mentions=(
                "num_mentions",
                "mean"
            ),

            reply_ratio=(
                "is_reply",
                "mean"
            ),

            retweet_ratio=(
                "is_retweet",
                "mean"
            ),

            replies_to_user_ratio=(
                "replies_to_user",
                "mean"
            ),
        )
        .reset_index()
    )

    logger.info(
        "Tweet aggregation complete. "
        "Created %s account rows.",
        len(tweet_features)
    )

    return tweet_features


# ============================================================
# BUILD FINAL MODEL DATASET
# ============================================================

def build_model_dataset(
    df_users: pd.DataFrame,
    df_tweets: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge account metadata with aggregated tweet behaviour.
    """

    if MODEL_DATASET_PATH.exists():

        logger.info(
            "Model dataset already exists. "
            "Loading checkpoint: %s",
            MODEL_DATASET_PATH
        )

        return pd.read_parquet(
            MODEL_DATASET_PATH
        )

    logger.info(
        "Creating final model dataset."
    )

    tweet_features = (
        aggregate_tweet_features(
            df_tweets
        )
    )

    model_df = df_users.merge(
        tweet_features,
        left_on="id",
        right_on="user_id",
        how="inner"
    )

    # user_id is only needed for the merge.
    # Keep id as account identifier but do not use it
    # as a model feature later.
    model_df = model_df.drop(
        columns=["user_id"],
        errors="ignore"
    )

    MODEL_DATASET_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model_df.to_parquet(
        MODEL_DATASET_PATH,
        index=False
    )

    logger.info(
        "Model dataset saved: %s",
        MODEL_DATASET_PATH
    )

    logger.info(
        "Model dataset rows: %s",
        len(model_df)
    )

    logger.info(
        "Model dataset columns: %s",
        len(model_df.columns)
    )

    logger.info(
        "Class distribution:\n%s",
        model_df[
            "Bot Label"
        ].value_counts()
    )

    return model_df


# ============================================================
# SELECT MODEL FEATURES
# ============================================================

def prepare_model_data(
    model_df: pd.DataFrame
):
    """
    Select only numerical features.

    Identifiers and the target variable are excluded.
    """

    numeric_df = model_df.select_dtypes(
        include=["number", "bool"]
    ).copy()

    y = numeric_df["Bot Label"]

    X = numeric_df.drop(
        columns=[
            "Bot Label",
        ],
        errors="ignore"
    )

    # Extra protection in case an identifier has been
    # converted to a numeric dtype somewhere in the pipeline.
    X = X.drop(
        columns=[
            "id",
            "user_id",
            "test_set_1",
            "test_set_2",
        ],
        errors="ignore"
    )

    logger.info(
        "Features selected for model: %s",
        list(X.columns)
    )

    logger.info(
        "Number of model features: %s",
        len(X.columns)
    )

    return X, y


# ============================================================
# MODEL PIPELINE
# ============================================================

def create_random_forest_pipeline(
    feature_columns: list[str]
) -> Pipeline:
    """
    Create a simple preprocessing + Random Forest pipeline.

    Missing numerical values are filled using the median.
    The imputer is fitted only on training data, preventing
    information leakage from the test set.
    """

    numeric_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(
                    strategy="median"
                ),
                feature_columns
            )
        ],
        remainder="drop"
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                numeric_preprocessor
            ),
            (
                "classifier",
                model
            ),
        ]
    )

    return pipeline


# ============================================================
# CROSS VALIDATION
# ============================================================

def run_cross_validation(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> dict:
    """
    Run 5-fold stratified cross-validation on training data.
    """

    logger.info(
        "Starting 5-fold stratified cross-validation."
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
        },
        n_jobs=-1
    )

    results = {
        "accuracy_mean": float(
            cv_results[
                "test_accuracy"
            ].mean()
        ),

        "accuracy_std": float(
            cv_results[
                "test_accuracy"
            ].std()
        ),

        "precision_mean": float(
            cv_results[
                "test_precision"
            ].mean()
        ),

        "precision_std": float(
            cv_results[
                "test_precision"
            ].std()
        ),

        "recall_mean": float(
            cv_results[
                "test_recall"
            ].mean()
        ),

        "recall_std": float(
            cv_results[
                "test_recall"
            ].std()
        ),

        "f1_mean": float(
            cv_results[
                "test_f1"
            ].mean()
        ),

        "f1_std": float(
            cv_results[
                "test_f1"
            ].std()
        ),
    }

    logger.info(
        "Cross-validation complete."
    )

    logger.info(
        "CV Accuracy: %.4f ± %.4f",
        results["accuracy_mean"],
        results["accuracy_std"]
    )

    return results


# ============================================================
# DUMMY BASELINE
# ============================================================

def evaluate_dummy_baseline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> dict:
    """
    Train a simple DummyClassifier baseline.
    """

    logger.info(
        "Training DummyClassifier baseline."
    )

    dummy = DummyClassifier(
        strategy="most_frequent"
    )

    dummy.fit(
        X_train,
        y_train
    )

    predictions = dummy.predict(
        X_test
    )

    results = {
        "accuracy": float(
            accuracy_score(
                y_test,
                predictions
            )
        ),

        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),
    }

    logger.info(
        "Dummy baseline accuracy: %.4f",
        results["accuracy"]
    )

    return results


# ============================================================
# TEST SET EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> tuple[dict, object]:
    """
    Evaluate the final trained model on the untouched test set.
    """

    predictions = model.predict(
        X_test
    )

    results = {
        "accuracy": float(
            accuracy_score(
                y_test,
                predictions
            )
        ),

        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),
    }

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    return results, matrix


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    matrix
) -> None:

    CONFUSION_MATRIX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    image = ax.imshow(
        matrix
    )

    ax.set_title(
        "Confusion Matrix"
    )

    ax.set_xlabel(
        "Predicted Label"
    )

    ax.set_ylabel(
        "True Label"
    )

    ax.set_xticks(
        [0, 1],
        labels=[
            "Human",
            "Bot"
        ]
    )

    ax.set_yticks(
        [0, 1],
        labels=[
            "Human",
            "Bot"
        ]
    )

    for row in range(
        matrix.shape[0]
    ):
        for column in range(
            matrix.shape[1]
        ):
            ax.text(
                column,
                row,
                str(
                    matrix[
                        row,
                        column
                    ]
                ),
                ha="center",
                va="center"
            )

    fig.colorbar(
        image,
        ax=ax
    )

    fig.tight_layout()

    fig.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    logger.info(
        "Confusion matrix saved: %s",
        CONFUSION_MATRIX_PATH
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(
    model: Pipeline,
    feature_names: list[str]
) -> None:

    classifier = (
        model.named_steps[
            "classifier"
        ]
    )

    importance = (
        classifier.feature_importances_
    )

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=True
        )
    )

    FEATURE_IMPORTANCE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    ax.set_title(
        "Random Forest Feature Importance"
    )

    ax.set_xlabel(
        "Importance"
    )

    ax.set_ylabel(
        "Feature"
    )

    fig.tight_layout()

    fig.savefig(
        FEATURE_IMPORTANCE_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    logger.info(
        "Feature importance saved: %s",
        FEATURE_IMPORTANCE_PATH
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_model_results(
    results: dict
) -> None:

    MODEL_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        MODEL_RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    logger.info(
        "Model results saved: %s",
        MODEL_RESULTS_PATH
    )


# ============================================================
# COMPLETE MODEL PIPELINE
# ============================================================

def run_model_pipeline(
    df_users: pd.DataFrame,
    df_tweets: pd.DataFrame
) -> None:

    logger.info(
        "Starting machine learning pipeline."
    )

    # ========================================================
    # CREATE / LOAD FINAL ACCOUNT-LEVEL DATASET
    # ========================================================

    model_df = build_model_dataset(
        df_users,
        df_tweets
    )

    X, y = prepare_model_data(
        model_df
    )

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    logger.info(
        "Train rows: %s | Test rows: %s",
        len(X_train),
        len(X_test)
    )

    logger.info(
        "Training class distribution:\n%s",
        y_train.value_counts()
    )

    logger.info(
        "Test class distribution:\n%s",
        y_test.value_counts()
    )

    # ========================================================
    # DUMMY BASELINE
    # ========================================================

    dummy_results = (
        evaluate_dummy_baseline(
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    model = (
        create_random_forest_pipeline(
            list(X.columns)
        )
    )

    # ========================================================
    # CROSS VALIDATION
    # ========================================================

    cv_results = (
        run_cross_validation(
            model,
            X_train,
            y_train
        )
    )

    # ========================================================
    # FINAL TRAINING
    # ========================================================

    logger.info(
        "Training final Random Forest model "
        "on the full training set."
    )

    model.fit(
        X_train,
        y_train
    )

    # ========================================================
    # TEST SET
    # ========================================================

    test_results, matrix = (
        evaluate_model(
            model,
            X_test,
            y_test
        )
    )

    logger.info(
        "Test Accuracy: %.4f",
        test_results["accuracy"]
    )

    logger.info(
        "Test Precision: %.4f",
        test_results["precision"]
    )

    logger.info(
        "Test Recall: %.4f",
        test_results["recall"]
    )

    logger.info(
        "Test F1: %.4f",
        test_results["f1"]
    )

    # ========================================================
    # TRAIN ACCURACY
    # ========================================================

    train_predictions = model.predict(
        X_train
    )

    train_accuracy = float(
        accuracy_score(
            y_train,
            train_predictions
        )
    )

    logger.info(
        "Train Accuracy: %.4f",
        train_accuracy
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    logger.info(
        "Model saved: %s",
        MODEL_PATH
    )

    # ========================================================
    # SAVE VISUALIZATIONS
    # ========================================================

    save_confusion_matrix(
        matrix
    )

    save_feature_importance(
        model,
        list(X.columns)
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results = {
        "dataset": {
            "total_accounts": int(
                len(model_df)
            ),

            "total_features": int(
                len(X.columns)
            ),

            "training_accounts": int(
                len(X_train)
            ),

            "test_accounts": int(
                len(X_test)
            ),
        },

        "dummy_baseline": (
            dummy_results
        ),

        "cross_validation": (
            cv_results
        ),

        "random_forest": {
            "train_accuracy":
                train_accuracy,

            "test_accuracy":
                test_results[
                    "accuracy"
                ],

            "precision":
                test_results[
                    "precision"
                ],

            "recall":
                test_results[
                    "recall"
                ],

            "f1":
                test_results[
                    "f1"
                ],
        },
    }

    save_model_results(
        results
    )

    logger.info(
        "Machine learning pipeline completed successfully."
    )