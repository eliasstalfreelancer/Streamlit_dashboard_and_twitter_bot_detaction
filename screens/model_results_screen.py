from pathlib import Path

import streamlit as st

from src.utils import load_json


# ============================================================
# PATHS
# ============================================================

MODEL_RESULTS_PATH = Path(
    "artifacts/model/results.json"
)

CONFUSION_MATRIX_PATH = Path(
    "artifacts/model/confusion_matrix.png"
)

FEATURE_IMPORTANCE_PATH = Path(
    "artifacts/model/feature_importance.png"
)


# ============================================================
# CACHED DATA
# ============================================================

@st.cache_data
def load_model_results():
    """
    Model results do not need to be loaded from disk
    on every Streamlit rerun.
    """

    return load_json(
        MODEL_RESULTS_PATH
    )


# ============================================================
# MODEL RESULT SCREEN
# ============================================================

def show_model_result_screen():

    # --------------------------------------------------------
    # Navigation
    # --------------------------------------------------------

    if st.button(
        "<-",
        key="back_button_model_results",
        width="content"
    ):
        st.session_state.page = "start"
        st.rerun()
        
    st.title("Model Results")

    st.info(
        "This page presents the performance of the trained "
        "Random Forest model. The model was evaluated using "
        "an untouched test set and 5-fold stratified "
        "cross-validation on the training data."
    )

    # --------------------------------------------------------
    # Load results
    # --------------------------------------------------------

    if not MODEL_RESULTS_PATH.exists():
        st.error(
            "Model results could not be found. "
            "Run the offline model pipeline first."
        )
        return

    results = load_model_results()

    dataset = results["dataset"]
    dummy = results["dummy_baseline"]
    cv = results["cross_validation"]
    model = results["random_forest"]

    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    st.subheader("Dataset")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accounts",
        dataset["total_accounts"]
    )

    col2.metric(
        "Features",
        dataset["total_features"]
    )

    col3.metric(
        "Training accounts",
        dataset["training_accounts"]
    )

    col4.metric(
        "Test accounts",
        dataset["test_accounts"]
    )

    st.divider()

    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.subheader("Random Forest Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Test Accuracy",
        f"{model['test_accuracy']:.2%}"
    )

    col2.metric(
        "Precision",
        f"{model['precision']:.2%}"
    )

    col3.metric(
        "Recall",
        f"{model['recall']:.2%}"
    )

    col4.metric(
        "F1 Score",
        f"{model['f1']:.2%}"
    )

    # --------------------------------------------------------
    # Train / test comparison
    # --------------------------------------------------------

    train_test_gap = (
        model["train_accuracy"]
        - model["test_accuracy"]
    )

    st.metric(
        "Train Accuracy",
        f"{model['train_accuracy']:.2%}",
        delta=f"{train_test_gap:.2%} train-test gap",
        delta_color="off"
    )

    st.caption(
        "A high training score is common for Random Forest models. "
        "The test set and cross-validation results are more important "
        "when evaluating generalisation."
    )

    st.divider()

    # ========================================================
    # TABS
    # ========================================================

    tab_overview, tab_cv, tab_visuals, tab_limitations = st.tabs(
        [
            "Model Comparison",
            "Cross Validation",
            "Visualisations",
            "Limitations"
        ]
    )

    # ========================================================
    # TAB 1 - MODEL COMPARISON
    # ========================================================

    with tab_overview:

        st.subheader(
            "Random Forest vs Dummy Baseline"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Random Forest Accuracy",
                f"{model['test_accuracy']:.2%}"
            )

        with col2:
            st.metric(
                "Dummy Accuracy",
                f"{dummy['accuracy']:.2%}"
            )

        improvement = (
            model["test_accuracy"]
            - dummy["accuracy"]
        )

        st.metric(
            "Accuracy improvement",
            f"{improvement:.2%}",
            delta=f"{improvement:.2%}"
        )

        st.write(
            "The DummyClassifier provides a simple baseline. "
            "Because the dataset is balanced between bot and human "
            "accounts, a model without useful predictive information "
            "would be expected to perform close to chance level."
        )

        with st.expander(
            "Show all DummyClassifier metrics"
        ):

            dummy_col1, dummy_col2, dummy_col3, dummy_col4 = (
                st.columns(4)
            )

            dummy_col1.metric(
                "Accuracy",
                f"{dummy['accuracy']:.2%}"
            )

            dummy_col2.metric(
                "Precision",
                f"{dummy['precision']:.2%}"
            )

            dummy_col3.metric(
                "Recall",
                f"{dummy['recall']:.2%}"
            )

            dummy_col4.metric(
                "F1",
                f"{dummy['f1']:.2%}"
            )

    # ========================================================
    # TAB 2 - CROSS VALIDATION
    # ========================================================

    with tab_cv:

        st.subheader(
            "5-Fold Stratified Cross Validation"
        )

        st.write(
            "Cross-validation was performed only on the training "
            "dataset. The separate test set remained untouched "
            "until the final model evaluation."
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Mean CV Accuracy",
                f"{cv['accuracy_mean']:.2%}"
            )

            st.metric(
                "Mean CV Precision",
                f"{cv['precision_mean']:.2%}"
            )

        with col2:

            st.metric(
                "Mean CV Recall",
                f"{cv['recall_mean']:.2%}"
            )

            st.metric(
                "Mean CV F1",
                f"{cv['f1_mean']:.2%}"
            )

        st.write(
            "### Stability"
        )

        st.metric(
            "Accuracy standard deviation",
            f"{cv['accuracy_std']:.2%}"
        )

        st.caption(
            "A relatively small standard deviation means that "
            "performance remained similar across the five folds."
        )

        with st.expander(
            "Show all cross-validation standard deviations"
        ):

            st.write(
                {
                    "Accuracy std":
                        cv["accuracy_std"],

                    "Precision std":
                        cv["precision_std"],

                    "Recall std":
                        cv["recall_std"],

                    "F1 std":
                        cv["f1_std"],
                }
            )

    # ========================================================
    # TAB 3 - VISUALISATIONS
    # ========================================================

    with tab_visuals:

        st.subheader(
            "Confusion Matrix"
        )

        if CONFUSION_MATRIX_PATH.exists():

            st.image(
                str(CONFUSION_MATRIX_PATH),
                caption=(
                    "Predicted classes compared with "
                    "the true test labels."
                )
            )

        else:
            st.warning(
                "Confusion matrix image was not found."
            )

        st.divider()

        st.subheader(
            "Feature Importance"
        )

        if FEATURE_IMPORTANCE_PATH.exists():

            st.image(
                str(FEATURE_IMPORTANCE_PATH),
                caption=(
                    "Feature importance calculated by "
                    "the Random Forest model."
                )
            )

        else:
            st.warning(
                "Feature importance image was not found."
            )

        st.info(
            "Feature importance shows how useful a feature was "
            "for splitting observations inside the Random Forest. "
            "It does not mean that the feature causes an account "
            "to be a bot."
        )

    # ========================================================
    # TAB 4 - LIMITATIONS
    # ========================================================

    with tab_limitations:

        st.subheader(
            "Model Limitations"
        )

        st.warning(
            "The high performance should not automatically be "
            "interpreted as equivalent performance on modern or "
            "previously unseen bot populations."
        )

        st.write(
            """
            The model is trained using accounts from the Cresci-2017
            dataset. Both the training and test accounts originate from
            the same dataset populations.

            This means that the model may learn characteristics that are
            specific to the genuine accounts and social spambots contained
            in this dataset.

            The model should therefore be interpreted as a demonstration
            of machine-learning integration in the Streamlit application,
            rather than as a production-ready detector for current X/Twitter
            accounts.
            """
        )

        with st.expander(
            "Why is the test accuracy so high?"
        ):

            st.write(
                """
                Several features in the dataset already show relatively
                strong relationships with the target class. The Random
                Forest can combine these signals and create more complex
                decision boundaries.

                Cross-validation performance is also close to the final
                test performance, which suggests that the result is not
                caused only by a fortunate train/test split.

                However, dataset-specific patterns may still make the
                classification task easier than detecting completely
                unknown bot behaviour in the real world.
                """
            )