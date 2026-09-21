from pathlib import Path

import pandas as pd

from src.data_processing import (
    load_tweets,
    load_users,
    df_to_summary_json,
    convert_id_collums_to_bolean,
    data_proccesing_convert_missing_vaules_into_csv,
    get_column_types,
    drop_non_correlating_columns,
    convert_binary_columns,
    
)
from src.model import ( 
    run_model_pipeline,
)


from src.data_preprocessing import (
    select_balanced_users_with_tweets,
)

from src.visualizations import (
    create_correlation_heatmap_and_save_heatmap,
)

from src.utils import (
    artifact_exists,
    save_json_if_missing,
)

from src.logger import get_logger


logger = get_logger(__name__)


# ============================================================
# PATHS
# ============================================================

USER_MISSING_PATH = Path(
    "data/proccessed/user_post_missing_vaules_converation.csv"
)

TWEET_MISSING_PATH = Path(
    "data/proccessed/tweets_post_missing_vaules_converation.csv"
)

BALANCED_USER_PATH = Path(
    "data/proccessed/balance_user.csv"
)

BALANCED_TWEET_PATH = Path(
    "data/proccessed/balance_tweets.csv"
)

PREPROCESSED_USER_PATH = Path(
    "data/proccessed/preprocessed_users.parquet"
)

PREPROCESSED_TWEET_PATH = Path(
    "data/proccessed/preprocessed_tweets.parquet"
)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    logger.info("Starting offline pipeline")

    # ========================================================
    # CHECKPOINT 1
    # FINAL PREPROCESSED DATA ALREADY EXISTS
    # ========================================================

    if (
        PREPROCESSED_USER_PATH.exists()
        and PREPROCESSED_TWEET_PATH.exists()
    ):

        logger.info(
            "Final preprocessed datasets already exist."
        )

        logger.info(
            "Skipping raw loading, EDA preprocessing, "
            "missing-value processing and balancing."
        )

        df_user = pd.read_parquet(
            PREPROCESSED_USER_PATH
        )

        df_tweets = pd.read_parquet(
            PREPROCESSED_TWEET_PATH
        )

        logger.info(
            "Preprocessed datasets loaded successfully."
        )

        logger.info(
            "Users: %s | Tweets: %s",
            len(df_user),
            len(df_tweets)
        )

        print(df_user.columns)
        print(df_tweets.columns)
        run_model_pipeline(df_user,df_tweets)
        return

    # ========================================================
    # CHECKPOINT 2
    # BALANCED DATA ALREADY EXISTS
    # ========================================================

    if (
        BALANCED_USER_PATH.exists()
        and BALANCED_TWEET_PATH.exists()
    ):

        logger.info(
            "Balanced datasets already exist."
        )

        logger.info(
            "Skipping raw preprocessing and balancing."
        )

        df_user = pd.read_csv(
            BALANCED_USER_PATH
        )

        df_tweets = pd.read_csv(
            BALANCED_TWEET_PATH
        )

        logger.info(
            "Balanced datasets loaded."
        )

    else:

        # ====================================================
        # CHECKPOINT 3
        # MISSING-VALUE PROCESSED DATA EXISTS
        # ====================================================

        if (
            USER_MISSING_PATH.exists()
            and TWEET_MISSING_PATH.exists()
        ):

            logger.info(
                "Missing-value processed datasets exist."
            )

            logger.info(
                "Skipping raw loading and missing-value preprocessing."
            )

            df_user = pd.read_csv(
                USER_MISSING_PATH
            )

            df_tweets = pd.read_csv(
                TWEET_MISSING_PATH
            )

            logger.info(
                "Missing-value processed datasets loaded."
            )

        else:

            # ================================================
            # NO CHECKPOINT FOUND
            # LOAD RAW DATA
            # ================================================

            logger.info(
                "No preprocessing checkpoint found."
            )

            logger.info(
                "Loading raw datasets."
            )

            df_tweets = load_tweets()
            df_user = load_users()

            logger.info(
                "Raw datasets loaded."
            )
            
            # ========================================================
            # MISSING VALUES SUMMARY
            # ========================================================

            user_missing_values = {
                column: int(value)
                for column, value in df_user.isna().sum().items()
                if value > 0
            }

            tweet_missing_values = {
                column: int(value)
                for column, value in df_tweets.isna().sum().items()
                if value > 0
            }

            save_json_if_missing(
                user_missing_values,
                "artifacts/missing_vaules_user.json"
            )

            save_json_if_missing(
                tweet_missing_values,
                "artifacts/missing_vaules_tweet.json"
            )

            logger.info(
                "Missing-value summaries created."
            )
            # ================================================
            # PRE-PROCESSING EDA SUMMARY
            # ================================================

            df_to_summary_json(
                df_user,
                df_tweets,
                "artifacts/pre_user_eda_summary.json",
                "artifacts/pre_tweets_eda_summary.json"
            )

            logger.info(
                "Pre-processing summaries complete."
            )

            # ================================================
            # CONVERT RAW ID FEATURES INTO BEHAVIOURAL FEATURES
            # ================================================

            df_tweets = (
                convert_id_collums_to_bolean(
                    df_tweets
                )
            )

            logger.info(
                "Tweet ID columns converted "
                "to behavioural features."
            )
            # ================================================
            # CONVERT BINARY COLUMNS
            # ================================================
            df_user = convert_binary_columns(df_user)
            df_tweets = convert_binary_columns(df_tweets)

            # ================================================
            # HANDLE MISSING VALUES
            # ================================================
            
            df_user, df_tweets = (
                data_proccesing_convert_missing_vaules_into_csv(
                    df_user,
                    df_tweets
                )
            )

            logger.info(
                "Missing-value preprocessing complete."
            )

        # ====================================================
        # INITIAL COLUMN TYPES
        # ====================================================

        tweets_column_list = (
            get_column_types(
                df_tweets
            )
        )

        user_column_list = (
            get_column_types(
                df_user
            )
        )

        save_json_if_missing(
            tweets_column_list,
            "artifacts/column_datatype_list_tweets.json"
        )

        save_json_if_missing(
            user_column_list,
            "artifacts/column_datatype_list_user.json"
        )

        # ====================================================
        # INITIAL HEATMAPS
        # ====================================================

        user_heatmap_path = Path(
            "data/viz/heatmap_users.png"
        )

        tweet_heatmap_path = Path(
            "data/viz/heatmap_tweets.png"
        )

        if not artifact_exists(
            user_heatmap_path
        ):

            create_correlation_heatmap_and_save_heatmap(
                df_user[
                    user_column_list[0]
                ],
                user_heatmap_path
            )

        if not artifact_exists(
            tweet_heatmap_path
        ):

            create_correlation_heatmap_and_save_heatmap(
                df_tweets[
                    tweets_column_list[0]
                ],
                tweet_heatmap_path
            )

        # ====================================================
        # SELECT USERS WITH TWEETS + BALANCE CLASSES
        # ====================================================

        logger.info(
            "Selecting accounts with tweet data "
            "and balancing Bot/Human classes."
        )

        df_user, df_tweets = (
            select_balanced_users_with_tweets(
                df_user,
                df_tweets
            )
        )

        BALANCED_USER_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df_user.to_csv(
            BALANCED_USER_PATH,
            index=False
        )

        df_tweets.to_csv(
            BALANCED_TWEET_PATH,
            index=False
        )

        logger.info(
            "Balanced datasets saved."
        )

    # ========================================================
    # FINAL PREPROCESSING
    # ========================================================

    logger.info(
        "Starting final preprocessing."
    )

    df_user = (
        drop_non_correlating_columns(
            df_user
        )
    )

    df_tweets = (
        drop_non_correlating_columns(
            df_tweets
        )
    )

    # ========================================================
    # GET UPDATED COLUMN TYPES
    # ========================================================

    tweets_column_list = (
        get_column_types(
            df_tweets
        )
    )

    user_column_list = (
        get_column_types(
            df_user
        )
    )

    # ========================================================
    # POST-PROCESSING EDA SUMMARY
    # ========================================================

    df_to_summary_json(
        df_user,
        df_tweets,
        "artifacts/post_user_eda_summary.json",
        "artifacts/post_tweets_eda_summary.json"
    )

    # ========================================================
    # POST-PROCESSING HEATMAP COLUMNS
    # ========================================================

    # ID is kept in the dataframe because we still need it
    # for merging later, but it should not be included in
    # correlation analysis.

    user_heatmap_columns = [
        column
        for column in user_column_list[0]
        if column != "id"
    ]

    tweet_heatmap_columns = [
        column
        for column in tweets_column_list[0]
        if column not in (
            "id",
            "user_id"
        )
    ]

    # ========================================================
    # POST-PROCESSING HEATMAPS
    # ========================================================

    post_user_heatmap_path = Path(
        "data/viz/post_heatmap_users.png"
    )

    post_tweet_heatmap_path = Path(
        "data/viz/post_heatmap_tweets.png"
    )

    if not artifact_exists(
        post_user_heatmap_path
    ):

        create_correlation_heatmap_and_save_heatmap(
            df_user[
                user_heatmap_columns
            ],
            post_user_heatmap_path
        )

    if not artifact_exists(
        post_tweet_heatmap_path
    ):

        create_correlation_heatmap_and_save_heatmap(
            df_tweets[
                tweet_heatmap_columns
            ],
            post_tweet_heatmap_path
        )

    # ========================================================
    # FINAL PREPROCESSING CHECKPOINT
    # ========================================================

    logger.info(
        "Saving final preprocessing checkpoint."
    )

    PREPROCESSED_USER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not artifact_exists(
        PREPROCESSED_USER_PATH
    ):

        df_user.to_parquet(
            PREPROCESSED_USER_PATH,
            index=False
        )

        logger.info(
            "Preprocessed user dataset saved: %s",
            PREPROCESSED_USER_PATH
        )

    if not artifact_exists(
        PREPROCESSED_TWEET_PATH
    ):

        df_tweets.to_parquet(
            PREPROCESSED_TWEET_PATH,
            index=False
        )

        logger.info(
            "Preprocessed tweet dataset saved: %s",
            PREPROCESSED_TWEET_PATH
        )

    logger.info(
        "Offline preprocessing pipeline completed successfully."
    )

    # ========================================================
    # NEXT STEP
    # ========================================================

    logger.info(
        "Data is ready for account-level feature engineering."
    )


if __name__ == "__main__":
    main()