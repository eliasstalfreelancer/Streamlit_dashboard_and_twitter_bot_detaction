from pathlib import Path

import pandas as pd

from src.logger import get_logger
from src.utils import (
    artifact_exists,
    save_json_if_missing,
)

logger = get_logger(__name__)

# ============================================================
# CONVERT BOOLEAN COLUMNS
# ============================================================


def convert_binary_columns(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    binary_columns = [
        "default_profile",
        "default_profile_image",
        "geo_enabled",
        "profile_use_background_image",
        "profile_background_tile",
        "is_translator",
        "follow_request_sent",
        "protected",
        "verified",
        "notifications",
        "contributors_enabled",
        "following",
    ]

    for column in binary_columns:

        if column not in df.columns:
            continue

        df[column] = (
            df[column]
            .replace({
                "": 0,
                "0": 0,
                "0.0": 0,
                0: 0,
                0.0: 0,

                "1": 1,
                "1.0": 1,
                1: 1,
                1.0: 1,

                True: 1,
                False: 0,
            })
            .fillna(0)
            .astype("int8")
        )

    return df

# ============================================================
# LOAD RAW DATA
# ============================================================

def load_tweets() -> pd.DataFrame:

    converters = {
        "id": str,
        "user_id": str,
        "in_reply_to_status_id": str,
        "in_reply_to_user_id": str,
        "retweeted_status_id": str,
        "place": str,
    }

    genuine = pd.read_csv(
        "data/raw/genuine_accounts/tweets.csv",
        encoding="latin-1",
        converters=converters,
        low_memory=False
    )

    bots = pd.read_csv(
        "data/raw/social_spambots_1/tweets.csv",
        encoding="latin-1",
        converters=converters,
        low_memory=False
    )

    genuine["Bot Label"] = 0
    bots["Bot Label"] = 1

    tweets = pd.concat(
        [genuine, bots],
        ignore_index=True
    )

    logger.info(
        "Tweets loaded successfully: %s rows",
        len(tweets)
    )

    return tweets


def load_users() -> pd.DataFrame:

    converters = {
        "id": str,
    }

    genuine = pd.read_csv(
        "data/raw/genuine_accounts/users.csv",
        converters=converters,
        low_memory=False
    )

    bots = pd.read_csv(
        "data/raw/social_spambots_1/users.csv",
        converters=converters,
        low_memory=False
    )

    genuine["Bot Label"] = 0
    bots["Bot Label"] = 1

    users = pd.concat(
        [genuine, bots],
        ignore_index=True
    )

    logger.info(
        "Users loaded successfully: %s rows",
        len(users)
    )

    return users
# ============================================================
# EDA SUMMARIES
# ============================================================

def df_user_summary(
    df: pd.DataFrame
) -> dict:

    summary = {
        "Total accounts": int(len(df)),
        "Unique accounts": int(
            df["id"].nunique()
        ),
        "Total features": int(
            len(df.columns)
        ),
        "Missing values": int(
            df.isna().sum().sum()
        ),
        "Duplicate feature combinations": int(
            df.duplicated().sum()
        ),
        "Bot to User Ratio": float(
            df["Bot Label"].mean()
        )
    }

    return summary


def df_posts_summary(
    df: pd.DataFrame
) -> dict:

    unique_accounts = (
        df[
            ["user_id", "Bot Label"]
        ]
        .drop_duplicates(
            subset="user_id"
        )
    )

    summary = {
        "Total posts": int(len(df)),
        "Unique accounts": int(
            df["user_id"].nunique()
        ),
        "Total features": int(
            len(df.columns)
        ),
        "Missing values": int(
            df.isna().sum().sum()
        ),
        "Duplicate feature combinations": int(
            df.duplicated().sum()
        ),

        # Ratio calculated from tweet rows
        "Bot Tweet Ratio": float(
            df["Bot Label"].mean()
        ),

        # Ratio calculated from unique accounts
        "Bot Account Ratio": float(
            unique_accounts[
                "Bot Label"
            ].mean()
        )
    }

    return summary


def get_missing_vaules(
    df: pd.DataFrame
) -> pd.Series:

    return df.isna().sum()


# ============================================================
# MISSING VALUES
# ============================================================

def handle_missing_values(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    # Only replace missing values in actual text columns.
    text_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    df[text_columns] = (
        df[text_columns]
        .fillna("")
    )

    return df


# ============================================================
# FEATURE CONVERSION
# ============================================================

def convert_id_collums_to_bolean(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    no_id_values = [
        "",
        "0",
        "0.0"
    ]

    df["is_reply"] = (
        ~df["in_reply_to_status_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .isin(no_id_values)
    ).astype(int)

    df["is_retweet"] = (
        ~df["retweeted_status_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .isin(no_id_values)
    ).astype(int)

    df["replies_to_user"] = (
        ~df["in_reply_to_user_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .isin(no_id_values)
    ).astype(int)

    return df


# ============================================================
# USER / TWEET RELATIONSHIP
# ============================================================

def users_without_tweets_stats(
    df_users: pd.DataFrame,
    df_tweets: pd.DataFrame
) -> dict:

    tweeting_ids = (
        df_tweets["user_id"]
        .dropna()
        .unique()
    )

    users_without_tweets = df_users[
        ~df_users["id"].isin(
            tweeting_ids
        )
    ]

    count = len(
        users_without_tweets
    )

    ratio = (
        count / len(df_users)
        if len(df_users) > 0
        else 0.0
    )

    return {
        "Users without tweet records":
            int(count),

        "Users without tweet records ratio":
            float(ratio)
    }


# ============================================================
# SAVE EDA SUMMARY JSON
# ============================================================

def df_to_summary_json(
    df_user: pd.DataFrame,
    df_tweets: pd.DataFrame,
    path_for_users_json: str,
    path_for_tweets_json: str
) -> None:

    # ---------------- USERS ----------------

    if artifact_exists(
        path_for_users_json
    ):
        logger.info(
            "Skipping user summary generation."
        )

    else:
        logger.info(
            "Creating user EDA summary."
        )

        user_summary = (
            df_user_summary(
                df_user
            )
        )

        missing_tweet_stats = (
            users_without_tweets_stats(
                df_user,
                df_tweets
            )
        )

        user_summary = (
            missing_tweet_stats
            | user_summary
        )

        save_json_if_missing(
            user_summary,
            path_for_users_json
        )

    # ---------------- TWEETS ----------------

    if artifact_exists(
        path_for_tweets_json
    ):
        logger.info(
            "Skipping tweet summary generation."
        )

    else:
        logger.info(
            "Creating tweet EDA summary."
        )

        tweet_summary = (
            df_posts_summary(
                df_tweets
            )
        )

        save_json_if_missing(
            tweet_summary,
            path_for_tweets_json
        )


# ============================================================
# REMOVE NON-USEFUL CORRELATION COLUMNS
# ============================================================

def drop_non_correlating_columns(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    manually_excluded_columns = {
        "in_reply_to_status_id",
        "retweeted_status_id",
        "in_reply_to_user_id",
        "test_set_1",
        "test_set_2",
    }

    columns_to_drop = [
        column
        for column in df.columns
        if (
            df[column]
            .dropna()
            .nunique() <= 1
            or column
            in manually_excluded_columns
        )
    ]

    logger.info(
        "Dropping non-correlating columns: %s",
        columns_to_drop
    )

    df = df.drop(
        columns=columns_to_drop
    )

    return df


# ============================================================
# COLUMN TYPES
# ============================================================

def get_column_types(
    df: pd.DataFrame
) -> tuple[
    list[str],
    list[str],
    list[str],
    list[str]
]:

    numeric_columns = []
    object_columns = []
    bool_columns = []
    datetime_columns = []

    for column in df.columns:
        dtype = df[column].dtype

        if dtype == "bool":
            bool_columns.append(
                column
            )

        elif dtype == "object":
            object_columns.append(
                column
            )

        elif "datetime" in str(dtype):
            datetime_columns.append(
                column
            )

        elif dtype.kind in "iuf":
            numeric_columns.append(
                column
            )

    return (
        numeric_columns,
        object_columns,
        bool_columns,
        datetime_columns
    )


# ============================================================
# SAVE / LOAD PROCESSED MISSING-VALUE DATA
# ============================================================

def data_proccesing_convert_missing_vaules_into_csv(
    df_user: pd.DataFrame,
    df_tweets: pd.DataFrame
) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:

    user_path = Path(
        "data/proccessed/"
        "user_post_missing_vaules_converation.csv"
    )

    tweet_path = Path(
        "data/proccessed/"
        "tweets_post_missing_vaules_converation.csv"
    )

    user_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # USERS
    # ========================================================

    if artifact_exists(user_path):

        logger.info(
            "Skipping user missing-value preprocessing. "
            "Loading existing file."
        )

        df_user_post = pd.read_csv(
            user_path
        )

    else:

        logger.info(
            "Processing missing values for users."
        )

        df_user_post = (
            handle_missing_values(
                df_user
            )
        )

        df_user_post.to_csv(
            user_path,
            index=False
        )

        logger.info(
            "Processed user CSV saved successfully: %s",
            user_path
        )

    # ========================================================
    # TWEETS
    # ========================================================

    if artifact_exists(tweet_path):

        logger.info(
            "Skipping tweet missing-value preprocessing. "
            "Loading existing file."
        )

        df_tweets_post = pd.read_csv(
            tweet_path,
            encoding="latin-1"
        )

    else:

        logger.info(
            "Processing missing values for tweets."
        )

        df_tweets_post = (
            handle_missing_values(
                df_tweets
            )
        )

        df_tweets_post.to_csv(
            tweet_path,
            index=False
        )

        logger.info(
            "Processed tweet CSV saved successfully: %s",
            tweet_path
        )

    return (
        df_user_post,
        df_tweets_post
    )