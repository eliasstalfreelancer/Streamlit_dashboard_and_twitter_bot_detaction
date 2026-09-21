import pandas as pd
from src.logger import *
logger = get_logger(__name__)

def select_balanced_users_with_tweets(
    df_users,
    df_tweets,
    random_state=42
):
    # Accounts som faktiskt finns i tweets
    tweet_user_ids = df_tweets["user_id"].unique()

    eligible_users = df_users[
        df_users["id"].isin(tweet_user_ids)
    ].copy()

    logger.info(
        "Accounts with tweet data: %s",
        len(eligible_users)
    )

    humans = eligible_users[
        eligible_users["Bot Label"] == 0
    ]

    bots = eligible_users[
        eligible_users["Bot Label"] == 1
    ]

    logger.info(
        "Available humans: %s | Available bots: %s",
        len(humans),
        len(bots)
    )

    # Minsta gruppen bestämmer storleken
    n_accounts = min(
        len(humans),
        len(bots)
    )

    human_sample = humans.sample(
        n=n_accounts,
        random_state=random_state
    )

    bot_sample = bots.sample(
        n=n_accounts,
        random_state=random_state
    )

    selected_users = pd.concat(
        [human_sample, bot_sample],
        ignore_index=True
    )

    # Blanda ordningen
    selected_users = selected_users.sample(
        frac=1,
        random_state=random_state
    ).reset_index(drop=True)

    selected_ids = selected_users["id"]

    # Behåll endast tweets från valda accounts
    selected_tweets = df_tweets[
        df_tweets["user_id"].isin(selected_ids)
    ].copy()

    logger.info(
        "Selected accounts: %s",
        len(selected_users)
    )

    logger.info(
        "Selected account distribution:\n%s",
        selected_users["Bot Label"].value_counts()
    )

    logger.info(
        "Tweets belonging to selected accounts: %s",
        len(selected_tweets)
    )

    return selected_users, selected_tweets