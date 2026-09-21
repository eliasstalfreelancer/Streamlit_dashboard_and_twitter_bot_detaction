import streamlit as st

from src.utils import load_json


def show_eda_screen():

    # Navigation
    if st.button(
        "<-",
        key="back_button",
        width = "content"
    ):
        st.session_state.page = "start"
        st.rerun()

    st.title("Explore Data")

    st.info(
        "This analysis focuses mainly on numerical features. "
        "Text analysis and NLP are intentionally excluded to keep "
        "the project within the scope of this Streamlit-focused assignment."
    )

    # ---------------------------------------------------------
    # Dataset overview - Tweets
    # ---------------------------------------------------------

    with st.expander(
        "Quick Dataset Summary of Posts Before and After Preprocessing"
    ):

        st.write("Before data processing")
        st.write(
            load_json(
                "artifacts/pre_tweets_eda_summary.json"
            )
        )

        st.write("After data processing")
        st.write(
            load_json(
                "artifacts/post_tweets_eda_summary.json"
            )
        )

        st.info(
            "The number of unique accounts decreases because the dataset "
            "is balanced to achieve a 50/50 ratio between bot and human accounts. "
            "The number of duplicate feature rows increases because text-based "
            "and identifying features are removed during preprocessing, which "
            "reduces the number of possible feature combinations."
        )

    # ---------------------------------------------------------
    # Dataset overview - Users
    # ---------------------------------------------------------

    with st.expander(
        "Quick Dataset Summary of Users Before and After Preprocessing"
    ):

        st.write("Before data processing")
        st.write(
            load_json(
                "artifacts/pre_user_eda_summary.json"
            )
        )

        st.write("After data processing")
        st.write(
            load_json(
                "artifacts/post_user_eda_summary.json"
            )
        )

        st.info(
            "A total of 29 features are removed from the user dataset. "
            "Only accounts with available tweet records are kept. "
            "The remaining accounts are then balanced to create a 50/50 "
            "split between bot and human accounts, resulting in 991 accounts "
            "per class."
        )
    # ========================================================
        # TABS
        # ========================================================
    
    tab_missing_values, initial_correlation_check, proccseesing_method, post_processing_correlation_check= st.tabs(
            [
                "Missing Values",
                "Initial Correlation Check",
                "Proccseesing method",
                "Post-Processing Correlation Check"
            ]
        )
    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------
    with tab_missing_values:
        st.subheader("Missing Values in User and Post Datasets")

        st.write("Users")
        st.bar_chart(
            load_json(
                "artifacts/missing_vaules_user.json"
            )
        )

        with st.expander(
            "Analysis of Missing Values in the User Dataset"
        ):
            st.write(
                "The missing values are mainly found in optional user-provided "
                "information or profile settings rather than representing data errors. "
                "For example, not providing a location is normal user behaviour and "
                "does not mean that the account should be removed."
            )

        st.write("Posts")
        st.bar_chart(
            load_json(
                "artifacts/missing_vaules_tweet.json"
            )
        )

        with st.expander(
            "Analysis of Missing Values in the Post Dataset"
        ):
            st.write(
                "The missing values are mainly related to optional tweet metadata "
                "or account settings. These values are handled during preprocessing "
                "instead of removing entire observations."
            )

    # ---------------------------------------------------------
    # Initial correlation check
    # ---------------------------------------------------------
    with initial_correlation_check:
        st.subheader("Initial Correlation Check")

        st.write("User Dataset")
        st.image(
            "data/viz/heatmap_users.png",
            width= "stretch"
        )

        st.info(
            "This heatmap is used as an initial sanity check before major "
            "feature processing. The purpose is to verify that the numerical "
            "features contain some relationship with the target variable before "
            "spending significant time on feature engineering."
        )

        st.write("Tweet Dataset")
        st.image(
            "data/viz/heatmap_tweets.png",
            width= "stretch"
        )

        st.info(
            "The tweet dataset also contains correlations with the target variable, "
            "which indicates that the available behavioural features may contain "
            "useful information for the model."
        )

    with proccseesing_method:
        st.write(
    """
    The data processing pipeline was designed to prepare the Cresci-2017
    dataset for account-level bot classification while keeping the machine
    learning part intentionally simple.

    First, raw user and tweet datasets were loaded and labelled as either
    human or bot accounts. Missing values were then handled based on the
    type of feature. Text-based fields were treated separately from numeric
    and binary values to avoid changing their data types.

    Only accounts with available tweet records were kept. The remaining
    accounts were balanced to an equal 50/50 distribution between human and
    bot accounts.

    Tweet-level data was then reduced to account-level behavioural features
    by grouping posts by user ID and calculating statistics such as average
    mentions, hashtags, URLs, retweets, favourites, reply ratio and retweet
    ratio.

    Columns used only as identifiers, dataset metadata, constant values or
    features without meaningful variation were excluded from the model.
    The processed user data and aggregated tweet behaviour were finally
    merged into one account-level dataset used for model training.

    Intermediate results were saved as reusable artifacts and Parquet
    checkpoints so that expensive preprocessing steps did not need to be
    repeated during model development or Streamlit reruns.
    """
)

    # ---------------------------------------------------------
    # Post-processing correlation check
    # ---------------------------------------------------------
    with post_processing_correlation_check:

        st.subheader("Post-Processing Correlation Check")
        st.info("""
    After preprocessing, the final dataset contains a combination of
    account-level features and aggregated posting behaviour.

    Account features such as follower count, following count, total posts,
    favourites, profile settings, verification status and UTC offset were
    kept because they describe the structure and configuration of the account.

    Tweet-level data was aggregated per account into behavioural features
    such as average mentions, hashtags, URLs, retweets, favourites, reply
    ratio and retweet ratio.

    Identifier columns, dataset-specific metadata, constant features and
    columns with no useful variation were removed. Text-based features were
    intentionally excluded to keep the project focused on Streamlit rather
    than NLP and advanced feature engineering.
    """)
        st.write("User Dataset")
        st.image(
            "data/viz/post_heatmap_users.png",
            width= "stretch"
        )

        st.info(
            "The post-processing heatmap is used to check how the relationships "
            "between the remaining numerical features changed after preprocessing."
        )

        st.write("Tweet Dataset")
        st.image(
            "data/viz/post_heatmap_tweets.png",
            width= "stretch"
        )

        st.info(
            "The processed tweet features still show relationships with the target "
            "variable, so the data can be used for the next stage of the project."
        )