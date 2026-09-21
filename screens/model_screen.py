import streamlit as st
from src.model import run_account_on_model,load_model

def show_model_screen():

    # Navigation
    if st.button(
        "<-",
        key="back_button",
        width="content"
    ):
        st.session_state.page = "start"
        st.rerun()
    st.markdown(
    """
    <style>
    div[data-testid="stForm"] {
        background-color: #0f142b;
        padding: 24px;
        border-radius: 15px;
        border: 1px solid #374151;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    st.subheader("Bot Detection")

    with st.form("bot_detector_form"):

        # ====================================================
        # ACCOUNT FEATURES
        # ====================================================

        st.write("### Account information")

        statuses_count = st.number_input(
            "Total posts",
            min_value=0,
            value=1000,
            help="Total number of posts made by the account."
        )

        followers_count = st.number_input(
            "Followers count",
            min_value=0,
            value=500
        )

        friends_count = st.number_input(
            "Following count",
            min_value=0,
            value=300
        )

        favourites_count = st.number_input(
            "Favourites count",
            min_value=0,
            value=200
        )

        listed_count = st.number_input(
            "Number of lists the account appears in",
            min_value=0,
            value=0
        )

        utc_offset_hours = st.number_input(
            "UTC offset (hours)",
            min_value=-12.0,
            max_value=14.0,
            value=0.0,
            step=0.5
        )

        utc_offset = utc_offset_hours * 3600

        # ====================================================
        # ACCOUNT SETTINGS
        # ====================================================

        st.write("### Account settings")

        default_profile = st.checkbox(
            "Default profile"
        )

        default_profile_image = st.checkbox(
            "Default profile image"
        )

        geo_enabled = st.checkbox(
            "Geo enabled"
        )

        profile_use_background_image = st.checkbox(
            "Uses profile background image"
        )

        profile_background_tile = st.checkbox(
            "Profile background tile enabled"
        )

        verified = st.checkbox(
            "Verified"
        )

        # ====================================================
        # TWEET BEHAVIOUR
        # ====================================================

        st.write("### Tweet behaviour")

        tweet_count = st.number_input(
            "Number of tweets analysed",
            min_value=1,
            value=2000,
            help=(
                "Number of tweets used to calculate "
                "the behavioural averages below."
            )
        )

        avg_retweet_count = st.number_input(
            "Average retweets per post",
            min_value=0.0,
            value=0.0,
            step=0.1
        )

        avg_favorite_count = st.number_input(
            "Average favourites per post",
            min_value=0.0,
            value=0.0,
            step=0.1
        )

        avg_hashtags = st.number_input(
            "Average hashtags per post",
            min_value=0.0,
            value=0.0,
            step=0.1
        )

        avg_urls = st.number_input(
            "Average URLs per post",
            min_value=0.0,
            value=0.0,
            step=0.1
        )

        avg_mentions = st.number_input(
            "Average mentions per post",
            min_value=0.0,
            value=1.0,
            step=0.1
        )

        reply_ratio = st.slider(
            "Reply ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.01,
            help="Share of analysed posts that are replies."
        )

        retweet_ratio = st.slider(
            "Retweet ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.01,
            help="Share of analysed posts that are retweets."
        )

        replies_to_user_ratio = st.slider(
            "Replies to user ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.01,
            help=(
                "Share of analysed posts that contain "
                "a reply to another user."
            )
        )

        # Submit should preferably be last
        submitted = st.form_submit_button(
            "Analyse account"
        )

    # ========================================================
    # RUN MODEL
    # ========================================================

    if submitted:

        account_data = {
            "statuses_count": statuses_count,
            "followers_count": followers_count,
            "friends_count": friends_count,
            "favourites_count": favourites_count,
            "listed_count": listed_count,

            "default_profile": int(default_profile),
            "default_profile_image": int(default_profile_image),
            "geo_enabled": int(geo_enabled),
            "profile_use_background_image": int(
                profile_use_background_image
            ),
            "profile_background_tile": int(
                profile_background_tile
            ),

            "utc_offset": utc_offset,
            "verified": int(verified),

            "tweet_count": tweet_count,
            "avg_retweet_count": avg_retweet_count,
            "avg_favorite_count": avg_favorite_count,
            "avg_hashtags": avg_hashtags,
            "avg_urls": avg_urls,
            "avg_mentions": avg_mentions,

            "reply_ratio": reply_ratio,
            "retweet_ratio": retweet_ratio,
            "replies_to_user_ratio": replies_to_user_ratio,
        }

        model = load_model()

        result = run_account_on_model(
            model,
            account_data
        )

        st.divider()

        st.metric(
            "Bot probability",
            f"{result['bot_probability']:.1%}"
        )

        if result["prediction"] == 1:
            st.error("Prediction: Bot")
        else:
            st.success("Prediction: Human")