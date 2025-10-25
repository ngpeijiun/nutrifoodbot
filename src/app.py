from typing import Callable

from dotenv import load_dotenv
import streamlit as st

load_dotenv()

PAGE_CLEANUP_REGISTRY_KEY = "_page_cleanup_registry"


def _run_page_teardowns(active_page_hash: str | None) -> None:
    registry: dict[str, Callable[[], None]] | None = st.session_state.get(
        PAGE_CLEANUP_REGISTRY_KEY
    )
    if not registry:
        return

    for page_hash, callback in list(registry.items()):
        if page_hash != active_page_hash:
            callback()


def _initialize_session_state() -> None:
    if "user_id" not in st.session_state:
        st.session_state.user_id = "1649954c-dca1-456d-a28f-4d4527c997d8"
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = st.session_state.user_id

_initialize_session_state()


chat_page = st.Page(
    "pages/1_chat.py",
    title="Chat",
    icon=":material/chat:"
)

run_ga_page = st.Page(
    "pages/3_ga_optimizer.py",
    title="Meal Planner (GA)",
    icon=":material/science:"
)

personalisation_page = st.Page(
    "pages/4_personalisation.py",
    title="Personalisation",
    icon=":material/tune:"
)

recommendations_page = st.Page(
    "pages/5_recommendations.py",
    title="Recommendations",
    icon=":material/restaurant:"
)

product_details_page = st.Page(
    "pages/6_product_details.py",
    title="Product Details",
    icon=":material/restaurant_menu:"
)

params = st.query_params
page_param = params.get("page", None)

if page_param == "6_product_details":
    pg = st.navigation([product_details_page])
else:
    pg = st.navigation([
        recommendations_page,
        chat_page,
        run_ga_page,
        personalisation_page,
    ])

_run_page_teardowns(getattr(pg, "_script_hash", None))

pg.run()
