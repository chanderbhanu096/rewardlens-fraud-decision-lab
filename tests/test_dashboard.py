import pytest
from streamlit.testing.v1 import AppTest


PAGES = {
    "0 · Start here": "Start here: should CoinQuest stop this reward?",
    "1 · Decision overview": "Where should RewardLens intervene?",
    "2 · Investigation queue": "Who should an analyst review first?",
    "3 · Traffic health": "Which traffic sources need attention?",
    "4 · Threshold decision lab": "What is the right intervention threshold?",
    "5 · Experiment readout": "Did the new fraud rule create net value?",
}


@pytest.mark.parametrize(("page", "expected_title"), PAGES.items())
def test_dashboard_page_renders_without_exception(
    page: str, expected_title: str
) -> None:
    app = AppTest.from_file("dashboard/app.py", default_timeout=30).run()
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception
    assert app.title[0].value == expected_title
