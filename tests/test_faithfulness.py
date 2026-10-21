"""Checking report figures against the query results.

The first test is the real case: a live run tabulated 2504 customers correctly and
then wrote "2,5004" in the sentence above the table.
"""

from __future__ import annotations

from analyst_assistant.safety import faithfulness

FRAMES = [
    {
        "rows": [
            {"state": "California", "customers": 2504, "spend_per_customer": 123.50},
            {"state": "Texas", "customers": 1575, "spend_per_customer": 121.15},
        ]
    }
]


def test_catches_a_garbled_number_from_a_real_run():
    answer = (
        "California sits at $123.50 across 2,5004 customers, while Texas customers "
        "trail at $121.15 across 1,575 customers."
    )
    report = faithfulness.check(answer, FRAMES)
    assert not report.ok
    assert "2,5004" in report.unverified


def test_passes_when_every_figure_is_in_the_data():
    answer = (
        "California sits at $123.50 across 2,504 customers, Texas at $121.15 across "
        "1,575 customers."
    )
    assert faithfulness.check(answer, FRAMES).ok


def test_currency_and_thousands_separators_are_handled():
    frames = [{"rows": [{"revenue": 1284900.37}]}]
    assert faithfulness.check("Revenue was $1,284,900.", frames).ok


def test_rounding_is_tolerated():
    frames = [{"rows": [{"aov": 84.1963}]}]
    assert faithfulness.check("AOV was $84.20.", frames).ok


def test_percentages_are_not_counted_as_unverified():
    """Derived figures legitimately do not appear in the results."""
    report = faithfulness.check("Revenue fell 12.4% against a base of 123.50.", FRAMES)
    assert report.ok


def test_small_numbers_are_ignored():
    """Years, list positions and small counts would be noise, not signal."""
    assert faithfulness.check("The top 3 states in 2024 saw 5 returns.", FRAMES).ok


def test_a_fabricated_large_figure_is_flagged():
    report = faithfulness.check("Revenue reached $9,999,999 last year.", FRAMES)
    assert not report.ok


def test_no_results_means_nothing_to_check():
    report = faithfulness.check("Revenue was $500,000.", [])
    assert report.ok
    assert report.checked == 0


def test_empty_answer_is_handled():
    assert faithfulness.check("", FRAMES).ok


def test_each_figure_is_counted_once():
    answer = "$123.50 and again $123.50 and once more $123.50."
    assert faithfulness.check(answer, FRAMES).checked == 1


def test_summary_is_readable_in_either_outcome():
    good = faithfulness.check("Spend was $123.50.", FRAMES)
    assert "traced to query results" in good.summary()

    bad = faithfulness.check("Spend was $777,777.", FRAMES)
    assert "not found in the results" in bad.summary()


def test_string_numbers_in_results_still_count():
    """BigQuery NUMERIC columns can arrive as strings."""
    frames = [{"rows": [{"revenue": "1284900.37"}]}]
    assert faithfulness.check("Revenue was $1,284,900.37.", frames).ok


# Answers written without querying anything


def test_figures_in_finds_invented_counts():
    """A live schema answer claimed 4,500 customers when nothing had been queried."""
    text = "We have 4,500 customers, 12,000 orders, and 2,500 products."
    assert faithfulness.figures_in(text) == ["4,500", "12,000", "2,500"]


def test_figures_in_ignores_years_and_small_numbers():
    assert faithfulness.figures_in("In 2024 we saw 3 categories grow.") == []


def test_figures_in_on_a_clean_schema_answer():
    text = (
        "You can ask about customers, orders and products, including where customers "
        "are based and how long delivery took. I can find the totals when you ask."
    )
    assert faithfulness.figures_in(text) == []
