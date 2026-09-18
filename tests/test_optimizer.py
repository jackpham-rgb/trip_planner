from recsys.content_scoring import ScoredOption
from recsys.optimizer import estimate_travel_hours, greedy_two_opt, milp_orienteering
from recsys.schema import Option


def make_option(id, est_hours=2.0, city_region="City", section="Region"):
    return Option(id=id, source_sheet="Test", section=section, city_region=city_region,
                  main_destination=id, est_hours=est_hours)


def test_estimate_travel_hours_same_city_is_cheap():
    a = make_option("a", city_region="Same")
    b = make_option("b", city_region="Same")
    c = make_option("c", city_region="Different", section="OtherRegion")
    assert estimate_travel_hours(a, b) < estimate_travel_hours(a, c)


def test_greedy_respects_time_budget():
    scored = [ScoredOption(option=make_option(f"o{i}", est_hours=3), score=float(i), features=None)
              for i in range(10)]
    itinerary = greedy_two_opt(scored, budget_hours=8.0)
    assert itinerary.total_hours <= 8.0 + 1e-6
    assert len(itinerary.order) > 0


def test_greedy_picks_higher_value_items_first_when_budget_is_tight():
    low = ScoredOption(option=make_option("low", est_hours=2), score=1.0, features=None)
    high = ScoredOption(option=make_option("high", est_hours=2), score=10.0, features=None)
    itinerary = greedy_two_opt([low, high], budget_hours=3.0)  # only room for one
    assert [o.id for o in itinerary.order] == ["high"]


def test_milp_matches_or_beats_greedy_value():
    scored = [ScoredOption(option=make_option(f"o{i}", est_hours=2 + (i % 3)), score=float(i + 1),
                            features=None) for i in range(8)]
    budget = 10.0
    greedy = greedy_two_opt(scored, budget)
    milp = milp_orienteering(scored, budget)
    assert milp.total_hours <= budget + 1e-6
    assert milp.total_value >= greedy.total_value - 1e-6


def test_milp_empty_candidates_returns_empty_itinerary():
    itinerary = milp_orienteering([], budget_hours=5.0)
    assert itinerary.order == []
    assert itinerary.total_value == 0.0
