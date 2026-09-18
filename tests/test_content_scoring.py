import numpy as np

from recsys.content_scoring import DEFAULT_WEIGHTS, diversify, featurize, filter_candidates, rank_by_rules
from recsys.content_scoring import ScoredOption
from recsys.schema import Context, Option


def make_option(**overrides) -> Option:
    base = dict(
        id="X-001", source_sheet="Cali", section="Test",
        main_destination="Test Place", specific_attraction="Test Attraction",
        type="Nature", cost="$$", priority="HIGH",
        viral_rating="Actually worth it", est_hours=4.0,
    )
    base.update(overrides)
    return Option(**base)


def test_filter_drops_over_budget():
    opts = [make_option(id="cheap", cost="$"), make_option(id="pricey", cost="$$$$")]
    ctx = Context(budget_level=1)  # allows Free/$
    kept = filter_candidates(opts, ctx, excluded_ids=set())
    assert [o.id for o in kept] == ["cheap"]


def test_filter_excludes_done_no_repeat():
    opts = [make_option(id="a"), make_option(id="b")]
    kept = filter_candidates(opts, Context(), excluded_ids={"a"})
    assert [o.id for o in kept] == ["b"]


def test_filter_trip_type_scopes_by_region_sheet():
    nearby = make_option(id="nearby", source_sheet="Cali", est_hours=3)
    far = make_option(id="far", source_sheet="East Asia", est_hours=2)  # short dwell, but not nearby
    kept_day = filter_candidates([nearby, far], Context(trip_type="hangout_nearby"), set())
    assert [o.id for o in kept_day] == ["nearby"]
    kept_long = filter_candidates([nearby, far], Context(trip_type="long_trip"), set())
    assert [o.id for o in kept_long] == ["far"]


def test_filter_trip_type_still_caps_long_dwell_for_nearby():
    nearby_but_long = make_option(id="long_dwell", source_sheet="Cali", est_hours=100)
    kept = filter_candidates([nearby_but_long], Context(trip_type="hangout_nearby"), set())
    assert kept == []


def test_featurize_outdoor_match_is_maximal_when_aligned():
    outdoor_opt = make_option(type="Adventure")  # OUTDOOR_LEAN["Adventure"] is high positive (schema.py)
    ctx_outdoor = Context(indoor_outdoor=1.0)
    ctx_indoor = Context(indoor_outdoor=-1.0)
    phi_match = featurize(outdoor_opt, ctx_outdoor)
    phi_mismatch = featurize(outdoor_opt, ctx_indoor)
    idx = 1  # indoor_outdoor_match
    assert phi_match[idx] > phi_mismatch[idx]
    assert phi_match[idx] > 0.9  # nearly perfectly aligned -> match score near 1


def test_budget_fit_penalizes_over_budget_items():
    ctx = Context(budget_level=0)  # only "Free" fits
    cheap = featurize(make_option(cost="Free"), ctx)
    expensive = featurize(make_option(cost="$$$$"), ctx)
    budget_idx = 4
    assert cheap[budget_idx] > expensive[budget_idx]


def test_rank_by_rules_orders_best_match_first():
    good = make_option(id="good", type="Adventure", cost="$", priority="MUST",
                        viral_rating="Actually worth it")
    bad = make_option(id="bad", type="Relaxation", cost="$$$$", priority="OPTIONAL",
                       viral_rating="Skip unless personally interested")
    ctx = Context(budget_level=1, indoor_outdoor=1.0, mood_adventurous=1.0, energy=0.9)
    ranked = rank_by_rules([bad, good], ctx, weights=DEFAULT_WEIGHTS)
    assert ranked[0].option.id == "good"


def test_featurize_returns_finite_vector():
    ctx = Context()
    phi = featurize(make_option(type=None, cost=None, priority=None, viral_rating=None), ctx)
    assert np.all(np.isfinite(phi))


def test_diversify_breaks_up_single_category_tunnel():
    scored = (
        [ScoredOption(option=make_option(id=f"nature{i}", type="Nature"), score=10 - i, features=None)
         for i in range(5)]
        + [ScoredOption(option=make_option(id="food1", type="Food"), score=6.0, features=None)]
    )
    top3 = diversify(scored, top_k=3, penalty=2.0)
    types = [s.option.type for s in top3]
    assert "Food" in types  # would be excluded by pure score ranking (6.0 < top-3 nature scores)


def test_diversify_returns_everything_sorted_when_top_k_covers_all():
    scored = [ScoredOption(option=make_option(id="a"), score=1.0, features=None),
              ScoredOption(option=make_option(id="b"), score=2.0, features=None)]
    result = diversify(scored, top_k=5)
    assert [s.option.id for s in result] == ["b", "a"]
