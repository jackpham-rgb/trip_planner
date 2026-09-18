import json
from datetime import date, timedelta

from recsys.state import BanditPosterior, UserState, load_state, save_state


def test_excluded_ids_only_includes_done_no_repeat():
    state = UserState()
    state.record_outcome("keep", would_repeat=True)
    state.record_outcome("drop", would_repeat=False)
    assert state.excluded_ids() == {"drop"}


def test_recovery_multiplier_is_low_right_after_and_recovers_over_time():
    state = UserState()
    state.record_outcome("fav", would_repeat=True, when=date.today())
    right_after = state.recovery_multiplier("fav", today=date.today())
    much_later = state.recovery_multiplier("fav", today=date.today() + timedelta(days=60))
    never_done = state.recovery_multiplier("brand_new")
    assert right_after < much_later
    assert much_later > 0.9
    assert never_done == 1.0


def test_save_and_load_state_round_trip(tmp_path):
    state = UserState()
    state.record_outcome("a", would_repeat=True, rating=4.5)
    state.bandit = BanditPosterior(A=[[1.0, 0.0], [0.0, 1.0]], b=[0.1, 0.2],
                                    feature_names=["x", "y"], n_updates=3)
    path = tmp_path / "state.json"
    save_state(state, path)

    loaded = load_state(path)
    assert loaded.items["a"].rating == 4.5
    assert loaded.items["a"].status == "done_repeat"
    assert loaded.bandit.n_updates == 3
    assert loaded.bandit.b == [0.1, 0.2]


def test_load_state_missing_file_returns_empty_state(tmp_path):
    state = load_state(tmp_path / "does_not_exist.json")
    assert state.items == {}
    assert state.bandit is None
