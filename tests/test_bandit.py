import numpy as np

from recsys.bandit import ThompsonBandit, reward_from_feedback


def test_prior_mean_matches_theta_hat_with_no_data():
    prior = np.array([0.0, 1.0, -1.0])
    bandit = ThompsonBandit(["bias", "a", "b"], prior_mean=prior)
    np.testing.assert_allclose(bandit.theta_hat, prior, atol=1e-9)


def test_update_shifts_theta_toward_rewarded_feature():
    prior = np.zeros(2)
    bandit = ThompsonBandit(["a", "b"], prior_mean=prior, lambda_reg=0.5)
    phi = np.array([1.0, 0.0])
    for _ in range(20):
        bandit.update(phi, reward=1.0)
    # theta for feature "a" should move up toward the reward it kept earning
    assert bandit.theta_hat[0] > 0.5
    # feature "b" was never touched, should stay at its prior (0)
    assert abs(bandit.theta_hat[1]) < 1e-6


def test_thompson_sampling_prefers_higher_value_arm_on_average():
    prior = np.zeros(2)
    bandit = ThompsonBandit(["good", "bad"], prior_mean=prior, lambda_reg=1.0, sigma=0.3)
    good_phi = np.array([1.0, 0.0])
    bad_phi = np.array([0.0, 1.0])
    for _ in range(30):
        bandit.update(good_phi, reward=1.0)
        bandit.update(bad_phi, reward=-1.0)

    rng = np.random.default_rng(0)
    picks_good = 0
    trials = 200
    for _ in range(trials):
        scores = bandit.rank([good_phi, bad_phi], rng)
        if scores[0] > scores[1]:
            picks_good += 1
    assert picks_good / trials > 0.9


def test_decay_pulls_posterior_back_toward_prior():
    prior = np.array([0.0])
    bandit = ThompsonBandit(["a"], prior_mean=prior, lambda_reg=1.0)
    phi = np.array([1.0])
    for _ in range(50):
        bandit.update(phi, reward=1.0)  # no decay: theta_hat should converge near 1
    theta_no_decay = bandit.theta_hat[0]

    bandit2 = ThompsonBandit(["a"], prior_mean=prior, lambda_reg=1.0)
    for _ in range(50):
        bandit2.update(phi, reward=1.0, gamma=0.8)  # heavy decay caps how far it can drift
    theta_with_decay = bandit2.theta_hat[0]

    assert theta_no_decay > theta_with_decay


def test_reward_from_feedback_rating_overrides_action():
    assert reward_from_feedback("accepted_repeat", rating=5) == 1.0
    assert reward_from_feedback("accepted_repeat", rating=1) == -1.0
    assert reward_from_feedback("accepted_repeat") == 1.0
    assert reward_from_feedback("accepted_no_repeat") < 0
    assert reward_from_feedback("skipped") < 0
    assert reward_from_feedback("unknown_action") == 0.0
