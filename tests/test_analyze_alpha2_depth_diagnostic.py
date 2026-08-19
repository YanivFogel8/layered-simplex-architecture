import math
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))

from analyze_alpha2_depth_diagnostic import corrected_fit, label_cost_fit  # noqa: E402
from data_scaling_experiment import DataScalingResult  # noqa: E402


def test_corrected_fit_recovers_two_term_curve():
    rows = []
    for n in (100, 200, 500, 1000, 2000):
        value = 1.7 * n ** -0.5 + 0.8 * n ** -1
        rows.append(DataScalingResult(
            d=10000, alpha=2.0, c_label="cstar", c_value=2.3, L=21, N=n,
            profile_samples=10, unique_profiles=10, union_unique_profiles=10,
            distinct_r_values=5, max_profile_part=4, entropy_bits=1.0,
            mean_log_q_nats=-1.0, model_penalty_bits=0.0,
            regret_bits=value * math.log(10000), regret_over_ln_d=value,
            stderr_bits=0.01, stderr_over_ln_d=0.001,
            nonconverged_profile_samples=0, failed_profile_samples=0,
            min_left_gap=1.0, min_right_gap=1.0, u_min=-80.0, u_max=45.0,
            elapsed_seconds=1.0,
        ))
    a, b, r2, rmse = corrected_fit(rows)
    assert math.isclose(a, 1.7, rel_tol=1e-10)
    assert math.isclose(b, 0.8, rel_tol=1e-10)
    assert math.isclose(r2, 1.0)
    assert rmse < 1e-12


def test_label_cost_fit_recovers_log_d_minus_half_log_n():
    rows = []
    for d in (10000, 100000):
        for n in (500, 1000, 5000, 20000):
            regret = n ** -0.5 * (1.8 * math.log(d) - 0.9 * math.log(n) + 0.4)
            rows.append(DataScalingResult(
                d=d, alpha=2.0, c_label="cstar", c_value=2.3, L=21, N=n,
                profile_samples=10, unique_profiles=10, union_unique_profiles=10,
                distinct_r_values=5, max_profile_part=4, entropy_bits=1.0,
                mean_log_q_nats=-1.0, model_penalty_bits=0.0,
                regret_bits=regret, regret_over_ln_d=regret/math.log(d),
                stderr_bits=0.01, stderr_over_ln_d=0.001,
                nonconverged_profile_samples=0, failed_profile_samples=0,
                min_left_gap=1.0, min_right_gap=1.0, u_min=-80.0, u_max=45.0,
                elapsed_seconds=1.0,
            ))
    a, b, intercept, r2, rmse = label_cost_fit(rows)
    assert math.isclose(a, 1.8, rel_tol=1e-10)
    assert math.isclose(b, -0.9, rel_tol=1e-10)
    assert math.isclose(intercept, 0.4, rel_tol=1e-10)
    assert math.isclose(b/a, -0.5, rel_tol=1e-10)
    assert math.isclose(r2, 1.0)
    assert rmse < 1e-12
