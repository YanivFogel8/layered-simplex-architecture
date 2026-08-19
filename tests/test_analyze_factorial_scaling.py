import math
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))

from analyze_factorial_scaling import normalized_prefactor  # noqa: E402
from data_scaling_experiment import DataScalingResult  # noqa: E402


def test_normalized_prefactor_removes_claimed_d_and_n_factors():
    alpha = 2.0
    d = 10000
    n = 400
    target = 1.7
    regret = target * math.log(d) * n ** -(1.0 - 1.0 / alpha)
    row = DataScalingResult(
        d=d, alpha=alpha, c_label="cstar", c_value=2.3, L=21, N=n,
        profile_samples=10, unique_profiles=10, union_unique_profiles=10,
        distinct_r_values=5, max_profile_part=4, entropy_bits=1.0,
        mean_log_q_nats=-1.0, model_penalty_bits=0.0, regret_bits=regret,
        regret_over_ln_d=regret/math.log(d), stderr_bits=0.01,
        stderr_over_ln_d=0.01/math.log(d), nonconverged_profile_samples=0,
        failed_profile_samples=0, min_left_gap=1.0, min_right_gap=1.0,
        u_min=-80.0, u_max=45.0, elapsed_seconds=1.0,
    )
    assert math.isclose(normalized_prefactor(row), target)
