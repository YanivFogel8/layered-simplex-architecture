from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))

from data_scaling_experiment import CSpec  # noqa: E402
from factorial_scaling_experiment import build_jobs  # noqa: E402


def test_factorial_jobs_form_cartesian_grid_with_stable_seeds():
    jobs = build_jobs(
        d_values=[1000, 10000],
        n_values=[50, 500, 5000],
        alphas=(1.0, 2.0),
        c_spec=CSpec("cstar", 2.0),
        profile_samples=20,
        profile_batch_size=8,
        seed=17,
        u_min=None,
        u_max=None,
        u_points=101,
        laguerre_order=16,
        chunk_size=32,
    )
    assert [(job["d"], job["N"]) for job in jobs] == [
        (1000, 50), (1000, 500), (1000, 5000),
        (10000, 50), (10000, 500), (10000, 5000),
    ]
    assert len({job["seed"] for job in jobs}) == len(jobs)
    assert all(job["alphas"] == (1.0, 2.0) for job in jobs)
