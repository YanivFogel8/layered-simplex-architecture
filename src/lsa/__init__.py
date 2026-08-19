"""Layered simplex architecture (LSA): code for the paper.

Paper: "A Layered Simplex Architecture for Large Alphabets"
(Feder, Fogel, Urbanke).  See README.md for the map from every figure
and table of the paper to the script that reproduces it.

The package has two numerical cores that check each other:

- ``lsa.mixture_weights``: the original reference implementation of the
  Appendix-B numerics (grid recursion for the moment functions, direct
  evaluation of the mixture weights ``q_lambda``).
- ``lsa.layered`` + ``lsa.fast_tables`` + ``lsa.mellin`` +
  ``lsa.universal_tables``: the production implementation (batched
  disk-cached tables, global-peak scan, Mellin--Barnes exact rows,
  designed anchor stores).  Validated against the reference; used by all
  corpus experiments.

Everything is measured in bits.
"""

# Prior sampling (Section 3, Figure 1)
from lsa.product_simplex import (
    ProductSimplexSampler,
    fill_product_simplex_sample,
    sample_product_simplex_distribution,
)

# Entropies and divergences, in bits
from lsa.metrics import entropy, kl_divergence

# Reference numerics (Appendix B; original implementation)
from lsa.mixture_weights import (
    ProductMomentTables,
    QLambdaResult,
    build_product_moment_tables,
    build_selected_product_moment_tables,
    compute_log_q_by_partition,
    log_q_lambda_closed_l1,
    log_q_lambda_grid,
    log_q_lambda_laplace,
)

# Profile weights A_lambda (Appendix A)
from lsa.pattern_weights import (
    HybridSaddlepointApproximation,
    SaddlepointApproximation,
    log_a_lambda_hybrid,
    log_a_lambda_saddlepoint,
    partition_multiplicities,
)

from lsa.transforms import largest_entries_first

# Production numerics and corpus experiments (Appendix B, Section 5)
from lsa.codelength import (
    C_STAR,
    DepthAveragedCodelength,
    default_l_max,
    depth_averaged_codelength,
    needed_r_values,
    profile_of,
)
from lsa.corpus import (
    empirical_conditional_entropy_bits,
    empirical_entropy_bits,
    load_tokens,
    prefix_counts,
)
from lsa.fast_tables import build_tables_fast

__version__ = "1.0.0"

__all__ = [
    "C_STAR",
    "DepthAveragedCodelength",
    "HybridSaddlepointApproximation",
    "ProductMomentTables",
    "ProductSimplexSampler",
    "QLambdaResult",
    "SaddlepointApproximation",
    "build_product_moment_tables",
    "build_selected_product_moment_tables",
    "build_tables_fast",
    "compute_log_q_by_partition",
    "default_l_max",
    "depth_averaged_codelength",
    "empirical_conditional_entropy_bits",
    "empirical_entropy_bits",
    "entropy",
    "fill_product_simplex_sample",
    "kl_divergence",
    "largest_entries_first",
    "load_tokens",
    "log_a_lambda_hybrid",
    "log_a_lambda_saddlepoint",
    "log_q_lambda_closed_l1",
    "log_q_lambda_grid",
    "log_q_lambda_laplace",
    "needed_r_values",
    "partition_multiplicities",
    "prefix_counts",
    "profile_of",
    "sample_product_simplex_distribution",
]
