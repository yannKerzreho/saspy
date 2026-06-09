"""Per-degree variance correction for Q initialisation."""

import jax.numpy as jnp


def _double_factorial(m: int) -> float:
    result = 1.0
    for i in range(1, m + 1, 2):
        result *= i
    return result


def _var_z_power(k: int) -> float:
    """Exact Var(z^k) for z ~ N(0, 1)."""
    e2k = _double_factorial(2 * k - 1)
    ek2 = 0.0 if k % 2 == 1 else _double_factorial(k - 1) ** 2
    return e2k - ek2


def q_degree_correction(q_degree: int, taylor_decay: float = 1.0) -> jnp.ndarray:
    """
    Per-degree scale factors for Q initialisation.

    Ensures equal variance contribution per active degree (taylor_decay=1),
    optionally tapered geometrically so lower degrees dominate (taylor_decay<1).
    The output is normalised so total Var(Q_raw(z)) = 1/2.

    Returns array of shape (q_degree+1,), dtype float32.
    Degree-0 (bias) entry = 1.0; it contributes zero variance.
    """
    n_active = max(q_degree, 1)

    base = [1.0] + [1.0 / (n_active * _var_z_power(k)) ** 0.5
                    for k in range(1, q_degree + 1)]

    if taylor_decay == 1.0:
        return jnp.array(base, dtype=jnp.float32)

    tapered = [base[k] * (taylor_decay ** (k - 1)) if k >= 1 else base[k]
               for k in range(q_degree + 1)]

    total = sum(0.5 * tapered[k] ** 2 * _var_z_power(k)
                for k in range(1, q_degree + 1))

    if total < 1e-12:
        out = [0.0] * (q_degree + 1)
        out[0] = 1.0
        if q_degree >= 1:
            out[1] = 1.0 / (n_active * _var_z_power(1)) ** 0.5
        return jnp.array(out, dtype=jnp.float32)

    norm = (0.5 / total) ** 0.5
    normalized = [tapered[k] * norm if k >= 1 else tapered[k]
                  for k in range(q_degree + 1)]
    return jnp.array(normalized, dtype=jnp.float32)
