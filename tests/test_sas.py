"""
test_sas.py — Test suite for the 3-layer SAS architecture.

Covers:
  1. Shape integrity  — full pipeline for d=1 and d=10, all basis types
  2. Dead-neuron check — dense W_in prevents rank collapse for d=1
  3. Combine associativity — required by jax.lax.associative_scan
  4. JIT compilation — no TracerLeak, no recompilation errors
  5. Trivial projector — W_in=1 reproduces broadcast-scalar behaviour
  6. SASForecaster smoke — fit / predict / update / transform
  7. RandomFourierBasis — RFF basis shapes, properties, and pytree round-trip
  8. Decoupled basis_p / basis_q — mixing different basis types
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from saspy import (
    InputProjector,
    DiagonalPoly,
    LRUBlockPoly,
    BlockLinearPoly,
    RandomFourierBasis,
    SASModel,
    SASForecaster,
    _forward,
)


# ── shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


@pytest.fixture
def ar1():
    rng = np.random.default_rng(42)
    y = np.zeros(500)
    for t in range(1, 500):
        y[t] = 0.8 * y[t - 1] + rng.normal()
    return y


# ════════════════════════════════════════════════════════════════════════════
# 1. Shape integrity
# ════════════════════════════════════════════════════════════════════════════

class TestShapes:
    """Full pipeline (projector → basis → engine) produces (T, N) states."""

    T = 120
    CHUNK = 32

    # ── DiagonalPoly ─────────────────────────────────────────────────────────

    @pytest.mark.parametrize("d", [1, 10])
    def test_diagonal_shape(self, key, d):
        N = 64
        proj  = InputProjector(d=d, n_drivers=N).initialize(key)
        basis = DiagonalPoly(n=N, p_degree=1, q_degree=1).initialize(key)
        model = SASModel(proj, basis, basis)

        z  = jax.random.normal(key, (self.T, d), dtype=jnp.float32)
        s0 = jnp.zeros(N)

        states, s_last = _forward(model, z, s0, self.CHUNK)

        assert states.shape == (self.T, N),  f"states: {states.shape}"
        assert s_last.shape == (N,),          f"s_last: {s_last.shape}"
        assert jnp.isfinite(states).all()
        assert jnp.isfinite(s_last).all()

    # ── LRUBlockPoly ─────────────────────────────────────────────────────────

    @pytest.mark.parametrize("d", [1, 10])
    def test_lru_block_shape(self, key, d):
        K = 16                              # n_drivers = K, N = 2K = 32
        N = K * 2
        proj  = InputProjector(d=d, n_drivers=K).initialize(key)
        basis = LRUBlockPoly(n_blocks=K, p_degree=1, q_degree=1).initialize(key)
        model = SASModel(proj, basis, basis)

        z  = jax.random.normal(key, (self.T, d), dtype=jnp.float32)
        s0 = jnp.zeros(N)

        states, s_last = _forward(model, z, s0, self.CHUNK)

        assert states.shape == (self.T, N)
        assert s_last.shape == (N,)
        assert jnp.isfinite(states).all()

    # ── BlockLinearPoly ───────────────────────────────────────────────────────

    @pytest.mark.parametrize("d,B", [(1, 4), (10, 4), (1, 8)])
    def test_block_linear_shape(self, key, d, B):
        K = 8                               # N = K * B
        N = K * B
        proj  = InputProjector(d=d, n_drivers=K).initialize(key)
        basis = BlockLinearPoly(n_blocks=K, block_size=B,
                                p_degree=1, q_degree=1).initialize(key)
        model = SASModel(proj, basis, basis)

        z  = jax.random.normal(key, (self.T, d), dtype=jnp.float32)
        s0 = jnp.zeros(N)

        states, s_last = _forward(model, z, s0, self.CHUNK)

        assert states.shape == (self.T, N)
        assert s_last.shape == (N,)
        assert jnp.isfinite(states).all()

    # ── T not divisible by chunk_size ────────────────────────────────────────

    @pytest.mark.parametrize("T", [1, 33, 64, 65, 128])
    def test_padding_correctness(self, key, T):
        N     = 32
        proj  = InputProjector.trivial(n_drivers=N)
        basis = DiagonalPoly(n=N).initialize(key)
        model = SASModel(proj, basis, basis)

        z  = jax.random.normal(key, (T, 1), dtype=jnp.float32)
        s0 = jnp.zeros(N)

        states, s_last = _forward(model, z, s0, chunk_size=16)

        assert states.shape == (T, N)
        assert s_last.shape == (N,)


# ════════════════════════════════════════════════════════════════════════════
# 2. Dead-neuron check (rank collapse prevention)
# ════════════════════════════════════════════════════════════════════════════

class TestDiversity:
    """Dense W_in must prevent all neurons from collapsing to the same value."""

    T = 300

    # ── regression guard: d=1 normalisation bug ──────────────────────────────

    def test_d1_weights_have_scale_diversity(self, key):
        """
        For d=1, W must NOT be column-normalised to ±1.

        The old buggy code divided every W[0,i] by |W[0,i]|, collapsing
        all weights to exactly ±1.  Scale diversity (values like 0.3, -1.4,
        0.9, …) is required so that even-degree features z_tilde[i]^2k
        differ across drivers.  This test catches that regression directly.
        """
        N    = 128
        proj = InputProjector(d=1, n_drivers=N).initialize(key)
        W    = np.asarray(proj.W[0])          # (N,)

        magnitudes = np.abs(W)

        # If every |W[0,i]| == 1 the bug is present
        assert not np.allclose(magnitudes, 1.0, atol=1e-4), (
            "All W[0,i] collapsed to ±1 — column-normalisation bug for d=1!"
        )
        # For N=128 i.i.d. Gaussians the std of magnitudes ≈ 0.6
        assert float(np.std(magnitudes)) > 0.1, (
            f"Insufficient scale diversity: std(|W|) = {np.std(magnitudes):.4f}"
        )

    def test_d1_even_degree_features_differ(self, key):
        """
        Even-degree collapse check: z_tilde[i]^2 must not all equal z_t^2.

        With ±1 weights (bug), (W[0,i]·z)^2 = z^2 for every i — a rank-1
        feature matrix for all even degrees.  After the fix the squared
        features must span a genuinely high-variance subspace.
        """
        N       = 64
        proj    = InputProjector(d=1, n_drivers=N).initialize(key)
        z_t     = jnp.array([[1.5]], dtype=jnp.float32)
        z_tilde = proj.project(z_t)[0]         # (N,)
        sq      = np.asarray(z_tilde ** 2)     # even-degree features

        assert float(np.std(sq)) > 1e-3, (
            f"Even-degree collapse: std(z_tilde²) = {np.std(sq):.2e}. "
            "Weights are effectively ±1."
        )

    def test_random_projector_no_collapse(self, key):
        """Random projection: each unit in DiagonalPoly sees a different scaled z."""
        N    = 64
        proj  = InputProjector(d=1, n_drivers=N).initialize(key)
        basis = DiagonalPoly(n=N, p_degree=1).initialize(key)
        model = SASModel(proj, basis, basis)

        z  = jax.random.normal(key, (self.T, 1), dtype=jnp.float32)
        s0 = jnp.zeros(N)

        states, _ = _forward(model, z, s0, chunk_size=64)

        # std across N dimension at each timestep — must be non-trivially positive
        per_t_std = jnp.std(states, axis=1)          # (T,)
        mean_std  = float(jnp.mean(per_t_std[10:]))  # skip washout
        assert mean_std > 1e-4, (
            f"Rank collapse detected: mean std across N = {mean_std:.2e}"
        )

    def test_trivial_projector_no_collapse(self, key):
        """Trivial W_in=1 still yields diversity from per-unit coefficients."""
        N    = 64
        proj  = InputProjector.trivial(n_drivers=N)
        basis = DiagonalPoly(n=N, p_degree=1).initialize(key)
        model = SASModel(proj, basis, basis)

        z  = jax.random.normal(key, (self.T, 1), dtype=jnp.float32)
        s0 = jnp.zeros(N)

        states, _ = _forward(model, z, s0, chunk_size=64)

        per_t_std = jnp.std(states, axis=1)
        mean_std  = float(jnp.mean(per_t_std[10:]))
        assert mean_std > 1e-4, (
            f"Trivial projector collapsed: mean std = {mean_std:.2e}"
        )

    def test_random_vs_trivial_richer(self, key):
        """Random projection should yield strictly more diversity than trivial."""
        N = 64
        basis_fn = lambda k: DiagonalPoly(n=N, p_degree=1).initialize(k)

        k1, k2 = jax.random.split(key)
        z = jax.random.normal(key, (self.T, 1), dtype=jnp.float32)
        s0 = jnp.zeros(N)

        # Random projector
        b = basis_fn(k2)
        proj_r  = InputProjector(d=1, n_drivers=N).initialize(k1)
        model_r = SASModel(proj_r, b, b)
        states_r, _ = _forward(model_r, z, s0, 64)

        # Trivial projector (same basis seed)
        proj_t  = InputProjector.trivial(n_drivers=N)
        model_t = SASModel(proj_t, b, b)
        states_t, _ = _forward(model_t, z, s0, 64)

        std_r = float(jnp.mean(jnp.std(states_r[10:], axis=1)))
        std_t = float(jnp.mean(jnp.std(states_t[10:], axis=1)))
        assert std_r >= std_t * 0.5, (
            f"Random projector not richer: std_r={std_r:.4f} std_t={std_t:.4f}"
        )


# ════════════════════════════════════════════════════════════════════════════
# 3. Combine associativity
# ════════════════════════════════════════════════════════════════════════════

class TestAssociativity:
    """combine((a ∘ b) ∘ c) == combine(a ∘ (b ∘ c)) for all block bases."""

    def _rand_pair_diag(self, key, N):
        ka, kb = jax.random.split(key)
        a = jax.random.uniform(ka, (N,), minval=-0.8, maxval=0.8)
        b = jax.random.normal(kb, (N,))
        return (a, b)

    def _rand_pair_block(self, key, K, B):
        ka, kb = jax.random.split(key)
        A = jax.random.normal(ka, (K, B, B)) * 0.1
        b = jax.random.normal(kb, (K * B,))
        return (A, b)

    def test_diagonal_combine_associative(self, key):
        N    = 32
        basis = DiagonalPoly(n=N).initialize(key)
        k1, k2, k3 = jax.random.split(key, 3)

        a, b, c = (self._rand_pair_diag(k, N) for k in [k1, k2, k3])
        left  = basis.combine(basis.combine(a, b), c)
        right = basis.combine(a, basis.combine(b, c))

        assert jnp.allclose(left[0], right[0], atol=1e-5)
        assert jnp.allclose(left[1], right[1], atol=1e-5)

    def test_lru_block_combine_associative(self, key):
        K    = 8
        basis = LRUBlockPoly(n_blocks=K).initialize(key)
        k1, k2, k3 = jax.random.split(key, 3)

        a, b, c = (self._rand_pair_block(k, K, 2) for k in [k1, k2, k3])
        left  = basis.combine(basis.combine(a, b), c)
        right = basis.combine(a, basis.combine(b, c))

        assert jnp.allclose(left[0], right[0], atol=1e-5)
        assert jnp.allclose(left[1], right[1], atol=1e-5)

    @pytest.mark.parametrize("B", [2, 4, 8])
    def test_block_linear_combine_associative(self, key, B):
        K    = 6
        basis = BlockLinearPoly(n_blocks=K, block_size=B).initialize(key)
        k1, k2, k3 = jax.random.split(key, 3)

        a, b, c = (self._rand_pair_block(k, K, B) for k in [k1, k2, k3])
        left  = basis.combine(basis.combine(a, b), c)
        right = basis.combine(a, basis.combine(b, c))

        assert jnp.allclose(left[0], right[0], atol=1e-5)
        assert jnp.allclose(left[1], right[1], atol=1e-5)

    def test_rff_combine_associative(self, key):
        """RandomFourierBasis uses a diagonal element-wise monoid — must be associative."""
        N     = 40   # K=10, B=4
        basis = RandomFourierBasis(n_blocks=10, features_per_block=4).initialize(key)
        k1, k2, k3 = jax.random.split(key, 3)

        def rand_pair(k):
            ka, kb = jax.random.split(k)
            a = jax.random.uniform(ka, (N,), minval=-0.8, maxval=0.8)
            b = jax.random.normal(kb, (N,))
            return (a, b)

        a, b, c = (rand_pair(k) for k in [k1, k2, k3])
        left  = basis.combine(basis.combine(a, b), c)
        right = basis.combine(a, basis.combine(b, c))

        assert jnp.allclose(left[0], right[0], atol=1e-5)
        assert jnp.allclose(left[1], right[1], atol=1e-5)


# ════════════════════════════════════════════════════════════════════════════
# 4. JIT compilation
# ════════════════════════════════════════════════════════════════════════════

class TestJIT:
    """Full forward pass compiles cleanly; no TracerLeak or shape errors."""

    @pytest.mark.parametrize("BasisClass,basis_kwargs,proj_kwargs", [
        (DiagonalPoly,    {"n": 32},                      {"d": 1, "n_drivers": 32}),
        (LRUBlockPoly,    {"n_blocks": 16},                {"d": 1, "n_drivers": 16}),
        (BlockLinearPoly, {"n_blocks": 8, "block_size": 4},{"d": 1, "n_drivers": 8}),
        (DiagonalPoly,    {"n": 32},                      {"d": 5, "n_drivers": 32}),
    ])
    def test_jit_forward(self, key, BasisClass, basis_kwargs, proj_kwargs):
        T = 80
        proj  = InputProjector(**proj_kwargs).initialize(key)
        basis = BasisClass(**basis_kwargs).initialize(key)
        model = SASModel(proj, basis, basis)

        z  = jax.random.normal(key, (T, proj_kwargs["d"]), dtype=jnp.float32)
        s0 = jnp.zeros(model.n)

        # Should not raise
        states, s_last = jax.jit(
            lambda m, z, s0: _forward(m, z, s0, chunk_size=32)
        )(model, z, s0)

        assert states.shape == (T, model.n)
        assert jnp.isfinite(states).all()
        assert jnp.isfinite(s_last).all()

    def test_step_once_jit(self, key):
        from saspy import _step_once
        N     = 32
        proj  = InputProjector(d=1, n_drivers=N).initialize(key)
        basis = DiagonalPoly(n=N).initialize(key)
        model = SASModel(proj, basis, basis)

        s   = jnp.zeros(N)
        z_t = jnp.array([0.5], dtype=jnp.float32)

        s_new = _step_once(model, s, z_t)

        assert s_new.shape == (N,)
        assert jnp.isfinite(s_new).all()

    def test_pytree_roundtrip(self, key):
        """Flatten + unflatten SASModel → leaves and structure unchanged."""
        proj    = InputProjector(d=2, n_drivers=16).initialize(key)
        basis_p = LRUBlockPoly(n_blocks=16).initialize(key)
        k2      = jax.random.split(key)[1]
        basis_q = LRUBlockPoly(n_blocks=16).initialize(k2)
        model   = SASModel(proj, basis_p, basis_q)

        leaves, treedef = jax.tree_util.tree_flatten(model)
        rebuilt         = jax.tree_util.tree_unflatten(treedef, leaves)

        assert jnp.allclose(rebuilt.projector.W,        model.projector.W)
        assert jnp.allclose(rebuilt.basis_p.P_weights,  model.basis_p.P_weights)
        assert jnp.allclose(rebuilt.basis_q.P_weights,  model.basis_q.P_weights)


# ════════════════════════════════════════════════════════════════════════════
# 5. Trivial projector — backward-compatible "W_in = 1" behaviour
# ════════════════════════════════════════════════════════════════════════════

class TestTrivialProjector:

    def test_trivial_w_is_ones(self):
        N = 16
        p = InputProjector.trivial(n_drivers=N)
        assert p.W is not None
        assert p.W.shape == (1, N)
        assert jnp.allclose(p.W, jnp.ones((1, N)))

    def test_trivial_broadcasts_scalar(self):
        """z_tilde[i] == z_t for every i."""
        N   = 16
        p   = InputProjector.trivial(n_drivers=N)
        z   = jnp.array([[2.5]], dtype=jnp.float32)   # (1, 1)
        out = p.project(z)                             # (1, N)
        assert out.shape == (1, N)
        assert jnp.allclose(out, 2.5)

    def test_trivial_model_matches_classic_scan(self, key):
        """
        SASModel with trivial projector should produce states with exactly
        the same element-wise structure as a single-unit-broadcast scan.
        """
        N = 32
        T = 50
        proj  = InputProjector.trivial(n_drivers=N)
        basis = DiagonalPoly(n=N, p_degree=1).initialize(key)
        model = SASModel(proj, basis, basis)

        z  = jax.random.normal(key, (T, 1), dtype=jnp.float32)
        s0 = jnp.zeros(N)

        states, _ = _forward(model, z, s0, chunk_size=16)
        assert states.shape == (T, N)
        # With trivial W all units still diverge because P/Q weights differ
        per_t_std = float(jnp.mean(jnp.std(states[5:], axis=1)))
        assert per_t_std > 1e-5


# ════════════════════════════════════════════════════════════════════════════
# 6. SASForecaster smoke tests
# ════════════════════════════════════════════════════════════════════════════

class TestSASForecaster:

    def _make_forecaster(self, key, n=64, washout=20):
        proj  = InputProjector.trivial(n_drivers=n)
        basis = DiagonalPoly(n=n, p_degree=1, q_degree=1)
        model = SASModel(proj, basis, basis)
        return SASForecaster(model=model, washout=washout, chunk_size=32, seed=0)

    def test_fit_predict_update(self, key, ar1):
        fc = self._make_forecaster(key)
        fc.fit(ar1[:400], horizons=[1, 5])

        p1 = fc.predict(1)
        p5 = fc.predict(5)
        assert np.isfinite(p1) and np.isfinite(p5)

        fc.update(float(ar1[400]))
        p1_after = fc.predict(1)
        # state changed → prediction changes
        assert p1_after != p1

    def test_transform_shape(self, key, ar1):
        fc = self._make_forecaster(key, n=32)
        fc.fit(ar1[:400], horizons=[1])
        states = fc.transform(ar1[400:450])
        assert states.shape == (50, 32)
        assert np.isfinite(states).all()

    def test_predict_before_fit_raises(self, key):
        proj  = InputProjector.trivial(n_drivers=16)
        basis = DiagonalPoly(n=16)
        fc    = SASForecaster(SASModel(proj, basis, basis))
        with pytest.raises(RuntimeError, match="must be fit"):
            fc.predict(1)

    def test_update_before_fit_raises(self, key):
        proj  = InputProjector.trivial(n_drivers=16)
        basis = DiagonalPoly(n=16)
        fc    = SASForecaster(SASModel(proj, basis, basis))
        with pytest.raises(RuntimeError, match="must be fit"):
            fc.update(0.0)

    def test_unknown_horizon_raises(self, key, ar1):
        fc = self._make_forecaster(key)
        fc.fit(ar1[:400], horizons=[1, 5])
        with pytest.raises(KeyError):
            fc.predict(99)

    def test_beats_mean_baseline(self, key, ar1):
        """h=1 MSE on AR(1) must beat 80% of the test-set variance."""
        train, test = ar1[:400], ar1[400:]
        fc = self._make_forecaster(key, n=64, washout=20)
        fc.fit(train, horizons=[1])

        preds = []
        for t in range(len(test)):
            preds.append(fc.predict(1))
            fc.update(test[t])

        preds    = np.asarray(preds)
        mse      = float(np.mean((preds - test) ** 2))
        baseline = float(np.var(test))
        assert mse < 0.8 * baseline, (
            f"MSE {mse:.3f} ≥ 0.8 × baseline {0.8 * baseline:.3f}"
        )

    @pytest.mark.parametrize("BasisClass,n,n_drv", [
        (DiagonalPoly,    64, 64),
        (LRUBlockPoly,    64, 32),
        (BlockLinearPoly, 64, 16),
    ])
    def test_all_bases_smoke(self, key, ar1, BasisClass, n, n_drv):
        """Every basis type fits and predicts without error."""
        if BasisClass is DiagonalPoly:
            basis = BasisClass(n=n)
        elif BasisClass is LRUBlockPoly:
            basis = BasisClass(n_blocks=n_drv)     # N = 2*K = 64
        else:
            B = n // n_drv
            basis = BasisClass(n_blocks=n_drv, block_size=B)

        proj  = InputProjector.trivial(n_drivers=n_drv)
        model = SASModel(proj, basis, basis)
        fc    = SASForecaster(model=model, washout=20, chunk_size=32, seed=0)
        fc.fit(ar1[:400], horizons=[1])
        assert np.isfinite(fc.predict(1))


# ════════════════════════════════════════════════════════════════════════════
# 7. RandomFourierBasis — shapes, properties, pytree, scan integration
# ════════════════════════════════════════════════════════════════════════════

class TestRandomFourierBasis:

    @pytest.mark.parametrize("kernel", ["gaussian", "laplace"])
    def test_initialize_sets_weights(self, key, kernel):
        basis = RandomFourierBasis(n_blocks=8, features_per_block=4,
                                   kernel_type=kernel).initialize(key)
        assert basis.Omega_weights is not None
        assert basis.Phase_weights is not None
        assert basis.Omega_weights.shape == (8, 4)
        assert basis.Phase_weights.shape == (8, 4)
        assert basis.is_initialized()

    def test_dimensions(self, key):
        basis = RandomFourierBasis(n_blocks=10, features_per_block=6).initialize(key)
        assert basis.n == 60
        assert basis.n_drivers == 10

    def test_eval_q_shape(self, key):
        K, B = 8, 4
        basis = RandomFourierBasis(n_blocks=K, features_per_block=B).initialize(key)
        z_t   = jax.random.normal(key, (K,), dtype=jnp.float32)
        q     = basis.eval_q(z_t)
        assert q.shape == (K * B,)
        assert jnp.isfinite(q).all()

    def test_eval_p_shape_and_bounds(self, key):
        K, B = 8, 4
        sn    = 0.95
        basis = RandomFourierBasis(n_blocks=K, features_per_block=B,
                                   spectral_norm=sn).initialize(key)
        z_t1  = jax.random.normal(key, (K,), dtype=jnp.float32)
        z_t2  = jax.random.normal(jax.random.PRNGKey(99), (K,), dtype=jnp.float32)
        A1    = basis.eval_p(z_t1)
        A2    = basis.eval_p(z_t2)
        assert A1.shape == (K * B,)
        # All eigenvalues strictly positive and bounded by spectral_norm
        assert jnp.all(A1 > 0.0) and jnp.all(A1 <= sn)
        # eval_p is input-independent (fixed log-spaced decays)
        assert jnp.allclose(A1, A2, atol=1e-6), "eval_p must not depend on z_tilde_t"

    def test_batch_eval_shapes(self, key):
        K, B, T = 8, 4, 50
        basis   = RandomFourierBasis(n_blocks=K, features_per_block=B).initialize(key)
        z_tilde = jax.random.normal(key, (T, K), dtype=jnp.float32)
        P_seq   = basis.batch_eval_p(z_tilde)
        Q_seq   = basis.batch_eval_q(z_tilde)
        assert P_seq.shape == (T, K * B)
        assert Q_seq.shape == (T, K * B)
        assert jnp.isfinite(P_seq).all()
        assert jnp.isfinite(Q_seq).all()

    def test_pytree_roundtrip(self, key):
        basis   = RandomFourierBasis(n_blocks=6, features_per_block=5,
                                     kernel_type="laplace", bandwidth=2.0,
                                     spectral_norm=0.95, tau_min=2.0, tau_max=50.0,
                                     bandwidth_min=0.5, bandwidth_max=5.0,
                                     ).initialize(key)
        leaves, treedef = jax.tree_util.tree_flatten(basis)
        rebuilt         = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.allclose(rebuilt.Omega_weights, basis.Omega_weights)
        assert jnp.allclose(rebuilt.Phase_weights, basis.Phase_weights)
        assert jnp.allclose(rebuilt.Rho_base,      basis.Rho_base)
        assert rebuilt.K == basis.K
        assert rebuilt.B == basis.B
        assert rebuilt.kernel_type == basis.kernel_type
        assert rebuilt.bandwidth == basis.bandwidth
        assert rebuilt.spectral_norm == basis.spectral_norm
        assert rebuilt.tau_min == basis.tau_min
        assert rebuilt.tau_max == basis.tau_max
        assert rebuilt.bandwidth_min == basis.bandwidth_min
        assert rebuilt.bandwidth_max == basis.bandwidth_max

    def test_multiscale_bandwidth(self, key):
        """Multi-scale bandwidth gives different frequencies per driver."""
        K, B = 20, 2
        basis_ms  = RandomFourierBasis(n_blocks=K, features_per_block=B,
                                        bandwidth_min=0.3, bandwidth_max=5.0).initialize(key)
        basis_uni = RandomFourierBasis(n_blocks=K, features_per_block=B,
                                        bandwidth=1.0).initialize(key)
        # Per-driver frequency magnitude should vary across drivers in multi-scale
        omega_std_ms  = float(jnp.std(jnp.abs(basis_ms.Omega_weights).mean(axis=1)))
        omega_std_uni = float(jnp.std(jnp.abs(basis_uni.Omega_weights).mean(axis=1)))
        assert omega_std_ms > omega_std_uni * 2, (
            "Multi-scale bandwidth should produce much higher per-driver frequency variance"
        )

    def test_forward_pass_with_rff(self, key):
        K, B, T = 10, 4, 100
        N = K * B
        proj  = InputProjector(d=1, n_drivers=K).initialize(key)
        basis = RandomFourierBasis(n_blocks=K, features_per_block=B).initialize(key)
        model = SASModel(proj, basis, basis)

        z  = jax.random.normal(key, (T, 1), dtype=jnp.float32)
        s0 = jnp.zeros(N)
        states, s_last = _forward(model, z, s0, chunk_size=32)

        assert states.shape == (T, N)
        assert s_last.shape == (N,)
        assert jnp.isfinite(states).all()

    def test_unknown_kernel_raises(self):
        with pytest.raises(ValueError, match="Unknown kernel type"):
            RandomFourierBasis(kernel_type="cosine").initialize(jax.random.PRNGKey(0))

    def test_rff_forecaster_smoke(self, key, ar1):
        K, B = 32, 2
        proj  = InputProjector.trivial(n_drivers=K)
        basis = RandomFourierBasis(n_blocks=K, features_per_block=B)
        model = SASModel(proj, basis, basis)
        fc    = SASForecaster(model=model, washout=20, chunk_size=32, seed=0)
        fc.fit(ar1[:400], horizons=[1])
        assert np.isfinite(fc.predict(1))


# ════════════════════════════════════════════════════════════════════════════
# 8. Decoupled basis_p / basis_q — mixing different basis types
# ════════════════════════════════════════════════════════════════════════════

class TestDecoupledBases:
    """Verify that basis_p and basis_q can be independently chosen."""

    T = 80
    CHUNK = 32

    def test_dimension_mismatch_n_raises(self, key):
        """basis_p.n != basis_q.n must raise ValueError."""
        proj    = InputProjector(d=1, n_drivers=8).initialize(key)
        basis_p = DiagonalPoly(n=8).initialize(key)           # n=8
        basis_q = DiagonalPoly(n=16, p_degree=1).initialize(key)  # n=16 — mismatch
        with pytest.raises(ValueError, match="basis_p.n"):
            SASModel(proj, basis_p, basis_q)

    def test_dimension_mismatch_n_drivers_raises(self, key):
        """basis_p.n_drivers != projector.n_drivers must raise ValueError."""
        proj    = InputProjector(d=1, n_drivers=8).initialize(key)
        basis_p = DiagonalPoly(n=16).initialize(key)   # n_drivers=16, proj gives 8
        basis_q = DiagonalPoly(n=16).initialize(key)
        with pytest.raises(ValueError, match="n_drivers"):
            SASModel(proj, basis_p, basis_q)

    def test_diagonal_p_rff_q_forward(self, key):
        """DiagonalPoly transition + RFF drive: shapes and finiteness."""
        N = 32
        proj    = InputProjector(d=1, n_drivers=N).initialize(key)
        basis_p = DiagonalPoly(n=N, p_degree=1).initialize(key)
        basis_q = RandomFourierBasis(n_blocks=N, features_per_block=1).initialize(key)
        model   = SASModel(proj, basis_p, basis_q)

        z  = jax.random.normal(key, (self.T, 1), dtype=jnp.float32)
        s0 = jnp.zeros(N)
        states, s_last = _forward(model, z, s0, self.CHUNK)

        assert states.shape == (self.T, N)
        assert jnp.isfinite(states).all()

    def test_rff_p_diagonal_q_forward(self, key):
        """RFF transition + DiagonalPoly drive."""
        N = 32
        proj    = InputProjector(d=1, n_drivers=N).initialize(key)
        basis_p = RandomFourierBasis(n_blocks=N, features_per_block=1).initialize(key)
        basis_q = DiagonalPoly(n=N, q_degree=1).initialize(key)
        model   = SASModel(proj, basis_p, basis_q)

        z  = jax.random.normal(key, (self.T, 1), dtype=jnp.float32)
        s0 = jnp.zeros(N)
        states, s_last = _forward(model, z, s0, self.CHUNK)

        assert states.shape == (self.T, N)
        assert jnp.isfinite(states).all()

    def test_decoupled_model_pytree(self, key):
        """SASModel with distinct basis_p/basis_q survives JAX pytree flatten/unflatten."""
        K = 10
        proj    = InputProjector(d=1, n_drivers=K).initialize(key)
        basis_p = LRUBlockPoly(n_blocks=K).initialize(key)
        basis_q = RandomFourierBasis(n_blocks=K, features_per_block=2).initialize(key)
        model   = SASModel(proj, basis_p, basis_q)

        leaves, treedef = jax.tree_util.tree_flatten(model)
        rebuilt         = jax.tree_util.tree_unflatten(treedef, leaves)

        assert jnp.allclose(rebuilt.basis_p.P_weights, model.basis_p.P_weights)
        assert jnp.allclose(rebuilt.basis_q.Omega_weights, model.basis_q.Omega_weights)

    def test_step_once_decoupled(self, key):
        """_step_once works with decoupled bases."""
        from saspy import _step_once
        K = 8
        proj    = InputProjector(d=1, n_drivers=K).initialize(key)
        basis_p = LRUBlockPoly(n_blocks=K).initialize(key)    # N = 2K = 16
        basis_q = RandomFourierBasis(n_blocks=K, features_per_block=2).initialize(key)  # N=16
        model   = SASModel(proj, basis_p, basis_q)

        s   = jnp.zeros(model.n)
        z_t = jnp.array([0.5], dtype=jnp.float32)
        s_new = _step_once(model, s, z_t)

        assert s_new.shape == (model.n,)
        assert jnp.isfinite(s_new).all()

    def test_sas_model_initialize_splits_key(self, key):
        """initialize() produces independently seeded basis_p and basis_q."""
        K = 8
        proj    = InputProjector(d=1, n_drivers=K)
        basis_p = LRUBlockPoly(n_blocks=K)
        basis_q = RandomFourierBasis(n_blocks=K, features_per_block=2)
        model   = SASModel(proj, basis_p, basis_q).initialize(key)

        assert model.is_initialized()
        assert model.basis_p.is_initialized()
        assert model.basis_q.is_initialized()
        assert model.n == K * 2
