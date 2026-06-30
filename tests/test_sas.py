"""Test suite for the SAS architecture (v0.3 — bounded features on [-1, 1]).

Covers:
  1. feature.py       — Cheb/Trig scalar+joint shapes and boundedness in [-1, 1]
  2. structures       — Diagonal / Block / Sparse P+Q: shapes, contractivity,
                        associativity, pytree round-trip
  3. model            — projection regimes (random W_in vs identity), domain
                        clip, dim-mismatch guards, encode/step
  4. engine           — forward shapes, chunk padding, JIT
  5. forecaster       — fit/predict/update/transform, [-1, 1] scaling, baselines
  6. decoupled P/Q    — mixing feature types and structures
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from saspy import (
    Cheb, Trig,
    DiagonalP, DiagonalQ,
    BlockP, BlockQ,
    SparseP, SparseQ,
    LowRankP, LowRankQ,
    SASModel, SASForecaster,
    _forward, _step_once,
)
from saspy.engine import _stream_scan, _fast_seq_scan


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


def _unit(key, shape):
    return jax.random.uniform(key, shape, minval=-1.0, maxval=1.0, dtype=jnp.float32)


# ════════════════════════════════════════════════════════════════════════════
# 1. feature.py
# ════════════════════════════════════════════════════════════════════════════

class TestFeatures:

    @pytest.mark.parametrize("spec", [
        Cheb(degree=3), Cheb(degree=2, cross_input=False),
        Trig(degree=3), Trig(degree=2, kernel="laplace", density_omega=0.5),
        Trig(degree=2, bandwidth_min=0.3, bandwidth_max=5.0),
    ])
    def test_scalar_shapes_and_bounds(self, key, spec):
        z = _unit(key, (7, 5))                       # (T, M)
        frozen = spec.init_scalar(key, 5)
        phi = spec.scalar_features(z, frozen)
        assert phi.shape == (7, 5, spec.n_scalar())
        assert jnp.abs(phi).max() <= 1.0 + 1e-5
        # feature 0 is the constant 1
        assert jnp.allclose(phi[..., 0], 1.0)

    @pytest.mark.parametrize("spec", [
        Cheb(degree=3), Cheb(degree=2, cross_input=False),
        Trig(degree=3), Trig(degree=2, kernel="laplace"),
    ])
    def test_joint_shapes_and_bounds(self, key, spec):
        K = 4
        z = _unit(key, (7, K))
        frozen = spec.init_joint(key, K)
        phi = spec.joint_features(z, frozen, K)
        assert phi.shape == (7, spec.n_joint(K))
        assert jnp.abs(phi).max() <= 1.0 + 1e-5
        assert jnp.allclose(phi[:, 0], 1.0)

    def test_trig_unknown_kernel_raises(self, key):
        with pytest.raises(ValueError, match="Unknown kernel"):
            Trig(kernel="cosine").init_joint(key, 3)


# ════════════════════════════════════════════════════════════════════════════
# 2. structures: shapes / contractivity / associativity / pytree
# ════════════════════════════════════════════════════════════════════════════

def _assoc(basis, mk_pair, key):
    k1, k2, k3 = jax.random.split(key, 3)
    a, b, c = mk_pair(k1), mk_pair(k2), mk_pair(k3)
    L = basis.combine(basis.combine(a, b), c)
    R = basis.combine(a, basis.combine(b, c))
    assert jnp.allclose(L[0], R[0], atol=1e-4)
    assert jnp.allclose(L[1], R[1], atol=1e-4)


def _roundtrip(basis):
    leaves, td = jax.tree_util.tree_flatten(basis)
    rebuilt = jax.tree_util.tree_unflatten(td, leaves)
    return rebuilt


class TestDiagonal:
    T, N = 30, 16

    @pytest.mark.parametrize("feat", [Cheb(2), Trig(2), Trig(1, kernel="laplace")])
    def test_p_contractive(self, key, feat):
        p = DiagonalP(self.N, feature=feat).initialize(key)
        A = p.batch_eval_p(_unit(key, (self.T, self.N)))
        assert A.shape == (self.T, self.N)
        assert jnp.abs(A).max() < 1.0
        assert p.n == self.N and p.n_drivers == self.N

    def test_p_associative_and_pytree(self, key):
        p = DiagonalP(self.N, feature=Cheb(2)).initialize(key)
        mk = lambda k: (_unit(k, (self.N,)) * 0.8,
                        jax.random.normal(jax.random.split(k)[0], (self.N,)))
        _assoc(p, mk, key)
        assert jnp.allclose(_roundtrip(p).P_weights, p.P_weights)

    def test_q_shapes(self, key):
        q = DiagonalQ(self.N, feature=Trig(2)).initialize(key)
        out = q.batch_eval_q(_unit(key, (self.T, self.N)))
        assert out.shape == (self.T, self.N)
        assert jnp.allclose(_roundtrip(q).Q_weights, q.Q_weights)


class TestBlock:
    T, K = 30, 8

    @pytest.mark.parametrize("mode,B", [("rotation", 2), ("orthogonal", 4)])
    def test_p_contractive(self, key, mode, B):
        p = BlockP(self.K, block_size=B, feature=Cheb(2), init_mode=mode).initialize(key)
        A = p.batch_eval_p(_unit(key, (self.T, self.K)))
        assert A.shape == (self.T, self.K, B, B)
        assert p.n == self.K * B
        rad = jnp.max(jnp.abs(jnp.linalg.eigvals(A.reshape(-1, B, B))))
        assert rad < 1.0 + 1e-4

    def test_rotation_requires_b2(self, key):
        with pytest.raises(ValueError, match="block_size=2"):
            BlockP(self.K, block_size=4, init_mode="rotation").initialize(key)

    def test_p_associative_and_pytree(self, key):
        B = 2
        p = BlockP(self.K, block_size=B, feature=Cheb(1)).initialize(key)
        mk = lambda k: (jax.random.normal(k, (self.K, B, B)) * 0.1,
                        jax.random.normal(jax.random.split(k)[0], (self.K * B,)))
        _assoc(p, mk, key)
        assert jnp.allclose(_roundtrip(p).P_weights, p.P_weights)

    def test_q_shapes(self, key):
        B = 2
        q = BlockQ(self.K, block_size=B, feature=Cheb(2)).initialize(key)
        out = q.batch_eval_q(_unit(key, (self.T, self.K)))
        assert out.shape == (self.T, self.K * B)


class TestSparse:
    T, N, K = 30, 24, 4

    @pytest.mark.parametrize("feat", [
        Cheb(2), Cheb(2, cross_input=False), Trig(2, bandwidth=2.0),
    ])
    def test_p_contractive(self, key, feat):
        p = SparseP(self.N, self.K, feature=feat, spectral_norm=0.9).initialize(key)
        A = p.batch_eval_p(_unit(key, (self.T, self.K)))
        assert A.shape == (self.T, self.N, self.N)
        rad = float(jnp.max(jnp.abs(jnp.linalg.eigvals(A))))
        assert rad < 1.0 + 1e-3

    def test_p_associative_and_pytree(self, key):
        p = SparseP(self.N, self.K, feature=Cheb(2)).initialize(key)
        mk = lambda k: (jax.random.normal(k, (self.N, self.N)) * 0.1,
                        jax.random.normal(jax.random.split(k)[0], (self.N,)))
        _assoc(p, mk, key)
        # sequential mode stores P_weights as a BCOO — compare densified.
        _dense = lambda P: P.todense() if hasattr(P, "todense") else P
        assert jnp.allclose(_dense(_roundtrip(p).P_weights), _dense(p.P_weights))

    def test_bcoo_matvec_matches_dense(self, key):
        """Sparse matvec_p (sequential) == dense apply(eval_p) (parallel)."""
        z = _unit(key, (self.K,))
        s = jax.random.normal(jax.random.split(key)[0], (self.N,))
        for feat in (Cheb(2), Trig(2, bandwidth=2.0)):
            seq = SparseP(self.N, self.K, feature=feat, training_mode="sequential").initialize(key)
            par = SparseP(self.N, self.K, feature=feat, training_mode="parallel").initialize(key)
            got  = seq.matvec_p(z, s)
            want = par.apply(par.eval_p(z), s)
            assert got.shape == (self.N,)
            assert jnp.allclose(got, want, atol=1e-4)

    def test_a_density_sparsifies(self, key):
        p = SparseP(self.N, self.K, feature=Cheb(2), A_density=0.1).initialize(key)
        A = p.eval_p(jnp.zeros(self.K))      # phi=[1,0,..] -> A = M_0
        frac = float((jnp.abs(A) > 1e-9).mean())
        assert frac < 0.35      # sparse + dense diagonal

    def test_q_shapes(self, key):
        q = SparseQ(self.N, self.K, feature=Trig(2)).initialize(key)
        out = q.batch_eval_q(_unit(key, (self.T, self.K)))
        assert out.shape == (self.T, self.N)


# ════════════════════════════════════════════════════════════════════════════
# 2b. LowRank (CP factorisation): A_t = M_0 + B·U diag(α) Vᵀ
# ════════════════════════════════════════════════════════════════════════════

class TestLowRank:
    T, N, K = 30, 24, 4

    @pytest.mark.parametrize("feat", [Cheb(3), Trig(2, bandwidth=1.0)])
    @pytest.mark.parametrize("mode,nd", [("map", 4), ("driver", 24)])
    def test_p_contractive(self, key, feat, mode, nd):
        # driver requires K==rank; map keeps them independent.
        p = LowRankP(self.N, nd, feature=feat, rank=nd, alpha_mode=mode,
                     spectral_norm=0.9).initialize(key)
        A = p.batch_eval_p(_unit(key, (self.T, nd)))
        assert A.shape == (self.T, self.N, self.N)
        # Rigorous budget guarantees (the soft ρ(A_t)<1 leaks at small N because
        # M_0 is scaled to spectral *radius* sn while ‖M_0‖₂ ≥ sn):
        #   (i) modulation spectral norm ≤ B,  (ii) ρ(A_t) ≤ ‖M_0‖₂ + B,
        #   (iii) backbone radius ρ(M_0) ≤ sn.
        B   = p.budget
        m0n = float(jnp.linalg.norm(p.M0, 2))
        mod = jnp.linalg.norm(A - p.M0[None], 2, axis=(1, 2))
        assert float(mod.max()) <= B + 1e-3
        rad = float(jnp.max(jnp.abs(jnp.linalg.eigvals(A))))
        assert rad <= m0n + B + 1e-3
        assert float(jnp.max(jnp.abs(jnp.linalg.eigvals(p.M0)))) <= 0.9 + 1e-3

    def test_driver_requires_kr(self):
        with pytest.raises(ValueError):
            LowRankP(self.N, 4, rank=8, alpha_mode="driver")
        with pytest.raises(ValueError):
            LowRankQ(self.N, 4, rank=8, alpha_mode="driver")

    def test_matvec_matches_dense(self, key):
        """Sequential matvec_p == dense eval_p @ s, for every mode/feature."""
        s = jax.random.normal(jax.random.split(key)[0], (self.N,))
        cases = [
            LowRankP(self.N, self.K, feature=Cheb(2), rank=8, alpha_mode="map"),
            LowRankP(self.N, self.N, feature=Cheb(3), rank=self.N, alpha_mode="driver"),
            LowRankP(self.N, self.N, feature=Trig(2), rank=self.N, alpha_mode="driver"),
        ]
        for spec in cases:
            p = spec.initialize(key)
            z = _unit(key, (p.n_drivers,))
            got, want = p.matvec_p(z, s), p.apply(p.eval_p(z), s)
            assert got.shape == (self.N,)
            assert jnp.allclose(got, want, atol=1e-4)

    def test_bcoo_M0_matches_dense(self, key):
        """sparse_M0 (BCOO, sequential) gives the same matvec as dense M_0."""
        s = jax.random.normal(jax.random.split(key)[0], (self.N,))
        z = _unit(key, (self.N,))
        dense  = LowRankP(self.N, self.N, feature=Cheb(3), rank=self.N,
                          alpha_mode="driver", sparse_M0=False).initialize(key)
        sparse = LowRankP(self.N, self.N, feature=Cheb(3), rank=self.N,
                          alpha_mode="driver", sparse_M0=True).initialize(key)
        assert hasattr(sparse.M0, "todense")            # stored as BCOO
        assert jnp.allclose(dense.matvec_p(z, s), sparse.matvec_p(z, s), atol=1e-4)

    def test_block_ortho_factors(self, key):
        """factor_blocks → sparse AND orthonormal U,V (exact spectral property)."""
        N, R = 60, 24
        for G in (4, 12):
            p = LowRankP(N, R, feature=Cheb(3), rank=R, alpha_mode="driver",
                         factor_blocks=G).initialize(key)
            assert jnp.allclose(p.U.T @ p.U, jnp.eye(R), atol=1e-4)   # orthonormal
            assert float((p.U != 0).mean()) < 1.0 / G + 0.05          # sparse ~1/G
            A = p.batch_eval_p(_unit(key, (self.T, R)))               # contractive
            assert float(jnp.max(jnp.abs(jnp.linalg.eigvals(A)))) <= 1.0 + 0.05

    def test_overcomplete_contractive(self, key):
        """R > N (overcomplete, spectral-normed factors) stays contractive."""
        p = LowRankP(self.N, 8, feature=Cheb(2), rank=3 * self.N,
                     alpha_mode="map", spectral_norm=0.9).initialize(key)
        A = p.batch_eval_p(_unit(key, (self.T, 8)))
        assert float(jnp.max(jnp.abs(jnp.linalg.eigvals(A)))) < 1.0 + 1e-3

    def test_no_backbone_contractive_with_leak(self, key):
        """backbone=False: M_0 is None, effective (1-leak)I + leak·A_t contractive."""
        p = LowRankP(self.N, self.N, feature=Cheb(3), rank=self.N,
                     alpha_mode="driver", backbone=False).initialize(key)
        assert p.M0 is None and p.is_initialized()
        A = p.eval_p(_unit(key, (self.N,)))
        for leak in (0.3, 1.0):
            Aeff = (1 - leak) * jnp.eye(self.N) + leak * A
            assert float(jnp.max(jnp.abs(jnp.linalg.eigvals(Aeff)))) < 1.0 + 1e-3

    def test_p_associative_and_pytree(self, key):
        p = LowRankP(self.N, self.N, feature=Cheb(3), rank=self.N,
                     alpha_mode="driver").initialize(key)
        mk = lambda k: (jax.random.normal(k, (self.N, self.N)) * 0.1,
                        jax.random.normal(jax.random.split(k)[0], (self.N,)))
        _assoc(p, mk, key)
        rt = _roundtrip(p)
        assert rt.alpha_mode == "driver" and rt.rank == self.N
        assert jnp.allclose(rt.U, p.U)

    @pytest.mark.parametrize("density_G", [None, 0.3])
    def test_q_shapes_and_drive(self, key, density_G):
        q = LowRankQ(self.N, self.N, feature=Cheb(3), rank=self.N,
                     alpha_mode="driver", density_G=density_G).initialize(key)
        out = q.batch_eval_q(_unit(key, (self.T, self.N)))
        assert out.shape == (self.T, self.N)
        # single-step matches batch
        z = _unit(key, (self.N,))
        assert jnp.allclose(q.eval_q(z), q.batch_eval_q(z[None])[0], atol=1e-4)

    def test_forecaster_end_to_end(self, key, ar1):
        """Driver-mode + sparse-G LowRank fits and forecasts a simple AR(1)."""
        N = 32
        model = SASModel(
            LowRankP(N, N, feature=Cheb(3), rank=N, alpha_mode="driver"),
            LowRankQ(N, N, feature=Cheb(3), rank=N, alpha_mode="driver", density_G=0.2),
            d=1)
        fc = SASForecaster(model=model, washout=20, chunk_size=8, seed=0,
                           scale_input=True, mode="autoreg").fit(ar1[:, None], horizons=[1])
        pred = np.atleast_1d(fc.predict(1))
        assert pred.shape == (1,) and np.isfinite(pred).all()

    @pytest.mark.parametrize("model_fn,d", [
        # LowRank driver (backbone) — the stacked [M_0;Vᵀ] fast path
        (lambda N: SASModel(LowRankP(N, 24, feature=Cheb(3), rank=24, alpha_mode="driver"),
                            LowRankQ(N, 24, feature=Cheb(3), rank=24, alpha_mode="driver",
                                     density_G=0.1), d=5, leak=0.25), 5),
        # LowRank no-backbone (M_0=None fallback path)
        (lambda N: SASModel(LowRankP(N, 24, feature=Cheb(3), rank=24, alpha_mode="driver",
                                     backbone=False),
                            LowRankQ(N, 24, feature=Cheb(3), rank=24, alpha_mode="driver"),
                            d=5, leak=0.4), 5),
        # LowRank BCOO M_0 (sparse_M0 fallback path)
        (lambda N: SASModel(LowRankP(N, 24, feature=Cheb(2), rank=24, alpha_mode="driver",
                                     sparse_M0=True),
                            LowRankQ(N, 24, feature=Cheb(2), rank=24, alpha_mode="driver"),
                            d=5), 5),
        # Sparse Cheb (joint) and Trig (per-driver)
        (lambda N: SASModel(SparseP(N, 3, feature=Cheb(3)), SparseQ(N, 3, feature=Cheb(3)),
                            leak=0.5), 3),
        (lambda N: SASModel(SparseP(N, 3, feature=Trig(2, bandwidth=1.0)),
                            SparseQ(N, 3, feature=Trig(2, bandwidth=1.0)), leak=0.5), 3),
    ])
    def test_fast_seq_scan_equals_stream_scan(self, key, model_fn, d):
        """The fast teacher-forced scan must equal the per-step streaming scan."""
        N = 28
        m = model_fn(N).initialize(key)
        z = _unit(key, (120, d))
        s0 = jnp.zeros(m.n)
        a, a_last = _stream_scan(m, s0, z)
        b, b_last = _fast_seq_scan(m, s0, z)
        assert jnp.allclose(a, b, atol=1e-4)
        assert jnp.allclose(a_last, b_last, atol=1e-4)


# ════════════════════════════════════════════════════════════════════════════
# 3. model: projection + domain clip + guards
# ════════════════════════════════════════════════════════════════════════════

class TestModel:

    def test_identity_projection_passthrough(self, key):
        N, K = 20, 4
        m = SASModel(SparseP(N, K), SparseQ(N, K)).initialize(key)
        assert m.W_in is None and m.input_dim == K
        z = _unit(key, (5, K))
        assert jnp.allclose(m.project(z), jnp.clip(z, -1, 1))

    def test_random_projection_shape(self, key):
        N = 32
        m = SASModel(DiagonalP(N), DiagonalQ(N), d=3).initialize(key)
        assert m.W_in.shape == (3, N) and m.input_dim == 3

    def test_projection_clips_to_unit(self, key):
        # d>1 projection can exceed [-1,1]; clip must bound it
        N = 64
        m = SASModel(DiagonalP(N), DiagonalQ(N), d=8).initialize(key)
        z = _unit(key, (50, 8))
        zt = m.project(z)
        assert jnp.abs(zt).max() <= 1.0 + 1e-6

    def test_d1_projection_unit_gain(self, key):
        # d=1 gains are ±1 → z_tilde = ±z stays in [-1,1] and uses the full
        # range; per-unit diversity comes from the basis, not the input scale.
        N = 128
        m = SASModel(DiagonalP(N), DiagonalQ(N), d=1).initialize(key)
        mags = np.abs(np.asarray(m.W_in[0]))
        assert np.allclose(mags, 1.0, atol=1e-5)
        z = _unit(key, (10, 1))
        assert jnp.abs(m.project(z)).max() <= 1.0 + 1e-6   # bounded, no clip needed

    def test_d_gt1_l1_normalized_in_range(self, key):
        # L1-normalised columns keep z_tilde in [-1,1] *before* the safety clip
        N = 64
        m = SASModel(DiagonalP(N), DiagonalQ(N), d=8).initialize(key)
        l1 = np.abs(np.asarray(m.W_in)).sum(axis=0)
        assert np.allclose(l1, 1.0, atol=1e-5)
        z = _unit(key, (50, 8))
        raw = z @ m.W_in            # pre-clip projection
        assert jnp.abs(raw).max() <= 1.0 + 1e-5

    def test_dim_mismatch_n_raises(self, key):
        with pytest.raises(ValueError, match="basis_p.n"):
            SASModel(DiagonalP(8), DiagonalQ(16))

    def test_dim_mismatch_drivers_raises(self, key):
        with pytest.raises(ValueError, match="n_drivers"):
            SASModel(SparseP(16, 4), SparseQ(16, 8))


# ════════════════════════════════════════════════════════════════════════════
# 4. engine: forward / padding / JIT
# ════════════════════════════════════════════════════════════════════════════

class TestEngine:
    T = 120

    def _model(self, key, kind):
        if kind == "diag":
            return SASModel(DiagonalP(48, feature=Cheb(1)),
                            DiagonalQ(48, feature=Cheb(1)), d=1).initialize(key), 1, 48
        if kind == "block":
            return SASModel(BlockP(24, 2, init_mode="rotation"),
                            BlockQ(24, 2), d=1).initialize(key), 1, 48
        return SASModel(SparseP(40, 3, feature=Cheb(2)),
                        SparseQ(40, 3, feature=Cheb(2))).initialize(key), 3, 40

    @pytest.mark.parametrize("kind", ["diag", "block", "sparse"])
    def test_forward_shape_finite(self, key, kind):
        m, d, N = self._model(key, kind)
        z = _unit(key, (self.T, d))
        states, s_last = _forward(m, z, jnp.zeros(N), 32)
        assert states.shape == (self.T, N) and s_last.shape == (N,)
        assert jnp.isfinite(states).all() and jnp.isfinite(s_last).all()

    @pytest.mark.parametrize("T", [1, 33, 64, 65, 128])
    def test_padding_correctness(self, key, T):
        m, _, N = self._model(key, "diag")
        z = _unit(key, (T, 1))
        states, _ = _forward(m, z, jnp.zeros(N), 16)
        assert states.shape == (T, N)

    def test_jit_and_step(self, key):
        m, d, N = self._model(key, "sparse")
        z = _unit(key, (60, d))
        f = jax.jit(lambda mm, zz, s0: _forward(mm, zz, s0, 32))
        states, s_last = f(m, z, jnp.zeros(N))
        assert states.shape == (60, N)
        s_new = _step_once(m, jnp.zeros(N), z[0])
        assert s_new.shape == (N,) and jnp.isfinite(s_new).all()

    def test_pytree_roundtrip_reproduces(self, key):
        m, d, N = self._model(key, "block")
        z = _unit(key, (50, d))
        st1, _ = _forward(m, z, jnp.zeros(N), 32)
        leaves, td = jax.tree_util.tree_flatten(m)
        rebuilt = jax.tree_util.tree_unflatten(td, leaves)
        st2, _ = _forward(rebuilt, z, jnp.zeros(N), 32)
        assert jnp.allclose(st1, st2)


# ════════════════════════════════════════════════════════════════════════════
# 5. forecaster
# ════════════════════════════════════════════════════════════════════════════

class TestForecaster:

    def _fc(self, n=64, washout=20, **kw):
        model = SASModel(DiagonalP(n, feature=Cheb(1)),
                         DiagonalQ(n, feature=Cheb(1)), d=1)
        return SASForecaster(model=model, washout=washout, chunk_size=32, seed=0, **kw)

    def test_fit_predict_update(self, ar1):
        fc = self._fc()
        fc.fit(ar1[:400], horizons=[1, 5])
        p1, p5 = fc.predict(1), fc.predict(5)
        assert np.isfinite(p1) and np.isfinite(p5)
        fc.update(float(ar1[400]))
        assert fc.predict(1) != p1

    def test_transform_shape(self, ar1):
        fc = self._fc(n=32)
        fc.fit(ar1[:400], horizons=[1])
        states = fc.transform(ar1[400:450])
        assert states.shape == (50, 32) and np.isfinite(states).all()

    def test_minmax_scaling_to_unit(self, ar1):
        # with scale_input=True the min-max transform must map train data to [-1,1]
        fc = self._fc(scale_input=True)
        fc.fit(ar1[:400], horizons=[1])
        center, half = fc._mu, fc._sigma
        scaled = (ar1[:400] - center) / half
        assert scaled.min() >= -1.0 - 1e-6 and scaled.max() <= 1.0 + 1e-6

    def test_default_no_scaling(self, ar1):
        # default scale_input=False ⇒ identity scaler (data assumed in [-1,1])
        fc = self._fc()
        fc.fit(ar1[:400], horizons=[1])
        assert fc._mu == 0.0 and fc._sigma == 1.0

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="must be fit"):
            self._fc().predict(1)

    def test_unknown_horizon_raises(self, ar1):
        fc = self._fc()
        fc.fit(ar1[:400], horizons=[1, 5])
        with pytest.raises(KeyError):
            fc.predict(99)

    def test_beats_mean_baseline(self, ar1):
        train, test = ar1[:400], ar1[400:]
        fc = self._fc(n=64, washout=20)
        fc.fit(train, horizons=[1])
        preds = []
        for t in range(len(test)):
            preds.append(fc.predict(1))
            fc.update(test[t])
        mse = float(np.mean((np.asarray(preds) - test) ** 2))
        assert mse < 0.8 * float(np.var(test))

    @pytest.mark.parametrize("model_fn", [
        lambda n: SASModel(DiagonalP(n, feature=Cheb(1)), DiagonalQ(n, feature=Cheb(1)), d=1),
        lambda n: SASModel(BlockP(n // 2, 2, init_mode="rotation"), BlockQ(n // 2, 2), d=1),
        lambda n: SASModel(SparseP(n, 1, feature=Cheb(2)), SparseQ(n, 1, feature=Cheb(2))),
    ])
    def test_all_structures_smoke(self, ar1, model_fn):
        fc = SASForecaster(model=model_fn(64), washout=20, chunk_size=32, seed=0)
        fc.fit(ar1[:400], horizons=[1])
        assert np.isfinite(fc.predict(1))

    def test_clip_output_keeps_autoreg_in_domain(self, ar1):
        # autoreg multi-step with clip_output must stay within the input range
        model = SASModel(DiagonalP(64, feature=Cheb(1)),
                         DiagonalQ(64, feature=Cheb(1)), d=1)
        fc = SASForecaster(model=model, washout=20, chunk_size=32, seed=0,
                           mode='autoreg', clip_output=True)
        fc.fit(ar1[:400], horizons=[1])
        lo, hi = ar1[:400].min(), ar1[:400].max()
        for h in range(1, 30):
            p = fc.predict(h)
            assert lo - 1e-6 <= p <= hi + 1e-6, (h, p)

    def test_multivariate_context(self, key):
        # 3-channel context driving a d=3 diagonal model
        T = 400
        rng = np.random.default_rng(0)
        ctx = np.clip(np.cumsum(rng.normal(size=(T, 3)) * 0.1, axis=0), -5, 5)
        target = ctx[:, 0] * 0.5 + ctx[:, 1] * 0.3
        model = SASModel(DiagonalP(48), DiagonalQ(48), d=3)
        fc = SASForecaster(model=model, washout=20, chunk_size=32, seed=0)
        fc.fit(target, horizons=[1], context=ctx)
        assert np.isfinite(fc.predict(1))


# ════════════════════════════════════════════════════════════════════════════
# 6. decoupled P / Q — feature & structure mixing
# ════════════════════════════════════════════════════════════════════════════

class TestDecoupled:
    T, N = 80, 32

    @pytest.mark.parametrize("featp,featq", [
        (Cheb(1), Trig(2)), (Trig(2), Cheb(2)), (Trig(1, kernel="laplace"), Cheb(1)),
    ])
    def test_diagonal_mixed_features(self, key, featp, featq):
        m = SASModel(DiagonalP(self.N, feature=featp),
                     DiagonalQ(self.N, feature=featq), d=1).initialize(key)
        states, _ = _forward(m, _unit(key, (self.T, 1)), jnp.zeros(self.N), 32)
        assert states.shape == (self.T, self.N) and jnp.isfinite(states).all()

    def test_sparse_chebP_trigQ(self, key):
        N, K = 32, 4
        m = SASModel(SparseP(N, K, feature=Cheb(2)),
                     SparseQ(N, K, feature=Trig(2))).initialize(key)
        states, _ = _forward(m, _unit(key, (self.T, K)), jnp.zeros(N), 32)
        assert states.shape == (self.T, N) and jnp.isfinite(states).all()

    def test_decoupled_pytree(self, key):
        N = 20
        m = SASModel(DiagonalP(N, feature=Cheb(2)),
                     DiagonalQ(N, feature=Trig(2)), d=1).initialize(key)
        leaves, td = jax.tree_util.tree_flatten(m)
        rebuilt = jax.tree_util.tree_unflatten(td, leaves)
        assert jnp.allclose(rebuilt.basis_p.P_weights, m.basis_p.P_weights)
        assert jnp.allclose(rebuilt.basis_q.Q_weights, m.basis_q.Q_weights)
