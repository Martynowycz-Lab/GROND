"""Direct-plus-target beam support.

The support score is a selection heuristic, not a guaranteed propagation-error
bound. Increasing a cap alone does not recover beams outside the source pool."""

from __future__ import annotations

import numpy as np


def ranked_candidates(scores, candidates, count):
    """Rank by descending score, break ties by pool index, then apply the cap.

    Selecting a partial partition first can discard earlier equally scored
    beams before the tie breaker sees them.
    """
    order = np.lexsort((candidates, -scores[candidates]))
    return candidates[order[:count]]


class BasisMixin:
    """One cohesive part of XDSBlochPort; initialized by the public facade."""

    def select_basis(self, target_hkl, omega_bragg, row_idx=None):
        """Return one row's beam basis as an (n, 3) int array: direct beam, target, then the strongest top-up beams.

        Top-up beams come from the pool with |S_g| <= max_support_sg at ``omega_bragg``. They are ranked by
        |U_g|^2 / (|S_g| + 1e-5), ties by pool order, until n_basis beams are chosen. With ``row_idx`` the pool U_g
        carry that row's dose attenuation.
        """

        sg_pool = self.sg_for_hkls(self.pool_hkl, omega_bragg)
        if row_idx is None:
            pool_ug = self.pool_ug
        else:
            pool_ug = self.pool_ug * self.dose_attenuation_for_inv_d2(self.pool_inv_d2, row_idx)
        score = np.abs(pool_ug) ** 2 / (np.abs(sg_pool) + 1e-5)
        ok = np.abs(sg_pool) <= float(self.cfg.max_support_sg)
        basis = [(0, 0, 0), tuple(int(v) for v in target_hkl)]
        seen = set(basis)
        candidates = np.flatnonzero(ok)
        needed = max(0, int(self.cfg.n_basis) - len(basis))
        if needed == 0:
            return np.asarray(basis, dtype=int)
        take = min(len(candidates), needed + 1)
        candidates = ranked_candidates(score, candidates, take)
        for pool_index in candidates:
            h = tuple(int(v) for v in self.pool_hkl[pool_index])
            if h in seen:
                continue
            basis.append(h)
            seen.add(h)
            if len(basis) >= int(self.cfg.n_basis):
                break
        return np.asarray(basis, dtype=int)
