from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# ---------------------------------------------------------------------------
# 0. The W implementation — library, never silently stubbed. (unchanged)
# ---------------------------------------------------------------------------
try:
    from rsuq.core.signals import credal_width as _credal_width   # the real one
    RSUQ_LIB = True
except Exception:                                                  # pragma: no cover
    RSUQ_LIB = False

    def _credal_width(m: np.ndarray, sizes: np.ndarray) -> np.ndarray:
        sizes = np.asarray(sizes, dtype=float)
        return (np.asarray(m, dtype=float) * (1.0 - 1.0 / sizes)).sum(axis=-1)


def credal_width(m: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    return _credal_width(m, sizes)


if not RSUQ_LIB:
    _W = "!" * 78
    print(f"\n{_W}\n!! RSUQ LIBRARY NOT FOUND — running STUB credal_width.\n"
          f"!! Results from this run are NOT paper-grade. Install rsuq or\n"
          f"!! re-run inside the rsuq repo before reporting numbers.\n{_W}\n")


# ---------------------------------------------------------------------------
# 1. The data contract for one cell.
# ---------------------------------------------------------------------------
@dataclass
class Cell:
    h: np.ndarray
    conf: np.ndarray
    err: np.ndarray
    name: str
    has_raw: bool = False
    n_pairs: int = 0


# ---------------------------------------------------------------------------
# 2. THE FOUR FUNCTIONS. Wired to run_repspace_deferral.py's real, working
#    implementations instead of raising NotImplementedError.
# ---------------------------------------------------------------------------
def load_cell(cache_path: str) -> Cell:
    from run_repspace_deferral import load_cache

    cells = load_cache(cache_path)
    (model_key, dataset_key), c = next(iter(cells.items()))

    h_pos, h_neg = c["h_pos"], c["h_neg"]
    N = h_pos.shape[0]
    h = np.vstack([h_pos, h_neg])
    err = np.r_[np.zeros(N), np.ones(N)].astype(int)   # 1 = h_wrong

    if "lp_pos" not in c or "lp_neg" not in c:
        raise ValueError(
            "cache has no logprob foil (lp_correct/lp_wrong) — conf would "
            "be a probe, not the real confidence foil. Refusing to proceed "
            "silently (matches run_tier01.py's own assert)."
        )
    conf = -np.r_[c["lp_pos"], c["lp_neg"]]   # FIX 1: higher = LESS confident

    return Cell(h=h, conf=conf, err=err,
                name=f"{model_key}/{dataset_key}",
                has_raw=False, n_pairs=int(N))


def build_frame(h: np.ndarray, K: int, seed: int):
    from run_repspace_deferral import build_repspace_frame
    return build_repspace_frame(h, K, seed)


def mass_head(h: np.ndarray, frame, mode: str, seed: int, tau: float = 1.0,
              y: np.ndarray | None = None,
              train_idx: np.ndarray | None = None) -> np.ndarray:
    from run_repspace_deferral import mass_geometric, train_mass_head

    if mode == "geometric":
        return mass_geometric(h, frame, tau=tau)

    if mode not in ("trained_unsupervised", "trained_supervised"):
        raise ValueError(f"unknown mode: {mode!r}")

    if train_idx is None:
        raise ValueError(
            f"mode={mode!r} needs train_idx (fit on a subset, score on all "
            f"of h) — e.g. from screen.single_split_indices(len(h), seed). "
            f"Not inferable from this call alone; pass it explicitly."
        )

    real_mode = "unsupervised" if mode == "trained_unsupervised" else "supervised"
    y_tr = None
    if real_mode == "supervised":
        if y is None:
            raise ValueError(
                "mode='trained_supervised' needs y (the (2N,) error labels) "
                "to fit train_mass_head — this is the one thing the original "
                "frozen signature had no slot for. Pass y=cell.err."
            )
        y_tr = y[train_idx]

    head = train_mass_head(h[train_idx], frame, real_mode, y_tr=y_tr, seed=seed)
    return head(h)


def frame_sizes(frame) -> np.ndarray:
    return np.asarray(frame["sizes"], dtype=float)
