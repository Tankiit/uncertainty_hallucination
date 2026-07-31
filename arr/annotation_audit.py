"""Audit the annotation target before any further panel work.

Motivation. analyse_hf.annotator_entropy_from_index() computes plug-in
Shannon entropy with no bias correction, and the three ARR datasets have
three different label-set sizes (3, 2, 5 by the observed entropy ceilings
of 1.098, 0.693, 1.584 in RESULTS_ALL.md section 6). Three consequences,
all of which need to be measured rather than assumed:

  1. Cross-dataset comparisons of rho are not on a common footing.
  2. The binary dataset's target is FOLDED (see fold_check.py).
  3. The counts-vs-labels branch in annotator_entropy_from_index can
     silently misparse. See check_label_column below.

Run this first. Nothing else in this package is trustworthy until it passes.
"""
import numpy as np

# TODO: fill from the real index.parquet. Do NOT assume -- the whole point
#       of this file is that these were never checked.
EXPECTED = {
    "chaosnli":       dict(support=3, n_annot=100),   # Nie et al. 2020: 100/item
    "chaosnli_alpha": dict(support=2, n_annot=100),   # same release
    "pavlick_nli":    dict(support=5, n_annot=None),  # TODO: unknown, blocks Table 1
}


def check_label_column(raw_values, dataset):
    """Is this column per-annotator labels, or already counts/probabilities?

    analyse_hf's heuristic tests `max(p) == int(max(p))` and `sum(p) != 1`,
    which CANNOT distinguish:

        [0,0,1,2,0,...]   (100 per-annotator labels, support 3)
        [78, 15, 7]       (counts over 3 labels)

    Fed the second, it bincounts them as labels and returns garbage entropy
    without raising.

    Discriminator: a label vector has size >> max(value); a count vector has
    size << max(value). Assert, do not infer.
    """
    sup = EXPECTED[dataset]["support"]
    report = dict(dataset=dataset, n_items=len(raw_values))
    sizes = np.array([len(np.ravel(v)) for v in raw_values])
    report["size_mode"] = int(np.bincount(sizes).argmax()) if sizes.size else 0
    report["size_varies"] = bool(sizes.min() != sizes.max())
    report["looks_like"] = ("counts/probs" if report["size_mode"] == sup
                            else "per-annotator labels")
    # TODO: raise if this disagrees with what annotator_entropy_from_index
    #       would have taken. That is the silent-failure case.
    return report


def audit(idx, dataset):
    """Return the row that belongs in the paper's Table 1.

    Must produce: n items, support size, annotator-count distribution
    (not just the mode), mean/max entropy, fraction at exactly zero.

    TODO: the annotator-count DISTRIBUTION is the load-bearing bit. If the
          count varies within a dataset, the plug-in bias varies per item
          and becomes an item-level confound in the target itself, not just
          a cross-dataset nuisance. That changes what Sec. Setup has to say
          from a sentence to a paragraph.
    """
    raise NotImplementedError


def entropy_variants(counts, support):
    """Target variants for the robustness table.

    h_plugin : current behaviour, what every existing number uses
    h_norm   : h_plugin / log(support) -- puts datasets on a common ceiling
    h_mm     : Miller-Madow, h_plugin + (K_obs - 1) / (2 N)
               where K_obs is the number of labels actually observed and N
               the annotator count for THAT item.

    TODO: h_mm needs a per-item N. Blocked on audit() above.
    TODO: decide which is PRIMARY before recomputing the panel. Changing it
          afterwards, once the numbers are visible, is choosing the result.
    """
    N = float(np.sum(counts))
    p = np.asarray(counts, float) / max(N, 1e-12)
    h_plugin = float(-(p * np.log(np.clip(p, 1e-12, None))).sum())
    k_obs = int((np.asarray(counts) > 0).sum())
    return dict(
        h_plugin=h_plugin,
        h_norm=h_plugin / np.log(support) if support > 1 else np.nan,
        h_mm=h_plugin + (k_obs - 1) / (2.0 * N) if N > 0 else np.nan,
    )
