"""When a heat map counts as an explanation. Plain Python, the policy in one place.

A method earns trust in two steps:
  1. it passes the sanity check (Adebayo et al., 2018): randomize the network's top layers and the map must change;
     a map that survives random weights is drawing the image, not the model;
  2. on this image, the map is stable: small perturbations of the input don't reshuffle what it highlights.
The thresholds are a stated policy, picked before looking at the results, not tuned to them.
"""

SANITY_MAX_SIMILARITY = 0.5  # rank correlation with the trained model's map after the top block is randomized
STABLE_MIN = 0.6  # mean top-k overlap between maps of slightly perturbed copies of the image


def passes_sanity(similarity_after_top_randomized: float) -> bool:
    return similarity_after_top_randomized < SANITY_MAX_SIMILARITY


def badge(passed_sanity: bool, stability: float) -> dict:
    if not passed_sanity:
        return {"level": "not an explanation",
                "why": "this method's map barely changes when the model's weights are randomized"}
    if stability < STABLE_MIN:
        return {"level": "unstable",
                "why": f"small changes to the image reshuffle what the map highlights (stability {stability:.2f})"}
    return {"level": "trusted", "why": f"passes the sanity check and is stable on this image ({stability:.2f})"}


def top_k_overlap(a: list[float], b: list[float], k: int) -> float:
    """Jaccard of the k highest-valued positions of two flattened maps."""
    ta, tb = _top(a, k), _top(b, k)
    return len(ta & tb) / len(ta | tb) if ta | tb else 1.0


def _top(m: list[float], k: int) -> set[int]:
    return set(sorted(range(len(m)), key=lambda i: -m[i])[:k])
