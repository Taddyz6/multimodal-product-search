"""Train-fitted affine CLIP calibration, including missing-image fallback.

Parameters are loaded, never re-estimated by the online service. Keep float64
intermediates until the final feature cast, matching the frozen implementation.
"""

import numpy as np


def mixture_std(params, beta):
    if not 0 <= beta <= 1:
        raise ValueError("Invalid beta")
    return float(
        np.sqrt(max(0.0, (1 - beta) ** 2 + beta**2 + 2 * beta * (1 - beta) * params["correlation"]))
    )


def calibrated_clip_scores(text, image, mask, params, beta, mask_aware):
    t = np.asarray(text, np.float64)
    i = np.asarray(image, np.float64)
    mask = np.asarray(mask, bool)
    if t.shape != i.shape or t.shape != mask.shape or (not 0 <= beta <= 1):
        raise ValueError("Invalid score/mask inputs")
    zt = (t - params["text"]["mean"]) / params["text"]["std"]
    zi = (i - params["image"]["mean"]) / params["image"]["std"]
    score = (1 - beta) * zt + beta * zi
    if mask_aware:
        missing = (
            mixture_std(params, beta)
            * (t - params["missing_text"]["mean"])
            / params["missing_text"]["std"]
        )
    else:
        missing = zt
    return params["text"]["mean"] + params["text"]["std"] * np.where(mask, score, missing)
