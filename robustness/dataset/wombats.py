import numpy as np
import random
from numpy.typing import NDArray

from wombats.anomalies.increasing import (
    GWN, Impulse, Step, Constant
)

ALL_ANOMALIES = [
    GWN,
    Impulse,
    Step,
    Constant,
    ]


def apply_random_wombats_anomaly(
    signal: NDArray,  # shape (W,)
    Xok_ref: NDArray, # shape (N, W) for fit
    delta_range: tuple[float, float],
) -> np.ndarray:
    """
    Applica una anomalia WOMBATS random a un segnale 1D
    """
    anomaly_cls = random.choice(ALL_ANOMALIES)
    delta = np.random.uniform(*delta_range)

    anomaly = anomaly_cls(delta)

    # alcune anomalie richiedono fit
    try:
        anomaly.fit(Xok_ref)
    except TypeError:
        pass

    return anomaly.distort(signal)
