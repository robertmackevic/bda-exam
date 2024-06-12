import time
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Any, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class Rendezvous:
    mmsi1: int = 0
    mmsi2: int = 0
    coords1: Tuple[float, float] = (0, 0)
    coords2: Tuple[float, float] = (0, 0)
    distance: float = float("inf")
    timestamp: pd.Timestamp = pd.Timestamp.now()


def timeit(function: Callable) -> Callable:
    @wraps(function)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        output = function(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function `{function.__name__}` took {total_time:.4f} seconds")
        return output

    return wrapper


def is_vessel_within_circle_pairwise(
        latitudes: NDArray,
        longitudes: NDArray,
        center: Tuple[float, float] = (55.225000, 14.245000),
        radius_km: float = 50
) -> NDArray:
    center_latitude, center_longitude = center
    delta_latitudes = np.radians(latitudes - center_latitude)
    delta_longitudes = np.radians(longitudes - center_longitude)
    x = (
            np.sin(delta_latitudes / 2) ** 2 +
            np.cos(np.radians(center_latitude)) * np.cos(np.radians(latitudes)) * np.sin(delta_longitudes / 2) ** 2
    )
    return (6371 * 2 * np.arctan2(np.sqrt(x), np.sqrt(1 - x))) <= radius_km
