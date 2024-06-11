import time
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Any
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from src.paths import DATA_DIR


@dataclass(frozen=True)
class ClosestPair:
    mmsi1: int = 0
    mmsi2: int = 0
    distance: float = float("inf")
    timestamp: pd.Timestamp = pd.Timestamp.now()


def timeit(function: Callable) -> Callable:
    @wraps(function)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        function(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function `{function.__name__}` took {total_time:.4f} seconds")
        return total_time

    return wrapper


def is_vessel_within_circle_pairwise(
        latitudes: NDArray,
        longitudes: NDArray,
        center: Tuple[float, float],
        radius: float
) -> NDArray:
    center_latitude, center_longitude = center
    delta_latitudes = np.radians(latitudes - center_latitude)
    delta_longitudes = np.radians(longitudes - center_longitude)
    a = (
            np.sin(delta_latitudes / 2) ** 2 +
            np.cos(np.radians(center_latitude)) * np.cos(np.radians(latitudes)) * np.sin(delta_longitudes / 2) ** 2
    )
    return (6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))) <= radius


def plot_vessel_trajectories(df: pd.DataFrame, closest_pair: ClosestPair) -> None:
    start_time = closest_pair.timestamp - pd.Timedelta(minutes=10)
    end_time = closest_pair.timestamp + pd.Timedelta(minutes=10)

    vessel1_data = df[
        (df["MMSI"] == closest_pair.mmsi1) &
        (df["Timestamp"] >= start_time) &
        (df["Timestamp"] <= end_time)
        ]
    vessel2_data = df[
        (df["MMSI"] == closest_pair.mmsi2) &
        (df["Timestamp"] >= start_time) &
        (df["Timestamp"] <= end_time)
        ]

    plt.figure(figsize=(10, 6))

    plt.plot(vessel1_data["Longitude"], vessel1_data["Latitude"], label=f"Vessel {closest_pair.mmsi1}")
    plt.plot(vessel2_data["Longitude"], vessel2_data["Latitude"], label=f"Vessel {closest_pair.mmsi2}")

    plt.scatter(vessel1_data["Longitude"], vessel1_data["Latitude"], c="blue", s=10)
    plt.scatter(vessel2_data["Longitude"], vessel2_data["Latitude"], c="red", s=10)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Trajectory of Vessels {closest_pair.mmsi1} and {closest_pair.mmsi2} around {closest_pair.timestamp}")
    plt.legend()
    plt.grid(True)
    plt.savefig(DATA_DIR / (closest_pair.timestamp.date().isoformat() + ".png"))
    plt.close()
