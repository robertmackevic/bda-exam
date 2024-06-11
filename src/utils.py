import time
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Any
from typing import Tuple

import numpy as np
import pandas as pd
from folium import Map, PolyLine, CircleMarker
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

    # Create a map centered around the average coordinates of the vessels
    mean_latitude = (vessel1_data["Latitude"].mean() + vessel2_data["Latitude"].mean()) / 2
    mean_longitude = (vessel1_data["Longitude"].mean() + vessel2_data["Longitude"].mean()) / 2
    folium_map = Map(location=[mean_latitude, mean_longitude], zoom_start=15)

    # Add vessel 1 trajectory
    vessel1_coords = list(zip(vessel1_data["Latitude"], vessel1_data["Longitude"]))
    PolyLine(vessel1_coords, color="red", weight=2.5, opacity=1, tooltip=f"{closest_pair.mmsi1}").add_to(folium_map)
    for coord in vessel1_coords:
        CircleMarker(location=coord, radius=3, color="red").add_to(folium_map)

    # Add vessel 2 trajectory
    vessel2_coords = list(zip(vessel2_data["Latitude"], vessel2_data["Longitude"]))
    PolyLine(vessel2_coords, color="blue", weight=2.5, opacity=1, tooltip=f"{closest_pair.mmsi2}").add_to(folium_map)
    for coord in vessel2_coords:
        CircleMarker(location=coord, radius=3, color="blue").add_to(folium_map)

    folium_map.save(DATA_DIR / (closest_pair.timestamp.date().isoformat() + ".html"))

    plt.figure(figsize=(10, 6))

    plt.plot(vessel1_data["Longitude"], vessel1_data["Latitude"], label=f"Vessel {closest_pair.mmsi1}", c="red")
    plt.plot(vessel2_data["Longitude"], vessel2_data["Latitude"], label=f"Vessel {closest_pair.mmsi2}", c="blue")

    plt.scatter(vessel1_data["Longitude"], vessel1_data["Latitude"], c="red", s=10)
    plt.scatter(vessel2_data["Longitude"], vessel2_data["Latitude"], c="blue", s=10)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Trajectory of Vessels {closest_pair.mmsi1} and {closest_pair.mmsi2} around {closest_pair.timestamp}")
    plt.legend()
    plt.grid(True)
    plt.savefig(DATA_DIR / (closest_pair.timestamp.date().isoformat() + ".png"))
    plt.close()
