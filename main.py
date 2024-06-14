"""
Script to find the two closest moving vessels in a particular sea area and visualize their trajectory
10 minutes before and after the rendezvous time.

Args:
    -w  --max-workers   (optional)  Number of parallel workers. (default: all available)
    -f  --num-files     (optional)  Number of files to process. (default: all available)

Note that by default all available workers will be used, which may lead to high memory consumption.
It's best to adjust this parameter based on the hardware.
"""
from argparse import Namespace, ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from os import listdir
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from tqdm import tqdm

from src.paths import DATA_DIR
from src.utils import timeit, is_vessel_within_circle_pairwise, Rendezvous
from src.visuals import plot_vessel_trajectories


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-w", "--max-workers", type=int, required=False, default=None)
    parser.add_argument("-f", "--num-files", type=int, required=False, default=None)
    return parser.parse_args()


def find_rendezvous_moment(
        filepath: Path,
        window_size: int = 30,
        step_size: int = 5,
        center: Tuple[float, float] = (55.225000, 14.245000),
        radius_km: float = 50
) -> Rendezvous:
    df = pd.read_csv(filepath)

    df.rename(columns={"# Timestamp": "Timestamp"}, inplace=True)
    # Only leave the columns which are needed for analysis
    df = df.filter(items=["Timestamp", "Latitude", "Longitude", "MMSI", "SOG"])
    # Filter out entries with incorrect coordinate values
    # Filter out entries which have a low or non-existing speed over ground (SOG)
    df = df[
        (df["Latitude"] >= -90) &
        (df["Latitude"] <= 90) &
        (df["Longitude"] >= -90) &
        (df["Longitude"] <= 90) &
        (df["SOG"] >= 1.0)
        ]
    # Filter out vessels that are not in the circle
    df = df[is_vessel_within_circle_pairwise(df["Latitude"].values, df["Longitude"].values, center, radius_km)]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S")

    rendezvous = Rendezvous()
    start_time = df["Timestamp"].min()
    end_time = df["Timestamp"].max()
    current_time = start_time

    # Go through the data with a sliding window technique
    while current_time <= end_time:
        # Group the vessels by time
        window_end = current_time + pd.Timedelta(seconds=window_size)
        window_df = df[(df["Timestamp"] >= current_time) & (df["Timestamp"] <= window_end)]

        # If less than 2 vessels in a group, then skip the calculation
        if len(window_df) < 2:
            current_time += pd.Timedelta(seconds=step_size)
            continue

        vessel_positions = np.radians(window_df[["Latitude", "Longitude"]].values)
        vessel_mmsi = window_df["MMSI"].values
        # Create a BallTree for efficient spatial queries using the Haversine metric
        tree = BallTree(vessel_positions, metric="haversine")

        # Query the BallTree to find the distance and index of the closest neighbor for each vessel
        # k=2 returns the distance and index of the two closest neighbors (including itself)
        distances, indices = tree.query(vessel_positions, k=2)

        for distance, index in zip(distances, indices):
            # If it's the same vessel, but at a slightly different time step, then skip the calculation
            if vessel_mmsi[index[0]] == vessel_mmsi[index[1]]:
                continue

            # Convert distance to kilometers
            distance = distance[1] * 6371

            if distance < rendezvous.distance:
                rendezvous = Rendezvous(
                    mmsi1=vessel_mmsi[index[0]],
                    mmsi2=vessel_mmsi[index[1]],
                    coords1=window_df[["Latitude", "Longitude"]].values[index[0]],
                    coords2=window_df[["Latitude", "Longitude"]].values[index[1]],
                    distance=distance,
                    timestamp=current_time
                )

        current_time += pd.Timedelta(seconds=step_size)

    plot_vessel_trajectories(df, rendezvous)
    return rendezvous


@timeit
def run(max_workers: Optional[int], num_files: Optional[int]) -> None:
    filepaths = [DATA_DIR / filename for filename in listdir(DATA_DIR) if filename.endswith(".csv")][:num_files]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(find_rendezvous_moment, filepaths), total=len(filepaths)))

    rendezvous = min(results, key=lambda result: result.distance)
    print("Rendezvous of the closest pair of vessels:")
    print(f"MMSI 1: {rendezvous.mmsi1} | Location {rendezvous.coords1}")
    print(f"MMSI 2: {rendezvous.mmsi2} | Location {rendezvous.coords2}")
    print(f"Distance: {rendezvous.distance * 1000:.3f} meters")
    print(f"Timestamp: {rendezvous.timestamp}")


if __name__ == "__main__":
    run(**vars(parse_args()))
