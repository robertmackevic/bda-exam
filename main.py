from argparse import Namespace, ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from os import listdir
from pathlib import Path
from typing import Optional

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


def find_rendezvous_moment(filepath: Path) -> Rendezvous:
    df = pd.read_csv(filepath)

    df.rename(columns={"# Timestamp": "Timestamp"}, inplace=True)
    df = df.filter(items=["Timestamp", "Latitude", "Longitude", "MMSI", "SOG"])
    df = df[
        (df["Latitude"] >= -90) &
        (df["Latitude"] <= 90) &
        (df["Longitude"] >= -90) &
        (df["Longitude"] <= 90) &
        (df["SOG"] >= 1.0)
        ]
    df = df[is_vessel_within_circle_pairwise(df["Latitude"].values, df["Longitude"].values)]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S")

    rendezvous = Rendezvous()

    for timestamp, group in df.groupby("Timestamp"):
        if len(group) < 2:
            continue

        vessel_positions = np.radians(group[["Latitude", "Longitude"]].values)
        vessel_mmsi = group["MMSI"].values
        tree = BallTree(vessel_positions, metric="haversine")
        distances, indices = tree.query(vessel_positions, k=2)

        for i, (distance, j) in enumerate(zip(distances, indices)):
            if vessel_mmsi[i] == vessel_mmsi[j[1]]:
                continue

            haversine_distance = distance[1] * 6371

            if haversine_distance < rendezvous.distance:
                rendezvous = Rendezvous(
                    mmsi1=vessel_mmsi[i],
                    mmsi2=vessel_mmsi[j[1]],
                    coords1=group[["Latitude", "Longitude"]].values[i],
                    coords2=group[["Latitude", "Longitude"]].values[j[1]],
                    distance=haversine_distance,
                    timestamp=timestamp
                )

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
