from argparse import Namespace, ArgumentParser
from os import listdir
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from src.paths import DATA_DIR
from src.utils import timeit, is_vessel_within_circle_pairwise, ClosestPair, plot_vessel_trajectories

CIRCLE_CENTER = (55.225000, 14.245000)
RADIUS_KM = 50


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-n", "--num-files", type=int, required=False, default=None)
    return parser.parse_args()


@timeit
def run(num_files: Optional[int]) -> None:
    filepaths = [DATA_DIR / filename for filename in listdir(DATA_DIR) if filename.endswith(".csv")][:num_files]

    for filepath in filepaths:
        df = pd.read_csv(filepath)

        df.rename(columns={"# Timestamp": "Timestamp"}, inplace=True)
        df = df.filter(items=["Timestamp", "Latitude", "Longitude", "MMSI"])
        df = df[(df["Latitude"] >= -90) & (df["Latitude"] <= 90) & (df["Longitude"] >= -90) & (df["Longitude"] <= 90)]
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S")

        df = df[is_vessel_within_circle_pairwise(
            df["Latitude"].values,
            df["Longitude"].values,
            CIRCLE_CENTER,
            RADIUS_KM
        )]

        closest_pair = ClosestPair()
        grouped = df.groupby("Timestamp")

        for timestamp, group in grouped:
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

                if haversine_distance >= closest_pair.distance:
                    continue

                closest_pair = ClosestPair(
                    mmsi1=vessel_mmsi[i],
                    mmsi2=vessel_mmsi[j[1]],
                    distance=haversine_distance,
                    timestamp=timestamp
                )

        plot_vessel_trajectories(df, closest_pair)
        print(f"FILE {filepath.name} | "
              f"Closest pair of vessels: {closest_pair.mmsi1} and {closest_pair.mmsi2} | "
              f"Distance {closest_pair.distance} | "
              f"Timestamp {closest_pair.timestamp}")


if __name__ == "__main__":
    run(**vars(parse_args()))
