import pandas as pd
from folium import Map, PolyLine, CircleMarker
from matplotlib import pyplot as plt

from src.paths import DATA_DIR
from src.utils import Rendezvous


def plot_vessel_trajectories(df: pd.DataFrame, rendezvous: Rendezvous) -> None:
    start_time = rendezvous.timestamp - pd.Timedelta(minutes=10)
    end_time = rendezvous.timestamp + pd.Timedelta(minutes=10)

    vessel1_data = df[
        (df["MMSI"] == rendezvous.mmsi1) &
        (df["Timestamp"] >= start_time) &
        (df["Timestamp"] <= end_time)
        ]
    vessel2_data = df[
        (df["MMSI"] == rendezvous.mmsi2) &
        (df["Timestamp"] >= start_time) &
        (df["Timestamp"] <= end_time)
        ]

    _plot_trajectories_folium(vessel1_data, vessel2_data, rendezvous)
    _plot_trajectories_pyplot(vessel1_data, vessel2_data, rendezvous)


def _plot_trajectories_folium(
        vessel1_data: pd.DataFrame,
        vessel2_data: pd.DataFrame,
        rendezvous: Rendezvous,
) -> None:
    # Create a map centered around the average coordinates of the vessels
    mean_latitude = (vessel1_data["Latitude"].mean() + vessel2_data["Latitude"].mean()) / 2
    mean_longitude = (vessel1_data["Longitude"].mean() + vessel2_data["Longitude"].mean()) / 2
    folium_map = Map(location=[mean_latitude, mean_longitude], zoom_start=15)

    # Add vessel 1 trajectory
    vessel1_coords = list(zip(vessel1_data["Latitude"], vessel1_data["Longitude"]))
    PolyLine(vessel1_coords, color="red", weight=2.5, opacity=1, tooltip=f"{rendezvous.mmsi1}").add_to(folium_map)
    for i, coord in enumerate(vessel1_coords):
        CircleMarker(coord, radius=3, color="red", tooltip=f"{vessel1_data["Timestamp"].iloc[i]}").add_to(folium_map)

    # Add vessel 2 trajectory
    vessel2_coords = list(zip(vessel2_data["Latitude"], vessel2_data["Longitude"]))
    PolyLine(vessel2_coords, color="blue", weight=2.5, opacity=1, tooltip=f"{rendezvous.mmsi2}").add_to(folium_map)
    for i, coord in enumerate(vessel2_coords):
        CircleMarker(coord, radius=3, color="blue", tooltip=f"{vessel2_data["Timestamp"].iloc[i]}").add_to(folium_map)

    # Add black circle markers at the closest points
    CircleMarker(rendezvous.coords1,
                 radius=10, color="black", fill=True, fill_color="black", fill_opacity=1,
                 tooltip=f"{rendezvous.timestamp}").add_to(folium_map)
    CircleMarker(rendezvous.coords2,
                 radius=10, color="black", fill=True, fill_color="black", fill_opacity=1,
                 tooltip=f"{rendezvous.timestamp}").add_to(folium_map)

    folium_map.save(DATA_DIR / (rendezvous.timestamp.date().isoformat() + ".html"))


def _plot_trajectories_pyplot(
        vessel1_data: pd.DataFrame,
        vessel2_data: pd.DataFrame,
        rendezvous: Rendezvous,
) -> None:
    plt.figure(figsize=(10, 6))

    plt.plot(vessel1_data["Longitude"], vessel1_data["Latitude"], label=f"{rendezvous.mmsi1}", c="red", zorder=1)
    plt.plot(vessel2_data["Longitude"], vessel2_data["Latitude"], label=f"{rendezvous.mmsi2}", c="blue", zorder=1)

    plt.scatter(vessel1_data["Longitude"], vessel1_data["Latitude"], c="red", s=10, zorder=2)
    plt.scatter(vessel2_data["Longitude"], vessel2_data["Latitude"], c="blue", s=10, zorder=2)

    plt.scatter(*rendezvous.coords1[::-1], c="black", s=50, label="Rendezvous", zorder=3)
    plt.scatter(*rendezvous.coords2[::-1], c="black", s=50, zorder=3)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Trajectory of Vessels {rendezvous.mmsi1} and {rendezvous.mmsi2} around {rendezvous.timestamp}")
    plt.legend()
    plt.grid(True)
    plt.savefig(DATA_DIR / (rendezvous.timestamp.date().isoformat() + ".png"))
    plt.close()
