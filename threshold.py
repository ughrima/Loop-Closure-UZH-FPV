import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# === STEP 1: Load Pose Data (CSV Format) ===
# File paths (change these to your actual file paths)
bag3_file = 'poses/poses_bag-3.csv'
bag7_file = 'poses/poses_bag-7.csv'

# Columns: Timestamp, PosX, PosY, PosZ, OriX, OriY, OriZ, OriW
bag3_df = pd.read_csv(bag3_file)
bag7_df = pd.read_csv(bag7_file)

# Extract only positions
bag3_xyz = bag3_df[['PosX', 'PosY', 'PosZ']].values
bag7_xyz = bag7_df[['PosX', 'PosY', 'PosZ']].values

# === STEP 2: Plot Trajectories ===
def plot_trajectories(bag3_xyz, bag7_xyz):
    plt.figure(figsize=(10, 7))
    plt.plot(bag3_xyz[:, 0], bag3_xyz[:, 1], label='Bag 3 (Reference)', color='blue')
    plt.plot(bag7_xyz[:, 0], bag7_xyz[:, 1], label='Bag 7 (Target)', color='red')
    plt.legend()
    plt.title("Trajectory Overlap: Bag 3 vs Bag 7")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

plot_trajectories(bag3_xyz, bag7_xyz)

# === STEP 3: Compute Min Distances from Bag 7 to Bag 3 ===
tree = cKDTree(bag3_xyz)
dists, indices = tree.query(bag7_xyz)

# Optional: Save matches for ground truth evaluation
matches = pd.DataFrame({
    'Bag7_Timestamp': bag7_df['Timestamp'],
    'Bag7_PosX': bag7_df['PosX'],
    'Bag7_PosY': bag7_df['PosY'],
    'Bag7_PosZ': bag7_df['PosZ'],
    'Bag3_Closest_Timestamp': bag3_df['Timestamp'].iloc[indices].values,
    'Distance': dists
})
matches.to_csv('ground_truth_loop_closures_bag7_to_bag3.csv', index=False)

# === STEP 4: Plot Histogram of Distances ===
def min_distance_histogram(dists):
    plt.figure(figsize=(8, 5))
    plt.hist(dists, bins=50, color='green', alpha=0.7)
    plt.title("Histogram of Min Distances (Bag 7 → Bag 3)")
    plt.xlabel("Distance (m)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()

min_distance_histogram(dists)

