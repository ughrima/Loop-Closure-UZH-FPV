import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# 1. Load Bag 3 and Bag 7 pose files
bag3 = pd.read_csv("/home/agrima/Desktop/new-approach/poses/poses_bag-3.csv")  # Replace with your path
bag7 = pd.read_csv("/home/agrima/Desktop/new-approach/poses/poses_bag-7.csv")

# Extract positions from the pose files (ignoring orientation)
bag3_xyz = bag3[['PosX', 'PosY', 'PosZ']].to_numpy()
bag7_xyz = bag7[['PosX', 'PosY', 'PosZ']].to_numpy()

# 2. Compute nearest distances from Bag 7 → Bag 3 using cKDTree
tree = cKDTree(bag3_xyz)
dists, indices = tree.query(bag7_xyz, k=1)

# 3. Build a DataFrame of closest pairs and distances
closest_matches = pd.DataFrame({
    'Bag7_Timestamp': bag7['Timestamp'],
    'Bag7_PosX': bag7['PosX'],
    'Bag7_PosY': bag7['PosY'],
    'Bag7_PosZ': bag7['PosZ'],
    'Bag3_Closest_Timestamp': bag3['Timestamp'].iloc[indices].values,
    'Bag3_PosX': bag3['PosX'].iloc[indices].values,
    'Bag3_PosY': bag3['PosY'].iloc[indices].values,
    'Bag3_PosZ': bag3['PosZ'].iloc[indices].values,
    'Distance': dists
})

# 4. Plot Histogram and CDF of distances
plt.figure(figsize=(12, 5))

# Histogram
plt.subplot(1, 2, 1)
plt.hist(dists, bins=100, range=(0, 2), color='steelblue')
plt.title("Histogram of Closest Distances")
plt.xlabel("Distance (m)")
plt.ylabel("Frequency")

# CDF
plt.subplot(1, 2, 2)
sorted_dists = np.sort(dists)
cdf = np.arange(len(dists)) / float(len(dists))
plt.plot(sorted_dists, cdf, color='darkgreen')
plt.title("CDF of Closest Distances")
plt.xlabel("Distance (m)")
plt.ylabel("Cumulative Probability")

plt.tight_layout()
plt.show()

# 5. Save ground truth for different thresholds (0.3m, 0.4m, 0.5m, 0.6m)
thresholds = [0.3, 0.4, 0.5, 0.6]
for thresh in thresholds:
    gt = closest_matches[closest_matches['Distance'] <= thresh]
    gt.to_csv(f"ground_truth_loop_closures_thresh_{thresh:.2f}m.csv", index=False)
    print(f"Saved {len(gt)} loop closures at threshold {thresh:.2f}m")

