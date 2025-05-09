# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.spatial import cKDTree

# # === STEP 1: Load Pose Data ===
# bag3_file = 'poses/poses_bag-3.csv'  # Reference (Oval)
# bag7_file = 'poses/poses_bag-7.csv'  # Target

# bag3_df = pd.read_csv(bag3_file)
# bag7_df = pd.read_csv(bag7_file)

# bag3_xyz = bag3_df[['PosX', 'PosY', 'PosZ']].values
# bag7_xyz = bag7_df[['PosX', 'PosY', 'PosZ']].values

# # === STEP 2: Build KD-Tree for Bag 3 ===
# tree = cKDTree(bag3_xyz)

# # === STEP 3: Query All Bag 7 Points for Closest Match in Bag 3 ===
# dists, indices = tree.query(bag7_xyz)

# # === STEP 4: Apply 0.5 Meter Threshold ===
# threshold = 0.5  # meters
# mask = dists < threshold

# # === STEP 5: Extract Matching Ground Truth Points ===
# loop_closures = pd.DataFrame({
#     'Bag7_Timestamp': bag7_df['Timestamp'][mask].values,
#     'Bag7_PosX': bag7_df['PosX'][mask].values,
#     'Bag7_PosY': bag7_df['PosY'][mask].values,
#     'Bag7_PosZ': bag7_df['PosZ'][mask].values,
#     'Bag3_Closest_Timestamp': bag3_df['Timestamp'].iloc[indices[mask]].values,
#     'Bag3_PosX': bag3_df['PosX'].iloc[indices[mask]].values,
#     'Bag3_PosY': bag3_df['PosY'].iloc[indices[mask]].values,
#     'Bag3_PosZ': bag3_df['PosZ'].iloc[indices[mask]].values,
#     'Distance': dists[mask]
# })

# # === STEP 6: Save to CSV ===
# loop_closures.to_csv('ground_truth_loop_closures_bag7_to_bag3_thresh_0.5m.csv', index=False)

# print(f"✅ Saved {len(loop_closures)} ground truth loop closures under 0.5m threshold.")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# === STEP 1: Load Pose Data ===
bag3_file = 'poses/poses_bag-3.csv'  # Reference (Oval)
bag7_file = 'poses/poses_bag-7.csv'  # Target

bag3_df = pd.read_csv(bag3_file)
bag7_df = pd.read_csv(bag7_file)

bag3_xyz = bag3_df[['PosX', 'PosY', 'PosZ']].values
bag7_xyz = bag7_df[['PosX', 'PosY', 'PosZ']].values

# === STEP 2: Build KD-Tree for Bag 3 ===
tree = cKDTree(bag3_xyz)

# === STEP 3: Query All Bag 7 Points for Closest Match in Bag 3 ===
dists, indices = tree.query(bag7_xyz)

# === STEP 4: Apply 0.5 Meter Threshold ===
threshold = 0.6 # meters
mask = dists < threshold

# === STEP 5: Extract Matching Ground Truth Points ===
loop_closures = pd.DataFrame({
    'Bag7_Timestamp': bag7_df['Timestamp'][mask].values,
    'Bag7_PosX': bag7_df['PosX'][mask].values,
    'Bag7_PosY': bag7_df['PosY'][mask].values,
    'Bag7_PosZ': bag7_df['PosZ'][mask].values,
    'Bag3_Closest_Timestamp': bag3_df['Timestamp'].iloc[indices[mask]].values,
    'Bag3_PosX': bag3_df['PosX'].iloc[indices[mask]].values,
    'Bag3_PosY': bag3_df['PosY'].iloc[indices[mask]].values,
    'Bag3_PosZ': bag3_df['PosZ'].iloc[indices[mask]].values,
    'Distance': dists[mask]
})

# === STEP 6: Save to CSV ===
loop_closures.to_csv('ground_truth_loop_closures_bag7_to_bag3_thresh_0.5m.csv', index=False)

# === STEP 7: Plot Ground Truth Loop Closures ===
plt.figure(figsize=(10, 6))

# Plot Bag 7 poses
plt.scatter(bag7_df['PosX'], bag7_df['PosY'], label='Bag 7 Poses', c='blue', s=1)

# Plot Bag 3 poses
plt.scatter(bag3_df['PosX'], bag3_df['PosY'], label='Bag 3 Poses', c='red', s=1)

# Plot Ground Truth Loop Closures (Bag 7)
plt.scatter(loop_closures['Bag7_PosX'], loop_closures['Bag7_PosY'], label='Ground Truth Loop Closures (Bag 7)', c='purple', s=30, marker='o')

# Plot Ground Truth Loop Closures (Bag 3)
plt.scatter(loop_closures['Bag3_PosX'], loop_closures['Bag3_PosY'], label='Ground Truth Loop Closures (Bag 3)', c='green', s=30, marker='x')

# Labels and title
plt.xlabel('X Position (meters)')
plt.ylabel('Y Position (meters)')
plt.title('Ground Truth Loop Closures: Bag 7 and Bag 3')
plt.legend()

# Show the plot
plt.show()

