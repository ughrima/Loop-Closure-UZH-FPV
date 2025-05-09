import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

# Load Bag 7 and Bag 3 trajectories
bag7_df = pd.read_csv('poses/poses_bag-7.csv')
bag3_df = pd.read_csv('poses/poses_bag-3.csv')

# Load predictions CSV file
predictions_df = pd.read_csv('predictions.csv')

# Function to compute the closest Bag 3 position
def compute_closest_bag3_pos(bag7_row, bag3_df):
    """Find the closest Bag 3 position to a given Bag 7 position."""
    # Calculate distances from Bag 7 position to all Bag 3 positions
    distances = cdist([bag7_row[['PosX', 'PosY', 'PosZ']].values], 
                      bag3_df[['PosX', 'PosY', 'PosZ']].values, metric='euclidean')
    min_idx = np.argmin(distances)  # Find index of closest point
    closest_bag3_row = bag3_df.iloc[min_idx]
    return closest_bag3_row['Timestamp'], closest_bag3_row[['PosX', 'PosY', 'PosZ']].values, distances[0, min_idx]

# Prepare lists to store new columns
closest_bag3_timestamps = []
closest_bag3_positions = []
distances = []

# For each predicted loop closure, find the closest Bag 3 position
for _, pred_row in predictions_df.iterrows():
    if pred_row['Predicted_LoopClosure'] == 1:
        # Find the closest Bag 3 position for this Bag 7 position
        timestamp, pos, dist = compute_closest_bag3_pos(bag7_df.loc[bag7_df['Timestamp'] == pred_row['Bag7_Timestamp']].iloc[0], bag3_df)
        closest_bag3_timestamps.append(timestamp)
        closest_bag3_positions.append(pos)
        distances.append(dist)
    else:
        # If no loop closure predicted, append NaN for Bag3-related columns
        closest_bag3_timestamps.append(np.nan)
        closest_bag3_positions.append([np.nan, np.nan, np.nan])
        distances.append(np.nan)

# Convert closest Bag 3 positions to separate columns
closest_bag3_positions = np.array(closest_bag3_positions)
closest_bag3_x = closest_bag3_positions[:, 0]
closest_bag3_y = closest_bag3_positions[:, 1]
closest_bag3_z = closest_bag3_positions[:, 2]

# Add the new columns to the predictions DataFrame
predictions_df['Bag3_Closest_Timestamp'] = closest_bag3_timestamps
predictions_df['Bag3_PosX'] = closest_bag3_x
predictions_df['Bag3_PosY'] = closest_bag3_y
predictions_df['Bag3_PosZ'] = closest_bag3_z
predictions_df['Distance'] = distances

# Reorder columns to match the desired output
predictions_df = predictions_df[['Bag7_Timestamp', 'Predicted_LoopClosure', 'BestMatch_Index_Bag3', 'Min_Distance',
                                   'Bag3_Closest_Timestamp', 'Bag3_PosX', 'Bag3_PosY', 'Bag3_PosZ', 'Distance']]

# Save the modified predictions to a new CSV file
predictions_df.to_csv('predictions_with_bag3.csv', index=False)
print("Predictions with closest Bag3 data saved to 'predictions_with_bag3.csv'")
