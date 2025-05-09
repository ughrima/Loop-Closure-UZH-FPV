import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load predicted data
predicted_data = pd.read_csv("/home/agrima/Desktop/new-approach/predictions_with_bag3.csv") 

# Filter for predicted loop closures (Predicted_LoopClosure == 1)
predicted_data = predicted_data[predicted_data['Predicted_LoopClosure'] == 1]

# Load ground truth data
ground_truth_data = pd.read_csv("/home/agrima/Desktop/new-approach/ground_truth_loop_closures_thresh_0.30m.csv")  # Replace with the path to your ground truth file

# Clean up column names to remove extra spaces
ground_truth_data.columns = ground_truth_data.columns.str.strip()
predicted_data.columns = predicted_data.columns.str.strip()


# ----3d--------

# Initialize 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the predicted loop closures (Predicted_LoopClosure == 1)
ax.scatter(predicted_data['Bag3_PosX'], predicted_data['Bag3_PosY'], predicted_data['Bag3_PosZ'], c='r', label='Predicted Loop Closures', marker='o')

# Plot the ground truth loop closures
ax.scatter(ground_truth_data['Bag3_PosX'], ground_truth_data['Bag3_PosY'], ground_truth_data['Bag3_PosZ'], c='b', label='Ground Truth Loop Closures', marker='^')

# Labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Title and legend
ax.set_title('Predicted Loop Closures vs Ground Truth')
ax.legend()

# Show the plot
plt.show()


# ---2d-----

# Initialize the plot
plt.figure(figsize=(8, 6))

# Plot the predicted loop closures (Predicted_LoopClosure == 1)
plt.scatter(predicted_data['Bag3_PosX'], predicted_data['Bag3_PosY'], c='r', label='Predicted Loop Closures', marker='o')

# Plot the ground truth loop closures
plt.scatter(ground_truth_data['Bag3_PosX'], ground_truth_data['Bag3_PosY'], c='b', label='Ground Truth Loop Closures', marker='^')

# Labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Predicted vs Ground Truth Loop Closures')

# Add a legend
plt.legend()

# Show the plot
plt.show()

