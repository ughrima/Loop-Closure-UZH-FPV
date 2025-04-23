import open3d as o3d
import os
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.linalg import orthogonal_procrustes
import matplotlib.pyplot as plt

# Function to load point clouds from a folder
def load_point_clouds(folder_path):
    pcds = []
    for i in range(3176):  # Assuming you have 3176 point clouds
        file_path = os.path.join(folder_path, f"cloud_{i}.pcd")
        if os.path.exists(file_path):  # Check if the file exists
            pcd = o3d.io.read_point_cloud(file_path)
            pcds.append(pcd)
        else:
            print(f"File cloud_{i}.pcd not found, skipping.")
    
    return pcds

# ICP Loop Closure Function
def icp_loop_closure(pcds, threshold=0.05, max_iterations=50):
    transformations = []
    icp_results = []
    
    # Perform ICP between successive point clouds
    for i in range(len(pcds) - 1):
        print(f"Processing ICP between cloud_{i} and cloud_{i+1}")
        
        # Perform ICP registration
        reg_icp = o3d.pipelines.registration.registration_icp(
            pcds[i], pcds[i + 1], threshold,
            np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        
        transformations.append(reg_icp.transformation)
        icp_results.append(reg_icp)
        
    return icp_results, transformations

# Gromov-Wasserstein Loop Closure (simplified)
def gromov_wasserstein_loop_closure(pcds):
    gw_results = []
    
    # Assuming pairwise distance metric for Gromov-Wasserstein computation
    for i in range(len(pcds) - 1):
        print(f"Processing Gromov-Wasserstein between cloud_{i} and cloud_{i+1}")
        
        # Convert point clouds to numpy arrays
        pcd_a = np.asarray(pcds[i].points)
        pcd_b = np.asarray(pcds[i + 1].points)
        
        # Compute pairwise distances
        dist_a = pairwise_distances(pcd_a, pcd_a)
        dist_b = pairwise_distances(pcd_b, pcd_b)
        
        # Simplified Gromov-Wasserstein (use Procrustes analysis as a proxy)
        _, rotation, _ = orthogonal_procrustes(dist_a, dist_b)
        
        gw_results.append(rotation)
        
    return gw_results

# Function to compare loop closure results with the CSV
def compare_with_csv(csv_path, icp_transformations, gw_results):
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Assume we are comparing position data (PosX, PosY, PosZ) with ICP and GW results
    positions = df[['PosX', 'PosY', 'PosZ']].values
    
    # Extract ICP transformations and apply them to the initial position
    icp_positions = [np.dot(transformation[:3, :3], positions[i]) + transformation[:3, 3] for i, transformation in enumerate(icp_transformations)]
    
    # Gromov-Wasserstein error (simplified as rotation difference)
    gw_positions = [np.dot(rotation, positions[i]) for i, rotation in enumerate(gw_results)]
    
    # Calculate error between ICP results and CSV data
    icp_positions = np.array(icp_positions)
    gw_positions = np.array(gw_positions)
    
    icp_error = np.linalg.norm(icp_positions - positions[:len(icp_positions)], axis=1)
    gw_error = np.linalg.norm(gw_positions - positions[:len(gw_positions)], axis=1)
    
    # Calculate precision, recall, and F1-score for ICP and GW (thresholding for simplification)
    threshold = 0.1  # 10cm threshold for correct position
    icp_precision = precision_score(icp_error < threshold, np.ones_like(icp_error))
    gw_precision = precision_score(gw_error < threshold, np.ones_like(gw_error))
    
    icp_recall = recall_score(icp_error < threshold, np.ones_like(icp_error))
    gw_recall = recall_score(gw_error < threshold, np.ones_like(gw_error))
    
    icp_f1 = f1_score(icp_error < threshold, np.ones_like(icp_error))
    gw_f1 = f1_score(gw_error < threshold, np.ones_like(gw_error))
    
    print(f"ICP Error: {icp_error}")
    print(f"Gromov-Wasserstein Error: {gw_error}")
    
    return icp_error, gw_error, icp_precision, gw_precision, icp_recall, gw_recall, icp_f1, gw_f1

# Function to plot comparison graph
def plot_comparison_graph(icp_error, gw_error, icp_precision, gw_precision, icp_recall, gw_recall, icp_f1, gw_f1):
    labels = ['ICP', 'Gromov-Wasserstein']
    precision = [icp_precision, gw_precision]
    recall = [icp_recall, gw_recall]
    f1 = [icp_f1, gw_f1]
    
    x = np.arange(len(labels))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 0.2, precision, 0.4, label='Precision')
    ax.bar(x + 0.2, recall, 0.4, label='Recall')
    
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Loop Closure Methods (ICP vs Gromov-Wasserstein)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
# Main function
def main():
    folder_path = "pcd-7"  # Path to your folder containing cloud_0.pcd, cloud_1.pcd, ...
    csv_path = "/home/agrima/Desktop/new-approach/ground_truth_poses.csv"  # Path to your CSV file
    
    # Load point clouds from folder
    pcds = load_point_clouds(folder_path)
    
    # Perform ICP loop closure
    print("Performing ICP Loop Closure...")
    icp_results, icp_transformations = icp_loop_closure(pcds)
    
    # Perform Gromov-Wasserstein loop closure
    print("Performing Gromov-Wasserstein Loop Closure...")
    gw_results = gromov_wasserstein_loop_closure(pcds)
    
    # Compare ICP loop closure results with the CSV
    print("Comparing ICP results with CSV...")
    icp_error, gw_error, icp_precision, gw_precision, icp_recall, gw_recall, icp_f1, gw_f1 = compare_with_csv(csv_path, icp_transformations, gw_results)
    
    # Plot comparison graph
    plot_comparison_graph(icp_error, gw_error, icp_precision, gw_precision, icp_recall, gw_recall, icp_f1, gw_f1)
    
if __name__ == "__main__":
    main()
