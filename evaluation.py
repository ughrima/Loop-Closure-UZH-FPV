# import pandas as pd
# import numpy as np
# from scipy.spatial.distance import cdist

# # Load predictions and ground truth files
# predictions_df = pd.read_csv('predictions_with_bag3.csv')
# ground_truth_df = pd.read_csv('/home/agrima/Desktop/new-approach/ground_truth_lc_bag-3v7.csv')

# # Define a threshold for matching loop closures
# threshold_distance = 0.5  # Threshold in meters

# # Initialize counters
# TP = 0  # True Positives
# FP = 0  # False Positives
# FN = 0  # False Negatives
# TN = 0  # True Negatives

# # Function to compute Euclidean distance between two positions
# def compute_distance(pos1, pos2):
#     return np.linalg.norm(pos1 - pos2)

# # Create a set of ground truth loop closures for easier matching
# ground_truth_set = set(zip(ground_truth_df['Bag7_Timestamp'], ground_truth_df['Bag3_Closest_Timestamp']))

# # Iterate over predictions
# for _, pred_row in predictions_df.iterrows():
#     bag7_timestamp = pred_row['Bag7_Timestamp']
#     predicted_closure = pred_row['Predicted_LoopClosure']
#     bag3_timestamp = pred_row['Bag3_Closest_Timestamp']
#     bag3_pos = np.array([pred_row['Bag3_PosX'], pred_row['Bag3_PosY'], pred_row['Bag3_PosZ']])
    
#     # Check if prediction is a loop closure
#     if predicted_closure == 1:
#         # Check if it's a match with ground truth within the threshold
#         matching_gt = ground_truth_df[
#             (ground_truth_df['Bag7_Timestamp'] == bag7_timestamp) &
#             (ground_truth_df['Bag3_Closest_Timestamp'] == bag3_timestamp)
#         ]
        
#         if not matching_gt.empty:
#             # Compute distance and check if it's within threshold
#             dist = compute_distance(bag3_pos, matching_gt[['Bag3_PosX', 'Bag3_PosY', 'Bag3_PosZ']].values[0])
#             if dist <= threshold_distance:
#                 TP += 1  # True Positive
#             else:
#                 FP += 1  # False Positive
#         else:
#             FP += 1  # False Positive
#     else:
#         # Check for false negatives (actual loop closure in ground truth but not predicted)
#         matching_gt = ground_truth_df[
#             (ground_truth_df['Bag7_Timestamp'] == bag7_timestamp) &
#             (ground_truth_df['Bag3_Closest_Timestamp'] == bag3_timestamp)
#         ]
        
#         if not matching_gt.empty:
#             FN += 1  # False Negative

# # Compute TN: Loop closures predicted as False and no corresponding GT loop closure
# all_timestamps_pred = set(predictions_df['Bag7_Timestamp'])
# all_timestamps_gt = set(ground_truth_df['Bag7_Timestamp'])
# FN_predicted = all_timestamps_pred - all_timestamps_gt
# TN = len(FN_predicted)  # Assuming all non-loop closure predictions are true negatives

# # Calculate metrics
# precision = TP / (TP + FP) if (TP + FP) > 0 else 0
# recall = TP / (TP + FN) if (TP + FN) > 0 else 0
# f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
# accuracy = (TP + TN) / (TP + TN + FP + FN)

# # Print the evaluation results
# print(f"True Positives (TP): {TP}")
# print(f"False Positives (FP): {FP}")
# print(f"False Negatives (FN): {FN}")
# print(f"True Negatives (TN): {TN}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1-Score: {f1_score:.4f}")
# print(f"Accuracy: {accuracy:.4f}")


import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree

def evaluate_loop_closures(predictions_path, ground_truth_path, threshold_distance=0.5):
    """
    Improved evaluation with:
    - KDTree-based timestamp matching
    - Accurate TN calculation
    - Temporal consistency verification
    """
    # Load data
    predictions_df = pd.read_csv(predictions_path)
    ground_truth_df = pd.read_csv(ground_truth_path)

    # Build KDTree for efficient timestamp matching
    gt_timestamps = ground_truth_df['Bag7_Timestamp'].values.reshape(-1, 1)
    gt_tree = KDTree(gt_timestamps)
    
    # Initialize counters
    TP = FP = FN = TN = 0
    TIME_TOLERANCE = 1  # seconds for timestamp matching

    # Temporal consistency check (require N consecutive detections)
    N_CONSECUTIVE = 2
    predictions_df['Confirmed_Prediction'] = 0
    
    for i in range(len(predictions_df)):
        if i >= N_CONSECUTIVE - 1:
            window = predictions_df.iloc[i-N_CONSECUTIVE+1:i+1]
            if all(window['Predicted_LoopClosure'] == 1):
                predictions_df.at[i, 'Confirmed_Prediction'] = 1

    # Main evaluation loop
    for _, pred_row in predictions_df.iterrows():
        pred_time = pred_row['Bag7_Timestamp']
        is_predicted = pred_row['Confirmed_Prediction'] == 1
        
        # Find nearest ground truth match
        dist, idx = gt_tree.query([[pred_time]], k=1)
        has_gt_match = (dist <= TIME_TOLERANCE)[0]
        
        if has_gt_match:
            gt_match = ground_truth_df.iloc[idx[0]]
            # Position verification
            pred_pos = np.array([pred_row['Bag3_PosX'], pred_row['Bag3_PosY'], pred_row['Bag3_PosZ']])
            gt_pos = np.array([gt_match['Bag3_PosX'], gt_match['Bag3_PosY'], gt_match['Bag3_PosZ']])
            pos_distance = np.linalg.norm(pred_pos - gt_pos)
            
            if is_predicted:
                if pos_distance <= threshold_distance:
                    TP += 1
                else:
                    FP += 1
            else:
                if pos_distance <= threshold_distance:
                    FN += 1
                else:
                    TN += 1
        else:
            if is_predicted:
                FP += 1
            else:
                # Only count as TN if no GT exists for this timestamp
                if pred_time not in ground_truth_df['Bag7_Timestamp'].values:
                    TN += 1

    # Calculate metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    return {
        'True_Positives': TP,
        'False_Positives': FP,
        'False_Negatives': FN,
        'True_Negatives': TN,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1_score,
        'Accuracy': accuracy
    }

if __name__ == "__main__":
    # Path configuration
    PREDICTIONS_FILE = 'predictions_with_bag3.csv'
    GROUND_TRUTH_FILE = '/home/agrima/Desktop/new-approach/ground_truth_loop_closures_thresh_0.60m.csv'
    
    # Run evaluation
    metrics = evaluate_loop_closures(PREDICTIONS_FILE, GROUND_TRUTH_FILE)
    
    # Print results
    print("\nImproved Evaluation Results:")
    print(f"True Positives (TP): {metrics['True_Positives']}")
    print(f"False Positives (FP): {metrics['False_Positives']}")
    print(f"False Negatives (FN): {metrics['False_Negatives']}")
    print(f"True Negatives (TN): {metrics['True_Negatives']}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1-Score: {metrics['F1_Score']:.4f}")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    