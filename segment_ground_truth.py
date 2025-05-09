import pandas as pd

def segment_loop_closures(matches_df, min_cluster_size=5, max_time_gap=1.0):
    """
    Segment the point-wise loop closure matches into temporally coherent loop closure segments.

    Args:
        matches_df: DataFrame with columns ['Bag7_Timestamp', 'Bag3_Closest_Timestamp']
        min_cluster_size: Minimum number of matches to consider a valid loop closure segment
        max_time_gap: Maximum allowed time gap (in seconds) between consecutive matches within a segment

    Returns:
        List of segments: [(bag7_start, bag7_end, bag3_start, bag3_end)]
    """
    matches_sorted = matches_df.sort_values('Bag7_Timestamp').reset_index(drop=True)
    segments = []
    start_idx = 0

    for i in range(1, len(matches_sorted)):
        dt_bag7 = matches_sorted.loc[i, 'Bag7_Timestamp'] - matches_sorted.loc[i - 1, 'Bag7_Timestamp']
        dt_bag3 = matches_sorted.loc[i, 'Bag3_Closest_Timestamp'] - matches_sorted.loc[i - 1, 'Bag3_Closest_Timestamp']

        # If time jumps in either Bag 7 or Bag 3, start a new cluster
        if dt_bag7 > max_time_gap or abs(dt_bag3) > max_time_gap:
            if i - start_idx >= min_cluster_size:
                segment = (
                    matches_sorted.loc[start_idx, 'Bag7_Timestamp'],
                    matches_sorted.loc[i - 1, 'Bag7_Timestamp'],
                    matches_sorted.loc[start_idx, 'Bag3_Closest_Timestamp'],
                    matches_sorted.loc[i - 1, 'Bag3_Closest_Timestamp']
                )
                segments.append(segment)
            start_idx = i

    # Final segment
    if len(matches_sorted) - start_idx >= min_cluster_size:
        segment = (
            matches_sorted.loc[start_idx, 'Bag7_Timestamp'],
            matches_sorted.loc[len(matches_sorted) - 1, 'Bag7_Timestamp'],
            matches_sorted.loc[start_idx, 'Bag3_Closest_Timestamp'],
            matches_sorted.loc[len(matches_sorted) - 1, 'Bag3_Closest_Timestamp']
        )
        segments.append(segment)

    return segments

# === Load the point-level loop closure matches ===
matches_df = pd.read_csv('ground_truth_lc_bag-3v7.csv')

# === Run clustering ===
segments = segment_loop_closures(matches_df)

# === Save to CSV ===
seg_df = pd.DataFrame(segments, columns=['Bag7_Start', 'Bag7_End', 'Bag3_Start', 'Bag3_End'])
seg_df.to_csv('ground_truth_segments_strict.csv', index=False)

print(f"✅ Found {len(seg_df)} high-quality loop closure segments.")
