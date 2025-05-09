import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import ot



def select_median_reference(segments):
    """Select segment with size closest to median size."""
    if not segments:
        raise ValueError("Segments list is empty.")
    sizes = [seg.shape[0] for seg in segments]
    median_size = np.median(sizes)
    ref_idx = np.argmin([abs(len(seg) - median_size) for seg in segments])
    return segments[ref_idx]

def lgw_embedding(X, Y, p=None, q=None, lambdaa=0.5, partial=True):
    """Compute LPGW embedding with optional partial matching.
    
    Args:
        X: Source point cloud (n x d)
        Y: Target point cloud (m x d)
        p: Source mass distribution (n,), defaults to uniform
        q: Target mass distribution (m,), defaults to uniform
        lambdaa: Regularization parameter for PGW
        partial: If True, use partial GW; else, use standard GW
    
    Returns:
        k_gamma: Embedding matrix (n x n)
        q_tilde: Transported mass distribution (n,)
        gamma_c_abs: Absolute value of mass creation term
        transported_mass: Total transported mass
    """
    if X.shape[0] == 0 or Y.shape[0] == 0:
        raise ValueError("Input point clouds cannot be empty.")
    
    # Normalize point clouds
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    Y = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-8)

    # Distance matrices (squared Euclidean)
    D_X = cdist(X, X, metric='euclidean') ** 2
    D_Y = cdist(Y, Y, metric='euclidean') ** 2

    # Normalize distances
    D_X /= D_X.max() + 1e-8
    D_Y /= D_Y.max() + 1e-8

    # Mass distributions
    n, m = X.shape[0], Y.shape[0]
    p = np.ones(n) / n if p is None else np.array(p) / np.sum(p)
    q = np.ones(m) / m if q is None else np.array(q) / np.sum(q)

    if p.shape[0] != n or q.shape[0] != m:
        raise ValueError("Mass distributions must match point cloud sizes.")
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Mass distributions must be non-negative.")

    if partial:
        m = min(p.sum(), q.sum())  # Partial matching mass
        try:
            gamma = ot.gromov.partial_gromov_wasserstein(
                D_X, D_Y, p, q, loss_fun='square_loss', m=m, reg_m=lambdaa
            )
        except Exception as e:
            raise RuntimeError(f"Failed to compute partial GW: {e}")
    else:
        try:
            gamma = ot.gromov.gromov_wasserstein(D_X, D_Y, p, q, 'square_loss')
        except Exception as e:
            raise RuntimeError(f"Failed to compute GW: {e}")

    if np.any(np.isnan(gamma)) or np.any(gamma < 0):
        raise ValueError("Invalid transport plan computed.")

    # Barycentric projection
    q_tilde = gamma.sum(axis=1)  # Marginal on X
    T_gamma = np.zeros_like(X)
    valid = q_tilde > 1e-8
    T_gamma[valid] = (gamma[valid] @ Y) / (q_tilde[valid, None] + 1e-8)
    T_gamma[~valid] = 0  # Set to zero if no mass transported

    # Projected distance matrix
    D_Y_proj = cdist(T_gamma, T_gamma, metric='euclidean') ** 2
    D_Y_proj /= D_Y_proj.max() + 1e-8

    # LPGW embedding
    k_gamma = D_X - D_Y_proj

    # Compute |gamma_c| = (sum p_i)^2 - (sum q_tilde_i)^2
    gamma_c_abs = p.sum()**2 - q_tilde.sum()**2
    transported_mass = q_tilde.sum()

    return k_gamma, q_tilde, gamma_c_abs, transported_mass

def lgw_distance(K1, K2, q1, q2, nu1_mass, nu2_mass, gamma_c1_abs, gamma_c2_abs, lambdaa=0.5, partial=True):
    """Compute LPGW distance with mass penalty (Equation 25).
    
    Args:
        K1, K2: Embedding matrices (n x n)
        q1, q2: Transported mass distributions (n,)
        nu1_mass, nu2_mass: Total masses of nu1, nu2
        gamma_c1_abs, gamma_c2_abs: Mass creation terms
        lambdaa: Regularization parameter
        partial: If True, include mass penalty
    
    Returns:
        Distance value
    """
    if K1.shape != K2.shape or q1.shape != q2.shape:
        raise ValueError("Embedding matrices and mass distributions must have same shape.")
    
    # Check lambda condition: 2*lambda >= max |g_Y1 - g_Y2|^2
    # For normalized distances, max |g_Y1 - g_Y2|^2 <= 1
    if partial and 2 * lambdaa < 1:
        print(f"Warning: 2*lambda ({2*lambdaa}) < 1; may not satisfy Equation 17.")

    # Compute q12 = q1 ∧ q2
    q12 = np.minimum(q1, q2)
    q12_sum = q12.sum()
    if q12_sum < 1e-8:
        raise ValueError("Intersection of transported masses is near zero.")
    q12_norm = q12 / q12_sum  # Normalize for discrete measure

    # Weighted L2 norm: sum((K1 - K2)^2 * q12_norm * q12_norm')
    weighted_diff = (K1 - K2)**2 * q12_norm[:, None] * q12_norm[None, :]
    embedding_term = np.sum(weighted_diff)

    if partial:
        # Mass penalty: lambda * (|nu1|^2 + |nu2|^2 - 2 |q12|^2)
        mu_prime_mass = q12_sum
        mass_penalty = lambdaa * (nu1_mass**2 + nu2_mass**2 - 2 * mu_prime_mass**2)
        return embedding_term + mass_penalty
    else:
        return embedding_term

def compute_lgw_distance_matrix(segments_7, segments_3, lambdaa=0.5, partial=True):
    """Compute pairwise LPGW distance matrix."""
    if not segments_7 or not segments_3:
        raise ValueError("Segment lists cannot be empty.")

    # Use median-sized segment as fixed reference
    ref_segment = select_median_reference(segments_3)

    # Precompute embeddings and masses for Bag7 segments
    lgw_7 = []
    masses_7 = []
    q_tildes_7 = []
    gamma_c_abs_7 = []
    for seg7 in segments_7:
        k_gamma, q_tilde, gamma_c_abs, mass = lgw_embedding(ref_segment, seg7, lambdaa=lambdaa, partial=partial)
        lgw_7.append(k_gamma)
        q_tildes_7.append(q_tilde)
        gamma_c_abs_7.append(gamma_c_abs)
        masses_7.append(mass)

    # Precompute embeddings and masses for Bag3 segments
    lgw_3 = []
    masses_3 = []
    q_tildes_3 = []
    gamma_c_abs_3 = []
    for seg3 in segments_3:
        k_gamma, q_tilde, gamma_c_abs, mass = lgw_embedding(ref_segment, seg3, lambdaa=lambdaa, partial=partial)
        lgw_3.append(k_gamma)
        q_tildes_3.append(q_tilde)
        gamma_c_abs_3.append(gamma_c_abs)
        masses_3.append(mass)

    # Compute distance matrix
    D = np.zeros((len(segments_7), len(segments_3)))
    for i, (k7, q7, nu1, gc1) in enumerate(zip(lgw_7, q_tildes_7, masses_7, gamma_c_abs_7)):
        for j, (k3, q3, nu2, gc2) in enumerate(zip(lgw_3, q_tildes_3, masses_3, gamma_c_abs_3)):
            D[i, j] = lgw_distance(k7, k3, q7, q3, nu1, nu2, gc1, gc2, lambdaa, partial)
    return D


def load_xyz_trajectory(csv_path):
    """Load trajectory from CSV file."""
    df = pd.read_csv(csv_path)
    if not {'PosX', 'PosY', 'PosZ', 'Timestamp'}.issubset(df.columns):
        raise ValueError("CSV must contain PosX, PosY, PosZ, and Timestamp columns.")
    return df[['PosX', 'PosY', 'PosZ']].values, df['Timestamp'].values

def segment_trajectory(traj, segment_length=3.0, fps=10, stride=1.0):
    """Segment trajectory into overlapping windows."""
    if len(traj) < 1:
        raise ValueError("Trajectory cannot be empty.")
    num_points = int(segment_length * fps)
    stride_pts = int(stride * fps)
    segments = []
    for start in range(0, len(traj) - num_points + 1, stride_pts):
        segment = traj[start:start+num_points]
        if segment.shape[0] == num_points:
            segments.append(np.array(segment))
    return segments

def compute_barycenter(segments):
    """Compute GW barycenter of segments."""
    if not segments:
        raise ValueError("Segments list is empty.")
    Ds = [cdist(seg, seg, metric='euclidean') ** 2 for seg in segments]
    try:
        barycenter = ot.gromov.barycenter(
            Ds,
            weights=[np.ones(seg.shape[0]) / seg.shape[0] for seg in segments],
            loss_fun='square_loss'
        )
        return barycenter
    except Exception as e:
        raise RuntimeError(f"Failed to compute barycenter: {e}")


def downsample_trajectory(traj, target_points=12000):
    """Downsample trajectory to approximately target_points uniformly."""
    if traj.shape[0] <= target_points:
        return traj
    indices = np.linspace(0, traj.shape[0] - 1, target_points).astype(int)
    return traj[indices]

def fps_downsample(points, target_points=5000):
    chosen = [np.random.randint(len(points))]
    for _ in range(target_points - 1):
        dists = np.min(cdist(points, points[chosen]), axis=1)
        next_idx = np.argmax(dists)
        chosen.append(next_idx)
    return points[sorted(chosen)]



# Load full trajectories
bag3_xyz, bag3_timestamps = load_xyz_trajectory('poses/poses_bag-3.csv')
bag7_xyz, bag7_timestamps = load_xyz_trajectory('poses/poses_bag-7.csv')

# 🔽 Downsample to 1000 points each
bag3_xyz = downsample_trajectory(bag3_xyz)
bag7_xyz = downsample_trajectory(bag7_xyz)

# Segment trajectories
segments_3 = segment_trajectory(bag3_xyz, segment_length=3.0, fps=10)
segments_7 = segment_trajectory(bag7_xyz, segment_length=3.0, fps=10)

# Adjust segment count to match
min_len = min(len(segments_7), len(segments_3))
segments_7 = segments_7[:min_len]
segments_3 = segments_3[:min_len]

# Compute LPGW distance matrix
print("Computing LPGW distance matrix...")
D = compute_lgw_distance_matrix(segments_7, segments_3, lambdaa=0.5, partial=True)

import matplotlib.pyplot as plt

# Visualizing the LPGW Distance Matrix
plt.imshow(D, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('LPGW Distance Matrix')
plt.xlabel('Bag3 Segments')
plt.ylabel('Bag7 Segments')
plt.show()

# Detect loop closures using dynamic threshold (5th percentile)
threshold = np.percentile(D, 1) 
loop_closure_flags = []
best_matches = []



for i in range(D.shape[0]):
    min_j = np.argmin(D[i])
    loop_closure_flags.append(int(D[i, min_j] < threshold))
    best_matches.append(min_j)



# Generate timestamps for Bag7 segments
segment_length = 3.0
fps = 10
num_points = int(segment_length * fps)
mid_indices = [start + num_points // 2 for start in range(0, len(bag7_xyz) - num_points + 1, int(1.0 * fps))]
segment_timestamps = bag7_timestamps[mid_indices][:len(loop_closure_flags)]

# Save predictions
predictions = pd.DataFrame({
    'Bag7_Timestamp': segment_timestamps,
    'Predicted_LoopClosure': loop_closure_flags,
    'BestMatch_Index_Bag3': best_matches,
    'Min_Distance': [D[i, j] for i, j in zip(range(len(best_matches)), best_matches)]
})
predictions.to_csv('predictions.csv', index=False)
print("Predictions saved to 'predictions.csv'")
np.save('lpgw_distance_matrix.npy', D)
import matplotlib.pyplot as plt
plt.imshow(D, cmap='hot', aspect='auto')
plt.colorbar()
plt.title("Loop Closure Distance Matrix")
plt.xlabel("Bag3 Segments")
plt.ylabel("Bag7 Segments")
# plt.show()