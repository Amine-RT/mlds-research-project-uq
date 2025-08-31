
# ==============================================================================
#                 SIMULATION :  DATA GENERATION FUNCTION
# ==============================================================================
import numpy as np
import torch

def generate_toy_data_v1(N_points=300, noise_std_smooth=0.05, noise_std_discont=0.3, test_split=0.25):
    """
    Generates 1D toy data with smooth regions of varying noise and density and includes a train/test split.
    """
    n_region_1_perc = 0.45 # % data in Region 1
    n_region_2_perc = 0.1  # % data in Region 2

    # Calculate points for each region based on percentages, ensuring total N_points
    n_region1 = int(N_points * n_region_1_perc)
    n_region2 = int(N_points * n_region_2_perc)
    n_region3 = N_points - n_region1 - n_region2


    # Generate data points in Region 1 (Dense, low noise)
    x_region1 = np.random.uniform(-6, -2, n_region1).astype(np.float32)
    y_region1 = np.sin(x_region1) + np.random.normal(0, noise_std_smooth, n_region1)

    # Generate data points in Region 2 (Sparse, high noise, linear interpolation)
    x_region2 = np.random.uniform(-2, 2, n_region2).astype(np.float32) # Reverted Region 2 length
    # The deterministic part is a straight line connecting sin(-2) and cos(2-4)
    y_region2 = np.sin(-2) + (np.cos(2 - 4) - np.sin(-2)) / (2 - (-2)) * (x_region2 - (-2)) + np.random.normal(0, noise_std_discont, n_region2)

    # Generate data points in Region 3 (Dense, low noise)
    x_region3 = np.random.uniform(2, 10, n_region3).astype(np.float32) # Extended Region 3 to 10
    y_region3 = np.cos(x_region3 - 4) + np.random.normal(0, noise_std_smooth, n_region3)


    # Combine data, sort by x, and convert to tensors
    x_data_combined = np.concatenate([x_region1, x_region2, x_region3])
    y_data_combined = np.concatenate([y_region1, y_region2, y_region3])
    sort_indices = np.argsort(x_data_combined)
    x_data_sorted = x_data_combined[sort_indices]
    y_data_sorted = y_data_combined[sort_indices]
    x_tensor = torch.from_numpy(x_data_sorted).unsqueeze(1)
    y_tensor = torch.from_numpy(y_data_sorted).unsqueeze(1)

    # Shuffle indices and split consistently
    indices = np.arange(N_points)
    np.random.shuffle(indices)
    split_idx = int(N_points * (1 - test_split))

    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    x_train = x_tensor[train_indices]
    y_train = y_tensor[train_indices]
    x_test = x_tensor[test_indices]
    y_test = y_tensor[test_indices]

    return x_train, y_train, x_test, y_test


def generate_toy_data_v2(N_points=200, noise_std_smooth=0.05, noise_std_discont=0.3, test_split=0.25):
    """
    Generates 1D toy data with regions of varying predictability.
    """
    x_data = np.sort(np.random.uniform(-6, 10, N_points)).astype(np.float32)
    y_data = np.zeros_like(x_data)

    # Region 1: Smooth sin wave, low noise
    mask1 = x_data <= -2
    y_data[mask1] = np.sin(x_data[mask1]) + np.random.normal(0, noise_std_smooth, np.sum(mask1))

    # Region 2: Discontinuity, higher noise
    mask2 = (x_data > -2) & (x_data < 2)
    y_data[mask2] = np.sign(x_data[mask2]) * 1.5 + np.random.normal(0, noise_std_discont, np.sum(mask2))
    # Ensure sign(0) is handled if x_data can be exactly 0, though uniform sampling makes it unlikely.
    # For simplicity, np.sign(0) = 0, which means y=0 at x=0 for the non-noisy part.

    # Region 3: Smooth cos wave, low noise
    mask3 = x_data >= 2
    y_data[mask3] = np.cos(x_data[mask3] - 4) + np.random.normal(0, noise_std_smooth, np.sum(mask3))

    # Convert to PyTorch tensors
    x_tensor = torch.from_numpy(x_data).unsqueeze(1) # Shape: [N_points, 1]
    y_tensor = torch.from_numpy(y_data).unsqueeze(1) # Shape: [N_points, 1]

    # Shuffle and split
    indices = np.arange(N_points)
    np.random.shuffle(indices)
    split_idx = int(N_points * (1 - test_split))

    x_train = x_tensor[indices[:split_idx]]
    y_train = y_tensor[indices[:split_idx]]
    x_test = x_tensor[indices[split_idx:]]
    y_test = y_tensor[indices[split_idx:]]

    return x_train, y_train, x_test, y_test