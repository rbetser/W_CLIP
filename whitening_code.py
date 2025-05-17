import torch
import matplotlib.pyplot as plt
import os
from PIL import Image
import clip  # Install with: pip install git+https://github.com/openai/CLIP.git


# -----------------------------------
# Function Definitions
# -----------------------------------

def white_transform(x: torch.Tensor) -> tuple:
    """
    Performs whitening transformation on the input tensor.

    Whitening transformation ensures that the transformed features have zero mean
    and unit covariance, reducing redundancy in the data.

    Args:
        x (torch.Tensor): Input feature matrix of shape (N, D), where N is the number of samples
                          and D is the number of features.

    Returns:
        tuple:
            - w_x (torch.Tensor): Whitened feature matrix.
            - w_mat (torch.Tensor): Whitening transformation matrix.
    """
    # Compute the mean of the dataset
    x_mean = x.mean(dim=0)
    xu = x - x_mean  # Centering the data

    # Compute covariance matrix
    Cov_matrix = torch.cov(xu.T)

    # Eigen decomposition of the covariance matrix
    eigenvalues_complex, eigenvectors = torch.linalg.eig(Cov_matrix)
    eigenvalues = eigenvalues_complex.real  # Convert complex eigenvalues to real
    eigenvalues_sorted, indices = torch.sort(eigenvalues, descending=True)

    # Compute the diagonal eigenvalue matrix and its inverse square root
    S = torch.diag(eigenvalues_sorted)
    S_inv_sqrt = torch.sqrt(torch.inverse(S))

    # Sort eigenvectors accordingly
    V = eigenvectors.real
    V = V[:, indices]

    # Compute whitening matrix
    w_mat = torch.matmul(V, S_inv_sqrt)
    w_x = torch.matmul(xu, w_mat)  # Apply whitening transformation

    return w_x, w_mat


def detect_high_corr_features(features: torch.Tensor, threshold: float = 0.9) -> tuple:
    """
    Identifies highly correlated features in a dataset based on a threshold.

    Args:
        features (torch.Tensor): Feature matrix of shape (N, D), where N is the number of samples
                                 and D is the number of features.
        threshold (float, optional): Correlation threshold for detecting redundant features. Defaults to 0.9.

    Returns:
        tuple:
            - high_corr (list): Indices of features to be removed.
            - high_corr_sim (list): Indices of features that they are highly correlated with.
    """
    high_corr = []  # Indices of features to be removed
    high_corr_sim = []  # Indices of features they are similar to

    # Compute correlation matrix
    corr_matrix = torch.corrcoef(features.T)

    # Iterate through features
    for i in range(features.shape[1]):
        # Find indices of features with correlation > threshold
        ind = torch.where(corr_matrix[i, i + 1:] > threshold)[0]
        if len(ind) > 0:
            for idx in ind:
                high_corr.append(i + 1 + idx)  # Adjusted index
                high_corr_sim.append(i)  # Original feature index

    # Remove duplicates
    pairs = list(zip(high_corr, high_corr_sim))
    if pairs:
        unique_pairs = list(set(pairs))
        unique_high_corr, unique_high_corr_sim = zip(*unique_pairs)
        return list(unique_high_corr), list(unique_high_corr_sim)

    return [], []


def plot_cov_matrix(data: torch.Tensor, title: str, plots_dir: str):
    """
    Plots the covariance matrix of a dataset.

    Args:
        data (torch.Tensor): Input feature matrix.
        title (str): Title of the plot.
        plots_dir (str): Directory where the plot will be saved.

    Returns:
        None
    """
    # Compute covariance matrix
    cov_matrix = torch.cov(data.T)

    # Plot covariance matrix
    plt.figure(figsize=(10, 6))
    plt.imshow(cov_matrix.numpy(), cmap='coolwarm', vmin=0, vmax=1)
    plt.colorbar()

    # Save plot
    plt.savefig(os.path.join(plots_dir, f'{title}.png'))
    plt.close()


# -----------------------------------
# Setup Device and Load CLIP Model
# -----------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CLIP model and its preprocessing pipeline
model, preprocess = clip.load("ViT-L/14", device=device)

# -----------------------------------
# File Paths
# -----------------------------------
w_name = 'w_mats/w_mat_test.pt'  # Path to save the whitening matrix
mean_name = 'w_mats/mean_test.pt'  # Path to save the mean features
plots_dir = 'w_mats'  # Directory to save covariance matrix plots

# -----------------------------------
# Load Image Dataset and Extract Features
# -----------------------------------
image_dir_path = '../MSCOCO/Images'  # Directory containing images
image_names = os.listdir(image_dir_path)  # List all images
images_path = [os.path.join(image_dir_path, image) for image in image_names]  # Construct full paths

# Extract features using CLIP
all_image_features = []
for image_path in images_path:
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    image_features = model.encode_image(image).squeeze(0).to(device, dtype=torch.float32)
    all_image_features.append(image_features.detach().cpu())

# Stack extracted features into a single tensor
all_image_features = torch.stack(all_image_features)

# Compute the mean of all image features
mean_features = all_image_features.mean(dim=0)
torch.save(mean_features, mean_name)  # Save mean features

# -----------------------------------
# Detect High-Correlation Features and Replace Them
# -----------------------------------
F_i = all_image_features - mean_features  # Center features
N, k = F_i.shape  # Number of samples (N) and feature dimensions (k)

# Detect highly correlated features
high_corr_i, high_corr_sim_i = detect_high_corr_features(F_i)

# Replace high-correlation features with Gaussian noise
for i in high_corr_i:
    F_i[:, i] = 0.1 * torch.randn(N)

# -----------------------------------
# Apply Whitening Transformation
# -----------------------------------
F_s_i, w_mat = white_transform(F_i)  # Whiten features
torch.save(w_mat, w_name)  # Save whitening matrix

# -----------------------------------
# Plot and Save Covariance Matrices
# -----------------------------------
plot_cov_matrix(F_s_i, "Whitened Image Covariance", plots_dir)
plot_cov_matrix(all_image_features, "Original Image Covariance", plots_dir)
