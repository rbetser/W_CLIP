import torch
import os
import matplotlib.pyplot as plt
from PIL import Image
import clip  # Install with: pip install git+https://github.com/openai/CLIP.git

# -----------------------------------
# Setup Device and Load CLIP Model
# -----------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model and preprocessing pipeline
model, preprocess = clip.load("ViT-L/14", device=device)

# -----------------------------------
# Load Precomputed Mean and Whitening Matrix
# -----------------------------------
mean_path = 'w_mats/mean_image_L14.pt'
w_mat_path = 'w_mats/w_mat_image_L14.pt'

# Load mean feature vector and whitening matrix
mean_features = torch.load(mean_path, map_location=device, weights_only=False)
w_mat = torch.load(w_mat_path, map_location=device, weights_only=False)

# Number of features (dimensionality)
N = len(mean_features)

# -----------------------------------
# Load Images from Directory
# -----------------------------------
image_dir_path = 'images_hands'  # Directory containing images
image_names = os.listdir(image_dir_path)  # List all image filenames
images_path = [os.path.join(image_dir_path, image) for image in image_names]  # Construct full paths

# -----------------------------------
# Process Images and Compute Log-Likelihood
# -----------------------------------
for image_path in images_path:
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # Extract CLIP features
    image_features = model.encode_image(image_tensor).squeeze(0).to(device, dtype=torch.float32)

    # Center the features and apply whitening transformation
    cntr_features = image_features - mean_features
    w_features = torch.matmul(cntr_features, w_mat)

    # Compute log-likelihood
    log_like = -0.5 * (N * torch.log(torch.tensor(2 * torch.pi, device=device)) + torch.sum(w_features**2))

    # Print log-likelihood
    print(f"Log-Likelihood for {os.path.basename(image_path)}: {log_like.item():.2f}")


