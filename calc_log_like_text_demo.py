import torch
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
mean_path = 'w_mats/mean_text_L14.pt'
w_mat_path = 'w_mats/w_mat_text_L14.pt'

# Load mean feature vector and whitening matrix
mean_features = torch.load(mean_path, map_location=device, weights_only=False)
w_mat = torch.load(w_mat_path, map_location=device, weights_only=False)

# Number of features (dimensionality)
N = len(mean_features)

# -----------------------------------
# Define Captions for Log-Likelihood Computation
# -----------------------------------
captions = [
    "Donald Duck talking to Minney Mouse at Disney World at Christmas.",
    "Donald Duck talking to a Mouse at Disney World at Christmas.",
    "A Duck talking to Minney Mouse at Disney World at Christmas.",
    "Donald Duck talking to Minney Mouse at Disney World.",
    "Donald Duck talking to Minney Mouse at Christmas."
]

# Uncomment and modify the following captions for different experiments
# captions = [
#     "A woman points a hair drier like it is a gun.",
#     "An old woman points a hair drier like it is a gun.",
#     "A bride points a hair drier like it is a gun.",
#     "Jenny points a hair drier like it is a gun."
# ]

# -----------------------------------
# Process Each Caption and Compute Log-Likelihood
# -----------------------------------
for caption in captions:
    # Tokenize the caption and move it to the correct device
    text = clip.tokenize(caption).to(device)

    # Encode the text into feature space
    text_features = model.encode_text(text).to(device, dtype=torch.float32)

    # Center and transform features using the whitening matrix
    cntr_features = text_features - mean_features
    w_features = torch.matmul(cntr_features, w_mat)

    # Compute log-likelihood using Gaussian distribution assumption
    log_like = -0.5 * (N * torch.log(torch.tensor(2 * torch.pi, device=device)) + torch.sum(w_features ** 2))

    # Print log-likelihood
    print(f'\"{caption}\" - Log Likelihood: {log_like.item():.2f}')
