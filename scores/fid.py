import numpy as np
import os
import scipy.linalg
import torch
from scores.inception import build_inception_model
from torchvision import transforms
from PIL import Image


def compute_fid(fake_features, real_features):
    """Computes FID based on the features extracted from fake and real data.

    Given the mean and covariance (m_f, C_f) of fake data and (m_r, C_r) of real
    data, the FID metric can be computed by

    d^2 = ||m_f - m_r||_2^2 + Tr(C_f + C_r - 2(C_f C_r)^0.5)

    Args:
        fake_features: The features extracted from fake data.
        real_features: The features extracted from real data.

    Returns:
        A real number, suggesting the FID value.
    """

    m_f = np.mean(fake_features, axis=0)
    C_f = np.cov(fake_features, rowvar=False)
    m_r = np.mean(real_features, axis=0)
    C_r = np.cov(real_features, rowvar=False)

    fid = np.sum((m_f - m_r) ** 2) + np.trace(C_f + C_r - 2 * scipy.linalg.sqrtm(np.dot(C_f, C_r)))
    return np.real(fid)


device = torch.device("cuda")

inception_model = build_inception_model(align_tf=True, transform_input=True).to(device).eval().requires_grad_(False)

transform = transforms.Compose(
    [
        transforms.ToTensor(),  # Converts the image to a PyTorch tensor
    ]
)

real_features = []
fake_features = []

real_dir = ""
fake_dir = ""


for img_path in os.listdir(real_dir):
    img = Image.open(os.path.join(real_dir, img_path)).convert("RGB")  # Ensure the image is in RGB format
    img_tensor = transform(img).unsqueeze(0)
    img_tensor = img_tensor.to(device)

    real_features.append(inception_model(img_tensor).cpu().numpy())

for img_path in os.listdir(fake_dir):
    img = Image.open(os.path.join(fake_dir, img_path)).convert("RGB")  # Ensure the image is in RGB format
    img_tensor = transform(img).unsqueeze(0)
    img_tensor = img_tensor.to(device)

    fake_features.append(inception_model(img_tensor).cpu().numpy())

print(np.array(real_features).shape)
print(np.array(fake_features).shape)

real_features = np.array(real_features)[:100].reshape(-1, 2048)
fake_features = np.array(fake_features)[:100].reshape(-1, 2048)

fid = compute_fid(fake_features, real_features)

print("FID:", fid)
