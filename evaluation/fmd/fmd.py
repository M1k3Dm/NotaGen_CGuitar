import os
import numpy as np
from scipy.linalg import sqrtm

gt_feature_folder = '' # ground truth
output_feature_folder = '' # generations

def load_features_from_folder(folder_path):
    """
    Load all .npy files from the given folder and return
    a single numpy array of shape (n_samples, embedding_dim).
    """
    feats = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith('.npy'):
            path = os.path.join(folder_path, fname)
            feats.append(np.load(path))
    if not feats:
        raise ValueError(f'No .npy files found in folder {folder_path}')
    return np.vstack(feats)


def calculate_frechet_distance(real_feats, gen_feats, eps=1e-6):
    """
    Compute the Frechet Distance between two distributions:
    FMD = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2 * sqrt(Sigma_r * Sigma_g))
    where mu are the means and Sigma the covariance matrices.
    """
    # Compute means
    mu_r = np.mean(real_feats, axis=0)
    mu_g = np.mean(gen_feats,  axis=0)
    # Compute covariances
    sigma_r = np.cov(real_feats, rowvar=False)
    sigma_g = np.cov(gen_feats,  rowvar=False)
    # Difference of means
    diff = mu_r - mu_g

    # Matrix square root of product of covariances
    covmean, _ = sqrtm(sigma_r.dot(sigma_g), disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Final Frechet Music Distance calculation
    fmd = diff.dot(diff) + np.trace(sigma_r + sigma_g - 2 * covmean)
    return np.real(fmd)


if __name__ == '__main__':
    real_features = load_features_from_folder(gt_feature_folder)
    gen_features  = load_features_from_folder(output_feature_folder)
    fmd_value = calculate_frechet_distance(real_features, gen_features)
    print(f'Frechet Music Distance (FMD): {fmd_value:.4f}')
