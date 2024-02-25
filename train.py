import os

import numpy as np
from scipy.stats import pearsonr as corr
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

from arguments import Arguments
from dataset import ImageDataset


def fit_pca(feature_extractor, data_loader, batch_size=200):
    # Define PCA parameters
    pca = IncrementalPCA(n_components=100, batch_size=batch_size)

    # Fit PCA to batch
    for _, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        # Extract features
        ft = feature_extractor(d)
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        # Fit PCA to batch
        pca.partial_fit(ft.detach().cpu().numpy())
    return pca


def extract_features(feature_extractor, data_loader, pca):
    features = []
    for _, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        # Extract features
        ft = feature_extractor(d)
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        # Apply PCA transform
        ft = pca.transform(ft.cpu().detach().numpy())
        features.append(ft)
    return np.vstack(features)


def train(args: Arguments):
    img_dir = os.path.join(args.data_dir, "training_split", "training_images")
    num_images = len(os.listdir(img_dir))
    num_val = int(np.round(num_images * args.val_split / 100.0))
    idxs = np.arange(num_images)
    val_idx, train_idx = idxs[:num_val], idxs[num_val:]

    train_dataset = ImageDataset(args, train_idx)
    val_dataset = ImageDataset(args, val_idx)
    print(f"{len(train_dataset) = }, {len(val_dataset) = }")

    train_dataloader = DataLoader(train_dataset, batch_size=128)
    val_dataloader = DataLoader(val_dataset, batch_size=128)

    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet')
    model.to(device=args.device)
    model.eval()

    model_layer = 'features.2'
    feature_extractor = create_feature_extractor(model, return_nodes=[model_layer])

    pca = fit_pca(feature_extractor, train_dataloader)
    train_features = extract_features(feature_extractor, train_dataloader, pca)
    val_features = extract_features(feature_extractor, val_dataloader, pca)

    del model, pca

    fmri_dir = os.path.join(args.data_dir, "training_split", "training_fmri")
    print("Loading left hemisphere training data...")
    lh_fmri = np.load(os.path.join(fmri_dir, "lh_training_fmri.npy"))
    print(
        f"LH data shape (Training stimulus images x LH voxels): {lh_fmri.shape}"
    )

    print("Loading right hemisphere training data...")
    rh_fmri = np.load(os.path.join(fmri_dir, "rh_training_fmri.npy"))
    print(
        f"RH data shape (Training stimulus images x RH voxels): {rh_fmri.shape}"
    )

    reg_lh = LinearRegression().fit(train_features, lh_fmri[train_idx])
    reg_rh = LinearRegression().fit(train_features, rh_fmri[train_idx])

    lh_fmri_val_pred = reg_lh.predict(val_features)
    rh_fmri_val_pred = reg_rh.predict(val_features)

    # Empty correlation array of shape: (LH vertices)
    lh_correlation = np.zeros(lh_fmri_val_pred.shape[1])
    # Correlate each predicted LH vertex with the corresponding ground truth vertex
    for v in tqdm(range(lh_fmri_val_pred.shape[1])):
        lh_correlation[v] = corr(lh_fmri_val_pred[:,v], lh_fmri[val_idx][:,v])[0]

    # Empty correlation array of shape: (RH vertices)
    rh_correlation = np.zeros(rh_fmri_val_pred.shape[1])
    # Correlate each predicted RH vertex with the corresponding ground truth vertex
    for v in tqdm(range(rh_fmri_val_pred.shape[1])):
        rh_correlation[v] = corr(rh_fmri_val_pred[:,v], rh_fmri[val_idx][:,v])[0]

    # print(f"Left mean corr = {np.mean(lh_correlation)}")
    print(f"Right mean corr = {np.mean(rh_correlation)}")


if __name__ == '__main__':
    args = Arguments(1, '../algonauts_2023_challenge_data', 0.1)
    train(args)
