import os
import pickle
from argparse import ArgumentParser

import numpy as np
import torch
from scipy.stats import pearsonr as corr
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader, random_split
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

from arguments import Arguments
from dataset import ImageDataset


def fit_pca(feature_extractor, data_loader, batch_size=256):
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
    save_prefix = f"runs/{args.model}_{'_'.join(args.layers)}/{args.run_id}"
    os.makedirs(save_prefix, exist_ok=True)
    print(f"\nSaving run data to {save_prefix}\n")

    pca_path = os.path.join(save_prefix, f"subj{args.subj}_{'_'.join(args.layers)}_pca.pkl")
    train_features_path = os.path.join(save_prefix, "train_features.npy")
    val_features_path = os.path.join(save_prefix, "val_features.npy")
    train_idx_path = os.path.join(save_prefix, "train_idx.npy")
    val_idx_path = os.path.join(save_prefix, "val_idx.npy")

    # Create train/val indices based on split
    print("\n==========\nCreating train/val indices\n==========")
    if os.path.exists(train_idx_path) and os.path.exists(val_idx_path):
        train_idx = np.load(train_idx_path)
        val_idx = np.load(val_idx_path)
        print("Loaded indices from disk")
    else:
        img_dir = os.path.join(args.data_dir, "training_split", "training_images")
        num_images = len(os.listdir(img_dir))
        num_val = int(np.round(num_images / 100 * args.val_split))
        idxs = np.arange(num_images)
        val_idx, train_idx = idxs[:num_val], idxs[num_val:]

        np.save(train_idx_path, train_idx)
        np.save(val_idx_path, val_idx)

    # Load image dataset and create dataloaders
    print("\n==========\nLoading image dataset\n==========")
    train_dataset = ImageDataset(args, train_idx)
    val_dataset = ImageDataset(args, val_idx)
    print(f"{len(train_dataset) = }, {len(val_dataset) = }")
    train_dataloader = DataLoader(train_dataset, batch_size=256)
    val_dataloader = DataLoader(val_dataset, batch_size=256)

    # Load image model and set to eval mode, and create feature extractor
    print("\n==========\nLoading image model\n==========")
    model = torch.hub.load("pytorch/vision:v0.10.0", args.model, pretrained=True)
    model.to(device=args.device)
    model.eval()
    feature_extractor = create_feature_extractor(model, return_nodes=args.layers)

    # Train PCA on image model output features
    print("\n==========\nTraining PCA\n==========")
    if os.path.exists(pca_path):
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)
        print("Loaded PCA from disk")
    else:
        pca = fit_pca(feature_extractor, train_dataloader)
        with open(pca_path, "wb") as f:
            pickle.dump(pca, f)

    # Extract PCA components from image model output features
    print("\n==========\nExtracting features using PCA components\n==========")
    if os.path.exists(train_features_path) and os.path.exists(val_features_path):
        train_features = np.load(train_features_path)
        val_features = np.load(val_features_path)
        print("Loaded features from disk")
    else:
        train_features = extract_features(feature_extractor, train_dataloader, pca)
        val_features = extract_features(feature_extractor, val_dataloader, pca)
        np.save(train_features_path, train_features)
        np.save(val_features_path, val_features)

    del model, pca

    fmri_dir = os.path.join(args.data_dir, "training_split", "training_fmri")
    print("Loading left hemisphere training data...")
    lh_fmri = np.load(os.path.join(fmri_dir, "lh_training_fmri.npy"))
    print(f"LH data shape (Training stimulus images x LH voxels): {lh_fmri.shape}")

    print("Loading right hemisphere training data...")
    rh_fmri = np.load(os.path.join(fmri_dir, "rh_training_fmri.npy"))
    print(f"RH data shape (Training stimulus images x RH voxels): {rh_fmri.shape}")

    for roi in args.rois:
        print(f"\n=========\nROI: {roi}\n=========")
        if roi == "all" or roi is None:
            lh_roi_mask = np.arange(lh_fmri.shape[1])
            rh_roi_mask = np.arange(rh_fmri.shape[1])
        else:
            if roi in ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]:
                roi_class = "prf-visualrois"
            elif roi in ["EBA", "FBA-1", "FBA-2", "mTL-bodies"]:
                roi_class = "floc-bodies"
            elif roi in ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"]:
                roi_class = "floc-faces"
            elif roi in ["OPA", "PPA", "RSC"]:
                roi_class = "floc-places"
            elif roi in ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]:
                roi_class = "floc-words"
            elif roi in ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]:
                roi_class = "streams"
            else:
                raise ValueError(f"Unknown ROI {roi}")

            lh_rois = np.load(os.path.join(args.data_dir, "roi_masks", f"lh.{roi_class}_challenge_space.npy"), allow_pickle=True)
            rh_rois = np.load(os.path.join(args.data_dir, "roi_masks", f"rh.{roi_class}_challenge_space.npy"), allow_pickle=True)

            mapping = np.load(os.path.join(args.data_dir, "roi_masks", f"mapping_{roi_class}.npy"), allow_pickle=True).item()
            inverse_mapping = {name: val for val, name in mapping.items()}

            lh_roi_mask = np.where(lh_rois == inverse_mapping[roi])[0]
            rh_roi_mask = np.where(rh_rois == inverse_mapping[roi])[0]

        lh_fmri_roi_masked = lh_fmri[:, lh_roi_mask]
        rh_fmri_roi_masked = rh_fmri[:, rh_roi_mask]

        reg_lh = LinearRegression().fit(train_features, lh_fmri_roi_masked[train_idx])
        reg_rh = LinearRegression().fit(train_features, rh_fmri_roi_masked[train_idx])

        lh_fmri_val_pred = reg_lh.predict(val_features)
        rh_fmri_val_pred = reg_rh.predict(val_features)

        # Empty correlation array of shape: (LH vertices)
        lh_correlation = np.zeros(lh_fmri_val_pred.shape[1])
        # Correlate each predicted LH vertex with the corresponding ground truth vertex
        for v in tqdm(range(lh_fmri_val_pred.shape[1])):
            lh_correlation[v] = corr(lh_fmri_val_pred[:, v], lh_fmri_roi_masked[val_idx][:, v])[0]

        # Empty correlation array of shape: (RH vertices)
        rh_correlation = np.zeros(rh_fmri_val_pred.shape[1])
        # Correlate each predicted RH vertex with the corresponding ground truth vertex
        for v in tqdm(range(rh_fmri_val_pred.shape[1])):
            rh_correlation[v] = corr(rh_fmri_val_pred[:, v], rh_fmri_roi_masked[val_idx][:, v])[0]

        print(f"Left mean corr = {np.mean(lh_correlation)}")
        print(f"Right mean corr = {np.mean(rh_correlation)}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--rois', nargs='+', default=["all"])
    cmd_args = parser.parse_args()
    
    # pool1
    # args = Arguments(1, '../algonauts_2023_challenge_data', 10, model='vgg19', layers=['features.4'], roi=sys.argv[1])

    # pool4
    args = Arguments(1, "../algonauts_2023_challenge_data", 10, model="vgg19", layers=["features.27"], rois=cmd_args.rois, run_id="40ccc02e")
    train(args)
