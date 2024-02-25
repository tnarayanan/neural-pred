import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from arguments import Arguments


# https://drive.google.com/drive/folders/17RyBAnvDhrrt18Js2VZqSVi_nZ7bn3G3?usp=drive_link
class ImageDataset(Dataset):
    def __init__(self, args: Arguments, indices: np.ndarray):
        self.img_dir = os.path.join(args.data_dir, "training_split", "training_images")

        self.img_paths = np.array(sorted(os.listdir(self.img_dir)))[indices]

        self.transform = transforms.Compose([
            transforms.Resize((224,224)), # resize the images to 224x224 pixels
            transforms.ToTensor(), # convert the images to a PyTorch tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
        ])

        self.args = args

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_paths[idx])
        img = Image.open(str(img_path)).convert('RGB')
        img = self.transform(img).to(device=self.args.device)
        return img
