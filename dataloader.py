import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class VideoInterpolationDataset(Dataset):
    """
    Custom dataset for video frame interpolation. Loads pairs of frames (previous and next).
    """
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.data = self._load_data()

    def _load_data(self):
        """
        Loads file paths for paired frames (frame_t and frame_t+1).
        """
        data = []
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path '{self.data_path}' does not exist.")
        
        for root, _, files in os.walk(self.data_path):
            files = sorted([f for f in files if f.endswith((".png", ".jpg"))])
            for i in range(len(files) - 1):
                data.append((os.path.join(root, files[i]), os.path.join(root, files[i + 1])))
        if not data:
            raise ValueError(f"No image pairs found in '{self.data_path}'")
        
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame0_path, frame2_path = self.data[idx]
        frame0, frame2 = Image.open(frame0_path), Image.open(frame2_path)

        if self.transform:
            frame0 = self.transform(frame0)
            frame2 = self.transform(frame2)

        return frame0, frame2

def get_data_loader(data_path, batch_size=16, shuffle=True, num_workers=4):
    """
    Utility function to get a DataLoader instance.
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Optional resizing for consistency
        transforms.ToTensor(),
    ])
    
    dataset = VideoInterpolationDataset(data_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
