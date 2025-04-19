import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ArabicSignLanguageDataset(Dataset):
    def __init__(self, img_dirs, transform=None):
        
        self.img_dirs = img_dirs
        self.transform = transform
        self.sessions = []
        self.labels = []

        # Collect all session paths and corresponding labels
        for dir_path in self.img_dirs:
            
            sign_folders = sorted(os.listdir(dir_path))  # Sort sign folders (labels)
            
            for sign_folder in sign_folders:
                
                sign_path = os.path.join(dir_path, sign_folder)
                sign_sessions = sorted(os.listdir(sign_path))  # Sort sessions within each sign
                
                for session in sign_sessions:
                    
                    session_path = os.path.join(sign_path, session)
                    self.sessions.append(session_path)
                    self.labels.append(sign_folder)  # Label is the sign folder's name


    def __len__(self):
        
        return len(self.sessions) 


    def __getitem__(self, idx):
        
        session_path = self.sessions[idx]  
        label = self.labels[idx]

        images = []
        image_files = sorted(os.listdir(session_path))  # sort images to maintain order
        
        for image_file in image_files:
            
            img_path = os.path.join(session_path, image_file)
            image = Image.open(img_path)
            
            if self.transform:
                image = self.transform(image)
                
            images.append(image)
            
        while len(images) < 32:
            images.append(images[-1])
            
        tensor_images = torch.stack(images) 
        
        return tensor_images, int(label) - 1


def test_train_split(root_dir):
    data_dirs = []
    
    for signer in sorted(os.listdir(root_dir)):
        
        if signer == ".ipynb_checkpoints" or signer == "labels.xlsx":
            continue
        
        signer_path = os.path.join(root_dir, signer)
        
        for split in ["train", "test"]:
            
            split_path = os.path.join(signer_path, split)
            data_dirs.append(split_path)
            
        return (data_dirs[::2], data_dirs[1::2])  # train  # test


def get_dataloaders(root_dir, batch_size):
    train_img_dir, test_img_dir = test_train_split(root_dir)

    # Define transformations as lists
    train_transform = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    test_transform = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]


    train_transforms = transforms.Compose(train_transform)
    test_transforms = transforms.Compose(test_transform)

    train_dataset = ArabicSignLanguageDataset(img_dirs=train_img_dir, transform=train_transforms)
    test_dataset = ArabicSignLanguageDataset(img_dirs=test_img_dir, transform=test_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader
