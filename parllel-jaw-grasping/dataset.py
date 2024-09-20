import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class RoboticTasksDataset(Dataset):
    def __init__(self, root_dir, split='train', augment=False):
        self.root_dir = root_dir
        self.split = split
        self.augment = augment
        self.heightmap_color_dir = os.path.join(root_dir, 'heightmap-color')
        self.heightmap_depth_dir = os.path.join(root_dir, 'heightmap-depth')
        self.label_dir = os.path.join(root_dir, 'label')
        self.camera_pose_dir = os.path.join(root_dir, 'camera-pose')
        
        split_file = os.path.join(root_dir, f'{split}-split.txt')
        with open(split_file, 'r') as f:
            self.image_names = [line.strip() for line in f]
        
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ]) if augment else None

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        
        # Load color heightmap
        color_path = os.path.join(self.heightmap_color_dir, f'{img_name}.png')
        color_img = Image.open(color_path).convert('RGB')
        
        # Load depth heightmap
        depth_path = os.path.join(self.heightmap_depth_dir, f'{img_name}.png')
        depth_img = Image.open(depth_path)
        
        if self.transform:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            color_img = self.transform(color_img)
            torch.manual_seed(seed)
            depth_img = self.transform(depth_img)
        
        color_img = transforms.ToTensor()(color_img)
        depth_img = transforms.ToTensor()(depth_img)
        
        # Load grasp labels
        label_path = os.path.join(self.label_dir, f'{img_name}.txt')
        with open(label_path, 'r') as f:
            grasp_labels = [list(map(float, line.strip().split())) for line in f]
        grasp_labels = torch.FloatTensor(grasp_labels)
        
        # Load camera pose
        pose_path = os.path.join(self.camera_pose_dir, f'{img_name}.txt')
        camera_pose = torch.FloatTensor(np.loadtxt(pose_path))
        
        return color_img, depth_img, grasp_labels, camera_pose
