import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import RoboticTasksDataset
from model import MultiTaskRoboticModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import ParameterGrid

def train(model, train_loader, val_loader, config, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    grasp_criterion = nn.MSELoss()
    pose_criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            color_img, depth_img, grasp_labels, camera_pose = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            grasp_pred, pose_est = model(color_img, depth_img)
            
            grasp_loss = 0
            for i in range(grasp_pred.size(0)):
                pred = grasp_pred[i].unsqueeze(0).repeat(grasp_labels[i].size(0), 1, 1)
                true = grasp_labels[i].unsqueeze(1)
                distances = torch.sum((pred - true) ** 2, dim=2)
                min_distances, _ = torch.min(distances, dim=1)
                grasp_loss += torch.mean(min_distances)
            grasp_loss /= grasp_pred.size(0)
            
            pose_loss = pose_criterion(pose_est, camera_pose)
            
            loss = config['grasp_weight'] * grasp_loss + config['pose_weight'] * pose_loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        val_loss = validate(model, val_loader, grasp_criterion, pose_criterion, config, device)
        scheduler.step(val_loss)
        
        print(f'Epoch [{epoch+1}/{config["num_epochs"]}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    return best_val_loss

def validate(model, val_loader, grasp_criterion, pose_criterion, config, device):
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            color_img, depth_img, grasp_labels, camera_pose = [b.to(device) for b in batch]
            grasp_pred, pose_est = model(color_img, depth_img)
            
            grasp_loss = 0
            for i in range(grasp_pred.size(0)):
                pred = grasp_pred[i].unsqueeze(0).repeat(grasp_labels[i].size(0), 1, 1)
                true = grasp_labels[i].unsqueeze(1)
                distances = torch.sum((pred - true) ** 2, dim=2)
                min_distances, _ = torch.min(distances, dim=1)
                grasp_loss += torch.mean(min_distances)
            grasp_loss /= grasp_pred.size(0)
            
            pose_loss = pose_criterion(pose_est, camera_pose)
            
            loss = config['grasp_weight'] * grasp_loss + config['pose_weight'] * pose_loss
            val_loss += loss.item()
    
    return val_loss / len(val_loader)

def hyperparameter_tuning(train_loader, val_loader, device):
    param_grid = {
        'learning_rate': [1e-3, 1e-4],
        'weight_decay': [1e-4, 1e-5],
        'num_epochs': [50, 100],
        'grasp_weight': [0.5, 1.0],
        'pose_weight': [0.5, 1.0],
        'num_experts': [3, 5]
    }
    
    grid = ParameterGrid(param_grid)
    best_config = None
    best_val_loss = float('inf')
    
    for config in grid:
        model = MultiTaskRoboticModel(num_experts=config['num_experts'])
        val_loss = train(model, train_loader, val_loader, config, device)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_config = config
    
    return best_config

def get_dataloader(root_dir, batch_size, split='train', augment=False):
    dataset = RoboticTasksDataset(root_dir, split, augment)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'), num_workers=4)
