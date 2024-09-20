import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

def evaluate(model, dataloader, device):
    model.eval()
    grasp_criterion = nn.MSELoss()
    pose_criterion = nn.MSELoss()
    
    total_grasp_loss = 0
    total_pose_loss = 0
    grasp_errors = []
    translation_errors = []
    rotation_errors = []
    num_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            color_img, depth_img, grasp_labels, camera_pose = [b.to(device) for b in batch]
            grasp_pred, pose_est = model(color_img, depth_img)
            
            # Compute grasp loss
            grasp_loss = 0
            for i in range(grasp_pred.size(0)):
                pred = grasp_pred[i].unsqueeze(0).repeat(grasp_labels[i].size(0), 1, 1)
                true = grasp_labels[i].unsqueeze(1)
                distances = torch.sum((pred - true) ** 2, dim=2)
                min_distances, _ = torch.min(distances, dim=1)
                grasp_loss += torch.mean(min_distances)
                grasp_errors.extend(min_distances.cpu().numpy())
            grasp_loss /= grasp_pred.size(0)
            
            pose_loss = pose_criterion(pose_est, camera_pose)
            
            # Compute translation and rotation errors
            pred_trans = pose_est[:, :3].cpu().numpy()
            true_trans = camera_pose[:, :3].cpu().numpy()
            pred_rot = pose_est[:, 3:].cpu().numpy()
            true_rot = camera_pose[:, 3:].cpu().numpy()
            
            translation_error = np.linalg.norm(pred_trans - true_trans, axis=1)
            translation_errors.extend(translation_error)
            
            for i in range(pred_rot.shape[0]):
                pred_r = Rotation.from_quat(pred_rot[i])
                true_r = Rotation.from_quat(true_rot[i])
                rotation_error = Rotation.magnitude(pred_r * true_r.inv())
                rotation_errors.append(rotation_error)
            
            total_grasp_loss += grasp_loss.item() * grasp_pred.size(0)
            total_pose_loss += pose_loss.item() * grasp_pred.size(0)
            num_samples += grasp_pred.size(0)
    
    avg_grasp_loss = total_grasp_loss / num_samples
    avg_pose_loss = total_pose_loss / num_samples
    
    grasp_rmse = np.sqrt(mean_squared_error(np.zeros_like(grasp_errors), grasp_errors))
    translation_rmse = np.sqrt(mean_squared_error(np.zeros_like(translation_errors), translation_errors))
    rotation_rmse = np.sqrt(mean_squared_error(np.zeros_like(rotation_errors), rotation_errors))
    
    print(f"Evaluation Results:")
    print(f"Average Grasp Loss: {avg_grasp_loss:.4f}")
    print(f"Average Pose Loss: {avg_pose_loss:.4f}")
    print(f"Grasp RMSE: {grasp_rmse:.4f}")
    print(f"Translation RMSE: {translation_rmse:.4f}")
    print(f"Rotation RMSE: {rotation_rmse:.4f}")
    
    return avg_grasp_loss, avg_pose_loss, grasp_rmse, translation_rmse, rotation_rmse

def visualize_results(model, dataloader, device, num_samples=5):
    model.eval()
    
    with torch.no_grad():
        for i, (color_img, depth_img, grasp_labels, camera_pose) in enumerate(dataloader):
            if i >= num_samples:
                break
            
            color_img, depth_img = color_img.to(device), depth_img.to(device)
            grasp_pred, pose_est = model(color_img, depth_img)
            

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(color_img[0].cpu().permute(1, 2, 0))
            plt.title("Color Image")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(color_img[0].cpu().permute(1, 2, 0))
            x1, y1, x2, y2 = grasp_pred[0].cpu().numpy()
            plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2)
            plt.scatter([x1, x2], [y1, y2], c='r', s=50)
            plt.title("Predicted Grasp")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"sample_result_{i+1}.png")
            plt.close()
            
            print(f"Sample {i+1}:")
            print(f"Predicted Grasp: {grasp_pred[0].cpu().numpy()}")
            print(f"Predicted Pose: {pose_est[0].cpu().numpy()}")
            print(f"True Grasp: {grasp_labels[0][0].numpy()}")
            print(f"True Pose: {camera_pose[0].numpy()}")
            print()

def compute_grasp_metrics(model, dataloader, device, iou_threshold=0.25):
    model.eval()
    total_samples = 0
    correct_grasps = 0
    
    with torch.no_grad():
        for batch in dataloader:
            color_img, depth_img, grasp_labels, _ = [b.to(device) for b in batch]
            grasp_pred, _ = model(color_img, depth_img)
            
            for i in range(grasp_pred.size(0)):
                pred = grasp_pred[i]
                true = grasp_labels[i]
                
                ious = compute_grasp_iou(pred.unsqueeze(0), true)
                max_iou = ious.max().item()
                
                if max_iou > iou_threshold:
                    correct_grasps += 1
                total_samples += 1
    
    grasp_accuracy = correct_grasps / total_samples
    print(f"Grasp Accuracy (IoU > {iou_threshold}): {grasp_accuracy:.4f}")
    return grasp_accuracy

def compute_grasp_iou(pred_grasp, true_grasps, gripper_width=20):
  
    px1, py1, px2, py2 = pred_grasp
    tx1, ty1, tx2, ty2 = true_grasps.unbind(dim=1)
    
    pred_rect = compute_grasp_rectangle(px1, py1, px2, py2, gripper_width)
    true_rects = compute_grasp_rectangle(tx1, ty1, tx2, ty2, gripper_width)
    
    x_left = torch.maximum(pred_rect[0], true_rects[:, 0])
    y_top = torch.maximum(pred_rect[1], true_rects[:, 1])
    x_right = torch.minimum(pred_rect[2], true_rects[:, 2])
    y_bottom = torch.minimum(pred_rect[3], true_rects[:, 3])
    
    intersection = torch.clamp(x_right - x_left, min=0) * torch.clamp(y_bottom - y_top, min=0)
    
    pred_area = (pred_rect[2] - pred_rect[0]) * (pred_rect[3] - pred_rect[1])
    true_areas = (true_rects[:, 2] - true_rects[:, 0]) * (true_rects[:, 3] - true_rects[:, 1])

    union = pred_area + true_areas - intersection
    iou = intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero
    
    return iou

def compute_grasp_rectangle(x1, y1, x2, y2, gripper_width):
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Compute grasp angle
    angle = torch.atan2(y2 - y1, x2 - x1)
    
    dx = gripper_width / 2 * torch.sin(angle)
    dy = gripper_width / 2 * torch.cos(angle)
    
    left = torch.minimum(x1, x2) - torch.abs(dx)
    right = torch.maximum(x1, x2) + torch.abs(dx)
    top = torch.minimum(y1, y2) - torch.abs(dy)
    bottom = torch.maximum(y1, y2) + torch.abs(dy)
    
    return torch.stack([left, top, right, bottom])


def analyze_pose_errors(translation_errors, rotation_errors):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(translation_errors, bins=20)
    plt.title("Translation Error Distribution")
    plt.xlabel("Error (m)")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 2, 2)
    plt.hist(rotation_errors, bins=20)
    plt.title("Rotation Error Distribution")
    plt.xlabel("Error (rad)")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig("pose_error_distribution.png")
    plt.close()

