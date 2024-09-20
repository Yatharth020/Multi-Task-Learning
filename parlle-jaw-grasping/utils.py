import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_learning_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Learning Curves')
    plt.savefig('learning_curves.png')
    plt.close()

def quaternion_to_euler(q):
    r = Rotation.from_quat(q)
    return r.as_euler('xyz', degrees=True)

def visualize_grasp(image, grasp):
    plt.imshow(image.permute(1, 2, 0))
    x1, y1, x2, y2 = grasp
    plt.plot([x1, x2], [y1, y2], 'r-')
    plt.scatter([x1, x2], [y1, y2], c='r', s=50)
    plt.title('Grasp Visualization')
    plt.axis('off')
    plt.show()

def compute_iou(pred_grasp, true_grasp):
    return 0.5  # Placeholder value
