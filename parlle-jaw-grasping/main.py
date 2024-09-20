import torch
from model import MultiTaskRoboticModel
from train import get_dataloader, hyperparameter_tuning, train
from evaluate import evaluate, visualize_results
from utils import set_seed, plot_learning_curves
import argparse

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = get_dataloader(args.data_dir, args.batch_size, split='train', augment=True)
    val_loader = get_dataloader(args.data_dir, args.batch_size, split='test')

    if args.tune:
        print("Starting hyperparameter tuning...")
        best_config = hyperparameter_tuning(train_loader, val_loader, device)
        print(f"Best configuration: {best_config}")
    else:
        best_config = {
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'num_epochs': args.epochs,
            'grasp_weight': args.grasp_weight,
            'pose_weight': args.pose_weight,
            'num_experts': args.num_experts
        }


    model = MultiTaskRoboticModel(num_experts=best_config['num_experts'])
    train_losses, val_losses = train(model, train_loader, val_loader, best_config, device)

    plot_learning_curves(train_losses, val_losses)

    print("Evaluating the model...")
    evaluate(model, val_loader, device)

    print("Visualizing results...")
    visualize_results(model, val_loader, device)

    torch.save(model.state_dict(), 'final_model.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel-Jaw Grasping Experiment")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--grasp_weight', type=float, default=1.0, help='Weight for grasp loss')
    parser.add_argument('--pose_weight', type=float, default=1.0, help='Weight for pose loss')
    parser.add_argument('--num_experts', type=int, default=3, help='Number of experts in adaptive routing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning')
    
    args = parser.parse_args()
    main(args)
