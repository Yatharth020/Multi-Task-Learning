import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveRoutingLayer(nn.Module):
    def __init__(self, in_features, num_experts, num_tasks):
        super(AdaptiveRoutingLayer, self).__init__()
        self.in_features = in_features
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        
        self.experts = nn.ModuleList([nn.Linear(in_features, in_features) for _ in range(num_experts)])
        self.routing_weights = nn.Parameter(torch.randn(num_tasks, num_experts))
        
    def forward(self, x, task_id):
        expert_outputs = torch.stack([expert(x) for expert in self.experts])
        routing_weights = F.softmax(self.routing_weights[task_id], dim=0)
        output = torch.sum(expert_outputs * routing_weights.view(self.num_experts, 1, 1), dim=0)
        return output

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Linear(256 * 8 * 8, 512)

    def forward(self, color, depth):
        x = torch.cat([color, depth], dim=1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.view(-1, 256 * 8 * 8)
        x = F.relu(self.fc(x))
        return x

class GraspPredictionModule(nn.Module):
    def __init__(self, input_dim):
        super(GraspPredictionModule, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 4)  # Output: (x1, y1, x2, y2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class PoseEstimationModule(nn.Module):
    def __init__(self, input_dim):
        super(PoseEstimationModule, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 7)  # 3 for translation, 4 for rotation (quaternion)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class MultiTaskRoboticModel(nn.Module):
    def __init__(self, num_experts=3):
        super(MultiTaskRoboticModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.adaptive_routing = AdaptiveRoutingLayer(512, num_experts, 2)
        self.grasp_prediction = GraspPredictionModule(512)
        self.pose_estimation = PoseEstimationModule(512)

    def forward(self, color, depth):
        features = self.feature_extractor(color, depth)
        grasp_features = self.adaptive_routing(features, 0)
        pose_features = self.adaptive_routing(features, 1)
        grasp_pred = self.grasp_prediction(grasp_features)
        pose_est = self.pose_estimation(pose_features)
        return grasp_pred, pose_est
