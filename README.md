# Multi-Task-Learning

## Parallel Jaw Grasping Experiments

This repository contains one part of the initial code for experiments with parallel jaw grasping using the dataset from Zeng et al. (2018) [1]. The experiments focus on multi-task learning for robotic grasping, combining grasp prediction and pose estimation.

### Key Features

- Multi-task learning framework for grasp prediction and pose estimation
- Adaptive routing mechanism for flexible feature utilization
- Hyperparameter tuning capabilities
- Comprehensive evaluation metrics including IoU for grasps

The experiments use the parallel-jaw grasping dataset provided by Zeng et al. This dataset includes:

- RGB-D images of input scenes
- Pre-computed heightmaps
- Camera intrinsics and poses
- Manually labeled grasp annotations

For full details on the dataset, please refer to the original paper [1] and the dataset website [2].


## References

[1] A. Zeng, S. Song, K. Yu, E. Donlon, F. R. Hogan, M. Bauza, D. Ma, O. Taylor, M. Liu, E. Romo, N. Fazeli, F. Alet, N. C. Dafle, R. Holladay, I. Morona, P. Q. Nair, D. Green, I. Taylor, W. Liu, T. Funkhouser, and A. Rodriguez, "Robotic Pick-and-Place of Novel Objects in Clutter with Multi-Affordance Grasping and Cross-Domain Image Matching," in Proceedings of the IEEE International Conference on Robotics and Automation, 2018.

[2] Parallel Jaw Grasping Dataset. [Online]. Available: https://vision.princeton.edu/projects/2017/arc/

## Acknowledgements

This code is based on the dataset provided by Zeng et al. [1]. We thank the authors for making their data publicly available for research purposes.
