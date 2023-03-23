# Centroid Distance Distillation for Effective Rehearsal in Continual Learning(icassp2023)  

Daofeng Liu, Fan Lyu, Linyan Li, Zhenping Xia, Fuyuan Hu  

Abstract
-----
Rehearsal, retraining on a stored small data subset of old tasks, has been proven effective in solving catastrophic forgetting in continual learning. However, due to the sampled data may have a large bias towards the original dataset, retraining them is susceptible to driving continual domain drift of old tasks in feature space, resulting in forgetting. In this paper, we focus on tackling the continual domain drift problem with centroid distance distillation. First, we propose a centroid caching mechanism for sampling data points based on constructed centroids to reduce the sample bias in rehearsal. Then, we present a centroid distance distillation that only stores the centroid distance to reduce the continual domain drift. The experiments on four continual learning datasets show the superiority of the proposed method, and the continual domain drift can be reduced.

Requirements
----
TensorFlow >= v1.9.0.

Training
---
Example runs are:

$ ./replicate_results.sh MNIST 6   /* Train MDMT-R on MNIST */

$ ./replicate_results.sh CIFAR 5   /* Train MDMT-R on CIFAR */

$ ./replicate_results.sh CUB 5     /* Train MDMT-R on CUB */

$ ./replicate_results.sh AWA 9     /* Train MDMT-R on AWA */
