# Group Equivariant CNNs Outperform Spatial Transformers on Tasks Which Require Rotation Invariance
## Candidate number: 1068707

This study aims to show that group equivariant CNNs
outperform spatial transformers, on tasks which demand rotation invariance, by providing theoretical background and  experimental performance comparison with detailed analysis.

#### Models implementations
The folder **models** contains implementations of group equivariant neural networks and spatial transformers. 
All layers are implemented from scratch and are located in **layers** subfolder. Similarly, implementations of interpolation-based lifting convolution kernels 
and group convolution kernels are in **kernels** subfolder. The implementation of localization network is in **localization_net.py**.
The discretized implementation of SO2 is in folder **groups**, in **discrete_so2.py**.

Group equivariant model is implemented in **group_equivariant_cnn.py**, while spatial transformer model is implemented in **spatial_transformer.py**.

#### Visualisations
Are implemented in **visualizations** folder.

#### Reproducibility
In the **results** folder, there are two folders - **no-rotations** and **rotations**. Each of those folders contains weights and training logs, 
for each of training configurations. Weights and training logs are grouped by model configurations and training configurations. 
By executing notebooks in the root folder, it is possible to reproduce all tables, visualizations and plots which were present in the 
submitted report. Training configuration is implemented in MNISTModule, which is located in **modules** folder, in **MNISTModule.py**.

#### References
The implementation in **models** folder is based on the following resources:
- [SE2CNN](https://github.com/tueimage/SE2CNN)
- [Tutorial on Group Convolutional Networks](https://colab.research.google.com/drive/1h7U15-qFC2yy6roRIfLPk5TSlo6sONsm)
- [Tutorial on Regular Group Convolutions](https://colab.research.google.com/github/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/DL2/Geometric_deep_learning/tutorial1_regular_group_convolutions.ipynb)
