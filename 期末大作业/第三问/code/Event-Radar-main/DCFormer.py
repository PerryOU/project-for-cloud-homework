# === Python代码文件: DCFormer.py (已修复) ===

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

class encodeMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(encodeMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.LeakyReLU()
    def forward(self, x):
        out = self.fc1(x); out = self.relu(out); out = self.fc2(out)
        return out

class DCTDetectionModel_MOE(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1, encode_model=encodeMLP, feature_out=32):
        super(DCTDetectionModel_MOE, self).__init__()
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.feature_layer = encode_model(embedding_dim, feature_out, 64)
        self.output = nn.Linear(feature_out, 2)
        self.dropout = dropout

    def forward(self, batch):
        """
        --- FIX: forward方法现在处理PyG的Batch对象来高效提取和池化图像特征 ---
        """
        # 1. 直接从batch对象访问批处理好的图像节点索引
        image_indices = batch.image_idx

        # 2. 提取所有图像节点的特征和它们对应的batch向量
        image_nodes_features = batch.x[image_indices]
        image_nodes_batch_vector = batch.batch[image_indices]

        # 3. 使用全局平均池化得到每个图的图像表示
        pooled_image_features = global_mean_pool(image_nodes_features, image_nodes_batch_vector, size=batch.num_graphs)

        # 4. 应用后续层
        pooled_image_features = self.bn(pooled_image_features)
        out = self.feature_layer(pooled_image_features)
        prob = self.output(out)
        return out, prob
