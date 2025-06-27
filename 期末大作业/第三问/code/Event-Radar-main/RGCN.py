# === Python代码文件: RGCN.py (已修复) ===

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, inits, global_mean_pool

class encodeMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(encodeMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.LeakyReLU()
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class multimodal_RGCN(nn.Module):
    def __init__(self, args, dropout=0.1, feature_out=32):
        super(multimodal_RGCN, self).__init__()
        self.args = args
        self.dropout = dropout
        self.embedding_dim = args.dim_node_features
        self.gnn_layers = nn.ModuleList()

        for _ in range(args.num_gnn_layers):
            conv = GCNConv(args.dim_node_features, args.dim_node_features, add_self_loops=True, normalize=True)
            self.gnn_layers.append(conv)

        self.Hbn = nn.BatchNorm1d(args.dim_node_features)
        self.generate_feature = nn.Sequential(
            nn.Linear(in_features=args.dim_node_features * 4, out_features=args.dim_node_features),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Linear(in_features=args.dim_node_features, out_features=self.embedding_dim),
        )
        self.bn2 = nn.BatchNorm1d(self.embedding_dim)
        self.feature_layer = encodeMLP(self.embedding_dim, feature_out, 64)
        self.output = nn.Linear(feature_out, 2)

    def forward(self, batch):
        H = batch.x
        edge_index = batch.edge_index

        for gnn in self.gnn_layers:
            H = gnn(x=H, edge_index=edge_index)
            H = F.leaky_relu(H)
        H = self.Hbn(H)

        # --- FIX: 直接从batch对象访问批处理好的索引属性 ---
        post_indices = batch.post_idx
        image_indices = batch.image_idx

        # 使用这些1D索引张量来安全地选择节点特征
        post_nodes_features = H[post_indices]
        image_nodes_features = H[image_indices]

        # 同样，获取这些节点对应的batch向量，用于池化
        post_nodes_batch_vector = batch.batch[post_indices]
        image_nodes_batch_vector = batch.batch[image_indices]

        # 使用全局平均池化得到每个图的表示
        post_feature = global_mean_pool(post_nodes_features, post_nodes_batch_vector, size=batch.num_graphs)
        image_feature = global_mean_pool(image_nodes_features, image_nodes_batch_vector, size=batch.num_graphs)

        # 融合特征并生成输出
        combine_feature = torch.cat(
            [post_feature, image_feature, post_feature - image_feature, post_feature * image_feature], dim=-1)
        final_feature_batch = self.generate_feature(combine_feature)
        final_feature_batch = self.bn2(final_feature_batch)
        out = self.feature_layer(final_feature_batch)
        prob = self.output(out)

        return out, prob
