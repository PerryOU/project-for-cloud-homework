# === Python代码文件: Model.py (最终修复版) ===

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

# 确保其他模块是从我们之前修复的版本导入的
from DCFormer import DCTDetectionModel_MOE
from RGCN import multimodal_RGCN
from decision import fushion_decision


# --- 辅助类 ---
class encodeMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(encodeMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.fc1(x);
        out = self.relu(out);
        out = self.fc2(out)
        return out


class FixedPooling(nn.Module):
    def __init__(self, fixed_size):
        super().__init__()
        self.fixed_size = fixed_size

    def forward(self, x):
        b, w, h = x.shape
        p_w = self.fixed_size * ((w + self.fixed_size - 1) // self.fixed_size) - w
        p_h = self.fixed_size * ((h + self.fixed_size - 1) // self.fixed_size) - h
        x = nn.functional.pad(x, (0, p_h, 0, p_w))
        pool_size = (((w + self.fixed_size - 1) // self.fixed_size), ((h + self.fixed_size - 1) // self.fixed_size))
        pool = nn.MaxPool2d(pool_size, stride=pool_size)
        return pool(x)


class LModel(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4, dropout=0.1, norm_first=True, layer_norm_eps=1e-5):
        super(LModel, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout,
                                                         batch_first=True)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(p=dropout)

    def forward(self, text_src):
        # --- FIX: 修复了LModel以确保它永远不会返回None ---
        # 这是一个标准的 pre-norm transformer block
        if self.norm_first:
            # Pre-LN: LayerNorm -> Attention -> Dropout -> Residual
            attn_output, attention_weight = self._sa_block(self.norm1(text_src))
            text = text_src + attn_output
        else:
            # Post-LN: Attention -> Dropout -> Residual -> LayerNorm
            attn_output, attention_weight = self._sa_block(text_src)
            text = self.norm1(text_src + attn_output)
        return text, attention_weight

    def _sa_block(self, text):
        # need_weights=True 确保总是返回注意力权重
        text, attention_weight = self.multihead_attention(text, text, text, need_weights=True)
        return self.dropout1(text), attention_weight


class FakeNewsDetection(nn.Module):
    def __init__(self, args, feature_out=32):
        super(FakeNewsDetection, self).__init__()
        self.args = args
        self.feature_out = feature_out
        embedding_dim = args.dim_node_features

        # --- FIX: 恢复了原始代码中对文本特征的预处理层 ---
        self.text_feature_mlp = nn.Sequential(
            nn.BatchNorm1d(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
        )

        # 分支特定的特征编码器
        self.text_fea = encodeMLP(embedding_dim, feature_out, 64)
        self.gcn_fea = multimodal_RGCN(self.args, feature_out=feature_out)
        self.dct_fea = DCTDetectionModel_MOE(embedding_dim, feature_out=feature_out)

        # 融合与决策模块
        self.fusion = LModel(embed_dim=feature_out)
        self.decision = fushion_decision(views=3, feature_out=feature_out)
        self.fixed_pooling = FixedPooling(fixed_size=6)
        self.ln = nn.LayerNorm(feature_out)

        # 最终的分类器
        self.mlp1 = nn.Linear(feature_out * 3 + 6 * 6, feature_out)
        self.mlp_classifier = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(feature_out, feature_out),
            nn.LeakyReLU(),
            nn.Linear(feature_out, 2),
        )

    def forward(self, batch, global_step):
        # 1. 从不同分支提取特征
        # GCN 分支
        gcn_out, _ = self.gcn_fea(batch)

        # 文本 (Emotion) 分支
        root_node_indices = batch.ptr[:-1]
        text_features = batch.x[root_node_indices]
        processed_text_features = self.text_feature_mlp(text_features)
        emo_out = self.text_fea(processed_text_features)

        # 图像 (DCT) 分支
        dct_out, _ = self.dct_fea(batch)

        # 2. 基于不确定性的决策
        outputs = [gcn_out, dct_out, emo_out]
        label = batch.y
        env_single, uncertain, certainloss = self.decision(outputs, label, global_step)

        # 3. 特征融合
        out_tensor = torch.stack(outputs, dim=1)
        out_tensor = torch.mul(out_tensor, 1 - uncertain.unsqueeze(-1))
        out_tensor = self.ln(out_tensor)

        fused_tensor, attention = self.fusion(out_tensor)

        attention = self.fixed_pooling(attention)

        y = torch.cat([fused_tensor.reshape(len(fused_tensor), self.feature_out * 3),
                       attention.reshape(len(attention), -1)], dim=1)

        # 4. 最终分类
        y_embed = self.mlp1(y)
        pred = self.mlp_classifier(y_embed)

        # 5. 返回所有trainer需要的值，并使用一致的命名
        return pred, y_embed, env_single, uncertain, certainloss, gcn_out, dct_out, emo_out
