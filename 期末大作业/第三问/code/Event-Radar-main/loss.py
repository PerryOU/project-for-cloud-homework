# === Python代码文件: loss.py (已修复) ===

import torch
import torch.nn as nn
import torch.nn.functional as F

class Bias_loss(nn.Module):
    def __init__(self, temperature=0.5):
        super(Bias_loss, self).__init__()
        self.temperature = temperature

    def forward(self, bias, features, gcn_feature, emofeature, dctfeature, index):
        """
        Args:
            bias (torch.Tensor): 每个样本的最小不确定性 (即最自信的偏置). Shape: [B, 1].
            features (torch.Tensor): 融合后的特征. Shape: [B, D].
            gcn_feature (torch.Tensor): GCN分支的特征. Shape: [B, D].
            emofeature (torch.Tensor): 文本分支的特征. Shape: [B, D].
            dctfeature (torch.Tensor): 图像分支的特征. Shape: [B, D].
            index (torch.Tensor): 最小不确定性所在的模态索引 {0, 1, 2}. Shape: [B, 1].
        """
        # --- 步骤0: 维度安全检查 ---
        if bias.dim() != 2 or bias.shape[1] != 1:
            raise ValueError(f"输入 'bias' 的维度错误! 期望 [B, 1], 实际 {bias.shape}")
        if index.dim() != 2 or index.shape[1] != 1:
            raise ValueError(f"输入 'index' 的维度错误! 期望 [B, 1], 实际 {index.shape}")

        # --- 步骤1: 特征归一化与相似度计算 ---
        features = F.normalize(features, p=2, dim=1)
        gcn_feature = F.normalize(gcn_feature, p=2, dim=1)
        emofeature = F.normalize(emofeature, p=2, dim=1)
        dctfeature = F.normalize(dctfeature, p=2, dim=1)

        # 使用 bmm 进行批处理矩阵乘法
        s_fg = torch.bmm(features.unsqueeze(1), gcn_feature.unsqueeze(2)).squeeze(-1) / self.temperature
        s_fd = torch.bmm(features.unsqueeze(1), dctfeature.unsqueeze(2)).squeeze(-1) / self.temperature
        s_fe = torch.bmm(features.unsqueeze(1), emofeature.unsqueeze(2)).squeeze(-1) / self.temperature

        # --- 步骤2: 归一化相似度分数 ---
        # s_sum 的形状: [B, 1, 3]
        s_sum = torch.stack([s_fg, s_fd, s_fe], dim=2)
        s_sum = F.softmax(s_sum, dim=2)

        # --- 步骤3: 构建偏置矩阵 (最关键的修正) ---
        base_bias = bias.expand(-1, 3)
        src = 1 - bias
        # bias_matrix 的最终形状是 [B, 3]
        bias_matrix = base_bias.scatter(1, index, src)

        # --- 步骤4: 计算加权相似度 ---
        # s_sum 的形状: [B, 1, 3]
        # bias_matrix.unsqueeze(1) 的形状: [B, 1, 3]
        # 逐元素相乘后求和
        logits = torch.sum(s_sum * bias_matrix.unsqueeze(1), dim=2)

        # --- 步骤5: 计算最终损失 ---
        loss = -torch.mean(torch.log(logits + 1e-10))

        return loss
