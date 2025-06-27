# === Python代码文件: decision.py (已修复) ===

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- KL 和 ce_loss 函数保持不变 ---
def KL(alpha, c, device):
    beta = torch.ones((1, c)).to(device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c, alpha.device)
    return (A + B)


class fushion_decision(nn.Module):
    def __init__(self, views, feature_out, lambda_epochs=50):
        super(fushion_decision, self).__init__()
        self.views = views
        self.lambda_epochs = lambda_epochs
        self.Classifiers = nn.ModuleList([Classifier(feature_out) for i in range(self.views)])

    def DSuncertain(self, alpha):
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(self.views):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = self.views / S[v]

        # --- FIX: 使用 torch.cat 得到正确的 [B, 3] 形状 ---
        # 将 [B, 1], [B, 1], [B, 1] 沿着维度1拼接，得到期望的 [B, 3]
        return torch.cat([u[0], u[1], u[2]], dim=1)

    def forward(self, X, y, global_step):
        evidence = self.infer(X)
        loss = 0
        alpha = dict()
        for v_num in range(len(X)):
            alpha[v_num] = evidence[v_num] + 1
            loss += ce_loss(y, alpha[v_num], 2, global_step, self.lambda_epochs)

        # uncertaincof 现在将是正确的 [B, 3] 形状
        uncertaincof = self.DSuncertain(alpha)
        loss = torch.mean(loss)
        return evidence, uncertaincof, loss

    def infer(self, input):
        evidence = dict()
        for v_num in range(self.views):
            evidence[v_num] = self.Classifiers[v_num](input[v_num])
        return evidence


class Classifier(nn.Module):
    def __init__(self, feature_out):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_out, feature_out),
            nn.ReLU(),
            nn.Linear(feature_out, feature_out // 2)
        )
        self.fcclass = nn.Linear(feature_out // 2, 2)
        self.fcevd = nn.Softplus()

    def forward(self, x):
        h = self.fc(x)
        h = self.fcclass(h)
        h = self.fcevd(h)
        return h
