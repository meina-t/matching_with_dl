import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from model.loss.efficiency import compute_ev
from model.loss.stability import compute_sv
from model.loss.strategy_proofness import compute_spv


class MatchingNet(nn.Module):
    """ Neural Network Module for Matching """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        device = cfg.device
        num_agents = self.cfg.num_agents
        num_hidden_nodes = self.cfg.num_hidden_nodes

        self.layers = nn.Sequential(
            nn.Linear(num_agents*num_agents*2, num_hidden_nodes),
            nn.LeakyReLU(),
            nn.Linear(num_hidden_nodes, num_hidden_nodes),
            nn.LeakyReLU(),
            nn.Linear(num_hidden_nodes, num_hidden_nodes),
            nn.LeakyReLU(),
            nn.Linear(num_hidden_nodes, num_hidden_nodes),
            nn.LeakyReLU(),
            nn.Linear(num_hidden_nodes, num_agents * num_agents)
            ).to(device)

    def forward(self, p, q):
        def sinkhorn_normalization(r, num_iters=10):
            for _ in range(num_iters):
                r = F.normalize(r, p=1, dim=1, eps=1e-8)  # 行方向に正規化
                r = F.normalize(r, p=1, dim=2, eps=1e-8)  # 列方向に正規化
            return r
    
        x = torch.stack([p, q], dim = -1)
        x = x.view(-1, self.cfg.num_agents ** 2 * 2)
        r = self.layers(x)
        r = r.view(-1, self.cfg.num_agents, self.cfg.num_agents)
        r = F.softplus(r)
        r = sinkhorn_normalization(r)
        return r
    
    



def train_model(cfg, model, data):
    """
    """
    device = cfg.device
    num_epochs = cfg.epochs
    lr = cfg.lr
    batch_size = cfg.batch_size

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 損失関数の重み付け
    lambda_spv = 1.0  # c_1 の重み初期値
    lambda_sv = 1.0  # c_2 の重み初期値

    rho = 1  # 重み付けのパラメータ
    
    print("Training started.")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        P, Q = data.get_batch(batch_size, epoch)
        r = model(P, Q)
        
        # 損失の計算
        spv = compute_spv(cfg, model, r, P, Q)  # 制約条件1の損失
        sv = compute_sv(cfg, r, P, Q)  # 制約条件2の損失
        objective_loss = compute_ev(cfg, r, P, Q, data)  # 目的関数

        # 総合損失
        loss_matrix = lambda_spv * spv + lambda_sv * sv + objective_loss
        total_loss = loss_matrix.sum()
        
        # 逆伝播とパラメータ更新
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # パラメータの更新
        lambda_spv += rho * spv.sum().item()
        lambda_sv += rho * sv.sum().item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch+1}")
            print(f"Total Loss: {total_loss.item()}")
            print(f"SPV: {spv.sum().item()}")
            print(f"SV: {sv.sum().item()}")
            print(f"Objective Loss(ev): {objective_loss.sum().item()}")
            print(f"Paramiters: lambda_spv = {lambda_spv}, lambda_sv = {lambda_sv}, rho = {rho}")
            print("---------------------------")

    print("Training completed.")
