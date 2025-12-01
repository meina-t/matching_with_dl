import torch

def compute_ev(cfg, r, p, q, non_weighted=False):
    """
    効率性損失を計算する。
    - r: マッチング確率行列 (batch_size x num_agents x num_agents)
    - p: 提案者の選好行列 (batch_size x num_agents x num_agents)
    - q: 受け手の選好行列 (batch_size x num_agents x num_agents)

    return: 効率性損失 (2d tensor, num_agents x num_agents)
    """
    batch_size = p.shape[0]
    num_agents = cfg.num_agents
    device = cfg.device

    efficiency_loss = []

    if non_weighted == False:
        lambda_weights = cfg.lambda_weights
    else:
        lambda_weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)

    cond = (p.unsqueeze(2) >= p.unsqueeze(-1)).float() 
    efficiency_per_batch = (r.unsqueeze(2) * cond).sum(dim=-1)

    efficiency_loss = -efficiency_per_batch.mean(dim=0) * lambda_weights
    
    return efficiency_loss