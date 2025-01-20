import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from model.data import Data

def compute_spv(cfg, model, r, p, q):
    """
    Computes the strategy proofness violation of the model

    Arguments:
        cfg: Config
        model: MatchingNet
        r: matching matrix [Batch_size * num_agents, *num_agents]
        p: men's preference [Batch_size, num_agents, num_agents]
        q: women's preference [Batch_size, num_agents, num_agents]
    returns:
        sp_v: torch.Tensor of shape [num_agents, num_agents]
    """
    num_agents = cfg.num_agents
    device = cfg.device
    G = Data(cfg)


    spv = torch.zeros((num_agents, num_agents), device=device)
    for agent_idx in range(num_agents):
        P_mis, Q_mis = G.compose_misreport(p, q, G.mis_array, agent_idx, is_P=True)
        p_mis, q_mis = torch.Tensor(P_mis).to(device), torch.Tensor(Q_mis).to(device)
        r_mis = model(p_mis.view(-1, num_agents, num_agents), q_mis.view(-1, num_agents, num_agents))
        r_mis = r_mis.view(p.shape[0], -1, num_agents, num_agents)

        r_mis_agent = r_mis[:, :, agent_idx, :]

        r_agent = r[:, agent_idx, :].to(device)
        r_agent = r_agent.repeat(1, r_mis_agent.shape[1]).view(r_mis_agent.shape[0], r_mis_agent.shape[1], r_mis_agent.shape[2])

        for f in range(num_agents):
            mask = torch.where(p[:, agent_idx, :] >= p[:, agent_idx, f].view(-1, 1), 1, 0).to(device)
            mask = mask.repeat(1, r_mis_agent.shape[1]).view(r_mis_agent.shape[0], r_mis_agent.shape[1], r_mis_agent.shape[2])
            spv[agent_idx, f] = ((r_mis_agent - r_agent) * mask).sum(-1).relu().sum(-1).mean()
    return spv
