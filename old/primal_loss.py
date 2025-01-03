import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from data import Data

from efficiency_loss import compute_efficiency_loss

def compute_t(r, p, q):
    """
    Compute the loss/metric t based on r, p, and q tensors.

    Args:
        r (torch.Tensor): Allocation matrix (batch_size x num_agents x num_agents)
        p (torch.Tensor): Preference matrix (batch_size x num_agents x num_agents)
        q (torch.Tensor): Preference matrix (batch_size x num_agents x num_agents)

    Returns:
        torch.Tensor: Computed tensor t with non-negative values.
    """
    # Compute wp (preference weight matrix for p)
    wp = torch.where(p[:, :, None, :] - p[:, :, :, None] > 0, 1.0, 0.0)
    
    # Compute wq (preference weight matrix for q)
    q_t = q.transpose(1, 2)  # Use a new variable for transposed q
    wq = torch.where(q_t[:, :, None, :] - q_t[:, :, :, None] > 0, 1.0, 0.0)
    
    # Compute t using einsum for both wp and wq
    t = (
        1 - r
        - torch.einsum('bic,bijc->bij', r, wp)
        - torch.einsum('bic,bijc->bij', r.transpose(1, 2), wq).transpose(2, 1)
    )
    
    # Ensure non-negative values
    return t.relu()


def compute_spv_w(cfg, model, r, p, q):
    num_agents = cfg.num_agents
    device = cfg.device
    G = Data(cfg)

    P,Q = p.to(device).detach().cpu().numpy().copy(),q.to(device).detach().cpu().numpy().copy()
    spv_w = torch.zeros((num_agents,num_agents)).to(device)
    for agent_idx in range(num_agents):
        P_mis, Q_mis = G.generate_all_misreports(P, Q, agent_idx = agent_idx, is_P = True, include_truncation = False)
        p_mis, q_mis = torch.Tensor(P_mis).to(device), torch.Tensor(Q_mis).to(device)
        r_mis = model(p_mis.view(-1, num_agents, num_agents), q_mis.view(-1, num_agents, num_agents))
        r_mis = r_mis.view(p.shape[0],-1,num_agents,num_agents)

        r_mis_agent = r_mis[:,:,agent_idx,:]

        r_agent = r[:,agent_idx,:]
        r_agent = r_agent.repeat(1,r_mis_agent.shape[1]).view(r_mis_agent.shape[0],r_mis_agent.shape[1],r_mis_agent.shape[2])

        for f in range(num_agents):
            mask = torch.where(p[:,agent_idx,:]>=p[:,agent_idx,f].view(-1,1),1,0)
            mask = mask.repeat(1,r_mis_agent.shape[1]).view(r_mis_agent.shape[0],r_mis_agent.shape[1],r_mis_agent.shape[2])
            spv_w[agent_idx,f] = ((r_mis_agent - r_agent)*mask).sum(-1).relu().sum(-1).mean()
    return spv_w

def compute_spv_f(cfg, model, r, p, q):
    num_agents = cfg.num_agents
    device = cfg.device
    G = Data(cfg)

    P,Q = p.to(device).detach().numpy().copy(),q.to(device).detach().numpy().copy()
    spv_f = torch.zeros((num_agents,num_agents)).to(device)
    for agent_idx in range(num_agents):
        P_mis, Q_mis = G.generate_all_misreports(P, Q, agent_idx = agent_idx, is_P = False, include_truncation = False)
        p_mis, q_mis = torch.Tensor(P_mis).to(device), torch.Tensor(Q_mis).to(device)
        r_mis = model(p_mis.view(-1, num_agents, num_agents), q_mis.view(-1, num_agents, num_agents))
        r_mis = r_mis.view(p.shape[0],-1,num_agents,num_agents)

        r_mis_agent = r_mis[:,:,:,agent_idx]

        r_agent = r[:,:,agent_idx]
        r_agent = r_agent.repeat(1,r_mis_agent.shape[1]).view(r_mis_agent.shape[0],r_mis_agent.shape[1],r_mis_agent.shape[2])

        for w in range(num_agents):
            mask = torch.where(q[:,:,agent_idx]>=q[:,w,agent_idx].view(-1,1),1,0)
            mask = mask.repeat(1,r_mis_agent.shape[1]).view(r_mis_agent.shape[0],r_mis_agent.shape[1],r_mis_agent.shape[2])
            spv_f[w,agent_idx] = ((r_mis_agent - r_agent)*mask).sum(-1).relu().sum(-1).mean()
    return spv_f



def compute_loss(cfg, model, r, p, q, lambd, rho):
    t = compute_t(r,p,q)
    spv_w = compute_spv_w(cfg,model,r,p,q)
#    spv_f = compute_spv_f(cfg,model,r,p,q) 受け手のspは今回無視

    constr_vio = spv_w#+spv_f

    obj = t.sum(-1).sum(-1).mean()

    efficiency_loss = compute_efficiency_loss(cfg, r, p, q)

    lambd = torch.Tensor(lambd).to(cfg.device)
    loss = obj + (constr_vio*lambd).sum() + 0.5*rho*constr_vio.square().sum() + efficiency_loss # 3項目は大きいものを強く抑制するため

    return loss, constr_vio, obj, efficiency_loss
