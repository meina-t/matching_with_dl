import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from itertools import permutations
from data import Data

def pref_to_num(p_agent):
    device = p_agent.device
    p_agent = p_agent.to('mps').detach().numpy().copy()
    num_agents = p_agent.shape[-1]
    p_agent = p_agent*num_agents
    all_pref = np.array(list(permutations(np.arange(num_agents))))
    return torch.Tensor(np.array([np.where((all_pref+1==pref).all(axis=1))[0][0] for pref in p_agent])).to(device).to(int)

def compute_xloss(x):
    return x.repeat(1,x.shape[1]).reshape((-1,x.shape[1],x.shape[1]))

def compute_yloss(y):
    return y.view(-1,1).repeat(1,y.shape[1]).reshape((-1,y.shape[1],y.shape[1]))
    
def compute_zloss(z,p,q):
    wp = torch.where(p[:, :, None, :] - p[:, :, :, None]<0,1,0).to(torch.float)
    wq = torch.where(q[:, :, None, :] - q[:, None, :, :]<0,1,0).to(torch.float)
    zloss = z + torch.einsum('bic,bijc->bij', z, wp) + torch.einsum('bic,biac->bac', z, wq)
    return zloss

def compute_uloss(cfg, model, u ,p, q):
    num_agents = cfg.num_agents
    batch_size = p.shape[0]
    device = cfg.device
    G = Data(cfg)
    all_prefs = torch.Tensor(np.array(list(permutations(np.arange(num_agents))))+1).to(device)/num_agents

    P = p.to('mps').detach().numpy().copy()
    Q = q.to('mps').detach().numpy().copy()
    ulosses = torch.zeros((batch_size,num_agents,num_agents)).to(device)

    for agent_idx in range(num_agents):
        P_mis, Q_mis = G.generate_all_misreports(P, Q, agent_idx = agent_idx, is_P = True, include_truncation = False)
        p_mis, q_mis = torch.Tensor(P_mis).to(device), torch.Tensor(Q_mis).to(device)

        _,_,_,u_mis,_ = model(p_mis.view(-1, num_agents, num_agents), q_mis.view(-1, num_agents, num_agents))
        u_mis = u_mis.view(-1,u.shape[2],num_agents,u.shape[2],num_agents)

        u_agent = u[:,agent_idx,:,:]

        u_mis_agent = u_mis[torch.arange(p.shape[0]),:,agent_idx,pref_to_num(p[:,agent_idx,:])]

        for f in range(num_agents):
            mask = torch.where(p[:,agent_idx,:]<p[:,agent_idx,f].view(-1,1),1,0)
            sum_agent = torch.einsum("bj,bij->b",mask.to(float),u_agent.to(float))

            mask_mis = torch.where(all_prefs[:,:]<all_prefs[:,f].view(-1,1),1,0)
            sum_mis_agent = torch.einsum("ij,bij->b",mask_mis.to(float),u_mis_agent.to(float))
            
            ulosses[:,agent_idx,f] = (sum_agent-sum_mis_agent)
        
    return ulosses

def compute_vloss(cfg, model, v, p, q):
    num_agents = cfg.num_agents
    batch_size = p.shape[0]
    device = cfg.device
    G = Data(cfg)
    all_prefs = torch.Tensor(np.array(list(permutations(np.arange(num_agents))))+1).to(device)/num_agents

    P = p.to('mps').detach().numpy().copy()
    Q = q.to('mps').detach().numpy().copy()
    vlosses = torch.zeros((batch_size,num_agents,num_agents)).to(device)

    for agent_idx in range(num_agents):
        P_mis, Q_mis = G.generate_all_misreports(P, Q, agent_idx = agent_idx, is_P = False, include_truncation = False)
        p_mis, q_mis = torch.Tensor(P_mis).to(device), torch.Tensor(Q_mis).to(device)

        _,_,_,_,v_mis = model(p_mis.view(-1, num_agents, num_agents), q_mis.view(-1, num_agents, num_agents))
        v_mis = v_mis.view(-1,v.shape[2],num_agents,v.shape[2],num_agents)

        v_agent = v[:,agent_idx,:,:]
        v_mis_agent = v_mis[torch.arange(q.shape[0]),:,agent_idx,pref_to_num(q[:,:,agent_idx])]

        for w in range(num_agents):
            mask = torch.where(q[:,:,agent_idx]<q[:,w,agent_idx].view(-1,1),1,0)
            sum_agent = torch.einsum("bj,bij->b", mask.to(float), v_agent.to(float))

            mask_mis = torch.where(all_prefs[:,:]<all_prefs[:,w].view(-1,1),1,0)
            sum_agent_mis = torch.einsum("ij,bij->b", mask_mis.to(float), v_mis_agent.to(float))

            vlosses[:,w,agent_idx] = sum_agent - sum_agent_mis

    return vlosses

def compute_constraint_vio(cfg, model, x, y, z, u, v, p, q):
    x_loss = compute_xloss(x)
    y_loss = compute_yloss(y)
    z_loss = compute_zloss(z,p,q)
    u_loss = compute_uloss(cfg,model,u,p,q)
    v_loss = compute_vloss(cfg,model,v,p,q)

    total_constraints = F.relu(-(x_loss+y_loss-z_loss-u_loss-v_loss))

    return total_constraints.mean(0)

def compute_loss(cfg, model, x, y, z, u, v, p, q, lambd, rho):
    lambd = torch.Tensor(lambd).to(cfg.device)
    obj = (x.sum(-1) + y.sum(-1) - z.sum(-1).sum(-1)).mean()
    constr_vio = compute_constraint_vio(cfg,model,x,y,z,u,v,p,q)
    return obj + (constr_vio*lambd).sum() + 0.5*rho*constr_vio.square().sum(), constr_vio, obj
