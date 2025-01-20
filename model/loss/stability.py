import torch
import logging

def compute_sv(cfg, r, p, q):
    """
    Compute the loss/metric t based on r, p, and q tensors.

    Args:
        r (torch.Tensor): Allocation matrix (batch_size x num_agents x num_agents)
        p (torch.Tensor): Preference matrix (batch_size x num_agents x num_agents)
        q (torch.Tensor): Preference matrix (batch_size x num_agents x num_agents)

    Returns:
        torch.Tensor: Computed tensor t with non-negative values.

    """
    device = cfg.device
    num_agents = cfg.num_agents

    # Compute wp (preference weight matrix for p)
    p_expanded = p.unsqueeze(-1)  # Shape: [batch, p1, q1, 1]
    p_diff = p_expanded - p.unsqueeze(-2)  # Shape: [batch, p1, q1, q1]

    # Apply max(., 0) element-wise
    #p_diff_clipped = torch.clamp(p_diff, min=0)  # Shape: [batch, p1, q1, q1]
    p_diff_clipped = (p_diff > 0).float()

    # Multiply r with p_diff_clipped along the appropriate dimension
    r_expanded = r.unsqueeze(-2)  # Shape: [batch, p1, 1, q1]
    weighted_diff = r_expanded * p_diff_clipped  # Shape: [batch, p1, q1, q1]

    # Sum over q' (dim=-1)
    wp = weighted_diff.sum(dim=-1)
    

    # Compute wq (preference weight matrix for q)
    r_transposed = r.transpose(-1, -2)  # Shape: [batch, q, p]

    # Expand q for broadcasting
    q_expanded = q.unsqueeze(-1)  # Shape: [batch, q, p, 1]
    q_diff = q_expanded - q.unsqueeze(-2)  # Shape: [batch, q, p, p]

    # Apply max(., 0) element-wise
    #q_diff_clipped = torch.clamp(q_diff, min=0)  # Shape: [batch, q, p, p]
    q_diff_clipped = (q_diff > 0).float()

    # Multiply r_transposed with q_diff_clipped along the appropriate dimension
    r_expanded = r_transposed.unsqueeze(-2)  # Shape: [batch, q, 1, p]
    weighted_diff = r_expanded * q_diff_clipped  # Shape: [batch, q, p, p]

    # Sum over p' (dim=-1)
    wq = weighted_diff.sum(dim=-1)  # Shape: [batch, q, p]

    # Transpose wq back to [batch, p, q]
    wq = wq.transpose(-1, -2)

    sv = wp * wq
    # Average over the batch dimension
    sv = sv.mean(dim=0)
    
    # Ensure non-negative values
    return sv
