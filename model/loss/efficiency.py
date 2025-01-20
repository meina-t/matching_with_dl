import torch
import itertools
from model.utils import is_pareto_dominates
from model.loss.stability import compute_sv

def compute_ev(cfg, r, p, q, data):
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

    stable_matchings = generate_stable_matchings(cfg, p, q, data)
    efficient_matchings = filter_efficient_stable_matchings(stable_matchings, p)

    for batch_idx in range(batch_size):
        p_batch = p[batch_idx]
        efficient_matchings_batch = efficient_matchings[batch_idx]

        pareto_dominated = False # pareto支配されていなければOK
        for mu in efficient_matchings_batch:
            mu_dominates_r = is_pareto_dominates(p_batch, mu, r[batch_idx])
            if mu_dominates_r:
                pareto_dominated = True
                break

        if not pareto_dominated:
            batch_efficiency_loss = torch.zeros((num_agents, num_agents), device=device)
        else:
            batch_efficiency_loss = torch.ones((num_agents, num_agents), device=device)
            batch_efficiency_loss_sum = batch_efficiency_loss.sum()
            for mu in efficient_matchings_batch:
                loss_for_matching = calc_efficiency_loss(r[batch_idx], mu, p_batch)
                if loss_for_matching.sum() < batch_efficiency_loss_sum:
                    batch_efficiency_loss_sum = loss_for_matching.sum()
                    batch_efficiency_loss = loss_for_matching
        efficiency_loss.append(batch_efficiency_loss)

    # 平均損失を返す
    return torch.stack(efficiency_loss).mean(dim=0)

def generate_stable_matchings(cfg, p, q, data):
    """
    batchに対して安定確定マッチングを生成する。
    - p: 提案者の選好行列 (3D tensor, batch_size x num_agents x num_agents)
    - q: 受け手の選好行列 (3D tensor, batch_size x num_agents x num_agents)
    - data.allpossible_matchings: 全てのマッチングのリスト (num_matchings x num_agents x num_agents)
    """
    all_possible_matchings = data.all_possible_matchings
    stable_matchings = []

    for batch_idx in range(p.size(0)):
        stable_matchings_batch = []
        p_batch = p[batch_idx]
        q_batch = q[batch_idx]
        for matching in all_possible_matchings:
            sv = compute_sv(cfg, matching, p_batch, q_batch)
            if sv.sum() == 0:
                stable_matchings_batch.append(matching)
        stable_matchings.append(stable_matchings_batch)

    return stable_matchings

def filter_efficient_stable_matchings(stable_matchings, p):
    """
    効率的安定マッチングをフィルタリングする。
    - stable_matchings: 安定確定マッチングのリスト(batch_size x num_agents x num_agents)
    - p: 提案者の選好行列 (3D tensor, batch * num_agents x num_agents)
    """
    efficient_matchings = []

    for batch_idx in range(p.shape[0]):
        stable_matchings_batch = stable_matchings[batch_idx]
        p_batch = p[batch_idx]
        efficient_matchings_batch = []
        for mu in stable_matchings_batch:
            is_efficient = True
            for mu_prime in stable_matchings_batch:
                if not torch.equal(mu, mu_prime):
                    pareto_dominated = is_pareto_dominates(p_batch, mu_prime, mu)
                    if pareto_dominated:
                        is_efficient = False
                        break
            if is_efficient:
                efficient_matchings_batch.append(mu)
        efficient_matchings.append(efficient_matchings_batch)
    return efficient_matchings


def calc_efficiency_loss(r_batch, mu, p_batch):
    """
    効率性損失を計算する。
    - r: マッチング確率行列 (2D tensor, num_agents x num_agents)
    - mu: 効率マッチング (2D tensor, num_agents x num_agents)
    - p: 提案者の選好行列 (2D tensor, num_agents x num_agents)
    - q: 受け手の選好行列 (2D tensor, num_agents x num_agents)
    """

    """
    loss = 0.0
    # i, jはマッチング相手
    for i in range(cfg.num_agents):
        for j in range(cfg.num_agents):
            # sは比較対象
            for s in range(cfg.num_agents):
                    if p[i, j] <= p[i, s]:
                        loss += matching[i, s] 
                        loss -= r[i, s]
    """
    prob_r = calc_favorable_match_prob(r_batch, p_batch)
    prob_mu = calc_favorable_match_prob(mu, p_batch)
    loss = prob_mu - prob_r               
    return loss

def calc_favorable_match_prob(r_batch, p_batch):
    """
    r: マッチング確率行列 (2D tensor, num_agents x num_agents)
    p: 提案者の選好行列 (2D tensor, num_agents x num_agents)
    
    戻り値:
    3x3テンソル: 各エージェントが相手またはそれ以上に好ましい相手とマッチする確率
    """
    num_agents = r_batch.size(0)
    result = torch.zeros_like(r_batch)
    
    for i in range(num_agents):
        for j in range(num_agents):
            # p[i][j] 以上の選好を持つ相手 s を見つける
            mask = p_batch[i] >= p_batch[i][j]
            # 対応する確率 r[i][s] を合計
            result[i][j] = r_batch[i][mask].sum()
    
    return result