import itertools
import numpy as np
import torch


def generate_stable_matchings(p, q):
    """
    安定確定マッチングを生成する。
    - p: 提案者の選好行列 (2D tensor, num_agents x num_agents)
    - q: 受け手の選好行列 (2D tensor, num_agents x num_agents)
    """
    num_agents = p.shape[0]
    stable_matchings = []

    # ブロッキングペアのチェックロジック
    for possible_matching in itertools.permutations(range(num_agents)):
        is_stable = True
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j:
                    # ブロッキングペアの条件をチェック
                    if (
                        (p[i, possible_matching[j]] > p[i, possible_matching[i]]).item()
                        and (q[possible_matching[j], i] > q[possible_matching[j], j]).item()
                    ):
                        is_stable = False
                        break
            if not is_stable:
                break
        if is_stable:
            stable_matchings.append(possible_matching)
    return stable_matchings

def filter_efficient_stable_matchings(stable_matchings, p, q):
    """
    効率的安定マッチングをフィルタリングする。
    - stable_matchings: 安定確定マッチングのリスト
    - p: 提案者の選好行列 (2D tensor, num_agents x num_agents)
    - q: 受け手の選好行列 (2D tensor, num_agents x num_agents)
    """
    efficient_matchings = []

    for mu in stable_matchings:
        is_efficient = True
        for mu_prime in stable_matchings:
            if mu != mu_prime:
                # 強いパレート支配の条件をチェック
                pareto_dominates = True
                for i in range(len(mu)):
                    if (
                        p[i, mu_prime[i]] <= p[i, mu[i]]
                        and q[mu_prime[i], i] <= q[mu[i], i]
                    ):
                        pareto_dominates = False
                        break
                if pareto_dominates:
                    is_efficient = False
                    break
        if is_efficient:
            efficient_matchings.append(mu)

    return efficient_matchings

def calc_efficiency_loss(cfg, r, mu, p, q):
    """
    効率性損失を計算する。
    - r: マッチング確率行列 (2D tensor, num_agents x num_agents)
    - mu: マッチング (1D list, num_agents)
    - p: 提案者の選好行列 (2D tensor, num_agents x num_agents)
    - q: 受け手の選好行列 (2D tensor, num_agents x num_agents)
    """
    # マッチングを2D tensorに変換
    matching = torch.zeros((cfg.num_agents, cfg.num_agents), dtype=torch.float32).to(cfg.device)
    for i, j in enumerate(mu):
        matching[i][j] = 1.0
    
    loss = 0.0
    # i, jはマッチング相手
    for i in range(cfg.num_agents):
        for j in range(cfg.num_agents):
            # sは比較対象
            for s in range(cfg.num_agents):
                    if p[i, j] <= p[i, s]:
                        loss += matching[i, s] 
                        loss -= r[i, s]
    return loss


def compute_efficiency_loss(cfg, r, p, q):
    """
    効率性損失を計算する。
    - r: マッチング確率行列 (batch_size x num_agents x num_agents)
    - p: 提案者の選好行列 (batch_size x num_agents x num_agents)
    - q: 受け手の選好行列 (batch_size x num_agents x num_agents)
    """
    num_agents = p.shape[1]
    batch_size = p.shape[0]
    device = cfg.device

    total_efficiency_loss = 0.0

    for batch_idx in range(batch_size):
        p_batch = p[batch_idx]
        q_batch = q[batch_idx]

        stable_matchings = generate_stable_matchings(p_batch, q_batch)
        efficient_matchings = filter_efficient_stable_matchings(stable_matchings, p_batch, q_batch)

    batch_efficiency_loss = float("inf")
    for matching in efficient_matchings:
        loss_for_matching = calc_efficiency_loss(cfg, r[batch_idx], matching, p_batch, q_batch)
        if loss_for_matching < batch_efficiency_loss:
            batch_efficiency_loss = loss_for_matching

        
        # バッチ全体の損失を合計
        total_efficiency_loss += batch_efficiency_loss

    # 平均損失を返す
    return total_efficiency_loss / batch_size
