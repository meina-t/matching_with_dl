import itertools
import numpy as np

def is_blocking_pair(P, Q, matching, i, j):
    """
    i と j がブロッキングペアであるかを判定。
    - P: 提案者の好み行列 (NumPy配列)
    - Q: 受け手の好み行列 (NumPy配列)
    - matching: 現在のマッチング (list of pairs)
    """
    current_partner_i = matching[i]
    current_partner_j = matching[j]

    # 各比較がスカラ値となるようにインデックスを調整
    return (
        Q[j, i] > Q[j, current_partner_j] and  # j が i を今の相手より好む
        P[i, j] > P[i, current_partner_i]      # i が j を今の相手より好む
    )


def find_stable_matchings(P, Q):
    """
    全ての安定確定マッチングを列挙。
    """
    num_agents = P.shape[0]
    stable_matchings = []
    all_possible_matchings = itertools.permutations(range(num_agents), num_agents)

    for matching in all_possible_matchings:
        matching_valid = True
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j and is_blocking_pair(P, Q, matching, i, j):
                    matching_valid = False
                    break
            if not matching_valid:
                break
        if matching_valid:
            stable_matchings.append(matching)
    return stable_matchings

def is_pareto_dominated(matching1, matching2, P, Q):
    """
    matching1 が matching2 にパレート支配されているか判定。
    """
    dominated = False
    for i in range(P.shape[0]):
        if not (P[i, matching1[i]] >= P[i, matching2[i]] and Q[matching1[i], i] >= Q[matching2[i], i]):
            return False
        if P[i, matching1[i]] < P[i, matching2[i]] or Q[matching1[i], i] < Q[matching2[i], i]:
            dominated = True
    return dominated

def find_efficient_stable_matchings(P, Q):
    """
    効率的安定マッチングを全て列挙。
    """
    stable_matchings = find_stable_matchings(P, Q)
    efficient_matchings = []
    for matching in stable_matchings:
        dominated = False
        for other_matching in stable_matchings:
            if matching != other_matching and is_pareto_dominated(matching, other_matching, P, Q):
                dominated = True
                break
        if not dominated:
            efficient_matchings.append(matching)
    return efficient_matchings

def compute_efficiency_loss(P, Q, r, efficient_matchings):
    """
    効率的安定マッチングに基づく efficiency_loss を計算。
    """
    num_agents = P.shape[0]
    min_loss = float('inf')
    for matching in efficient_matchings:
        loss = 0
        for i in range(num_agents):
            for j in range(num_agents):
                r_ij = r[:, i, j]  # 提案者iが受け手jとマッチする確率
                preferred_partners = (P[i, :] >= P[i, j]).astype(float)
                preferred_partners_prob = (r[:, i, :] * torch.Tensor(preferred_partners).to(r.device)).sum(-1)

                efficient_partner_prob = (matching[i] == j).astype(float)
                loss += (preferred_partners_prob - efficient_partner_prob).sum()
        min_loss = min(min_loss, loss)
    return min_loss
