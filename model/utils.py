import torch
import numpy as np

def is_pareto_dominates(p, mu, nu):
    """
    return True if mu paretodominates nu in proposers' preference matrix

    Args:
        p: Proposers' preference matrix. [num_agents x num_agents]
        mu: Matching matrix. [num_agents x num_agents]
        nu: Matching matrix. [num_agents x num_agents]
    """
    pareto_dominates = True
    strict_flag = False
    num_agents = p.shape[0]
    
    for i in range(num_agents):
        for j in range(num_agents):
            # p[i][j] 以上の選好を持つ相手 s を見つける
            mask = p[i] >= p[i][j]
            # 対応する確率 r[i][s] を合計
            r_mu = mu[i][mask].sum()
            r_nu = nu[i][mask].sum()
            if r_mu > r_nu:
                strict_flag = True
            elif r_mu < r_nu:
                pareto_dominates = False
                break
    if not strict_flag:
        pareto_dominates = False
    return pareto_dominates

def da_with_t(p, q):
    """
    - p: 提案者の選好行列 (batch_size x num_agents x num_agents)
    - q: 受け手の選好行列 (batch_size x num_agents x num_agents)
    """
    num_agents = p.shape[1]
    batch_size = p.shape[0]
    device = 'mps'

    r = []
    for batch_idx in range(batch_size):
        p_batch = p[batch_idx]
        q_batch = q[batch_idx]

        # Convert PyTorch tensors to NumPy arrays
        p_array = p_batch.cpu().numpy() if isinstance(p, torch.Tensor) else np.array(p)
        q_array = q_batch.cpu().numpy() if isinstance(q, torch.Tensor) else np.array(q)
    
        tie_broken_q = tie_break(q_array)
        num_tie_breaks = len(tie_broken_q)
        total_result = np.zeros((3, 3))
    
        for qt in tie_broken_q:
            result = deferred_acceptance(p_array, qt)  # Pass NumPy arrays
            total_result += result

        output =  total_result / num_tie_breaks
        r.append(output)
    r = torch.from_numpy(np.array(r)).float().to(device)
    return r

def tie_break(q_batch):
    """
    3x3テンソルに対してタイブレーク処理を行う関数。

    Args:
        tensor: 3x3のNumPy配列。

    Returns:
        タイブレーク後のテンソルを格納したNumPy配列のリスト。
    """

    results = [q_batch]
    for i in range(3):  # 各行に対して処理
        row = q_batch[i].copy()
        unique_values, counts = np.unique(row, return_counts=True)
        tie_indices = np.where(counts > 1)[0]

        if len(tie_indices) == 0:
            continue

        new_results = []
        for result in results:
            current_row = result[i].copy()
            if len(tie_indices) == 1: # 同じ値が2個の場合
                value_index = np.where(row == unique_values[tie_indices[0]])[0]
                for j in range(2):
                    new_row = current_row.copy()
                    new_row[value_index[j]] += 0.1
                    new_result = result.copy()
                    new_result[i] = new_row
                    new_results.append(new_result)
            elif len(tie_indices) == 3: # 同じ値が3個の場合
                for p in itertools.permutations([0.2,0.1,0.0]):
                    new_row = current_row.copy()
                    for k in range(3):
                        new_row[k] += p[k]
                    new_result = result.copy()
                    new_result[i] = new_row
                    new_results.append(new_result)
        results = new_results
    return np.array(results)

def deferred_acceptance(p, q, n=3):
    """
    Deferred Acceptance Algorithm.

    Args:
        p: Proposers' preference matrix. n*n
        q: Reviewers' preference matrix. n*n
        n: Number of proposers/reviewers.

    Returns:
        Matching output matrix.
    """
    # Create a copy of p to avoid modifying the original preference matrix
    p_copy = p.copy()

    matching_output = np.zeros((n, n), dtype=int)
    remain_p = list(range(n))

    while remain_p:
        pi = remain_p.pop(0)
        qi = np.argmax(p_copy[pi])

        if matching_output[:, qi].sum() == 0:
            matching_output[pi][qi] = 1
        else:
            current_match = np.where(matching_output[:, qi] == 1)[0][0]
            if q[qi][pi] > q[qi][current_match]:
                matching_output[current_match][qi] = 0
                matching_output[pi][qi] = 1
                remain_p.append(current_match)
            else:
                remain_p.append(pi)

        # Update the COPY of p
        p_copy[pi][qi] = -1

    return matching_output