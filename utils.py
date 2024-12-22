import itertools
import numpy as np
import torch
from tqdm import tqdm
from primal_loss import compute_spv_w, compute_t
from efficiency_loss import compute_efficiency_loss

device = "mps" 

def generate_all_patterns():
    patterns = []
    
    # 種類1: no tie (3,2,1の順列)
    no_tie_patterns = list(itertools.permutations([3, 2, 1]))
    patterns.extend(no_tie_patterns)
    
    # 種類2: one tie
    # 2-1: 二つが2, 一つが1
    one_tie_2_1 = set(itertools.permutations([2, 2, 1]))
    patterns.extend(one_tie_2_1)
    
    # 2-2: 一つが1, 二つが2 (同じ結果になるため省略可)
    one_tie_2_2 = set(itertools.permutations([2, 1, 1]))
    patterns.extend(one_tie_2_2)
    
    # 種類3: fulltie (1,1,1)
    full_tie_pattern = [(1, 1, 1)]
    patterns.extend(full_tie_pattern)
    
    return patterns

def normalize_tuples(tuples_list):
    normalized_list = []
    for tpl in tuples_list:
        total = sum(tpl)
        if total != 0:
            normalized_list.append(tuple(x / total for x in tpl))
        else:
            normalized_list.append(tpl)  # 合計が0の場合はそのまま追加
    return normalized_list

def deferred_acceptance(p, q, n=3):
    """
    Deferred Acceptance Algorithm.

    Args:
        p: Proposers' preference matrix.
        q: Reviewers' preference matrix.
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


def da_with_t(p, q):
    """
    - p: 提案者の選好行列 (batch_size x num_agents x num_agents)
    - q: 受け手の選好行列 (batch_size x num_agents x num_agents)
    """
    num_agents = p.shape[1]
    batch_size = p.shape[0]
    device = "mps"

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

def convert_to_float(df):
    def try_convert(value):
        """ 各要素を float に変換する補助関数 """
        try:
            return float(value)
        except (ValueError, TypeError):
            return value

    return df.applymap(try_convert)

def apply_features(cfg, model,df):
    model_efficiency_losses = []
    model_stability_losses = []
    model_sp_losses = []
    da_efficiency_losses = []
    da_stability_losses = []
    da_sp_losses = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        p = torch.tensor([row['p']], dtype=torch.float32).to(device)
        q = torch.tensor([row['q']], dtype=torch.float32).to(device)

        model_output = model(p, q)
        model_efficiency_loss = compute_efficiency_loss(cfg, model_output, p, q).cpu().item()
        model_stability_loss = compute_t(model_output, p, q).mean().cpu().detach().numpy()
        model_sp_loss = compute_spv_w(cfg, model, model_output, p, q).mean().cpu().detach().numpy()
        da_output = da_with_t(p, q)
        da_efficiency_loss = compute_efficiency_loss(cfg, da_output, p, q).cpu().detach().numpy()
        da_stability_loss = compute_t(da_output, p, q).mean().cpu().detach().numpy()
        da_sp_loss = compute_spv_w(cfg, da_with_t, da_output, p, q).mean().cpu().detach().numpy()

        model_efficiency_losses.append(model_efficiency_loss)
        model_stability_losses.append(model_stability_loss)
        model_sp_losses.append(model_sp_loss)
        da_efficiency_losses.append(da_efficiency_loss)
        da_stability_losses.append(da_stability_loss)
        da_sp_losses.append(da_sp_loss)

    df['model_efficiency_loss'] = model_efficiency_losses
    df['model_stability_loss'] = model_stability_losses
    df['model_sp_loss'] = model_sp_losses
    df['da_efficiency_loss'] = da_efficiency_losses
    df['da_stability_loss'] = da_stability_losses
    df['da_sp_loss'] = da_sp_losses
    return df