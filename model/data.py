import itertools
import numpy as np
import torch
import random

class Data(object):
    """
    A class for generating data for the matching problem
    """
    
    def __init__(self, cfg): 
        self.cfg = cfg
        self.num_agents = cfg.num_agents
        self.corr = cfg.corr
        self.device = cfg.device

        self.mis_array = torch.tensor(self.generate_mis_array(include_truncation = False), device = self.device, dtype=torch.float32)
        self.all_possible_matchings = self.generate_all_possible_matchings(num_agents = self.num_agents)
    
    def generate_mis_array(self, allow_tie = True, include_truncation = False):
        """ 
        Generates all possible rankings 
        Arguments
            include_truncation: Whether to include truncations or only generate complete rankings
        Returns:
            P_mis: [m, num_agents]
                where m = N! if complete, (N+1)! if truncations are included
        """
        if allow_tie:
            if include_truncation is False:
                m = np.array(list(itertools.product(np.arange(self.num_agents), repeat=self.num_agents))) + 1.0
            else:
                m = np.array(list(itertools.product(np.arange(self.num_agents + 1), repeat=self.num_agents)))
                m = (m - m[:, -1:])[:, :-1]
        else:
            if include_truncation is False:
                m = np.array(list(itertools.permutations(np.arange(self.num_agents)))) + 1.0
            else:
                m = np.array(list(itertools.permutations(np.arange(self.num_agents + 1))))
                m = (m - m[:, -1:])[:, :-1]
            
        return m/self.num_agents
    
    def generate_all_possible_matchings(self, num_agents = 3):
        """
        Generates all possible matchings for a given number of agents
        Arguments:
            num_agents: Number of agents
        Returns:
            matchings: [num_matchings, num_agents, num_agents]
        """
        permutations = list(itertools.permutations(range(num_agents)))

        # Create matchings as a single numpy array
        matchings = np.zeros((len(permutations), num_agents, num_agents))
        for idx, perm in enumerate(permutations):
            for i in range(num_agents):
                matchings[idx, i, perm[i]] = 1

        # Convert to torch.Tensor
        return torch.tensor(matchings, dtype=torch.float32, device=self.device)

    def get_batch(self, batch_size, seed,  prob = 0, corr = None):
        """
        Samples a batch of data from training
        Arguments
            batch_size: number of samples
            prob: probability of truncation
        Returns
            P: Men's preferences, 
                P_{ij}: How much Man-i prefers to be Women-j
            Q: Women's preferences,
                Q_{ij}: How much Woman-j prefers to be with Man-i
        """
        random.seed(seed)
        if corr is None: corr = self.corr
        
        N = batch_size * self.num_agents
        
        P = self.sample_ranking(N, prob)
        Q = self.sample_ranking_with_ties(N, prob) 
        
        P = P.reshape(-1, self.num_agents, self.num_agents)                           
        Q = Q.reshape(-1, self.num_agents, self.num_agents)
                
        if corr > 0.00:
            P_common = self.sample_ranking(batch_size, prob).reshape(batch_size, 1, self.num_agents)
            Q_common = self.sample_ranking_with_ties(batch_size, prob).reshape(batch_size, 1, self.num_agents)
        
            P_idx = np.random.binomial(1, corr, [batch_size, self.num_agents, 1])
            Q_idx = np.random.binomial(1, corr, [batch_size, self.num_agents, 1])
        
            P = P * (1 - P_idx) + P_common * P_idx
            Q = Q * (1 - Q_idx) + Q_common * Q_idx
                
        return P, Q
    

    def sample_ranking(self, N, prob=0):
        """ 
        Samples ranked lists.

        Args:
            N (int): Number of samples.
            prob (float): Probability of truncation. Currently not used but can be integrated for truncation logic.

        Returns:
            torch.Tensor: Ranked list of shape [N, num_agents].
        """
        num_agents = self.num_agents

        # Efficiently sample permutations
        p_list = np.array(list(itertools.permutations(range(1, num_agents + 1))))
        if prob > 0:
            # Optional truncation logic (currently placeholder)
            truncation_mask = np.random.rand(N, num_agents) < prob
            p_list = np.where(truncation_mask, 0, p_list)

        # Randomly select N samples
        indices = np.random.choice(len(p_list), size=N, replace=True)
        samples = p_list[indices]
        samples = samples / np.sum(samples, axis=1, keepdims=True)

        # Convert to tensor and normalize
        return torch.tensor(samples, dtype=torch.float32, device=self.device)

        
    def sample_ranking_with_ties(self, N, prob = 0):
        """ 
        Samples ranked lists with ties
        Arguments
            N: Number of samples
            prob: Probability of truncation       
        Returns:
            Ranked List of shape [N, Num_agents] (with ties)
        """
        num_agents = self.num_agents

        num_agents = self.num_agents

        # Generate samples directly as a NumPy array
        samples = np.random.randint(1, num_agents + 1, size=(N, num_agents))

        # Normalize each row if needed
        samples = samples / samples.sum(axis=1, keepdims=True)

        return torch.tensor(samples, dtype=torch.float32, device=self.device)
    
    ################ Used in calculating efficiency loss ################
    def compose_misreport(self, p, q, m, agent_idx, is_P = True):
        """ Composes mis-report
        Arguments:
            P: Men's preference, [Batch_size, num_agents, num_agents]
            Q: Women's preference [Batch_size, num_agents, num_agents]
            M: Ranked List of mis_reports
                    either [num_misreports, num_agents]
                    or [batch_size, num_misreports, num_agents]                    
            agent_idx: Agent-idx that is mis-reporting
            is_P: if True, Men[agent-idx] misreporting 
                    else, Women[agent-idx] misreporting
                    
        Returns:
            P_mis, Q_mis: [batch-size, num_misreports, num_agents, num_agents]
            
        """
        
        num_misreports = m.shape[-2]
        P_mis = p.unsqueeze(1).repeat(1, num_misreports, 1, 1).to(self.device)
        Q_mis = q.unsqueeze(1).repeat(1, num_misreports, 1, 1).to(self.device)
        
        if is_P: P_mis[:, :, agent_idx, :] = m
        else: Q_mis[:, :, :, agent_idx] = m
        
        return P_mis, Q_mis