class HParams:
    def __init__(self, num_agents = 3, num_hidden_nodes = 32, batch_size = 32, epochs = 10000, corr = 0.0, device = 'mps', lr = 0.01):
        self.num_agents = num_agents
        self.num_hidden_nodes = num_hidden_nodes
        self.batch_size = batch_size
        self.epochs = epochs
        self.corr = corr
        self.device = device
        self.lr = lr
        

