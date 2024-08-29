import torch
import pickle as pkl
from generate_graph import *
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

def get_metrics(x: torch.tensor, is_initialized: bool):
    """
    This function either loads a precomputed dictionary of similarity/distance matrices from a 
    file or computes these matrices using various metrics. It's main function is to save running
    time when running test on the same metric (every metric should be computed once).
    
    Args:
        x: A 2D tensor of shape (num_nodes, num_features) representing the node features.
        is_initialized: A flag indicating whether to load a precomputed dictionary of 
                        similarity/distance matrices  or to compute the matrices.

    Returns:
            A dictionary where keys are metric names and values are the corresponding 2D 
            arrays containing similarity/distance values between nodes.
    """
    if is_initialized:
        with open('metrics_dict.pkl', 'rb') as f:
            return pkl.load(f)
    else:
        metrics_dict = dict()
        for metric in ['euclidian', 'max_norm', 'city_block', 'cosine', 'chord']:
            metrics_dict[metric] = get_similarities(x, metric)
        return metrics_dict

class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels, output_dim, seed=1):
        super().__init__()
        torch.cuda.manual_seed(seed)
        self.conv1 = SAGEConv((-1, -1), hidden_channels//2, aggr="mean", normalize=True)
        self.conv2 = SAGEConv((-1, -1), hidden_channels//4, aggr="mean", normalize=True)
        self.conv3 = SAGEConv((-1, -1), output_dim, aggr="mean", normalize=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)


def get_accuracy(data_df: pd.DataFrame, metric: str, seed, lower_threshold, upper_threshold, is_initialized=True):
    x, y, edges = construct_graph(data_df, [metric, lower_threshold, upper_threshold], is_initialized)
    train_indeces = list(range(0,400)) + list(range(500,900)) + list(range(1000,1400)) + list(range(1500,1900))
    valid_indeces = list(range(400,450)) + list(range(900,950)) + list(range(1400,1450)) + list(range(1900,1950))
    test_indeces = list(range(450,500)) + list(range(950,1000)) + list(range(1450,1500)) + list(range(1950,2000))
    train_mask = torch.tensor([1 if i in train_indeces else 0 for i in range(x.shape[0])], dtype=torch.bool)
    valid_mask = torch.tensor([1 if i in valid_indeces else 0 for i in range(x.shape[0])], dtype=torch.bool)
    test_mask = torch.tensor([1 if i in test_indeces else 0 for i in range(x.shape[0])], dtype=torch.bool)
    data = Data(x=x, y=y, edge_index=edges, train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)

    output_dim = len(set(data.y.tolist()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphSAGE(x.shape[1], output_dim, seed).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)

    def train():
        model.train()
        optimizer.zero_grad()
        F.nll_loss(model(data.x, data.edge_index)[train_mask], data.y[train_mask]).backward()
        optimizer.step()

    def test():
        model.eval()
        logits = model(data.x, data.edge_index)
        accs = []
        for mask in [train_mask, valid_mask, test_mask]:
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        return accs
    
    best_val_acc = test_acc = 0
    val_acc_lst, test_acc_lst = [], []
    for epoch in tqdm(range(100)):
        train()
        if epoch % 10 == 0 or epoch == 99:
            _, val_acc, tmp_test_acc = test()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            # log = 'Epoch: {:03d}, Val: {:.4f}, Test: {:.4f}'
            val_acc_lst.append(val_acc)
            test_acc_lst.append(test_acc)
            # print(log.format(epoch, best_val_acc, test_acc))

    return val_acc_lst, test_acc_lst
    