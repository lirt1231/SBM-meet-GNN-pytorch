import os.path as osp
import pickle as pkl
from typing import Dict, List

import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx


class GraphDataset:
    def __init__(self, dataset: str, split_name: str) -> None:
        tx, allx, graph, test_idx, split_data = self._parse_files(dataset, split_name)
        node_features = self._get_full_node_features(allx, tx, test_idx)
        self.node_features = self._csr_to_torch(node_features)
        self.adj_orig = self._get_adj_mat(graph)
        """
        `adj_train`: scipy.sparse.csr_matrix (num_nodes, num_nodes)
        `edges_val`, `edges_val_neg`, `edges_test`, `edges_test_neg`: np.ndarray (N, 2)
        """
        adj_train, self.edges_val, self.edges_val_neg, self.edges_test, self.edges_test_neg = \
            split_data['adj_train'][0], split_data['val_edges'], split_data['val_edges_false'], \
            split_data['test_edges'], split_data['test_edges_false']
        self.adj_train = self._csr_to_torch(adj_train)

    @property
    def num_nodes(self) -> int:
        return self.adj_train.size(0)

    @property
    def num_features(self) -> int:
        return self.node_features.size(1)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict:
        return {
            "node_features": self.node_features,
            "adj_train": self.adj_train,
            "edges_val": self.edges_val,
            "edges_val_neg": self.edges_val_neg,
            "edges_test": self.edges_test,
            "edges_test_neg": self.edges_test_neg,
        }

    def collate(self, data: dict) -> Dict[str, torch.Tensor]:
        """Construct label and label mask."""
        labels = torch.FloatTensor(self.adj_orig.toarray())
        require_keys = ["edges_val", "edges_val_neg", "edges_test", "edges_test_neg"]
        label_mask = self.construct_adj_from_edges(np.vstack([data[k] for k in require_keys]))

        return {
            "x": data["node_features"],
            "adj_mat": data["adj_train"],
            "labels": labels.flatten(),
            "label_mask": label_mask.flatten(),
        }

    def construct_adj_from_edges(self, edges: np.ndarray) -> torch.BoolTensor:
        """Construct adjacency matrix from list of edges.

        Args:
            `edges`: numpy array in shape (num_edges, 2)

        Returns:
            Adjacency matrix of torch.BoolTensor in shape (num_nodes, num_nodes)
        """
        adj_mat = torch.zeros(self.adj_train.size(), dtype=torch.bool)
        adj_mat[edges[:, 0], edges[:, 1]] = True

        return adj_mat

    def _parse_files(self, dataset: str, split_name: str) -> tuple:
        """
        Returns:
            `tx`, `allx`: sparse matrices in CSR format, test and training node features
            `test_idx`: list, indices of test nodes in graph,
            `split_data`: edges partition for training, valid and test
        """
        folder = osp.join("./data", dataset)
        file_names, files = ['tx', 'allx', 'graph'], []
        for name in file_names:
            path = osp.join(folder, f"ind.{dataset.lower()}.{name}")
            with open(path, 'rb') as file:
                files.append(pkl.load(file, encoding='latin1'))

        path = osp.join(folder, f"ind.{dataset.lower()}.test.index")
        with open(path, 'r') as file:
            test_idx = list(map(lambda x: int(x.strip()), file))

        split_data = np.load(
            osp.join(folder, split_name),
            allow_pickle=True, encoding='latin1'
        )

        return *files, test_idx, split_data

    def _get_full_node_features(
        self,
        allx: sp.csr_matrix,
        tx: sp.csr_matrix,
        test_idx: List[int]
    ) -> sp.csr_matrix:
        """
        The indices of test instances in graph are from #x to #x + #tx - 1, with the same order as in tx, while the size of `tx` is smaller than the range of `test_idx`.
        """
        test_idx_range = max(test_idx) - min(test_idx) + 1
        num_nodes = test_idx_range + allx.shape[0]
        num_features = allx.shape[1]

        features = sp.lil_matrix((num_nodes, num_features))
        features[:allx.shape[0], :] = allx
        features[test_idx, :] = tx

        return features

    def _get_adj_mat(self, graph: Dict[str, List[int]]) -> sp.csr_matrix:
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj.eliminate_zeros()
        return adj

    def _csr_to_torch(self, csr_matrix: sp.csr_matrix) -> torch.Tensor:
        coo_matrix = csr_matrix.tocoo()
        return torch.sparse_coo_tensor(
            np.vstack([coo_matrix.row, coo_matrix.col]),
            coo_matrix.data, coo_matrix.shape, dtype=torch.float32
        )
