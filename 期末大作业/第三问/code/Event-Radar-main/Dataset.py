import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

class NewsDataset(Dataset):
    def __init__(self, graph_data, data_list):
        super(NewsDataset, self).__init__()
        self.data_list = data_list
        self.nodes_features, self.edges_data, _, _, _, self.type2nidxs = graph_data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        instance = self.data_list[index]
        item_id = instance["Id"]
        label = instance["label"]

        nodes = self.nodes_features[item_id]
        edges_structure = self.edges_data[item_id]
        type2nidxs = self.type2nidxs[item_id]

        # --- FIX: 将节点类型索引存储为独立的张量属性 ---
        # PyG的DataLoader会自动识别这些属性，并在批处理时正确地拼接和偏移索引。
        graph_obj = Data(
            x=nodes.float(),
            edge_index=edges_structure['index'],
            y=torch.LongTensor([label]),
            id=item_id,
            # 将字典分解为独立的属性
            post_idx=torch.tensor(type2nidxs['post_subgraph'], dtype=torch.long),
            image_idx=torch.tensor(type2nidxs['image_subgraph'], dtype=torch.long),
            num_nodes=nodes.shape[0]
        )
        return graph_obj

    @staticmethod
    def collate_fn(samples):
        # 使用PyG标准的批处理函数，它现在能正确处理我们新增的 post_idx 和 image_idx 属性。
        batch = Batch.from_data_list(samples)
        return batch