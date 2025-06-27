import random
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
import os
import torch
from transformers import logging
from typing import Dict
from torch.utils.data import DataLoader
import copy
import warnings

# --- 确保您的项目中有这些对应的文件 ---
# 您需要确保 Dataset.py 和 Model.py 也已经更新，以处理图数据
from Dataset import NewsDataset
from Model import FakeNewsDetection
from trainer import Trainer
from tester import Tester

# --- 初始化设置 ---
random.seed(1107)
torch.manual_seed(1107)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1107)

warnings.filterwarnings("ignore")
logging.set_verbosity_warning()
logging.set_verbosity_error()

TRAIN = "train"
DEV = "eval"
TEST = "test"
SPLITS = [TRAIN, DEV, TEST]


def pickle_reader(path):
    """一个简单的pickle文件读取函数"""
    print(f"Reading file: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def main(args):
    #### 初始化 DataLoader
    # 1. 加载包含 'Id' 和 'label' 的 pkl 文件
    data_paths = {split: args.data_dir / f"{split}.pkl" for split in SPLITS}
    data = {split: pickle_reader(path) for split, path in data_paths.items()}

    # --- MODIFICATION 1: 正确加载共享的图数据 ---
    # 新的数据预处理脚本将所有图数据（节点特征、边等）保存在一个共享文件里。
    print("Loading shared graph data...")
    graph_data_path = args.data_dir / "graph_data.pkl"
    shared_graph_data = pickle_reader(graph_data_path)
    # 将加载的单个图数据对象分配给所有 splits，以便 Dataset 可以使用它。
    graph_data = {split: shared_graph_data for split in SPLITS}
    print("Graph data loaded and assigned to all splits.")
    # --- END MODIFICATION 1 ---

    # 创建 Dataset 实例
    # **重要**: 这里的 NewsDataset 必须被修改为能够接收 graph_data 和 data_list，
    # 并根据 item['Id'] 从 graph_data 中提取相应的图信息。
    datasets: Dict[str, NewsDataset] = {
        split: NewsDataset(graph_data[split], split_data)
        for split, split_data in data.items()
    }

    # 根据模式 (train/test) 创建 DataLoader
    tr_set, dev_set, test_set = None, None, None
    tr_size, dev_size, test_size = 0, 0, 0

    if args.mode == 0:  # 训练模式
        tr_dataset = datasets[TRAIN]
        tr_size = len(tr_dataset)
        print(f"Train set size: {tr_size}")
        tr_set = DataLoader(
            tr_dataset, batch_size=args.batch_size, collate_fn=tr_dataset.collate_fn,
            shuffle=True, drop_last=True,
            num_workers=args.num_workers, pin_memory=True
        )

        dev_dataset = datasets[DEV]
        dev_size = len(dev_dataset)
        print(f"Dev set size: {dev_size}")
        dev_set = DataLoader(
            dev_dataset, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn,
            shuffle=False, drop_last=False,
            num_workers=args.num_workers, pin_memory=True
        )
    else:  # 测试模式
        test_dataset = datasets[TEST]
        test_size = len(test_dataset)
        print(f"Test set size: {test_size}")
        test_set = DataLoader(
            test_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn,
            shuffle=False, drop_last=False,
            num_workers=args.num_workers, pin_memory=True
        )

    # --- MODIFICATION 2: 修正模型初始化 ---
    # 模型现在应该只依赖于 args 对象来获取所有配置，特别是 `args.dim_node_features`
    # 移除了已经废弃的 `embedding_dim` 参数
    classifier = FakeNewsDetection(args)
    # --- END MODIFICATION 2 ---
    classifier.to(args.device)

    # 确保输出目录存在
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == 0:  # 训练/验证流程
        # 保存本次运行的参数
        args_dict = copy.deepcopy(vars(args))
        with open(args.output_dir / f"param_{args.name}.txt", mode="w") as f:
            f.write("============ Parameters ============\n")
            print("============ Parameters =============")
            for k, v in args_dict.items():
                line = f"{k}: {v}\n"
                f.write(line)
                print(line.strip())
            print("======================================")

        trainer = Trainer(args, classifier, tr_set, tr_size, dev_set, dev_size)
        trainer.train()
    else:  # 测试流程
        # **重要**: 这里的 Tester 可能也需要修改，以适配新的数据和评估逻辑
        tester = Tester(args, classifier, test_set, test_size)
        tester.test()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default="./dataset/", help="Directory to the processed dataset")
    parser.add_argument("--cache_dir", type=Path, default="./cache/", help="Directory to the preprocessed caches.")
    parser.add_argument("--ckpt_dir", type=Path, default="./ckpt/", help="Directory to save the model checkpoints.")
    parser.add_argument("--output_dir", type=Path, default="./output/", help="Directory to save the run outputs (logs, params).")
    parser.add_argument("--test_path", type=Path, help="Path to load a specific model checkpoint for testing.")

    # --- MODIFICATION 3: 更新核心模型维度参数 ---
    # 节点特征维度现在默认是768，以匹配BERT-base模型的输出
    parser.add_argument('--dim_node_features', type=int, default=768, help="Dimension of node features. Should be 768 for BERT-base.")
    # --- 移除了已废弃的参数: --embedding_dim, --text_size ---

    # 模型架构相关参数 (请根据你的 Model.py 进行调整)
    parser.add_argument("--num_gcn", type=int, default=3, help="Number of GCN layers or similar graph layers.")
    parser.add_argument('--num_gnn_layers', type=int, default=2, help="Number of layers in the main GNN.")
    # 其他超参数 (请根据你的模型进行调整)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.2)

    # 训练超参数
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader. 0 for main process.")

    # 运行设置
    parser.add_argument("--device", type=str, help="Device to use (e.g., 'cuda:0' or 'cpu')", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--name", type=str, default="run_01", help="A name for the current run, used for saving files.")
    parser.add_argument("--mode", type=int, help="0 for training, 1 for testing", default=0)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
