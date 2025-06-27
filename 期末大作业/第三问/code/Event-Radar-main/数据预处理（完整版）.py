import pandas as pd
import pickle
import os
from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image
import warnings
from sklearn.model_selection import train_test_split

# --- 新增的、符合论文要求的库 ---
import spacy
from transformers import BertTokenizer, BertModel
import clip

warnings.filterwarnings("ignore")


class DataProcessor:
    def __init__(self, csv_path, image_dir, output_dir):
        """
        初始化数据处理器 (修改后)
        """
        # 1. 设备设置 (GPU优先)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # 2. 加载论文所需的核心模型
        print("Loading models (BERT, CLIP, SpaCy)... This may take a moment.")
        try:
            # NER模型 (SpaCy)
            self.nlp = spacy.load("en_core_web_lg")
            # CLIP模型及预处理器
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self.clip_model.eval()
            # BERT模型及分词器
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
            self.bert_model.eval()
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Please ensure you have installed all required packages and have a stable internet connection.")
            raise

        print("Models loaded successfully.")

        # 3. 路径和参数设置 (使用传入的参数)
        self.csv_path = csv_path
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.supported_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.gif']
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 定义统一的特征维度，以BERT的输出为准
        self.feature_dim = self.bert_model.config.hidden_size  # bert-base-uncased 是 768

    def find_image_file(self, image_id):
        """根据image_id查找对应的图片文件"""
        for ext in self.supported_extensions:
            # 确保image_id是字符串，并去除可能存在的 .jpg 等后缀
            image_stem = Path(str(image_id)).stem
            image_path = self.image_dir / f"{image_stem}{ext}"
            if image_path.exists():
                return image_path
        return None

    def get_bert_embedding(self, text, max_length=128):
        """使用BERT获取文本的[CLS] token嵌入，并移动到CPU"""
        if not text or not isinstance(text, str):
            # 对于空文本，返回一个CPU上的零向量
            return torch.zeros(self.feature_dim)

        inputs = self.bert_tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=max_length, padding='max_length'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.bert_model(**inputs)

        # 使用[CLS] token的输出来代表整个句子的嵌入，并立即移动到CPU
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
        return cls_embedding

    def extract_text_and_entity_features(self, text):
        """使用SpaCy进行实体识别，并用BERT提取帖子和实体的特征"""
        post_text = str(text) if pd.notna(text) else ""

        # 1. 提取帖子原文的BERT特征
        post_embedding = self.get_bert_embedding(post_text)

        # 2. 使用SpaCy提取实体
        doc = self.nlp(post_text)
        entities = [ent.text for ent in doc.ents]
        # 保持顺序去重
        unique_entities = sorted(list(set(entities)), key=entities.index)

        # 3. 提取每个实体的BERT特征
        entity_embeddings = []
        if unique_entities:
            for entity in unique_entities:
                entity_embeddings.append(self.get_bert_embedding(entity))
            entity_embeddings_tensor = torch.stack(entity_embeddings)
        else:
            # 如果没有实体，创建一个(0, dim)的空tensor
            entity_embeddings_tensor = torch.empty(0, self.feature_dim)

        return post_embedding, unique_entities, entity_embeddings_tensor

    def get_clip_image_features(self, image_path):
        """加载图片并使用CLIP提取特征，统一维度后返回CPU tensor (已修复dtype问题)"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)

            # 归一化特征以获得更好的相似度比较
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # 检查并统一特征维度
            clip_output_dim = image_features.shape[-1]
            if clip_output_dim != self.feature_dim:
                # 使用一个线性投影层来匹配维度 (只在第一次使用时创建)
                if not hasattr(self, 'clip_projection'):
                    print(
                        f"CLIP output dim ({clip_output_dim}) != BERT dim ({self.feature_dim}). Creating projection layer.")
                    # 确保线性层与BERT模型在同一设备上
                    self.clip_projection = torch.nn.Linear(clip_output_dim, self.feature_dim).to(self.device)

                # ---【核心修改】---
                # 在送入线性层前，将半精度(float16)的image_features转换为单精度(float32)
                # 以匹配线性层默认的 float32 类型，解决 "Half and Float" 错误。
                image_features = self.clip_projection(image_features.float())
                # ---【修改结束】---

            return image_features.squeeze(0).cpu()

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return torch.zeros(self.feature_dim)  # 返回CPU上的零向量

    def process_image_ids_clip(self, image_ids_str):
        """处理image_id(s)字符串，返回CLIP特征列表"""
        if pd.isna(image_ids_str):
            return []

        image_ids = [id.strip() for id in str(image_ids_str).split(',') if id.strip()]
        clip_features = []

        for image_id in image_ids:
            image_path = self.find_image_file(image_id)
            if image_path:
                clip_feature = self.get_clip_image_features(image_path)
                clip_features.append(clip_feature)
            else:
                print(f"Image not found: {image_id}, adding zero vector.")
                clip_features.append(torch.zeros(self.feature_dim))

        return clip_features

    def create_graph_structure_from_features(self, post_embedding, entity_embeddings, image_embeddings):
        """根据真实的特征构建符合论文要求的图结构"""
        # 将图像特征列表转换为tensor
        if image_embeddings:
            # 确保所有特征都是float32，以防万一
            image_embeddings = [t.float() for t in image_embeddings]
            image_embeddings_tensor = torch.stack(image_embeddings)
        else:
            image_embeddings_tensor = torch.empty(0, self.feature_dim)

        # 节点特征拼接顺序：[帖子节点, 实体节点..., 图片节点...]
        node_features = torch.cat(
            [post_embedding.unsqueeze(0).float(), entity_embeddings.float(), image_embeddings_tensor.float()],
            dim=0
        )

        num_nodes = node_features.shape[0]
        num_entities = entity_embeddings.shape[0]
        num_images = image_embeddings_tensor.shape[0]

        # 创建边索引 (双向边)
        edge_index = []
        # 1. 连接帖子与所有实体
        for i in range(num_entities):
            entity_node_idx = 1 + i
            edge_index.extend([[0, entity_node_idx], [entity_node_idx, 0]])

        # 2. 连接帖子与所有图片
        for i in range(num_images):
            image_node_idx = 1 + num_entities + i
            edge_index.extend([[0, image_node_idx], [image_node_idx, 0]])

        # 如果图中只有一个节点（只有帖子），或没有边，则创建一个自环以避免空边
        if num_nodes > 0 and not edge_index:
            edge_index_tensor = torch.LongTensor([[0], [0]])
        else:
            edge_index_tensor = torch.LongTensor(edge_index).t().contiguous()

        # 创建类型到节点索引的映射
        type2nidx = {
            'post_subgraph': [0] if num_nodes > 0 else [],
            'entity_subgraph': list(range(1, 1 + num_entities)),
            'image_subgraph': list(range(1 + num_entities, num_nodes)),
            'all_nodes': list(range(num_nodes))
        }

        # 论文未明确提及边权重，我们将其设为None
        edges_structure = {'index': edge_index_tensor, 'weight': None}

        return node_features, edges_structure, type2nidx

    def process_csv_data(self):
        """处理CSV数据，生成所有样本的图数据"""
        df = pd.read_csv(self.csv_path)
        print(f"Read {len(df)} rows from {self.csv_path}.")

        processed_data_list = []
        nodes_features_dict, edges_data_dict, type2nidxs_dict = {}, {}, {}

        for idx, row in df.iterrows():
            print(f"Processing progress: {idx + 1}/{len(df)}")

            # 使用CSV中的'Id'列或行索引作为唯一标识
            data_id = row.get('Id', row.get('id', idx))

            # 1. 文本和实体特征提取
            post_embedding, _, entity_embeddings = self.extract_text_and_entity_features(row['post_text'])

            # 2. 图像特征提取
            image_embeddings = self.process_image_ids_clip(row.get('image_id(s)'))

            # 3. 构建图结构
            node_feats, edges_structure, type2nidx = self.create_graph_structure_from_features(
                post_embedding, entity_embeddings, image_embeddings
            )

            # 4. 存储该样本的图数据
            nodes_features_dict[data_id] = node_feats
            edges_data_dict[data_id] = edges_structure
            type2nidxs_dict[data_id] = type2nidx

            # 5. 创建数据列表项 (只包含ID和标签，训练时通过ID查找图数据)
            data_item = {
                'Id': data_id,
                'label': 1 if row['label'] == 'real' else 0,
            }
            processed_data_list.append(data_item)

        # 将所有图数据打包成一个元组
        graph_data_tuple = (
            nodes_features_dict,
            edges_data_dict,  # 简化，所有边类型共用一个字典
            edges_data_dict,
            edges_data_dict,
            edges_data_dict,
            type2nidxs_dict
        )

        return processed_data_list, graph_data_tuple

    def split_and_save_data(self, data_list, graph_data, train_ratio=0.7, eval_ratio=0.15, test_ratio=0.15):
        """分割数据并保存为PKL文件"""
        if not abs(train_ratio + eval_ratio + test_ratio - 1.0) < 1e-6:
            raise ValueError("Train, eval, and test ratios must sum to 1.")

        random.seed(1107)
        np.random.seed(1107)

        # 为了分层采样，我们需要标签
        labels = [item['label'] for item in data_list]

        # 使用索引进行分割
        indices = list(range(len(data_list)))

        # 确保数据集足够大可以进行划分
        if len(indices) < 3:
            raise ValueError("Dataset is too small to be split into train, eval, and test sets.")

        # 分层采样，先分出训练集
        try:
            train_indices, temp_indices = train_test_split(
                indices, test_size=(eval_ratio + test_ratio),
                stratify=[labels[i] for i in indices], random_state=1107
            )

            # 再从剩余部分分出验证集和测试集
            if temp_indices:
                temp_labels = [labels[i] for i in temp_indices]
                relative_test_ratio = test_ratio / (eval_ratio + test_ratio)
                eval_indices, test_indices = train_test_split(
                    temp_indices, test_size=relative_test_ratio,
                    stratify=temp_labels, random_state=1107
                )
            else:  # 如果数据集过小，可能没有temp_indices
                eval_indices, test_indices = [], []

        except ValueError as e:
            print(
                f"Warning: Could not stratify split, possibly due to small dataset or class imbalance. Splitting without stratification. Error: {e}")
            # 如果分层采样失败（例如某个类别样本太少），则进行普通随机划分
            train_indices, temp_indices = train_test_split(indices, test_size=(eval_ratio + test_ratio),
                                                           random_state=1107)
            if temp_indices:
                relative_test_ratio = test_ratio / (eval_ratio + test_ratio)
                eval_indices, test_indices = train_test_split(temp_indices, test_size=relative_test_ratio,
                                                              random_state=1107)
            else:
                eval_indices, test_indices = [], []

        splits = {
            'train': [data_list[i] for i in train_indices],
            'eval': [data_list[i] for i in eval_indices],
            'test': [data_list[i] for i in test_indices]
        }

        # --- 保存文件 ---
        # 1. 保存共享的图数据
        graph_path = self.output_dir / "graph_data.pkl"
        with open(graph_path, 'wb') as f:
            pickle.dump(graph_data, f)
        print(f"\nSaved shared graph data to {graph_path}")

        # 2. 保存各划分的ID和标签列表
        for split_name, split_data in splits.items():
            if not split_data:
                print(f"  - {split_name} set is empty. Skipping file creation.")
                continue

            data_path = self.output_dir / f"{split_name}.pkl"
            with open(data_path, 'wb') as f:
                pickle.dump(split_data, f)
            print(f"Saved {split_name} data list to {data_path}")

            # 打印统计信息
            s_labels = [item['label'] for item in split_data]
            fake_count = s_labels.count(0)
            real_count = s_labels.count(1)
            print(f"  - {split_name} set size: {len(split_data)} (fake: {fake_count}, real: {real_count})")

    def process_and_save(self):
        """完整的处理和保存流程"""
        print("\n--- Starting Data Processing ---")
        data_list, graph_data = self.process_csv_data()

        print("\n--- Splitting and Saving Data ---")
        self.split_and_save_data(data_list, graph_data)

        print("\n--- Data Processing Finished! ---")
        print("Generated files are in:", self.output_dir)
        print("  - graph_data.pkl (Contains all node features and graph structures, shared by all sets)")
        print("  - train.pkl")
        print("  - eval.pkl")
        print("  - test.pkl")


def main():
    """主函数，使用您原始代码中定义的路径"""
    # --- 使用您在原始代码中提供的路径 ---
    data_dir = r"D:\大三下\云计算与大数据分析\大作业\期末大作业\data"
    csv_path = os.path.join(data_dir, "训练集1.csv")
    image_dir = r"D:\大三下\云计算与大数据分析\大作业\期末大作业\data\train_images"
    output_dir = "./dataset"  # 输出到当前目录下的dataset文件夹

    # 检查文件和文件夹是否存在，提供更明确的错误信息
    if not os.path.exists(csv_path):
        print(f"FATAL ERROR: CSV file not found at the specified path: {csv_path}")
        print("Please check if the file exists and the path is correct.")
        return

    if not os.path.exists(image_dir):
        print(f"FATAL ERROR: Image directory not found at the specified path: {image_dir}")
        print("Please check if the folder exists and the path is correct.")
        return

    print(f"CSV file path: {csv_path}")
    print(f"Image folder path: {image_dir}")
    print(f"Output folder path: {output_dir}")

    # 创建数据处理器并执行完整的处理和保存流程
    processor = DataProcessor(csv_path, image_dir, output_dir)
    processor.process_and_save()


if __name__ == "__main__":
    main()
