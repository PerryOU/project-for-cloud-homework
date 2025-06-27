# === Python代码文件: visualize_embeddings.py (新文件) ===

import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
from pathlib import Path


def visualize_tsne(embedding_path: Path, output_dir: Path, plot_title: str):
    """
    加载pickle文件中的嵌入，使用t-SNE进行降维，并绘制可视化结果。
    """
    if not embedding_path.exists():
        print(f"错误: 找不到嵌入文件: {embedding_path}")
        return

    print(f"正在加载嵌入文件: {embedding_path}...")
    with open(embedding_path, "rb") as f:
        data_dict = pickle.load(f)

    # 从字典中提取嵌入和标签
    embeddings = []
    labels = []
    for item_id, value in data_dict.items():
        # 确保嵌入是NumPy array格式
        embedding = value["embedding"].cpu().numpy() if hasattr(value["embedding"], 'cpu') else value["embedding"]
        embeddings.append(embedding)
        labels.append(value["label"])

    X = np.array(embeddings)
    y = np.array(labels)

    if X.shape[0] < 2:
        print("错误: 嵌入数量不足，无法进行t-SNE分析。")
        return

    print(f"数据加载完成. 特征维度: {X.shape}, 标签数量: {len(y)}")
    print("正在进行 t-SNE 降维... (这可能需要一些时间)")

    # 执行 t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X)

    print("t-SNE 降维完成. 正在绘制图像...")

    # 绘制散点图
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 10))

    # 使用Seaborn调色板，并为标签创建图例
    palette = sns.color_palette("deep", 2)
    scatter = sns.scatterplot(
        x=X_tsne[:, 0],
        y=X_tsne[:, 1],
        hue=y,
        palette=palette,
        legend='full',
        alpha=0.7,
        s=50  # 点的大小
    )

    # 自定义图例
    handles, _ = scatter.get_legend_handles_labels()
    scatter.legend(handles, ['Label 0 (e.g., Real)', 'Label 1 (e.g., Fake)'], title='Category')

    plt.title(plot_title, fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.grid(True)

    # 保存图像
    output_filename = embedding_path.stem + "_tsne.png"
    save_path = output_dir / output_filename
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"t-SNE 可视化图像已保存到: {save_path}")


def main():
    parser = ArgumentParser(description="t-SNE 可视化脚本，用于分析模型生成的特征嵌入。")
    parser.add_argument("--output_dir", type=Path, default="./output/",
                        help="保存结果的目录 (应与 main.py 的设置一致)。")
    parser.add_argument("--run_name", type=str, default="run_01", help="当前运行的名称 (用于定位测试报告文件，可选)。")
    args = parser.parse_args()

    # 自动定位由 tester.py 生成的 .pkl 文件
    final_embedding_file = args.output_dir / "y_embed_test.pkl"
    gcn_embedding_file = args.output_dir / "gcn_embed_test.pkl"

    print(f"将在输出目录 '{args.output_dir}' 中查找嵌入文件...")

    visualize_tsne(
        embedding_path=final_embedding_file,
        output_dir=args.output_dir,
        plot_title="t-SNE Visualization of Fused Embeddings (Test Set)"
    )

    print("-" * 30)

    visualize_tsne(
        embedding_path=gcn_embedding_file,
        output_dir=args.output_dir,
        plot_title="t-SNE Visualization of GCN Branch Embeddings (Test Set)"
    )


if __name__ == "__main__":
    main()
