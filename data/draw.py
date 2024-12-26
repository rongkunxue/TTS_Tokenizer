import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
import seaborn as sns

# 假设你的 embedding 和 decoder 是以下实例
embedding = torch.nn.Embedding(num_embeddings=8192, embedding_dim=512)  # 示例 embedding

# 1. 提取码本
codebook = embedding.weight.detach().cpu().numpy()  # (num_embeddings, embedding_dim)

# 2. t-SNE 可视化
def visualize_tsne(codebook, save_path="tsne_visualization.png"):
    tsne = TSNE(n_components=2, random_state=42)
    codebook_2d = tsne.fit_transform(codebook)

    plt.figure(figsize=(8, 6))
    plt.scatter(codebook_2d[:, 0], codebook_2d[:, 1], c=range(len(codebook)), cmap="viridis")
    plt.colorbar(label="Code Index")
    plt.title("t-SNE Visualization of Embedding Codebook")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig(save_path, dpi=300)
    plt.close()

visualize_tsne(codebook)

# 3. 热力图
def visualize_heatmap(codebook, save_path="heatmap.png"):
    plt.figure(figsize=(12, 6))
    sns.heatmap(codebook, cmap="viridis", annot=False)
    plt.title("Embedding Codebook Heatmap")
    plt.xlabel("Feature Dimension")
    plt.ylabel("Code Index")
    plt.savefig(save_path, dpi=300)
    plt.close()

visualize_heatmap(codebook)
