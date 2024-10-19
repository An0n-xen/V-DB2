import numpy as np
import seaborn as sns
from annoy import AnnoyIndex
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from openai import OpenAI, BadRequestError
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)


class VectorStore:
    def __init__(self, model="text-embedding-ada-002") -> None:
        self.model = model
        self.vector_dim = 1536
        self.annoy_index = AnnoyIndex(self.vector_dim, metric="angular")
        self.metadata = []
        self.texts = []
        self.current_index = 0
        self.client = OpenAI()
        self.embeddings = []
        self.cluster_labels = None

    def _get_embedding(self, text):
        response = self.client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding

    def add_text(self, text, metadata=None):
        embedding = self._get_embedding(text)
        self.annoy_index.add_item(self.current_index, embedding)
        self.metadata.append(metadata)
        self.texts.append(text)
        self.embeddings.append(embedding)
        self.current_index += 1

    def build(self, n_trees=10):
        self.annoy_index.build(n_trees)

    def save(self, file_path):
        self.annoy_index.save(file_path)

    def load(self, file_path):
        self.annoy_index.load(file_path)

    def search(self, query_text, top_k=5):
        if self.current_index == 0:
            return []

        query_embedding = self._get_embedding(query_text)
        indices, distance = self.annoy_index.get_nns_by_vector(
            query_embedding, top_k, include_distances=True
        )

        results = [
            {
                "index": idx,
                "distance": dist,
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
            }
            for idx, dist in zip(indices, distance)
        ]

        return results

    def _cluster_embeddings(self, n_clusters=5):
        if self.current_index == 0:
            print("No embeddings to cluster.")
            return None

        embeddings = np.array(self.embeddings)
        k_means = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_labels = k_means.fit_predict(embeddings)
        return self.cluster_labels

    def visualize_embeddings(self, perplexity=30, n_iter=1000, n_clusters=5):

        if self.current_index == 0:
            print("No embeddings to visualize.")
            return

        # Perform clustering
        if (
            self.cluster_labels is None
            or len(self.cluster_labels) != self.current_index
        ):
            self.cluster_labels = self._cluster_embeddings(n_clusters)

        # cluster_labels = self._cluster_embeddings(n_clusters)

        # Perform t-SNE
        embeddings = np.array(self.embeddings)
        tsne = TSNE(
            n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=42
        )
        embeddings_2d = tsne.fit_transform(embeddings)

        # Set up colors and markers
        colors = sns.color_palette("husl", n_colors=n_clusters)
        markers = [
            "o",
            "s",
            "^",
            "D",
            "v",
            "<",
            ">",
            "p",
            "*",
            "H",
        ]  # 10 different marker shapes

        # Plot
        plt.figure(figsize=(15, 10))
        for i in range(n_clusters):
            cluster_points = embeddings_2d[self.cluster_labels == i]
            plt.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                c=[colors[i]],
                marker=markers[i % len(markers)],
                label=f"Cluster {i}",
                s=100,  # Increase marker size
                alpha=0.7,
            )

        # Annotate points with text
        for i, txt in enumerate(self.texts):
            plt.annotate(
                txt[:20] + "...",
                (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                fontsize=8,
                alpha=0.7,
            )

        plt.title("t-SNE visualization of embeddings with clusters", fontsize=16)
        plt.xlabel("t-SNE feature 0", fontsize=12)
        plt.ylabel("t-SNE feature 1", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
        plt.tight_layout()
        plt.show()
