import os

import numpy as np
import torch

from src.models.greenlight_gt_timeseries_module import GreenlightGTTimeSeriesLitModule

from dotenv import load_dotenv

from torch import nn, Tensor

from src.utils.custom import read_json_file
from src.utils.greenlight_scaler import GreenlightScaler

from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

load_dotenv()


def read_experimented_c_leakage_values(
    relative_path: str = "scripts/c_leakage_configs.json",
) -> Tensor:
    base_dir = os.getenv("BASE_DIR")
    c_leakage_configs_json_path = os.path.join(base_dir, relative_path)
    json_data = read_json_file(c_leakage_configs_json_path)
    normalized_c_leakage_values = []
    scaler = GreenlightScaler()
    c_leakage_scaling_range = scaler.parameter_scaling_ranges["cLeakage"]
    for k, v in json_data.items():
        c_leakage_val = v["cLeakage"]
        normalized_c_leakage_value = (c_leakage_val - c_leakage_scaling_range[0]) / (
            c_leakage_scaling_range[1] - c_leakage_scaling_range[0]
        )
        normalized_c_leakage_values.append(normalized_c_leakage_value)
    normalized_c_leakage_values = np.array(
        sorted(normalized_c_leakage_values), dtype=np.float32
    )
    return torch.from_numpy(normalized_c_leakage_values)


@torch.inference_mode()
def get_embeddings(projector: nn.Module, normalized_c_leakage_vals: Tensor) -> Tensor:
    return projector(normalized_c_leakage_vals.view(-1, 1))


def compute_pairwise_distances(embeddings: Tensor) -> np.ndarray:
    x = embeddings.cpu().numpy()
    dists = squareform(pdist(x, metric="cosine"))
    plt.imshow(dists, cmap="viridis")
    plt.colorbar(label="Cosine Distance")
    plt.title("Pairwise ExoPrompt Embedding Distances")
    plt.savefig("pairwise_exo_embedding_cosine_distance", bbox_inches="tight")
    plt.show()
    return dists


def pca_for_visualization(embeddings: Tensor, input_t: Tensor) -> np.ndarray:
    x = embeddings.cpu().numpy()
    input_t = input_t.cpu().numpy()
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(x)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=input_t, cmap="viridis")
    plt.colorbar(label="cLeakage")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("ExoPrompt Embedding Space")
    plt.savefig("exo_embedding_pca", bbox_inches="tight")
    plt.show()
    return X_pca


if __name__ == "__main__":
    system_mode = os.getenv("MODE")
    if system_mode == "macos":
        cleakage_best_exo_model_ckpt = "/Users/gsoykan/Desktop/yanan-desktop/wur-phd-2024/physinet/logs/custom/2025-06-06_16-45-32-best-cleakage-exo/checkpoints/epoch_049.ckpt"
        device = torch.device("mps")
    elif system_mode == "server":
        cleakage_best_exo_model_ckpt = "/gpfs/home2/gsoykan/physinet/logs/train/runs/2025-06-06_16-45-32/checkpoints/epoch_049.ckpt"
        device = torch.device("cuda")
    else:
        raise ValueError("Unknown system mode: {}".format(system_mode))
    model_lit_module = GreenlightGTTimeSeriesLitModule.load_from_checkpoint(
        cleakage_best_exo_model_ckpt
    )
    model_lit_module.eval()
    exo_prompt_projector = model_lit_module.net.enc_embedding.exo_prompt_projector
    exo_prompt_projector.to(device)

    input_t = read_experimented_c_leakage_values().to(device)
    embeddings = get_embeddings(exo_prompt_projector, input_t)

    p_dists = compute_pairwise_distances(embeddings)
    x_pca = pca_for_visualization(embeddings, input_t)

    print(p_dists, x_pca)
