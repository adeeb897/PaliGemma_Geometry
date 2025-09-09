"""
plotting functions for visualizing token embeddings
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_theme(
    context="paper",
    style="white",
    palette="colorblind",
    font="DejaVu Sans",
    font_scale=1.75,
)


def plot_category_clusters(
    category1,
    category2,
    dirs,
    embeddings,
    normalize: bool=True,
    orthogonal=True,
    alpha=0.15,
    s=8,
    target_alpha=0.8,
    target_s=25,
    fontsize=12,
):
    """
    Plot two categories with optimal separation using their LDA directions.
    """
    _, ax = plt.subplots(figsize=(10, 8))

    # Use each category's LDA direction for maximum separation
    dir1 = dirs[category1]["lda"].to(torch.float32)
    dir2 = dirs[category2]["lda"].to(torch.float32)

    if normalize:
        dir1 = dir1 / dir1.norm()
        dir2 = dir2 / dir2.norm()
    if orthogonal:
        dir1 = dir1 / dir1.norm()
        dir2 = dir2 - (dir2 @ dir1) * dir1
        dir2 = dir2 / dir2.norm()

    # Collect image embeddings for the two categories
    category_embeds = {}
    all_embeds = []
    category_indices = {category1: [], category2: []}
    current_idx = 0

    for category in [category1, category2]:
        tokens = embeddings[category]
        if tokens.dim() == 3:
            tokens = tokens.view(-1, tokens.shape[-1])

        category_embeds[category] = tokens
        all_embeds.append(tokens)
        end_idx = current_idx + len(tokens)
        category_indices[category] = list(range(current_idx, end_idx))
        current_idx = end_idx

    image_matrix = torch.cat(all_embeds, dim=0).to(torch.float32)

    # Project tokens onto the category directions
    proj1 = image_matrix @ dir1
    proj2 = image_matrix @ dir2

    # Plot all points in light gray first
    ax.scatter(
        proj1.cpu().numpy(),
        proj2.cpu().numpy(),
        alpha=alpha,
        color="lightgray",
        s=s,
        zorder=1,
    )

    # Color code the two categories
    colors = ["#2E86AB", "#A23B72"]  # blue and purple
    legend_handles = []

    for i, (category, indices) in enumerate(category_indices.items()):
        color = colors[i]

        # Convert indices to tensor
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        category_proj1 = proj1[indices_tensor]
        category_proj2 = proj2[indices_tensor]

        ax.scatter(
            category_proj1.cpu().numpy(),
            category_proj2.cpu().numpy(),
            alpha=target_alpha,
            color=color,
            s=target_s,
            label=category.capitalize(),
            zorder=2,
        )

        legend_handles.append(mpatches.Patch(color=color, label=category.capitalize()))

    # Add legend
    ax.legend(handles=legend_handles, loc="upper right", fontsize=fontsize)

    # Add reference lines
    ax.axhline(y=0, color="black", alpha=0.3, linestyle="--", linewidth=1)
    ax.axvline(x=0, color="black", alpha=0.3, linestyle="--", linewidth=1)

    # Labels and title
    ax.set_xlabel(f"{category1.capitalize()} Direction", fontsize=fontsize)
    ax.set_ylabel(f"{category2.capitalize()} Direction", fontsize=fontsize)
    ax.set_title(
        f"{category1.capitalize()} vs {category2.capitalize()} Separation",
        fontsize=fontsize + 2,
    )

    # Clean up the plot
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()
