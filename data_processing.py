import os
from model import Model
from PIL import Image
from sklearn.covariance import ledoit_wolf
import torch
import inflect

p = inflect.engine()

class DataProcessing:

    def __init__(self, model: Model):
        self.model = model
        self.device = model.device

    def get_image_tokens(
        self, categories: dict[str, list[str]], image_paths: dict[str, list[str]]
    ) -> dict[str, torch.Tensor]:
        """
        Process categories with embeddings.
        """
        image_embeddings: dict[str, torch.Tensor] = {}

        for category in categories:
            print(f"Processing category: {category}")
            image_files = image_paths.get(category, [])
            if image_files:
                # Load images
                images = [Image.open(path) for path in image_files]

                # Get and store embeddings
                image_embeddings[category] = self.model.get_visual_embeddings(images)

        return image_embeddings

    def _estimate_single_dir_from_embeddings(self, category_embeddings: torch.Tensor):
        """
        Estimate a direction vector from a collection of embeddings.
        """
        # Ensure float32
        category_embeddings = category_embeddings.to(torch.float32)

        if category_embeddings.shape[0] < 2:
            # Return zero tensors if not enough samples
            D = category_embeddings.shape[1]
            return torch.zeros(
                D, dtype=torch.float32, device=category_embeddings.device
            ), torch.zeros(D, dtype=torch.float32, device=category_embeddings.device)

        if category_embeddings.dim() == 3:
            category_embeddings = category_embeddings.view(-1, category_embeddings.shape[-1])

        category_mean = category_embeddings.mean(dim=0)

        # Estimate covariance with regularization
        category_np = category_embeddings.detach().cpu().numpy()
        cov, _ = ledoit_wolf(category_np)
        cov = torch.tensor(cov, device=category_embeddings.device, dtype=torch.float32)

        # Compute direction using regularized inverse
        pseudo_inv = torch.linalg.pinv(cov)
        lda_dir = pseudo_inv @ category_mean
        lda_dir = lda_dir / torch.norm(lda_dir)

        # Scale by projection magnitude
        lda_dir = (category_mean @ lda_dir) * lda_dir

        return lda_dir, category_mean

    def estimate_dirs_from_embeddings(
        self, embeddings: dict[str, torch.Tensor], overall_category: str
    ):
        print("Estimating directions for embeddings...")

        # Compute lda and mean for each category
        dirs = {
            k: {"lda": v[0], "mean": v[1]}
            for k, v in (
                (key, self._estimate_single_dir_from_embeddings(val))
                for key, val in embeddings.items()
                if val is not None
            )
        }

        # Compute for all embeddings combined
        all_embeddings_flattened = [a for k, v in embeddings.items() for a in v]
        lda_dir, category_mean = self._estimate_single_dir_from_embeddings(
            torch.stack(all_embeddings_flattened)
        )
        dirs.update({overall_category: {"lda": lda_dir, "mean": category_mean}})
        embeddings.update({overall_category: torch.stack(all_embeddings_flattened)})

        return dirs

    def save_embeddings(self, embeddings: dict[str, torch.Tensor], file_path: str):
        """
        Save the embeddings to a file.
        """
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        torch.save(embeddings, file_path)

    def load_embeddings(self, file_path: str) -> dict[str, torch.Tensor]:
        """
        Load the embeddings from a file.
        """
        if not os.path.exists(file_path):
            return {}

        with open(file_path, "rb") as f:
            return torch.load(f)