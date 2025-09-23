from typing import List
import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoTokenizer,
)
from PIL import Image

class Model:

    def __init__(self, model_name: str, device: torch.device):
        self.model_name = model_name
        self.device = device
        self.model = AutoModelForImageTextToText.from_pretrained(
            # self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map=self.device,
        )

        self._get_vocab()
        self._get_causal_transform()


    def _get_vocab(self):
        """
        Returns vocab dict and ordered vocab list.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.vocab_dict: dict[str, int] = tokenizer.get_vocab()
        self.vocab_list: list[str | None] = [None] * (max(self.vocab_dict.values()) + 1)
        for word, index in self.vocab_dict.items():
            self.vocab_list[index] = word
        return self.vocab_dict, self.vocab_list

    def _get_causal_transform(self):
        """
        Get the causal inner product transformation.
        This transforms the model's unembedding space.
        """

        with torch.no_grad():
            # Get the model's unembedding matrix (output embeddings)
            gamma = (
                self.model.get_output_embeddings().weight.detach().to(torch.float32)
            )  # Shape: [vocab_size, d_model]
            W, _ = gamma.shape

            # Center the unembeddings
            self.gamma_bar = torch.mean(gamma, dim=0)
            centered_gamma = gamma - self.gamma_bar

            # Compute covariance matrix
            Cov_gamma = centered_gamma.T @ centered_gamma / W

            # Compute whitening transformation (Cov^{-1/2})
            eigenvalues, eigenvectors = torch.linalg.eigh(Cov_gamma)
            # Add small epsilon for numerical stability
            eigenvalues = eigenvalues + 1e-6
            self.inv_sqrt_Cov_gamma = (
                eigenvectors @ torch.diag(1 / torch.sqrt(eigenvalues)) @ eigenvectors.T
            )
            self.sqrt_Cov_gamma = (
                eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T
            )

            # Apply transformation to get g(y)
            self.g = centered_gamma @ self.inv_sqrt_Cov_gamma

            return self.g, self.inv_sqrt_Cov_gamma, self.sqrt_Cov_gamma, self.gamma_bar

    def get_text_embeddings(
        self, categories: list[str], data: dict[str, list[str]]
    ) -> dict[str, torch.Tensor]:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        text_embedding: dict[str, torch.Tensor] = {}
        for category in categories:
            print(f"Tokenizing category: {category}")
            text_list = []
            for word in data[category]:
                text_list.extend(tokenizer(word))
            text_embedding[category] = torch.cat(text_list, dim=0)
        return text_embedding

    def get_visual_embeddings(
        self,
        images: List[Image.Image],
    ) -> torch.Tensor:
        """
        Extract visual embeddings and project them to language space
        """
        embeddings_list = []
        processor = AutoProcessor.from_pretrained(self.model_name, use_fast=True)

        with torch.no_grad():
            for image in images:
                if image.mode != "RGB":
                    image = image.convert("RGB")

                # # Get vision embeddings
                inputs = processor(
                    text="<image>",
                    images=[image],
                    return_tensors="pt",
                    do_rescale=True,
                    do_normalize=True,
                ).to(self.device)

                vision_outputs = self.model.vision_tower(inputs["pixel_values"])
                embeddings = self.model.multi_modal_projector(
                    vision_outputs.last_hidden_state
                )  # [1, num_patches, d_model]
                embeddings_list.append(embeddings.squeeze(0))  # [num_patches, d_model]

        embeddings = torch.cat(embeddings_list, dim=0)  # [num_patches*n, d_model]
        return embeddings

    def find_nearest_tokens(self, image_embeds, tokenizer=None, top_k=5):

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Normalize embeddings for cosine similarity
        image_embeds_norm = torch.nn.functional.normalize(image_embeds.float(), dim=-1)
        text_embeds_norm = torch.nn.functional.normalize(self.g.float(), dim=-1)

        # Compute similarity scores
        similarities = torch.matmul(image_embeds_norm, text_embeds_norm.T)

        # Get top-k most similar tokens for each image embedding
        top_k_indices = torch.topk(similarities, k=top_k, dim=-1).indices

        # Convert token IDs to text
        results = []
        for i, token_ids in enumerate(top_k_indices):
            if token_ids.dim() == 0:
                token_ids = token_ids.unsqueeze(0)
            tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
            results.append(tokens)

        return results
