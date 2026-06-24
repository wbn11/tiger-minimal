import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from tiger_min.rqvae.model import RqvaeModel
from tiger_min.rqvae.semantic_id_dedup import deduplicate_semantic_ids
from tiger_min.utils import ensure_dir, save_json, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train minimal RQ-VAE item tokenizer.")
    parser.add_argument("--item-embeddings", default="data/processed/toy/item_embeddings.pt")
    parser.add_argument("--output-dir", default="data/processed/toy")
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-quantizer-layers", type=int, default=3)
    parser.add_argument("--codebook-size", type=int, default=4)
    parser.add_argument("--commitment-weight", type=float, default=0.25)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--kmeans-max-iter", type=int, default=100)
    parser.add_argument("--kmeans-n-init", type=int, default=10)
    return parser


def load_item_embeddings(path: str | Path) -> torch.Tensor:
    item_embeddings = torch.load(path, map_location="cpu")
    if not isinstance(item_embeddings, torch.Tensor):
        raise TypeError("item_embeddings file must contain a torch.Tensor.")
    if item_embeddings.ndim != 2:
        raise ValueError("item_embeddings must be a 2D tensor.")
    if item_embeddings.shape[0] == 0:
        raise ValueError("item_embeddings cannot be empty.")
    if torch.isnan(item_embeddings).any().item():
        raise ValueError("item_embeddings contains NaN.")
    return item_embeddings.float()


def build_model(
    input_dim: int,
    latent_dim: int,
    hidden_dim: int,
    num_quantizer_layers: int,
    codebook_size: int,
    commitment_weight: float,
    reconstruction_weight: float,
) -> RqvaeModel:
    return RqvaeModel(
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_quantizer_layers=num_quantizer_layers,
        codebook_size=codebook_size,
        hidden_dim=hidden_dim if hidden_dim > 0 else None,
        commitment_weight=commitment_weight,
        reconstruction_weight=reconstruction_weight,
    )


def train_rqvae(
    model: RqvaeModel,
    item_embeddings: torch.Tensor,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    kmeans_max_iter: int,
    kmeans_n_init: int,
) -> list[dict[str, float]]:
    set_seed(seed)

    model.to(device)
    full_item_embeddings = item_embeddings.to(device)
    model.initialize_codebooks_with_kmeans(
        full_item_embeddings,
        seed=seed,
        max_iter=kmeans_max_iter,
        n_init=kmeans_n_init,
    )

    dataset = TensorDataset(item_embeddings)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch_metrics: list[dict[str, float]] = []
    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        total_reconstruction_loss = 0.0
        total_quantizer_loss = 0.0
        total_codebook_loss = 0.0
        total_commitment_loss = 0.0
        num_examples = 0

        for (batch_embeddings,) in dataloader:
            batch_embeddings = batch_embeddings.to(device)
            output = model(batch_embeddings)

            optimizer.zero_grad()
            output.loss.backward()
            optimizer.step()

            batch_size_actual = batch_embeddings.shape[0]
            total_loss += float(output.loss.item()) * batch_size_actual
            total_reconstruction_loss += (
                float(output.reconstruction_loss.item()) * batch_size_actual
            )
            total_quantizer_loss += float(output.quantizer_loss.item()) * batch_size_actual
            total_codebook_loss += float(output.codebook_loss.item()) * batch_size_actual
            total_commitment_loss += float(output.commitment_loss.item()) * batch_size_actual
            num_examples += batch_size_actual

        epoch_metrics.append(
            {
                "loss": total_loss / num_examples,
                "reconstruction_loss": total_reconstruction_loss / num_examples,
                "quantizer_loss": total_quantizer_loss / num_examples,
                "codebook_loss": total_codebook_loss / num_examples,
                "commitment_loss": total_commitment_loss / num_examples,
            }
        )

    return epoch_metrics


@torch.no_grad()
def export_semantic_ids(
    model: RqvaeModel,
    item_embeddings: torch.Tensor,
    output_dir: str | Path,
    device: torch.device,
) -> dict:
    output_path = ensure_dir(output_dir)
    model.eval()

    output = model(item_embeddings.to(device))
    base_semantic_ids = output.code_ids.cpu()
    dedup_result = deduplicate_semantic_ids(base_semantic_ids)
    semantic_ids = dedup_result.semantic_ids.cpu()

    torch.save(base_semantic_ids, output_path / "base_semantic_ids.pt")
    torch.save(semantic_ids, output_path / "semantic_ids.pt")
    save_json(dedup_result.meta, output_path / "semantic_id_meta.json")

    return {
        "base_semantic_ids_path": str(output_path / "base_semantic_ids.pt"),
        "semantic_ids_path": str(output_path / "semantic_ids.pt"),
        "semantic_id_meta_path": str(output_path / "semantic_id_meta.json"),
        "export_loss": float(output.loss.item()),
        "export_reconstruction_loss": float(output.reconstruction_loss.item()),
        "export_quantizer_loss": float(output.quantizer_loss.item()),
        "deduplication": dedup_result.meta,
    }


def save_checkpoint(
    model: RqvaeModel,
    output_dir: str | Path,
    model_args: dict,
) -> str:
    output_path = ensure_dir(output_dir)
    state_dict = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }
    checkpoint_path = output_path / "rqvae.pt"
    torch.save(
        {
            "model_args": model_args,
            "state_dict": state_dict,
        },
        checkpoint_path,
    )
    return str(checkpoint_path)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    item_embeddings = load_item_embeddings(args.item_embeddings)
    model_args = {
        "input_dim": int(item_embeddings.shape[1]),
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "num_quantizer_layers": args.num_quantizer_layers,
        "codebook_size": args.codebook_size,
        "commitment_weight": args.commitment_weight,
        "reconstruction_weight": args.reconstruction_weight,
    }
    model = build_model(**model_args)

    epoch_metrics = train_rqvae(
        model=model,
        item_embeddings=item_embeddings,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=device,
        kmeans_max_iter=args.kmeans_max_iter,
        kmeans_n_init=args.kmeans_n_init,
    )
    export_meta = export_semantic_ids(
        model=model,
        item_embeddings=item_embeddings,
        output_dir=args.output_dir,
        device=device,
    )
    checkpoint_path = save_checkpoint(
        model=model,
        output_dir=args.output_dir,
        model_args=model_args,
    )

    train_meta = {
        "item_embeddings_path": str(args.item_embeddings),
        "output_dir": str(args.output_dir),
        "checkpoint_path": checkpoint_path,
        "num_items": int(item_embeddings.shape[0]),
        "embedding_dim": int(item_embeddings.shape[1]),
        "device": str(device),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "kmeans_max_iter": args.kmeans_max_iter,
        "kmeans_n_init": args.kmeans_n_init,
        "model_args": model_args,
        "epoch_metrics": epoch_metrics,
        "final_epoch": epoch_metrics[-1] if epoch_metrics else None,
        "export": export_meta,
    }
    save_json(train_meta, Path(args.output_dir) / "rqvae_train_meta.json")
    print(train_meta)


if __name__ == "__main__":
    main()
