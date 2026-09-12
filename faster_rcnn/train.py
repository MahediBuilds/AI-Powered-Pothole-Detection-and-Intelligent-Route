import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faster_rcnn.dataset import PotholeDetectionDataset, collate_fn
from faster_rcnn.model import create_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "dataset" / "final"
RUNS_DIR = PROJECT_ROOT / "runs" / "faster_rcnn"


def train_one_epoch(model, loader, optimizer, device, epoch):

    model.train()

    total_loss = 0.0

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch}"
    )

    for images, targets in progress:

        images = [
            image.to(device)
            for image in images
        ]

        targets = [
            {
                key: value.to(device)
                for key, value in target.items()
            }
            for target in targets
        ]

        # Faster R-CNN returns a dictionary of losses
        loss_dict = model(
            images,
            targets
        )

        losses = sum(
            loss for loss in loss_dict.values()
        )

        optimizer.zero_grad()

        losses.backward()

        optimizer.step()

        loss_value = losses.item()

        total_loss += loss_value

        progress.set_postfix(
            loss=f"{loss_value:.4f}"
        )

    return total_loss / max(1, len(loader))


@torch.no_grad()
def validate_loss(model, loader, device):

    # torchvision detection models return losses
    # only while in training mode.
    model.train()

    total_loss = 0.0

    progress = tqdm(
        loader,
        desc="Validation"
    )

    for images, targets in progress:

        images = [
            image.to(device)
            for image in images
        ]

        targets = [
            {
                key: value.to(device)
                for key, value in target.items()
            }
            for target in targets
        ]

        loss_dict = model(
            images,
            targets
        )

        losses = sum(
            loss for loss in loss_dict.values()
        )

        total_loss += losses.item()

        progress.set_postfix(
            loss=f"{losses.item():.4f}"
        )

    return total_loss / max(1, len(loader))


def main():

    parser = argparse.ArgumentParser(
        description="Train Faster R-CNN for pothole detection"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=2,
        help="Images per batch"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.005
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run only 3 epochs"
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("FASTER R-CNN TRAINING")
    print("=" * 60)

    print(f"Device: {device}")

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        memory = (
            torch.cuda.get_device_properties(0)
            .total_memory
            / 1024**3
        )

        print(
            f"VRAM: {memory:.2f} GB"
        )

    # --------------------------------------------------
    # Epoch configuration
    # --------------------------------------------------

    epochs = 3 if args.smoke_test else args.epochs

    print(
        f"\nEpochs: {epochs}"
    )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    train_dataset = PotholeDetectionDataset(
        DATASET_ROOT,
        split="train"
    )

    val_dataset = PotholeDetectionDataset(
        DATASET_ROOT,
        split="val"
    )

    # --------------------------------------------------
    # DataLoaders
    # --------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    print(
        f"\nTraining images   : {len(train_dataset)}"
    )

    print(
        f"Validation images : {len(val_dataset)}"
    )

    print(
        f"Batch size        : {args.batch}"
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    print("\nCreating model...")

    model = create_model()

    model.to(device)

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0005
    )

    # Reduce learning rate when validation loss
    # stops improving.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=2
    )

    # --------------------------------------------------
    # Output directory
    # --------------------------------------------------

    run_name = (
        "pothole_faster_rcnn_smoke"
        if args.smoke_test
        else "pothole_faster_rcnn"
    )

    output_dir = RUNS_DIR / run_name

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    best_model_path = (
        output_dir / "best_model.pth"
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    best_val_loss = float("inf")

    print("\nStarting training...")
    print("=" * 60)

    for epoch in range(1, epochs + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch
        )

        val_loss = validate_loss(
            model,
            val_loader,
            device
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print("\n" + "-" * 60)

        print(
            f"Epoch {epoch}/{epochs}"
        )

        print(
            f"Train loss : {train_loss:.4f}"
        )

        print(
            f"Val loss   : {val_loss:.4f}"
        )

        print(
            f"Learning rate : {current_lr:.6f}"
        )

        # --------------------------------------------------
        # Save best model
        # --------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                },
                best_model_path
            )

            print(
                f"✓ New best model saved: {best_model_path}"
            )

    # --------------------------------------------------
    # Final
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"Best validation loss: {best_val_loss:.4f}"
    )

    print(
        f"Best model: {best_model_path}"
    )

    if args.smoke_test:

        print(
            "\nSmoke test completed successfully."
        )

        print(
            "Run without --smoke-test for full training."
        )


if __name__ == "__main__":
    main()