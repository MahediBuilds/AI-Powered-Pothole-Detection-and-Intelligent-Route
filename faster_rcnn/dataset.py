from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class PotholeDetectionDataset(Dataset):

    def __init__(self, root, split="train"):
        self.root = Path(root)
        self.split = split

        self.images_dir = self.root / "images" / split
        self.labels_dir = self.root / "labels" / split

        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.images_dir}"
            )

        if not self.labels_dir.exists():
            raise FileNotFoundError(
                f"Label directory not found: {self.labels_dir}"
            )

        self.image_paths = sorted(self.images_dir.glob("*.jpg"))

        if not self.image_paths:
            raise FileNotFoundError(
                f"No JPG images found in {self.images_dir}"
            )

        print(
            f"{split.upper()} dataset loaded: "
            f"{len(self.image_paths)} images"
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert("RGB")

        width, height = image.size

        label_path = self.labels_dir / f"{image_path.stem}.txt"

        boxes = []
        labels = []

        if label_path.exists():

            with open(label_path, "r") as f:

                for line in f:

                    values = line.strip().split()

                    if len(values) < 7:
                        continue

                    # First value is YOLO class ID
                    # Remaining values are polygon coordinates:
                    #
                    # x1 y1 x2 y2 x3 y3 ...
                    coords = list(map(float, values[1:]))

                    x_coords = coords[0::2]
                    y_coords = coords[1::2]

                    # Convert normalized YOLO coordinates
                    # back to pixel coordinates
                    x_coords = [x * width for x in x_coords]
                    y_coords = [y * height for y in y_coords]

                    xmin = min(x_coords)
                    ymin = min(y_coords)
                    xmax = max(x_coords)
                    ymax = max(y_coords)

                    # Ignore invalid / zero-area boxes
                    if xmax <= xmin or ymax <= ymin:
                        continue

                    boxes.append([
                        xmin,
                        ymin,
                        xmax,
                        ymax
                    ])

                    # Faster R-CNN reserves class 0 for background.
                    #
                    # Therefore:
                    # 0 = background
                    # 1 = pothole
                    labels.append(1)

        if boxes:

            boxes = torch.as_tensor(
                boxes,
                dtype=torch.float32
            )

            labels = torch.as_tensor(
                labels,
                dtype=torch.int64
            )

        else:

            # Faster R-CNN must also support images
            # containing no potholes.
            boxes = torch.zeros(
                (0, 4),
                dtype=torch.float32
            )

            labels = torch.zeros(
                (0,),
                dtype=torch.int64
            )

        image_id = torch.tensor([idx])

        area = (
            (boxes[:, 3] - boxes[:, 1]) *
            (boxes[:, 2] - boxes[:, 0])
        )

        iscrowd = torch.zeros(
            (boxes.shape[0],),
            dtype=torch.int64
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        # Convert PIL image to PyTorch tensor
        image = F.to_tensor(image)

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    dataset_root = PROJECT_ROOT / "dataset" / "final"

    dataset = PotholeDetectionDataset(
        dataset_root,
        split="train"
    )

    print("\nTesting dataset...")

    image, target = dataset[0]

    print("\nImage:")
    print("Shape:", image.shape)

    print("\nTarget:")
    print("Boxes:")
    print(target["boxes"])

    print("\nLabels:")
    print(target["labels"])

    print("\nAreas:")
    print(target["area"])

    print("\nDataset test successful.")