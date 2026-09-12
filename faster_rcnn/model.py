import sys
from pathlib import Path

import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


NUM_CLASSES = 2
# 0 = background
# 1 = pothole


def create_model():

    # Load Faster R-CNN with a pretrained ResNet-50 FPN backbone
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

    model = fasterrcnn_resnet50_fpn(
        weights=weights
    )

    # Get the number of input features going into
    # Faster R-CNN's classification head
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the original COCO classification head.
    #
    # COCO has 91 classes, but our problem only has:
    # 0 = background
    # 1 = pothole
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        NUM_CLASSES
    )

    return model


def main():

    print("=" * 60)
    print("FASTER R-CNN MODEL")
    print("=" * 60)

    # Select GPU if available
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nCreating Faster R-CNN model...")

    model = create_model()
    model.to(device)

    print("Model created successfully.")

    print("\nModel:")
    print(model)

    print("\nNumber of classes:", NUM_CLASSES)
    print("Class 0: background")
    print("Class 1: pothole")

    print("\nModel test successful.")


if __name__ == "__main__":
    main()