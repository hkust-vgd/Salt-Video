# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the Apache-2.0 license found in the LICENSE file in the root directory of segment_anything repository and source tree.
# Adapted from onnx_model_example.ipynb in the segment_anything repository.
# Please see the original notebook for more details and other examples and additional usage.
import os
import argparse
import cv2
from tqdm import tqdm
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

def main(checkpoint_path, model_type, device, images_folder, embeddings_folder):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    for image_name in tqdm(os.listdir(images_folder)):
        out_path = os.path.join(embeddings_folder, os.path.splitext(image_name)[0] + ".npy")
        if os.path.exists(out_path):
            continue
        image_path = os.path.join(images_folder, image_name)
        image = cv2.imread(image_path)
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)

            image_embedding = predictor.get_image_embedding().cpu().numpy()
            # out_path = os.path.join(embeddings_folder, os.path.splitext(image_name)[0] + ".npy")
            np.save(out_path, image_embedding)
        except:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, default="./saves/sam_vit_h_4b8939.pth")
    parser.add_argument("--model_type", type=str, default="default")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset-path", type=str, default="./video_seqs/shark")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    model_type = args.model_type
    device = args.device
    dataset_path = args.dataset_path

    images_folder = os.path.join(dataset_path, "images")
    embeddings_folder = os.path.join(dataset_path, "embeddings")
    if not os.path.exists(embeddings_folder):
        os.makedirs(embeddings_folder)

    main(checkpoint_path, model_type, device, images_folder, embeddings_folder)
