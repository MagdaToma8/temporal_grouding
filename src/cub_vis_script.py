import json
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--captions_path',
        type=str,
        default='data/CUB_200_2011/captions/bird.json',
        help='Path to the captions JSON file'
    )
    return parser.parse_args()



def main():
    args = parse_args()

    with open(args.captions_path, 'r') as f:
        data = json.load(f)

    plt.figure(figsize=(20, 4))

    np.random.seed(11)
    random_indices = np.random.randint(0, len(data['img_paths']), 4)
    for i, ind in enumerate(random_indices):
        plt.subplot(1, 4, i+1)

        img_path = data['img_paths'][ind]
        img = Image.open(img_path)
        plt.imshow(img)

        caption = data['captions'][ind]
        plt.title(caption, wrap=True)
        plt.axis('off')

    plt.tight_layout()
    dir = os.path.join(
        'results',
        'plots',
        os.path.basename(args.captions_path).split('.')[0]
    )
    os.makedirs(dir, exist_ok=True)
    plt.savefig(os.path.join(dir, 'caption_visualization.png'))
    plt.close()


if __name__ == "__main__":
    main()
