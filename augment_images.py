import cv2
import random
import numpy as np
import argparse
import sys

from pathlib import Path


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def augment_image(image, transformation):
    """Apply a randomly selected image augmentation."""

    if transformation == "brightness":
        value = random.randint(15, 40)
        return cv2.convertScaleAbs(
            image, alpha=1.0, beta=value
        )

    elif transformation == "darkness":
        value = random.randint(-40, -15)
        return cv2.convertScaleAbs(
            image, alpha=1.0, beta=value
        )

    elif transformation == "contrast":
        alpha = random.uniform(1.15, 1.4)
        return cv2.convertScaleAbs(
            image, alpha=alpha, beta=0
        )

    elif transformation == "gamma":
        gamma = random.uniform(0.7, 1.3)

        lookup_table = np.array([
            min(
                255,
                max(
                    0,
                    int(
                        ((i / 255.0) ** (1.0 / gamma))
                        * 255
                    )
                )
            )
            for i in range(256)
        ], dtype=np.uint8)

        return cv2.LUT(image, lookup_table)

    elif transformation == "horizontal_flip":
        return cv2.flip(image, 1)

    elif transformation == "slight_blur":
        kernel = random.choice([3, 5])

        return cv2.GaussianBlur(
            image,
            (kernel, kernel),
            sigmaX=0.5
        )

    elif transformation == "brightness_contrast":
        alpha = random.uniform(0.9, 1.2)
        beta = random.randint(-20, 25)

        return cv2.convertScaleAbs(
            image, alpha=alpha, beta=beta
        )

    elif transformation == "zoom_in":
        height, width = image.shape[:2]

        scale = random.uniform(1.05, 1.15)

        crop_width = int(width / scale)
        crop_height = int(height / scale)

        x = (width - crop_width) // 2
        y = (height - crop_height) // 2

        cropped = image[
            y:y + crop_height,
            x:x + crop_width
        ]

        return cv2.resize(
            cropped,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

    elif transformation == "small_rotation":
        height, width = image.shape[:2]

        angle = random.uniform(-7, 7)

        matrix = cv2.getRotationMatrix2D(
            (width / 2, height / 2),
            angle,
            1.0
        )

        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

    return image


def main():

    parser = argparse.ArgumentParser(
        description="Augment underwater buoy images without labels."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Folder containing original images"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Folder where augmented images will be saved"
    )

    parser.add_argument(
        "--count",
        type=int,
        default=180,
        help="Number of additional images to generate"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not input_dir.is_dir():
        sys.exit(f"ERROR: Input directory not found: {input_dir}")

    if args.count < 1:
        sys.exit("ERROR: Count must be greater than zero.")

    images = sorted([
        p for p in input_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not images:
        sys.exit("ERROR: No images found in the input folder.")

    # Prevent accidental overwriting of existing augmented images.
    output_dir.mkdir(parents=True, exist_ok=True)

    transformations = [
        "brightness",
        "darkness",
        "contrast",
        "gamma",
        "horizontal_flip",
        "slight_blur",
        "brightness_contrast",
        "zoom_in",
        "small_rotation"
    ]

    print("=" * 50)
    print("UNDERWATER BUOY IMAGE AUGMENTATION")
    print("=" * 50)
    print(f"Source images: {len(images)}")
    print(f"Images to generate: {args.count}")
    print(f"Output directory: {output_dir}")
    print()

    created = 0
    counts = {}

    for index in range(1, args.count + 1):

        # Select a random original image.
        source = random.choice(images)

        image = cv2.imread(str(source))

        if image is None:
            print(f"Skipping unreadable image: {source.name}")
            continue

        transformation = random.choice(transformations)

        augmented = augment_image(
            image,
            transformation
        )

        filename = (
            f"aug_{index:04d}_"
            f"{source.stem}_"
            f"{transformation}.jpg"
        )

        output_path = output_dir / filename

        success = cv2.imwrite(
            str(output_path),
            augmented,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        if not success:
            print(f"Failed to save: {filename}")
            continue

        counts[transformation] = (
            counts.get(transformation, 0) + 1
        )

        created += 1

        if created % 20 == 0:
            print(f"Progress: {created}/{args.count}")

    print("\n" + "=" * 50)
    print("AUGMENTATION COMPLETED")
    print("=" * 50)
    print(f"Images created: {created}")
    print(f"Saved to: {output_dir}")

    print("\nAugmentation summary:")

    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")

    print("\nNo labels were created.")
    print("You can annotate the generated images in Roboflow.")


if __name__ == "__main__":
    main()
