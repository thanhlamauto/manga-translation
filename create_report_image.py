"""
Create a horizontal composite image showing all pipeline stages for reporting.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def add_label(image, text, font_size=60):
    """Add a label at the top of the image."""
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Try to use a system font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except:
        font = ImageFont.load_default()

    # Get text size
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Draw white background for text
    padding = 20
    draw.rectangle(
        [(0, 0), (pil_img.width, text_height + padding * 2)],
        fill='white'
    )

    # Draw text centered
    x = (pil_img.width - text_width) // 2
    y = padding
    draw.text((x, y), text, fill='black', font=font)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def resize_to_same_height(images, target_height=2000):
    """Resize all images to the same height while maintaining aspect ratio."""
    resized = []
    for img in images:
        h, w = img.shape[:2]
        scale = target_height / h
        new_width = int(w * scale)
        resized_img = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_AREA)
        resized.append(resized_img)
    return resized


def create_horizontal_composite(image_dir, output_path, target_height=2000):
    """Create a horizontal composite image from all stage outputs."""
    image_dir = Path(image_dir)

    # Define the order and labels
    stages = [
        ("0_original.png", "Original"),
        ("1_detection.png", "Detection"),
        ("2_inpainted.png", "Inpainted"),
        ("3_final.png", "Translated"),
        ("colorized.png", "Colorized"),
    ]

    print(f"Loading images from: {image_dir}")

    images = []
    labels = []

    for filename, label in stages:
        filepath = image_dir / filename
        if not filepath.exists():
            print(f"⚠ Warning: {filename} not found, skipping...")
            continue

        print(f"  Loading {filename}...")
        img = cv2.imread(str(filepath))
        if img is None:
            print(f"  ✗ Failed to load {filename}")
            continue

        images.append(img)
        labels.append(label)

    if not images:
        raise ValueError("No images loaded!")

    print(f"\n✓ Loaded {len(images)} images")
    print(f"Resizing to height={target_height}px...")

    # Resize all to same height
    resized_images = resize_to_same_height(images, target_height)

    # Add labels
    print("Adding labels...")
    labeled_images = []
    for img, label in zip(resized_images, labels):
        labeled_img = add_label(img, label)
        labeled_images.append(labeled_img)

    # Concatenate horizontally
    print("Concatenating images...")
    composite = np.hstack(labeled_images)

    # Save
    print(f"Saving composite to: {output_path}")
    cv2.imwrite(output_path, composite)

    h, w = composite.shape[:2]
    file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB

    print(f"\n✓ SUCCESS!")
    print(f"  Dimensions: {w}x{h}px")
    print(f"  File size: {file_size:.1f} MB")
    print(f"  Output: {output_path}")

    return composite


def main():
    parser = argparse.ArgumentParser(
        description="Create horizontal composite image for pipeline report"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing stage images"
    )
    parser.add_argument(
        "--output",
        default="pipeline_report.png",
        help="Output composite image path (default: pipeline_report.png)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=2000,
        help="Target height for all images in pixels (default: 2000)"
    )

    args = parser.parse_args()

    create_horizontal_composite(
        args.input_dir,
        args.output,
        args.height
    )


if __name__ == "__main__":
    main()
