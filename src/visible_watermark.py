import cv2
import numpy as np
from PIL import Image, ImageDraw

def add_watermark(image_path, watermark_text, output_path):
    """
    Add a visible watermark to an image

    Args:
        image_path (str): Path to the input image
        watermark_text (str): Text to use as watermark
        output_path (str): Path to save the watermarked image
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image")

    # Get image dimensions
    height, width = image.shape[:2]

    # Create a copy of the image
    watermarked = image.copy()

    # Set watermark properties
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = width * 0.001  # Scale font based on image width
    thickness = max(1, int(width * 0.002))  # Scale thickness based on image width

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        watermark_text, font, font_scale, thickness
    )

    # Calculate position (bottom right corner)
    x = width - text_width - 10
    y = height - 10

    # Add semi-transparent background for better visibility
    overlay = watermarked.copy()
    cv2.putText(
        overlay, watermark_text, (x, y), font, font_scale,
        (255, 255, 255), thickness + 2  # White outline
    )
    cv2.putText(
        overlay, watermark_text, (x, y), font, font_scale,
        (0, 0, 0), thickness  # Black text
    )

    # Blend the watermark with the original image
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, watermarked, 1 - alpha, 0, watermarked)

    # Save the watermarked image
    cv2.imwrite(output_path, watermarked)

# Example usage
if __name__ == "__main__":
    img = Image.new('RGB', (800, 600), color='lightblue')
    draw = ImageDraw.Draw(img)
    draw.ellipse([300, 200, 500, 400], fill='yellow')  # Draw a sun
    img.save('test.jpg')
    add_watermark("test.jpg", "Blochain 2024", "test_visible_watermark.jpg")