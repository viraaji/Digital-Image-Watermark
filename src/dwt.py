import numpy as np
from PIL import Image
import pywt
import argparse

def preprocess_image(image_path):
    img = Image.open(image_path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img = img.convert('L')
    width, height = img.size
    new_width = width - (width % 2)
    new_height = height - (height % 2)
    if new_width != width or new_height != height:
        img = img.resize((new_width, new_height))
    return np.array(img)

def embed_watermark(img_array, text):
    coeffs = pywt.dwt2(img_array, 'haar')
    cA, (cH, cV, cD) = coeffs
    binary = ''.join(format(ord(c), '08b') for c in text)
    for i, bit in enumerate(binary):
        row, col = i // cD.shape[1], i % cD.shape[1]
        cD[row, col] += 0.1 if bit == '1' else -0.1
    watermarked = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
    return np.uint8(np.clip(watermarked, 0, 255))

def extract_text(watermarked, original, text_len):
    _, (_, _, cD_w) = pywt.dwt2(watermarked, 'haar')
    _, (_, _, cD_o) = pywt.dwt2(original, 'haar')
    bits = ''
    for i in range(text_len * 8):
        row, col = i // cD_w.shape[1], i % cD_w.shape[1]
        bits += '1' if cD_w[row, col] > cD_o[row, col] else '0'
    return ''.join(chr(int(bits[i:i+8], 2)) for i in range(0, len(bits), 8))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--text', required=True)
    parser.add_argument('--output', default='watermarked.png')
    args = parser.parse_args()

    img_array = preprocess_image(args.image)
    watermarked = embed_watermark(img_array, args.text)
    extracted = extract_text(watermarked, img_array, len(args.text))

    Image.fromarray(watermarked).save(args.output)
    print(f"Original text: {args.text}")
    print(f"Extracted text: {extracted}")

if __name__ == "__main__":
    main()