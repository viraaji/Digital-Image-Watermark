import numpy as np
import cv2
from scipy.fft import dct, idct
import os

class LSBWatermarker:
    def embed(self, image_path, watermark_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        wm = cv2.imread(watermark_path, 0)
        if wm is None:
            raise ValueError(f"Could not read watermark: {watermark_path}")

        wm = cv2.resize(wm, (img.shape[1], img.shape[0]))

        plane = np.zeros(img.shape[0:2], dtype=np.uint8)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                plane[i,j] = (wm[i,j] & 0x80) >> 7
                img[i,j,0] = (img[i,j,0] & 0xFE) | plane[i,j]

        return img

    def extract(self, watermarked_path):
        img = cv2.imread(watermarked_path)
        if img is None:
            raise ValueError(f"Could not read watermarked image: {watermarked_path}")

        extracted = np.zeros(img.shape[0:2], dtype=np.uint8)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                extracted[i,j] = (img[i,j,0] & 0x01) << 7
        return extracted

class DCTWatermarker:
    def embed(self, image_path, watermark_path, alpha=0.1):
        img = cv2.imread(image_path, 0)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        wm = cv2.imread(watermark_path, 0)
        if wm is None:
            raise ValueError(f"Could not read watermark: {watermark_path}")

        wm = cv2.resize(wm, (img.shape[1]//8, img.shape[0]//8))
        img_float = np.float32(img)
        dct_coeffs = dct(dct(img_float.T).T)

        for i in range(wm.shape[0]):
            for j in range(wm.shape[1]):
                dct_coeffs[i,j] += alpha * wm[i,j]

        watermarked = idct(idct(dct_coeffs.T).T)
        return np.uint8(watermarked)

    def extract(self, watermarked_path, original_path, alpha=0.1):
        watermarked = cv2.imread(watermarked_path, 0)
        if watermarked is None:
            raise ValueError(f"Could not read watermarked image: {watermarked_path}")

        original = cv2.imread(original_path, 0)
        if original is None:
            raise ValueError(f"Could not read original image: {original_path}")

        dct_wm = dct(dct(np.float32(watermarked).T).T)
        dct_orig = dct(dct(np.float32(original).T).T)

        extracted = (dct_wm - dct_orig) / alpha
        extracted = extracted[:watermarked.shape[0]//8, :watermarked.shape[1]//8]
        return np.uint8(extracted)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def test_watermarks():
    output_dir = 'watermark_output'
    ensure_dir(output_dir)

    host_path = os.path.join(output_dir, 'host.png')
    watermark_path = os.path.join(output_dir, 'watermark.png')

    host = np.zeros((512, 512), dtype=np.uint8)
    cv2.putText(host, "Original", (150, 250), cv2.FONT_HERSHEY_COMPLEX, 2, 255, 2)
    cv2.imwrite(host_path, host)

    watermark = np.zeros((256, 256), dtype=np.uint8)
    cv2.putText(watermark, "SECRET", (50, 125), cv2.FONT_HERSHEY_COMPLEX, 1, 255, 2)
    cv2.imwrite(watermark_path, watermark)

    lsb = LSBWatermarker()
    dct = DCTWatermarker()

    try:
        lsb_watermarked = lsb.embed(host_path, watermark_path)
        lsb_watermarked_path = os.path.join(output_dir, 'watermarked_lsb.png')
        cv2.imwrite(lsb_watermarked_path, lsb_watermarked)

        lsb_extracted = lsb.extract(lsb_watermarked_path)
        cv2.imwrite(os.path.join(output_dir, 'extracted_lsb.png'), lsb_extracted)

        dct_watermarked = dct.embed(host_path, watermark_path)
        dct_watermarked_path = os.path.join(output_dir, 'watermarked_dct.png')
        cv2.imwrite(dct_watermarked_path, dct_watermarked)

        dct_extracted = dct.extract(dct_watermarked_path, host_path)
        cv2.imwrite(os.path.join(output_dir, 'extracted_dct.png'), dct_extracted)

        print("Watermarking completed successfully")

    except Exception as e:
        print(f"Error during watermarking: {str(e)}")

if __name__ == "__main__":
    test_watermarks()