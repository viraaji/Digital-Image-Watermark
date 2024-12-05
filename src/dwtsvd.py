import numpy as np
import cv2
import pywt
from numpy.linalg import svd

class DwtSvdWatermarker:
    def embed(self, host_image, watermark, alpha=0.1):
        host = cv2.imread(host_image, 0)
        mark = cv2.imread(watermark, 0)
        mark = cv2.resize(mark, (host.shape[1]//2, host.shape[0]//2))

        coeffs2 = pywt.dwt2(host, 'haar')
        LL, (LH, HL, HH) = coeffs2

        U_h, S_h, V_h = svd(LL)
        U_w, S_w, V_w = svd(mark)

        S_wm = S_h + alpha * S_w
        LL_wm = np.dot(U_h, np.dot(np.diag(S_wm), V_h))

        coeffs_wm = LL_wm, (LH, HL, HH)
        watermarked = pywt.idwt2(coeffs_wm, 'haar')

        return np.uint8(watermarked)

    def extract(self, watermarked_image, original_image, alpha=0.1):
        watermarked = cv2.imread(watermarked_image, 0)
        original = cv2.imread(original_image, 0)

        coeffs2_wm = pywt.dwt2(watermarked, 'haar')
        LL_wm, _ = coeffs2_wm

        coeffs2 = pywt.dwt2(original, 'haar')
        LL, _ = coeffs2

        U_wm, S_wm, V_wm = svd(LL_wm)
        U, S, V = svd(LL)

        S_w = (S_wm - S) / alpha
        extracted = np.dot(U_wm, np.dot(np.diag(S_w), V_wm))

        return np.uint8(extracted)

def create_test_images():
    host = np.zeros((512, 512), dtype=np.uint8)
    cv2.putText(host, "Original", (150, 250), cv2.FONT_HERSHEY_COMPLEX, 2, 255, 2)
    cv2.imwrite('host.png', host)

    watermark = np.zeros((256, 256), dtype=np.uint8)
    cv2.putText(watermark, "SECRET", (50, 125), cv2.FONT_HERSHEY_COMPLEX, 1, 255, 2)
    cv2.imwrite('watermark.png', watermark)

def test_watermark():
    create_test_images()
    watermarker = DwtSvdWatermarker()

    watermarked = watermarker.embed('host.png', 'watermark.png', alpha=0.1)
    cv2.imwrite('watermarked_dwt_svd.png', watermarked)

    extracted = watermarker.extract('watermarked_dwt_svd.png', 'host.png', alpha=0.1)
    cv2.imwrite('extracted_dwt_svd.png', extracted)

test_watermark()