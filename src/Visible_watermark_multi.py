import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import numpy as np
from PIL import Image, ImageDraw

img = Image.new('RGB', (800, 600), color='lightblue')
draw = ImageDraw.Draw(img)
draw.ellipse([300, 200, 500, 400], fill='yellow')
img.save('test.jpg')


class Watermarker:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.height, self.width = self.image.shape[:2]

    def create_logo(self, text, size=(200, 200)):
        logo = np.zeros((size[0], size[1], 4), dtype=np.uint8)
        center = (size[0]//2, size[1]//2)
        radius = min(size[0], size[1])//3

        cv2.circle(logo, center, radius, (255, 255, 255, 255), -1)
        cv2.circle(logo, center, radius-2, (0, 0, 0, 255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = radius * 0.02
        text_size = cv2.getTextSize(text, font, font_scale, 2)[0]
        text_pos = (center[0] - text_size[0]//2, center[1] + text_size[1]//2)

        cv2.putText(logo, text, text_pos, font, font_scale, (0, 0, 0, 255), 2)
        return logo

    def add_text_watermark(self, text, position='bottom-right', opacity=0.7):
        overlay = self.image.copy()
        font = cv2.FONT_HERSHEY_COMPLEX
        font_scale = self.width * 0.001
        thickness = max(1, int(self.width * 0.002))

        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

        if position == 'bottom-right':
            x = self.width - text_width - 10
            y = self.height - 10
        elif position == 'top-right':
            x = self.width - text_width - 10
            y = text_height + 10

        cv2.putText(overlay, text, (x, y), font, font_scale, (255, 255, 255), thickness + 2)
        cv2.putText(overlay, text, (x, y), font, font_scale, (0, 0, 0), thickness)

        return cv2.addWeighted(overlay, opacity, self.image, 1 - opacity, 0)

    def add_diagonal_text(self, text, opacity=0.3):
        overlay = self.image.copy()
        font = cv2.FONT_HERSHEY_COMPLEX
        font_scale = self.width * 0.003
        thickness = max(2, int(self.width * 0.004))

        angle = math.atan2(self.height, self.width)
        diagonal_length = math.sqrt(self.width**2 + self.height**2)

        matrix = cv2.getRotationMatrix2D((self.width/2, self.height/2), math.degrees(angle), 1)
        cv2.putText(overlay, text, (int(self.width*0.2), int(self.height*0.5)), font, font_scale, (200, 200, 200), thickness)

        return cv2.addWeighted(overlay, opacity, self.image, 1 - opacity, 0)

    def add_tiled_pattern(self, text, opacity=0.1):
        overlay = self.image.copy()
        font = cv2.FONT_HERSHEY_COMPLEX
        font_scale = self.width * 0.001
        thickness = 1

        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

        for y in range(0, self.height, text_height * 4):
            for x in range(0, self.width, text_width * 4):
                cv2.putText(overlay, text, (x, y), font, font_scale, (128, 128, 128), thickness)

        return cv2.addWeighted(overlay, opacity, self.image, 1 - opacity, 0)

    def add_logo_watermark(self, scale=0.2, position='bottom-right'):
        logo = self.create_logo('©')

        w_width = int(self.width * scale)
        w_height = int(logo.shape[0] * (w_width / logo.shape[1]))
        logo = cv2.resize(logo, (w_width, w_height))

        if position == 'bottom-right':
            x = self.width - w_width - 10
            y = self.height - w_height - 10

        result = self.image.copy()
        alpha = logo[:, :, 3] / 255.0
        for c in range(3):
            result[y:y+w_height, x:x+w_width, c] = \
                (1-alpha) * result[y:y+w_height, x:x+w_width, c] + \
                alpha * logo[:, :, c]
        return result

watermarker = Watermarker('test.jpg')
result1 = watermarker.add_text_watermark('Blockchain')
result2 = watermarker.add_diagonal_text('Blockchain')
result3 = watermarker.add_tiled_pattern('BlockchainY')
result4 = watermarker.add_logo_watermark()

cv2.imwrite('text_watermark.jpg', result1)
cv2.imwrite('diagonal_watermark.jpg', result2)
cv2.imwrite('tiled_watermark.jpg', result3)
cv2.imwrite('logo_watermark.jpg', result4)