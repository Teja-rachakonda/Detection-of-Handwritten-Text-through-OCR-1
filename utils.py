import cv2
import numpy as np

def preprocess_image(image_path, manual_thresh=None):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if manual_thresh is not None:
        _, img = cv2.threshold(img, manual_thresh, 255, cv2.THRESH_BINARY_INV)
    return img

def segment_lines(img, thresh=2):
    horizontal_sum = np.sum(img < 128, axis=1)
    lines = []
    start = None
    for i, val in enumerate(horizontal_sum):
        if val > thresh and start is None:
            start = i
        elif val <= thresh and start is not None:
            end = i
            if end - start > 10:
                lines.append(img[start:end, :])
            start = None
    if start is not None:
        lines.append(img[start:, :])
    return lines

def segment_words(line_img):
    contours, _ = cv2.findContours(line_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    word_images = []
    sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    for cnt in sorted_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 5:
            word_img = line_img[y:y+h, x:x+w]
            word_images.append(word_img)
    return word_images

def filter_ocr_output(text, min_len=2, max_repeat=2):
    import re
    lines = text.split('\n')
    seen = {}
    filtered = []
    for line in lines:
        line = re.sub(r'[^a-zA-Z0-9\s,.\-\'#]', '', line).strip()
        if len(line) < min_len:
            continue
        count = seen.get(line, 0)
        if count < max_repeat:
            filtered.append(line)
            seen[line] = count + 1
    return "\n".join(filtered)
