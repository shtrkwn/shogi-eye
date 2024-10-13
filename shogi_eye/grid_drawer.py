# shogi_eye/grid_drawer.py

import cv2

def draw_grid(image, grid_size=9, color=(0, 0, 255), thickness=2):
    """Draw a grid on the given image."""
    height, width = image.shape[:2]

    # 各マスの幅と高さを計算
    cell_width = width // grid_size
    cell_height = height // grid_size

    # 横方向の線を描画
    for i in range(1, grid_size):
        y = i * cell_height
        cv2.line(image, (0, y), (width, y), color, thickness)

    # 縦方向の線を描画
    for i in range(1, grid_size):
        x = i * cell_width
        cv2.line(image, (x, 0), (x, height), color, thickness)

    return image
