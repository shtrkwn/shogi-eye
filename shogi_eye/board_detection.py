# shogi_eye/board_detection.py

import cv2
import numpy as np

def preprocess_image(image):
    """Convert to grayscale and apply edge detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # グレースケールに変換
    blur = cv2.GaussianBlur(gray, (5, 5), 0)       # ノイズを減らすためにぼかす
    edges = cv2.Canny(blur, 50, 150)               # エッジ検出
    return edges

def find_shogi_board_contour(edges):
    """Find the contour of the Shogi board."""
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 最大の輪郭を探す（将棋盤が画像全体に対して最も大きいと仮定）
    max_contour = max(contours, key=cv2.contourArea)
    
    # 輪郭を近似して4つの頂点を持つ多角形にする
    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    
    if len(approx) == 4:
        return approx
    else:
        return None  # 四角形が見つからなかった場合

def get_perspective_transform(image, contour):
    """Transform the detected Shogi board contour into a square image."""
    # 頂点を整理して左上、右上、右下、左下にマッピング
    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    # 左上、右下を特定
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下

    # 右上、左下を特定
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下

    # 変換後の画像サイズを定義
    max_width = max(int(np.linalg.norm(rect[0] - rect[1])), int(np.linalg.norm(rect[2] - rect[3])))
    max_height = max(int(np.linalg.norm(rect[0] - rect[3])), int(np.linalg.norm(rect[1] - rect[2])))

    # 目的の正方形の座標
    dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype="float32")

    # 透視変換行列を計算
    M = cv2.getPerspectiveTransform(rect, dst)

    # 画像を変換
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped
