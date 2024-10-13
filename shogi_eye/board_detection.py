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
    """Perform perspective transform to get a top-down view of the Shogi board."""
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

    # 目的の正方形の座標
    max_width = max(int(np.linalg.norm(rect[0] - rect[1])), int(np.linalg.norm(rect[2] - rect[3])))
    max_height = max(int(np.linalg.norm(rect[0] - rect[3])), int(np.linalg.norm(rect[1] - rect[2])))

    dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype="float32")

    # 透視変換行列を計算
    M = cv2.getPerspectiveTransform(rect, dst)

    # 画像を変換して切り抜き
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped

def crop_max_contour(image):
    """Find the largest contour in the transformed image and crop it."""
    # グレースケール変換とエッジ検出
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # 輪郭検出
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return image  # 輪郭が見つからなければ何もしない

    # 最大の輪郭を取得
    max_contour = max(contours, key=cv2.contourArea)

    # 輪郭の外接矩形を取得して、その領域を切り抜く
    x, y, w, h = cv2.boundingRect(max_contour)
    cropped = image[y:y+h, x:x+w]

    return cropped

def find_shogi_board(image):
    """Detect the Shogi board, transform it, and crop out the excess margin."""
    # ステップ1: エッジ検出
    edges = preprocess_image(image)
    
    # ステップ2: 将棋盤の輪郭を見つける
    contour = find_shogi_board_contour(edges)
    if contour is None:
        return None
    
    # ステップ3: 透視変換を行い、上から見た状態にする
    transformed_board = get_perspective_transform(image, contour)
    
    # ステップ4: 最大の矩形領域を切り抜き、余白を除去
    cropped_board = crop_max_contour(transformed_board)

    return cropped_board
