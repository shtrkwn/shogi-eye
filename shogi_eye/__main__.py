import cv2
import click
import signal
import sys

from shogi_eye.board_detection import preprocess_image, find_shogi_board_contour, get_perspective_transform
from shogi_eye.piece_recognition import match_template

def signal_handler(sig, frame):
    """Handle Ctrl-C (SIGINT) signal."""
    print("\nProcess interrupted. Closing windows...")
    cv2.destroyAllWindows()
    sys.exit(0)

# Ctrl-C (SIGINT) シグナルをキャッチするためのハンドラーを設定
signal.signal(signal.SIGINT, signal_handler)

@click.command()
@click.option('--image', required=True, help='Path to the Shogi board image')
def main(image):
    """Main function to process the Shogi board image."""
    # 画像を読み込む
    img = cv2.imread(image)
    if img is None:
        print(f"Error: Could not load image at {image}")
        return

    # ステップ1: エッジ検出と将棋盤検出
    edges = preprocess_image(img)
    contour = find_shogi_board_contour(edges)

    if contour is not None:
        warped_image = get_perspective_transform(img, contour)
        cv2.imshow("Warped Shogi Board", warped_image)
        
        # 'q'キーでウィンドウを閉じる
        print("Press 'q' to close the window or use Ctrl-C to terminate.")
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q'キーで終了
                break
    else:
        print("Shogi board contour not found.")

    # ウィンドウを閉じる
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
