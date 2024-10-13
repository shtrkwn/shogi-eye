import cv2
import click
from shogi_eye.board_detection import find_shogi_board
from shogi_eye.grid_drawer import draw_grid

@click.command()
@click.option('--image', required=True, help='Path to the Shogi board image')
def main(image):
    """Main function to process the Shogi board image and draw a grid."""
    # 画像を読み込む
    img = cv2.imread(image)
    if img is None:
        print(f"Error: Could not load image at {image}")
        return

    # ステップ1: 将棋盤の検出と余白除去
    board = find_shogi_board(img)
    if board is None:
        print("Shogi board not found.")
        return

    # ステップ2: 各マスに赤い線を引く（グリッド描画）
    grid_image = draw_grid(board)

    # 結果を表示
    cv2.imshow("Shogi Board with Grid", grid_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
