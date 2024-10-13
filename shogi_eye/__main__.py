import click
import cv2

@click.command()
@click.option('--image', required=True, help='Path to the Shogi board image')
def display_image(image):
    """Load and display the image using OpenCV."""
    img = cv2.imread(image)

    if img is None:
        click.echo(f"Error: Could not load image at {image}")
        return

    click.echo("Displaying the image... Press any key to close the window.")
    
    cv2.imshow("Shogi Board", img)

    # Use a loop to wait for a key press and close the window
    while True:
        key = cv2.waitKey(1)  # Check for a key press with a short delay
        if key != -1:  # Any key pressed will break the loop
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_image()
