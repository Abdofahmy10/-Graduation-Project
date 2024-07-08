import cv2
import os


def capture_frames(video_path, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate frame interval in milliseconds
    frame_interval_ms = int(1000 / fps)

    # Initialize variables
    frame_count = 0
    success = True

    while success:
        # Read frame from video
        success, frame = cap.read()

        if frame_count % fps == 0:
            # Generate filename based on frame count
            filename = f"frame_{frame_count // fps}.jpg"

            # Save frame as image in output directory
            cv2.imwrite(os.path.join(output_dir, filename), frame)

        # Increment frame count
        frame_count += 1

        # Wait for the next frame
        cv2.waitKey(frame_interval_ms)

    # Release video capture
    cap.release()


# Example usage
video_path = "videos/bad.mp4"
output_dir = "out"

capture_frames(video_path, output_dir)