import cv2
import numpy as np

def draw_robotic_ui(frame):
    # Get frame dimensions
    height, width, _ = frame.shape

    # Draw a crosshair in the center
    cv2.line(frame, (width // 2, 0), (width // 2, height), (0, 255, 0), 1)
    cv2.line(frame, (0, height // 2), (width, height // 2), (0, 255, 0), 1)

    # Draw a bounding box in the center
    box_size = 50
    top_left = (width // 2 - box_size, height // 2 - box_size)
    bottom_right = (width // 2 + box_size, height // 2 + box_size)
    cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)

    # Add text annotations
    cv2.putText(frame, "Robotic Vision UI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Press 'q' to quit", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Add circles to simulate object detection points
    for i in range(3):
        center = (np.random.randint(0, width), np.random.randint(0, height))
        cv2.circle(frame, center, 5, (0, 0, 255), -1)

    return frame

def apply_sobel(frame, sobel_mode):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if sobel_mode == 'xy':
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        sobel_combined = cv2.magnitude(sobel_x, sobel_y)
        return cv2.convertScaleAbs(sobel_combined)
    elif sobel_mode == 'x':
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        return cv2.convertScaleAbs(sobel_x)
    elif sobel_mode == 'y':
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        return cv2.convertScaleAbs(sobel_y)
    return gray

def main():
    # Capture video from the default camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    sobel_mode = None
    show_sobel = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to a smaller size
        frame = cv2.resize(frame, (640, 480))  # Resize to 640x480

        # Apply Sobel filter if enabled
        if show_sobel:
            frame = apply_sobel(frame, sobel_mode)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Draw robotic vision UI on the frame
        frame = draw_robotic_ui(frame)

        # Display the resulting frame
        cv2.imshow('Robotic Vision UI', frame)

        # Handle key press events
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            show_sobel = not show_sobel
            if not show_sobel:
                sobel_mode = None
        elif key == ord('z'):
            if show_sobel:
                sobel_mode = 'xy'
        elif key == ord('x'):
            if show_sobel:
                sobel_mode = 'x'
        elif key == ord('c'):
            if show_sobel:
                sobel_mode = 'y'

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
