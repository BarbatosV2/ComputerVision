import cv2
import numpy as np
import os
import time

def draw_robotic_ui(frame, recording, start_time):
    height, width, _ = frame.shape
    cv2.line(frame, (width // 2, 0), (width // 2, height), (0, 255, 0), 1)
    cv2.line(frame, (0, height // 2), (width, height // 2), (0, 255, 0), 1)
    box_size = 50
    top_left = (width // 2 - box_size, height // 2 - box_size)
    bottom_right = (width // 2 + box_size, height // 2 + box_size)
    cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
    cv2.putText(frame, "Robotic Vision UI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Press 'q' to quit", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw capture button with double loop
    center_capture = (width - 30, 30)
    cv2.circle(frame, center_capture, 20, (0, 0, 0), -1)  # Black circle for outline
    cv2.circle(frame, center_capture, 18, (255, 255, 255), -1)  # White circle for capture
    cv2.circle(frame, center_capture, 16, (0, 0, 0), 2)  # Inner outline

    # Draw record button with double loop
    center_record = (width - 30, 70)
    cv2.circle(frame, center_record, 20, (0, 0, 0), -1)  # Black circle for outline
    cv2.circle(frame, center_record, 18, (0, 0, 255), -1)  # Red circle for record
    cv2.circle(frame, center_record, 16, (0, 0, 0), 2)  # Inner outline

    # Show recording status and time
    if recording:
        elapsed_time = int(time.time() - start_time)
        cv2.circle(frame, (width - 30, height - 30), 10, (0, 0, 255), -1)  # Red dot for recording
        cv2.putText(frame, f"REC {elapsed_time}s", (width - 100, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

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

def apply_negative(frame):
    return cv2.bitwise_not(frame)

def apply_thermal(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

def apply_optical_flow(prev_frame, current_frame, prev_points):
    if prev_frame is None or prev_points is None:
        return current_frame, None, None

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    next_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, prev_points, None)
    good_new = next_points[status == 1]
    good_old = prev_points[status == 1]
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        current_frame = cv2.line(current_frame, (a, b), (c, d), (0, 255, 0), 2)
        current_frame = cv2.circle(current_frame, (a, b), 5, (0, 0, 255), -1)
    return current_frame, good_new.reshape(-1, 1, 2), status

def apply_canny(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def apply_cartoon(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 10)
    color = cv2.bilateralFilter(frame, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def apply_face_detection(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return frame

def apply_motion_detection(prev_frame, current_frame):
    if prev_frame is None:
        return current_frame, current_frame.copy()
    diff = cv2.absdiff(prev_frame, current_frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return current_frame, current_frame.copy()

def apply_hough_lines(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return frame

def apply_hough_circles(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, gray.shape[0] / 8, param1=100, param2=30, minRadius=1, maxRadius=30)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
    return frame

def apply_histogram_equalization(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

def apply_background_subtraction(frame):
    global bg_subtractor
    if 'bg_subtractor' not in globals():
        bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    fg_mask = bg_subtractor.apply(frame)
    return cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

def apply_aruco_markers(frame):
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters_create()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    return frame

def main():
    # Create capture directory if it doesn't exist
    capture_dir = "capture"
    if not os.path.exists(capture_dir):
        os.makedirs(capture_dir)

    # Use DirectShow backend
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    sobel_mode = None
    show_sobel = False
    show_negative = False
    show_thermal = False
    show_optical_flow = False
    show_canny = False
    show_cartoon = False
    show_face_detection = False
    show_motion_detection = False
    show_hough_lines = False
    show_hough_circles = False
    show_histogram_equalization = False
    show_background_subtraction = False
    show_aruco_markers = False

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    prev_frame = None
    prev_points = None

    recording = False
    out = None
    start_time = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal recording, out, start_time
        width = frame.shape[1]
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if the capture button is clicked
            if (width - 50) < x < (width - 10) and 10 < y < 50:
                capture_path = os.path.join(capture_dir, f"capture_{int(time.time())}.png")
                cv2.imwrite(capture_path, frame)
                print(f"Captured image saved at {capture_path}")

            # Check if the record button is clicked
            elif (width - 50) < x < (width - 10) and 50 < y < 90:
                if recording:
                    recording = False
                    out.release()
                    print("Recording stopped.")
                else:
                    recording = True
                    start_time = time.time()
                    video_path = os.path.join(capture_dir, f"record_{int(start_time)}.avi")
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))
                    print(f"Recording started. Video will be saved at {video_path}")

    cv2.namedWindow('Robotic Vision UI')
    cv2.setMouseCallback('Robotic Vision UI', mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.resize(frame, (640, 480))  # Resize to 640x480

        if show_sobel:
            frame = apply_sobel(frame, sobel_mode)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if show_negative:
            frame = apply_negative(frame)

        if show_thermal:
            frame = apply_thermal(frame)

        if show_optical_flow:
            frame, prev_points, _ = apply_optical_flow(prev_frame, frame, prev_points)

        if show_canny:
            frame = apply_canny(frame)

        if show_cartoon:
            frame = apply_cartoon(frame)

        if show_face_detection:
            frame = apply_face_detection(frame, face_cascade)

        if show_motion_detection:
            frame, prev_frame = apply_motion_detection(prev_frame, frame)

        if show_hough_lines:
            frame = apply_hough_lines(frame)

        if show_hough_circles:
            frame = apply_hough_circles(frame)

        if show_histogram_equalization:
            frame = apply_histogram_equalization(frame)

        if show_background_subtraction:
            frame = apply_background_subtraction(frame)

        if show_aruco_markers:
            frame = apply_aruco_markers(frame)

        frame = draw_robotic_ui(frame, recording, start_time)

        cv2.imshow('Robotic Vision UI', frame)

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
        elif key == ord('r'):
            show_negative = not show_negative
        elif key == ord('t'):
            show_thermal = not show_thermal
        elif key == ord('o'):
            show_optical_flow = not show_optical_flow
            if show_optical_flow and prev_frame is not None:
                prev_points = cv2.goodFeaturesToTrack(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        elif key == ord('n'):
            show_canny = not show_canny
        elif key == ord('m'):
            show_cartoon = not show_cartoon
        elif key == ord('f'):
            show_face_detection = not show_face_detection
        elif key == ord('d'):
            show_motion_detection = not show_motion_detection
        elif key == ord('l'):
            show_hough_lines = not show_hough_lines
        elif key == ord('k'):
            show_hough_circles = not show_hough_circles
        elif key == ord('h'):
            show_histogram_equalization = not show_histogram_equalization
        elif key == ord('b'):
            show_background_subtraction = not show_background_subtraction
        elif key == ord('a'):
            show_aruco_markers = not show_aruco_markers
        elif key == ord(' '):  # Space bar to capture image
            capture_path = os.path.join(capture_dir, f"capture_{int(time.time())}.png")
            cv2.imwrite(capture_path, frame)
            print(f"Captured image saved at {capture_path}")
        elif key == ord('v'):  # 'v' key to toggle video recording
            if recording:
                recording = False
                out.release()
                print("Recording stopped.")
            else:
                recording = True
                start_time = time.time()
                video_path = os.path.join(capture_dir, f"record_{int(start_time)}.avi")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))
                print(f"Recording started. Video will be saved at {video_path}")
        
        if recording:
            out.write(frame)

    cap.release()
    if recording:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
