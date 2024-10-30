import cv2
import numpy as np
import os
import time
import tkinter as tk
from tkinter import Button, Label, Frame
from PIL import Image, ImageTk
import cv2.aruco as aruco  

def draw_robotic_ui(frame, recording, start_time, capture_indicator):
    height, width, _ = frame.shape
    cv2.line(frame, (width // 2, 0), (width // 2, height), (0, 255, 0), 1)
    cv2.line(frame, (0, height // 2), (width, height // 2), (0, 255, 0), 1)
    center = (width // 2, height // 2)
    radius = 70
    cv2.circle(frame, center, radius, (0, 255, 0), 1)
    cv2.putText(frame, "Press 'q' to quit", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.circle(frame, (width // 2, height // 2), 2, (0, 0, 255), -1)

    # Small ticks at 45-degree intervals
    tick_length = 15
    angles = [45, 135, 225, 315]
    for angle in angles:
        radians = np.deg2rad(angle)
        x_outer = int(center[0] + radius * np.cos(radians))
        y_outer = int(center[1] - radius * np.sin(radians))
        x_inner = int(center[0] + (radius - tick_length) * np.cos(radians))
        y_inner = int(center[1] - (radius - tick_length) * np.sin(radians))
        cv2.line(frame, (x_inner, y_inner), (x_outer, y_outer), (0, 0, 255), 3)

    # Record and Capture Buttons
    center_capture = (width - 30, 30)
    cv2.circle(frame, center_capture, 20, (0, 0, 0), -1)  
    cv2.circle(frame, center_capture, 18, (255, 255, 255), -1)  
    
    capture_color = (0, 255, 0) if capture_indicator and time.time() - capture_indicator < 0.5 else (255, 255, 255)
    cv2.circle(frame, center_capture, 18, capture_color, -1)  # Green when capturing, white otherwise
    
    center_record = (width - 30, 70)
    cv2.circle(frame, center_record, 20, (0, 0, 0), -1)
    cv2.circle(frame, center_record, 18, (0, 0, 255), -1)

    # Capture Indicator - Blinking
    #if capture_indicator is not None:
        #capture_elapsed_time = int(time.time() - capture_indicator)
        #if capture_elapsed_time < 2:  # Show indicator for 2 seconds after capture
            #if capture_elapsed_time % 2 == 0:
                #cv2.circle(frame, (10, 10), 10, (0, 255, 0), -1)  # Green dot at top left corner

    # Recording Indicator
    if recording:
        elapsed_time = int(time.time() - start_time)
        if elapsed_time % 2 == 0:
            cv2.circle(frame, (width - 30, height - 30), 10, (0, 0, 255), -1)
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
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)  # Correct method for creating dictionary
    aruco_params = aruco.DetectorParameters()  # Instantiate DetectorParameters directly
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    if ids is not None:
        frame = aruco.drawDetectedMarkers(frame, corners, ids)
    return frame

def apply_segmentation(frame, k=2):
    pixel_values = frame.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(frame.shape)
    return segmented_image

def main():
    global frame, center_capture, center_record, recording, out, start_time, capture_indicator  # Declare frame as global
    capture_dir = "capture"
    if not os.path.exists(capture_dir):
        os.makedirs(capture_dir)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Define capture and record button positions
    center_capture = (640 - 30, 30)  # Update based on your frame size
    center_record = (640 - 30, 70)   # Update based on your frame size

    sobel_mode = None
    show_sobel = False
    show_negative = False
    recording = False
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
    show_segmentation = False

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    start_time = None
    out = None
    capture_indicator = None  # Initialize capture_indicator here

    # Tkinter GUI Setup
    root = tk.Tk()
    root.title("Robotic Vision UI")

        # Define the mouse click function here
    def mouse_click(event):
        global recording, out, start_time, capture_indicator  # Declare these variables as global
        # Check if the click is in the capture area
        if (event.x - center_capture[0]) ** 2 + (event.y - center_capture[1]) ** 2 < 18 ** 2:
            capture_path = os.path.join(capture_dir, f"capture_{int(time.time())}.png")
            cv2.imwrite(capture_path, frame)  # Save the current frame
            capture_indicator = time.time()  # Update capture indicator time
            print(f"Captured image saved at {capture_path}")

        # Check if the click is in the record area
        if (event.x - center_record[0]) ** 2 + (event.y - center_record[1]) ** 2 < 18 ** 2:
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

    # Bind mouse click events to the main window
    root.bind("<Button-1>", mouse_click)

    # Toggle functions for buttons
    def toggle_sobel():
        nonlocal show_sobel
        show_sobel = not show_sobel
        sobel_label.config(text=f"Sobel: {'ON' if show_sobel else 'OFF'}")

    def set_sobel_mode(mode):
        nonlocal sobel_mode
        if show_sobel:
            sobel_mode = mode
            sobel_mode_label.config(text=f"Sobel Mode: {mode}")

    def toggle_negative():
        nonlocal show_negative
        show_negative = not show_negative
        negative_label.config(text=f"Negative: {'ON' if show_negative else 'OFF'}")
        
    def toggle_thermal():
        nonlocal show_thermal
        show_thermal = not show_thermal
        thermal_label.config(text=f"Thermal: {'ON' if show_thermal else 'OFF'}")
        
    def toggle_canny():
        nonlocal show_canny
        show_canny = not show_canny
        canny_label.config(text=f"Canny: {'ON' if show_canny else 'OFF'}")
        
    def toggle_optic_flow():
        nonlocal show_optical_flow
        show_optical_flow = not show_optical_flow
        optic_flow_label.config(text=f"Canny: {'ON' if show_optical_flow else 'OFF'}")
        
    def toggle_canny():
        nonlocal show_canny
        show_canny = not show_canny
        canny_label.config(text=f"Canny: {'ON' if show_canny else 'OFF'}")

    # Status frame for filters
    status_frame = Frame(root)
    status_frame.pack(side="right")

    sobel_label = Label(status_frame, text="Sobel: OFF")
    sobel_label.pack(pady=5)

    sobel_mode_label = Label(status_frame, text="Sobel Mode: None")
    sobel_mode_label.pack(pady=5)

    negative_label = Label(status_frame, text="Negative: OFF")
    negative_label.pack(pady=5)
    
    thermal_label = Label(status_frame, text="Thermal: OFF")
    thermal_label.pack(pady=5)
    
    optic_flow_label = Label(status_frame, text="Optic Flow: OFF")
    optic_flow_label.pack(pady=5)
    
    canny_label = Label(status_frame, text="Canny: OFF")
    canny_label.pack(pady=5)

    # Button Frame
    button_frame = Frame(root)
    button_frame.pack(side="left")

    Button(button_frame, text="Toggle Sobel", command=toggle_sobel).pack(pady=5)
    Button(button_frame, text="Sobel X", command=lambda: set_sobel_mode('x')).pack(pady=5)
    Button(button_frame, text="Sobel Y", command=lambda: set_sobel_mode('y')).pack(pady=5)
    Button(button_frame, text="Sobel XY", command=lambda: set_sobel_mode('xy')).pack(pady=5)
    Button(button_frame, text="Toggle Negative", command=toggle_negative).pack(pady=5)
    Button(button_frame, text="Toggle Thermal", command=toggle_thermal).pack(pady=5)
    Button(button_frame, text="Toggle Optic Flow", command=toggle_optic_flow).pack(pady=5)
    Button(button_frame, text="Toggle Canny", command=toggle_canny).pack(pady=5)

    def update_frame():
        global frame  # Access the global frame variable
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            root.destroy()
            return

        frame = cv2.resize(frame, (640, 480))

        if show_sobel and sobel_mode:
            frame = apply_sobel(frame, sobel_mode)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if show_negative:
            frame = apply_negative(frame)
        
        if show_thermal:
            frame = apply_thermal(frame)
            
        if show_canny:
            frame = apply_canny(frame)

        frame = draw_robotic_ui(frame, recording, start_time, capture_indicator)

        # Convert OpenCV image to Tkinter format
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)

        if recording:
            out.write(frame)

        root.after(10, update_frame)

    # Keyboard event handling
    def key_handler(event):
        global frame, recording, out, start_time, capture_indicator  # Access the global frame variable
        if event.char == 'q':
            root.destroy()
        elif event.char == 'e':
            toggle_sobel()
        elif event.char == 'z':
            set_sobel_mode('xy')
        elif event.char == 'x':
            set_sobel_mode('x')
        elif event.char == 'c':
            set_sobel_mode('y')
        elif event.char == 'r':
            toggle_negative()
        elif event.char == 't':
            toggle_thermal()
        elif event.char == 't':
            toggle_canny()
        elif event.char == ' ':
            capture_path = os.path.join(capture_dir, f"capture_{int(time.time())}.png")
            cv2.imwrite(capture_path, frame)  # Save the current frame
            capture_indicator = time.time()  # Update capture indicator time
            print(f"Captured image saved at {capture_path}")
        elif event.char == 'v':
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

    # Label for camera feed
    camera_label = Label(root)
    camera_label.pack()
    root.bind("<Key>", key_handler)
    update_frame()
    root.mainloop()

    # Clean up
    cap.release()
    if recording and out is not None:
        out.release()

if __name__ == "__main__":
    main()
