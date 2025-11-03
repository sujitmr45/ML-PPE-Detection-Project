import cv2
import time
from ultralytics import YOLO

# Load trained PPE detection model
model = YOLO(r"C:\Users\Dell\OneDrive\Desktop\indoai\best90.pt")  # replace with your path

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

frame_count = 0
total_inference_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame")
        break

    # Resize frame for faster inference (optional, adjust as needed)
    resized_frame = cv2.resize(frame, (640, 640))

    # Start timer
    start_time = time.time()
    results = model(resized_frame)
    end_time = time.time()

    inference_time = end_time - start_time
    total_inference_time += inference_time
    frame_count += 1
    fps = 1 / inference_time

    # Annotate frame
    annotated_frame = results[0].plot()
    
    # Count number of detected boxes
    num_boxes = len(results[0].boxes)
    
    # Display info
    cv2.putText(annotated_frame, f"Detections: {num_boxes}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Inference: {inference_time*1000:.1f} ms | FPS: {fps:.1f}", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Display frame
    cv2.imshow("PPE Detection - Press 'q' to quit", annotated_frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Average inference time and FPS
if frame_count > 0:
    avg_inference_time = total_inference_time / frame_count
    avg_fps = 1 / avg_inference_time
    print(f"\nAverage inference time per frame: {avg_inference_time*1000:.2f} ms")
    print(f"Average FPS: {avg_fps:.2f}")
    
    if avg_fps >= 15:
        print("✅ Suitable for industrial real-time use")
    else:
        print("⚠️ May be too slow for industrial real-time use. Consider smaller model or GPU.")

cap.release()
cv2.destroyAllWindows()



