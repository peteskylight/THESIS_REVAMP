import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from fpdf import FPDF
from collections import defaultdict

# Load YOLOv8 Pose Model
model = YOLO("yolov8m-pose.pt")

# Video input and output
video_path = r"C:\Users\USER\Desktop\WORKING_THESIS\RESOURCES\ACTUAL SURVEY\MARCH 7\MARCH 7 VIDEO (4).mp4"
output_video_path = r"C:\Users\USER\Desktop\MIGRATION_APP\APP_TESTING\VISUALIZATION_TESTING\HEAD_VISUALIZATION\NOSE_BASIS.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Data storage for angle visualization
tracked_persons = defaultdict(dict)
frame_count = 0

def calculate_angle(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return np.degrees(np.arctan2(dy, dx))

def classify_head_posture(nose_angle, head_tilt, left_ear_detected, right_ear_detected):
    if left_ear_detected and not right_ear_detected:
        return "Looking Right"
    elif right_ear_detected and not left_ear_detected:
        return "Looking Left"
    
    if (100 > head_tilt > 75) and (95 > nose_angle> -30):
        return "Looking Up"
    elif (-85 > head_tilt > -170) and (110 > nose_angle > 100):
        return "Looking Down"
    elif (-60 > head_tilt > -175) and (75 > nose_angle > 5):
        return "Looking Left"
    elif (-50 > head_tilt > -175) and (140 > nose_angle > 100):
        return "Looking Right"
    else:
        return "Facing Forward"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.track(frame, persist=True, iou=0.5, agnostic_nms = True)
    for result in results:
        for i, person in enumerate(result.keypoints.xy.numpy()):
            nose, left_eye, right_eye, left_ear, right_ear = person[0], person[1], person[2], person[3], person[4]
            
            # Ensure track ID stability
            if result.boxes.id is not None:
                person_id = int(result.boxes.id.numpy()[i])
            else:
                person_id = hash(tuple(nose)) % 10000
            
            # Define keypoints dynamically based on position
            person_x = nose[0]
            if person_x < frame_width * 0.33:
                visible_ear, opposite_ear = right_ear, left_ear
                visible_eye = right_eye
            elif person_x > frame_width * 0.66:
                visible_ear, opposite_ear = left_ear, right_ear
                visible_eye = left_eye
            else:
                visible_ear, opposite_ear = left_ear, right_ear
                visible_eye = left_eye
            
            # Handle missing ears
            if np.isnan(opposite_ear).any():
                opposite_ear = nose + (nose - visible_eye)
            
            # Calculate angles
            eye_mid = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
            nose_angle = round(calculate_angle(eye_mid, nose), 2)
            ear_mid = ((left_ear[0] + right_ear[0]) / 2, (left_ear[1] + right_ear[1]) / 2)
            head_tilt = round(calculate_angle(nose, ear_mid), 2)

            # Check if ears are detected (not NaN or missing)
            left_ear_detected = not np.isnan(left_ear).any()
            right_ear_detected = not np.isnan(right_ear).any()

            # Get head posture
            posture = classify_head_posture(nose_angle, head_tilt, left_ear_detected, right_ear_detected)


            
            # Store data
            tracked_persons[person_id][frame_count] = {'nose_angle': nose_angle, 'head_tilt': head_tilt}
            
            # Draw bounding box and text
            x1, y1, x2, y2 = map(int, result.boxes.xyxy.numpy()[i])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {person_id}: {posture}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.putText(frame, f"Nose Angle: {nose_angle}", (x1, y1 - 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            
            cv2.putText(frame, f"Head Tilt: {head_tilt}", (x1, y1 - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            
            # Draw only head keypoints
            for kp in [nose, left_eye, right_eye, left_ear, right_ear]:
                cv2.circle(frame, tuple(kp.astype(int)), 2, (0,255,0), -1)
    
    out.write(frame)
    frame_count += 1

cap.release()
out.release()

# Generate PDF with stacked graphs
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
person_ids = sorted(tracked_persons.keys())
num_pages = (len(person_ids) + 4) // 5

for page in range(num_pages):
    fig, axes = plt.subplots(10, 1, figsize=(8, 12), sharex=True)
    fig.subplots_adjust(hspace=0.3)
    persons_on_page = person_ids[page * 5:(page + 1) * 5]
    
    for i, person_id in enumerate(persons_on_page):
        frames = sorted(tracked_persons[person_id].keys())
        if not frames:
            continue
        times = np.array(frames) / fps
        nose_angles = [tracked_persons[person_id][f]['nose_angle'] for f in frames]
        head_tilts = [tracked_persons[person_id][f]['head_tilt'] for f in frames]
        
        axes[2*i].plot(times, nose_angles, label=f"Person {person_id} Nose Angle", color='blue')
        axes[2*i].set_ylabel("Nose Angle (°)")
        axes[2*i].legend()
        
        axes[2*i+1].plot(times, head_tilts, label=f"Person {person_id} Head Tilt", linestyle='dashed', color='red')
        axes[2*i+1].set_ylabel("Head Tilt (°)")
        axes[2*i+1].legend()
    
    axes[-1].set_xlabel("Time (s)")
    plt.suptitle("Head Angles Over Time (Stacked View)")
    plt.tight_layout()
    graph_image = f"head{page}.png"
    plt.savefig(graph_image)
    plt.close()
    pdf.add_page()
    pdf.image(graph_image, x=10, y=10, w=100)

pdf.output(r"C:\Users\USER\Desktop\MIGRATION_APP\APP_TESTING\VISUALIZATION_TESTING\HEAD_VISUALIZATION\NOSE_BASIS.pdf")
print("Processing complete. Video and PDF generated.")