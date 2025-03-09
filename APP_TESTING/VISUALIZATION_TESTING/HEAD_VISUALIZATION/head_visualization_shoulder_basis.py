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
output_video_path = r"C:\Users\USER\Desktop\MIGRATION_APP\APP_TESTING\VISUALIZATION_TESTING\HEAD_VISUALIZATION\SHOULDER_BASED.mp4"
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

def classify_head_posture(nose, eyes_mid, ears_mid, shoulders_mid):
    """
    Classifies head posture based on nose, eye midpoint, ear midpoint, and shoulder midpoint.
    """
    head_relative_angle = calculate_angle(eyes_mid, nose)
    head_relative_tilt = calculate_angle(nose, ears_mid)
    shoulder_tilt = calculate_angle(shoulders_mid, ears_mid)
    
    # Adjust angles relative to shoulder tilt for more accuracy
    head_relative_tilt = head_relative_tilt - shoulder_tilt

    # Looking Up: Nose is significantly above shoulders, and head tilt is positive
    if nose[1] < shoulders_mid[1] - 20 and head_relative_tilt > 15:
        return "Looking Up"
    
    # Looking Down: Nose is below shoulders, and head tilt is negative
    elif nose[1] > shoulders_mid[1] + 20 and head_relative_tilt < -15:
        return "Looking Down"

    # Looking Left: Nose crosses over the left shoulder
    elif nose[0] < shoulders_mid[0] - 15:
        return "Looking Left"

    # Looking Right: Nose crosses over the right shoulder
    elif nose[0] > shoulders_mid[0] + 15:
        return "Looking Right"

    # If none of the above, assume Facing Forward
    return "Facing Forward"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.track(frame, persist=True, iou=0.5, agnostic_nms=True)
    
    for result in results:
        for i, person in enumerate(result.keypoints.xy.numpy()):
            nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder = \
                person[0], person[1], person[2], person[3], person[4], person[5], person[6]

            # Ensure track ID stability
            if result.boxes.id is not None:
                person_id = int(result.boxes.id.numpy()[i])
            else:
                person_id = hash(tuple(nose)) % 10000

            # Compute midpoints
            eyes_mid = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
            ears_mid = ((left_ear[0] + right_ear[0]) / 2, (left_ear[1] + right_ear[1]) / 2)
            shoulders_mid = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)

            head_relative_angle = round(calculate_angle(nose, shoulders_mid),2)
            head_relative_tilt = round(calculate_angle(ears_mid, shoulders_mid), 2)
            
            # Handle missing keypoints (if one ear is missing, estimate from available keypoints)
            if np.isnan(left_ear).any():
                left_ear = nose + (nose - right_eye)
            if np.isnan(right_ear).any():
                right_ear = nose + (nose - left_eye)

            # Classify head posture
            posture = classify_head_posture(nose, eyes_mid, ears_mid, shoulders_mid)

            # Store data
            tracked_persons[person_id][frame_count] = {'head_relative_angle': head_relative_angle,
                                                       'head_relative_tilt': head_relative_tilt}

            # Draw bounding box and text
            x1, y1, x2, y2 = map(int, result.boxes.xyxy.numpy()[i])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {person_id}: {posture}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.putText(frame, f"Nose Angle: {head_relative_angle}", (x1, y1 - 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            
            cv2.putText(frame, f"Head Tilt: {head_relative_tilt}", (x1, y1 - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)


            # Draw head keypoints
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
        nose_angles = [tracked_persons[person_id][f]['head_relative_angle'] for f in frames]
        head_tilts = [tracked_persons[person_id][f]['head_relative_tilt'] for f in frames]

        axes[2*i].plot(times, nose_angles, label=f"Person {person_id} Nose Angle", color='blue')
        axes[2*i+1].plot(times, head_tilts, label=f"Person {person_id} Head Tilt", linestyle='dashed', color='red')
    
    plt.tight_layout()
    graph_image = f"head{page}.png"
    plt.savefig(graph_image)
    pdf.add_page()
    pdf.image(graph_image, x=10, y=10, w=100)

pdf.output(r"C:\Users\USER\Desktop\MIGRATION_APP\APP_TESTING\VISUALIZATION_TESTING\HEAD_VISUALIZATION\SHOULDER_BASED.pdf")
print("Processing complete. Video and PDF generated.")
