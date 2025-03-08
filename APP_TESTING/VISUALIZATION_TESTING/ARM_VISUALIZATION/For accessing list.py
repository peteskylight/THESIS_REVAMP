import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from fpdf import FPDF

# Load YOLO models for detection and pose estimation
det_model = YOLO("yolov8n.pt")  # Human detection model
pose_model = YOLO("yolov8n-pose.pt")  # Pose estimation model

# Load video
video_path = r"C:\Users\USER\Desktop\WORKING_THESIS\RESOURCES\ACTUAL SURVEY\MARCH 7\MARCH 7 VIDEO (1).mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(r"C:\Users\USER\Desktop\MIGRATION_APP\APP_TESTING\VISUALIZATION_TESTING\ARM_VISUALIZATION\outputs\LASTAttempt3.mp4",
cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Store tracking info
detected_human_list = []  # Stores bounding boxes per frame
pose_tracking_list = []  # Stores keypoints per frame

# Dictionary for storing pose data for PDF
tracked_persons = {}

def calculate_angle(a, b, c):
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        return None
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return None
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return round(angle, 2)

# First pass: Detect and track humans
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = det_model.track(frame, persist=True, classes=0, iou=0.5, agnostic_nms=True)
    frame_tracking = {}
    
    for r in results:
        if r.boxes is None:
            continue
        
        for box, conf, track_id in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.id):
            if track_id is None:
                continue
            frame_tracking[int(track_id)] = box.cpu().numpy()
    
    detected_human_list.append(frame_tracking)

cap.release()
print("Human detection and tracking complete.")

# Second pass: Pose estimation using tracked bounding boxes
cap = cv2.VideoCapture(video_path)
frame_no = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_no >= len(detected_human_list):
        break
    
    frame_tracking = detected_human_list[frame_no]
    pose_tracking = {}
    
    for track_id, box in frame_tracking.items():
        x1, y1, x2, y2 = map(int, box)
        cropped_person = frame[y1:y2, x1:x2]
        
        pose_results = pose_model.predict(cropped_person)
        for r in pose_results:
            if r.keypoints is None:
                continue
            
            
            keypoints = r.keypoints.xy.cpu().numpy()
            if keypoints.shape[0] < 11 or keypoints.shape[1] < 2:
                continue

            
            pose_tracking[track_id] = keypoints
            
            if track_id not in tracked_persons:
                tracked_persons[track_id] = {}
            tracked_persons[track_id][frame_no] = keypoints
    
    pose_tracking_list.append(pose_tracking)
    frame_no += 1

    out.write(frame)  # Save the processed frame to output video

out.release()
cap.release()

print("Pose estimation complete.")

# -----------------------------------
# Generate PDF with Stacked Graphs (5 Persons Per Page)
# -----------------------------------
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)

# Prepare angle data for plotting
person_ids = sorted(tracked_persons.keys())
num_persons = len(person_ids)
num_pages = (num_persons + 4) // 5  # Groups of 5 persons per page

for page in range(num_pages):
    fig, axes = plt.subplots(10, 1, figsize=(8, 12), sharex=True)  # 2 plots per person, 5 persons per page
    fig.subplots_adjust(hspace=0.3)

    persons_on_page = person_ids[page * 5:(page + 1) * 5]
    for i, person_id in enumerate(persons_on_page):
        frames = sorted(tracked_persons[person_id].keys())
        times = np.array(frames) / fps  # Convert frame number to seconds

        shoulder_angles_r, shoulder_angles_l = [], []
        elbow_angles_r, elbow_angles_l = [], []

        for f in frames:
            keypoints = tracked_persons[person_id][f]
            shoulder_r, shoulder_l = keypoints[6], keypoints[5]
            elbow_r, elbow_l = keypoints[8], keypoints[7]
            wrist_r, wrist_l = keypoints[10], keypoints[9]

            shoulder_angle_r = calculate_angle(elbow_r, shoulder_r, shoulder_l)
            shoulder_angle_l = calculate_angle(elbow_l, shoulder_l, shoulder_r)
            elbow_angle_r = calculate_angle(wrist_r, elbow_r, shoulder_r)
            elbow_angle_l = calculate_angle(wrist_l, elbow_l, shoulder_l)

            shoulder_angles_r.append(shoulder_angle_r)
            shoulder_angles_l.append(shoulder_angle_l)
            elbow_angles_r.append(elbow_angle_r)
            elbow_angles_l.append(elbow_angle_l)

        axes[2*i].plot(times, shoulder_angles_r, label=f"Person {person_id} Right Shoulder", color='blue')
        axes[2*i].plot(times, shoulder_angles_l, label=f"Person {person_id} Left Shoulder", linestyle='dashed', color='red')
        axes[2*i+1].plot(times, elbow_angles_r, label=f"Person {person_id} Right Elbow", color='blue')
        axes[2*i+1].plot(times, elbow_angles_l, label=f"Person {person_id} Left Elbow", linestyle='dashed', color='red')
    
    plt.tight_layout()
    graph_image = f"stacked_graph_page_{page}.png"
    plt.savefig(graph_image, dpi=300)

    plt.close()

    pdf.add_page()
    pdf.image(graph_image, x=10, y=30, w=190)

pdf.output(r"C:\Users\USER\Desktop\MIGRATION_APP\APP_TESTING\VISUALIZATION_TESTING\ARM_VISUALIZATION\outputs\LASTAttempt3.pdf")
print("Processing complete. Video and PDF generated.")
