import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from fpdf import FPDF

# Load YOLOv8 pose estimation model
pose_model = YOLO("yolov8n-pose.pt")

# Load video
video_path = r"C:\Users\USER\Desktop\WORKING_THESIS\RESOURCES\ACTUAL SURVEY\MARCH 7\MARCH 7 VIDEO (1).mp4"  # Change to your video path
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(r"C:\Users\USER\Desktop\MIGRATION_APP\APP_TESTING\VISUALIZATION_TESTING\ARM_VISUALIZATION\outputs\SANAAttempt3.mp4",
cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Store tracking info
tracked_persons = {}
frame_no = 0

def calculate_angle(a, b, c):
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        return None  # Skip invalid calculations
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return None  # Avoid division by zero
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return round(angle, 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = pose_model.track(frame, persist=True, classes=0, iou=0.5, agnostic_nms=True)
    person_pose = {}
    
    for r in results:
        if r.boxes is None or r.keypoints is None:
            continue
        
        for box, keypoints, conf, track_id in zip(r.boxes.xyxy, r.keypoints.xy, r.boxes.conf, r.boxes.id):
            if track_id is None:
                continue
            track_id = int(track_id)
            keypoints = keypoints.cpu().numpy()
            if track_id not in person_pose or conf > person_pose[track_id][0]:
                person_pose[track_id] = (conf, box, keypoints)
    
    for track_id, (_, box, keypoints) in person_pose.items():
        shoulder_r, shoulder_l = keypoints[6], keypoints[5]
        elbow_r, elbow_l = keypoints[8], keypoints[7]
        wrist_r, wrist_l = keypoints[10], keypoints[9]
        
        if np.any(shoulder_r == 0) or np.any(shoulder_l == 0) or np.any(elbow_r == 0) or np.any(elbow_l == 0) or np.any(wrist_r == 0) or np.any(wrist_l == 0):
            continue
        
        shoulder_angle_r = calculate_angle(elbow_r, shoulder_r, shoulder_l)
        shoulder_angle_l = calculate_angle(elbow_l, shoulder_l, shoulder_r)
        elbow_angle_r = calculate_angle(wrist_r, elbow_r, shoulder_r)
        elbow_angle_l = calculate_angle(wrist_l, elbow_l, shoulder_l)
        
        if track_id not in tracked_persons:
            tracked_persons[track_id] = {}
        tracked_persons[track_id][frame_no] = keypoints
        
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        for point in [shoulder_r, shoulder_l, elbow_r, elbow_l, wrist_r, wrist_l]:
            cv2.circle(frame, tuple(point.astype(int)), 5, (0, 255, 0), -1)
        
        for pair in [(wrist_r, elbow_r), (elbow_r, shoulder_r), (wrist_l, elbow_l), (elbow_l, shoulder_l)]:
            cv2.line(frame, tuple(pair[0].astype(int)), tuple(pair[1].astype(int)), (255, 0, 0), 2)
        
        cv2.putText(frame, f"{shoulder_angle_r}°", tuple(shoulder_r.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(frame, f"{shoulder_angle_l}°", tuple(shoulder_l.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(frame, f"{elbow_angle_r}°", tuple(elbow_r.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(frame, f"{elbow_angle_l}°", tuple(elbow_l.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    out.write(frame)
    frame_no += 1
    #cv2.imshow("Processed Video", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Processing complete. Video and PDF generated.")



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

        # Shoulder angle plot
        axes[2*i].plot(times, shoulder_angles_r, label=f"Person {person_id} Right Shoulder", color='blue')
        axes[2*i].plot(times, shoulder_angles_l, label=f"Person {person_id} Left Shoulder", linestyle='dashed', color='red')
        axes[2*i].set_ylabel("Shoulder Angle (°)")
        axes[2*i].legend()
        
        # Elbow angle plot
        axes[2*i+1].plot(times, elbow_angles_r, label=f"Person {person_id} Right Elbow", color='blue')
        axes[2*i+1].plot(times, elbow_angles_l, label=f"Person {person_id} Left Elbow", linestyle='dashed', color='red')
        axes[2*i+1].set_ylabel("Elbow Angle (°)")
        axes[2*i+1].legend()

    axes[-1].set_xlabel("Time (s)")
    plt.suptitle("Joint Angles Over Time (Stacked View)")
    plt.tight_layout()
    graph_image = f"stacked_graph_page_{page}.png"
    plt.savefig(graph_image, dpi=300)

    plt.close()

    pdf.add_page()
    pdf.image(graph_image, x=10, y=30, w=190)

pdf.output(r"C:\Users\USER\Desktop\MIGRATION_APP\APP_TESTING\VISUALIZATION_TESTING\ARM_VISUALIZATION\outputs\SANAAttempt3.pdf")
print("Processing complete. Video and PDF generated.")