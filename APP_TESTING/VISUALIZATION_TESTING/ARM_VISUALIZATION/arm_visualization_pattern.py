import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from ultralytics import YOLO

# Load YOLOv8 Pose model
model = YOLO("yolov8n-pose.pt")

# Skeleton connections for visualization (Arms & Shoulders only)
SKELETON_CONNECTIONS = [(5, 7), (7, 9), (6, 8), (8, 10)]  # Left & Right arms

def calculate_angle(p1, p2, p3):
    """Calculate the angle formed by three points (in degrees)."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Avoid division by zero
    if norm_v1 == 0 or norm_v2 == 0:
        return 0  # Return 0 degrees if the vectors collapse to a point

    cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def draw_skeleton_and_angles(frame, keypoints, person_id, angle_data, frame_count):
    """Draw bounding boxes, keypoints, skeleton lines, and annotate angles on arms and shoulders."""

    if keypoints is None or len(keypoints) < 11:
        return

    # Extract key arm and shoulder keypoints
    right_shoulder, right_elbow, right_wrist = keypoints[5], keypoints[7], keypoints[9]
    left_shoulder, left_elbow, left_wrist = keypoints[6], keypoints[8], keypoints[10]

    if np.any(np.isnan([right_shoulder, right_elbow, right_wrist, left_shoulder, left_elbow, left_wrist])):
        return

    # Calculate angles for arms
    right_elbow_wrist_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    right_shoulder_elbow_angle = calculate_angle(keypoints[6], right_shoulder, right_elbow)
    left_elbow_wrist_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    left_shoulder_elbow_angle = calculate_angle(keypoints[5], left_shoulder, left_elbow)

    # Store angle data for export
    angle_data.append((frame_count, person_id, {
        "Right Elbow-Wrist": right_elbow_wrist_angle,
        "Right Shoulder-Elbow": right_shoulder_elbow_angle,
        "Left Elbow-Wrist": left_elbow_wrist_angle,
        "Left Shoulder-Elbow": left_shoulder_elbow_angle,
    }))

    # Bounding box for the person
    x_min, y_min = np.min(keypoints[:, 0]), np.min(keypoints[:, 1])
    x_max, y_max = np.max(keypoints[:, 0]), np.max(keypoints[:, 1])

    # Draw bounding box with track ID
    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
    cv2.putText(frame, f"ID: {person_id}", (int(x_min), int(y_min) - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw keypoints only for arms and shoulders
    for point in [right_shoulder, right_elbow, right_wrist, left_shoulder, left_elbow, left_wrist]:
        cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 255, 255), -1)  # Yellow keypoints

    # Draw skeleton for arms only
    for (p1, p2) in SKELETON_CONNECTIONS:
        x1, y1 = map(int, keypoints[p1])
        x2, y2 = map(int, keypoints[p2])
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Display angles next to the correct keypoints
    text_offset = 15
    cv2.putText(frame, f"{int(right_elbow_wrist_angle)}°", (int(right_elbow[0]) + text_offset, int(right_elbow[1])), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f"{int(right_shoulder_elbow_angle)}°", (int(right_shoulder[0]) + text_offset, int(right_shoulder[1])), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, f"{int(left_elbow_wrist_angle)}°", (int(left_elbow[0]) - text_offset * 2, int(left_elbow[1])), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f"{int(left_shoulder_elbow_angle)}°", (int(left_shoulder[0]) - text_offset * 2, int(left_shoulder[1])), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
def export_angles_to_pdf(angle_data, filename):
    """Export time-based joint angle graphs for multiple persons."""
    with PdfPages(filename) as pdf:
        times = list(range(len(angle_data)))
        people_ids = sorted(set(person_id for _, person_id, _ in angle_data))

        for person_id in people_ids:
            fig, ax = plt.subplots(figsize=(10, 5))

            right_shoulder_angles = [data[2]["Right Shoulder-Elbow"] for data in angle_data if data[1] == person_id]
            left_shoulder_angles = [data[2]["Left Shoulder-Elbow"] for data in angle_data if data[1] == person_id]
            right_elbow_angles = [data[2]["Right Elbow-Wrist"] for data in angle_data if data[1] == person_id]
            left_elbow_angles = [data[2]["Left Elbow-Wrist"] for data in angle_data if data[1] == person_id]

            ax.plot(times[:len(right_shoulder_angles)], right_shoulder_angles, label="Right Shoulder", color='blue')
            ax.plot(times[:len(left_shoulder_angles)], left_shoulder_angles, label="Left Shoulder", linestyle='dashed', color='red')
            ax.plot(times[:len(right_elbow_angles)], right_elbow_angles, label="Right Elbow", color='cyan')
            ax.plot(times[:len(left_elbow_angles)], left_elbow_angles, label="Left Elbow", linestyle='dashed', color='magenta')

            ax.set_xlabel("Time (frames)")
            ax.set_ylabel("Angle (degrees)")
            ax.set_title(f"Person {person_id} Joint Angles Over Time")
            ax.legend()
            ax.grid()

            pdf.savefig(fig)
            plt.close()

    print(f"Angles exported to {filename}")

def process_video(input_video, output_video, output_pdf):
    cap = cv2.VideoCapture(input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    angle_data = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        keypoints_list = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else []

        for person_id, keypoints in enumerate(keypoints_list):
            draw_skeleton_and_angles(frame, keypoints, person_id, angle_data, frame_count)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    export_angles_to_pdf(angle_data, output_pdf)


process_video(r"C:\Users\USER\Desktop\WORKING_THESIS\RESOURCES\ACTUAL SURVEY\MARCH 7\MARCH 7 VIDEO (1).mp4",
              r"C:\Users\USER\Desktop\MIGRATION_APP\APP_TESTING\VISUALIZATION_TESTING\ARM_VISUALIZATION\outputs\Attempt1.mp4",
              r"C:\Users\USER\Desktop\MIGRATION_APP\APP_TESTING\VISUALIZATION_TESTING\ARM_VISUALIZATION\outputs\Attempt1.pdf")



