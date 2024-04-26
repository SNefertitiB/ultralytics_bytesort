"""
use past measurements to predict future velocity / location
"""

from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

FORECAST_LENGTH = 10
FORECAST_COLOR = (0,0, 230)
FORECAST_SIZE = 5
FRAME_X = 640
FRAME_Y = 384

# Load the YOLOv8 model
model = YOLO('best_yolov8_base.pt')

# Open the video file
video_path = "vid.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()

        # added try except block
        # I think an error is thrown because custom dataset does not
        # include trained objects
        try:
            track_ids = results[0].boxes.id.int().cpu().tolist()
        except AttributeError:
            track_ids = []

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # Draw the forcast points
            if len(track)>1:
                last = track[-1]
                penultimate = track[-2]
                velocity_x = last[0] - penultimate[0]
                velocity_y = last[1] - penultimate[1]
                endpoint_x = penultimate[0] + (FORECAST_LENGTH*velocity_x)
                if endpoint_x > FRAME_X:
                    endpoint_x = FRAME_X
                elif endpoint_x < 0:
                    endpoint_x = 0
                endpoint_y = penultimate[1] + (FORECAST_LENGTH*velocity_y)
                if endpoint_y > FRAME_Y:
                    endpoint_y = FRAME_Y
                elif endpoint_y < 0:
                    endpoint_y = 0
                end = np.array([endpoint_x,endpoint_y]).astype(np.uint32)
                start = np.array([penultimate[0],penultimate[1]]).astype(np.uint32)
                if track_id == 2:
                    print(f"ID:{track_id} \n\tstart: {start}\n\tend:{end}")
                    print(f"\t end x{endpoint_x}, end y{endpoint_y}")
                cv2.line(annotated_frame, start, end, FORECAST_COLOR, FORECAST_SIZE)
                # for f in range(FORECAST_LENGTH):
                #     offset = 1+f
                #     point = (last[0] + offset*velocity_x, last[1] + offset*velocity_y)
                #     size = 5 + (3*offset)   #further out the forcast, the greater the uncertainty
                #     cv2.polylines(annotated_frame, point, isClosed=False, color=FORECAST_COLOR, thickness=size)


        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()