# Bytesort Tracking with Ultralytics
This repo was created to see if tracking using Ultralytics and Yolov8 works when not connected to wifi
(it does). 
## To run
Add an an mp4 file called 'vid.mp4' to the repo.

Add custom .pt file (best_yolov8_base.pt) or update ```model = YOLO('yolov8n.pt')``` to run using generic pretrained Yolov8 model.

 Note: ```results = model.track(frame, persist=True, tracker="bytetrack.yaml, verbose=False")``` to stop printing to console 
 

# The two .py files
bytesort.py: just runs the tracking algo against the custom dataset

plot_over_time.py: draws a line to show movement of objects over time