# Bytesort Tracking with Ultralytics
This repo was created to see if tracking using Ultralytics and Yolov8 works when not connected to wifi
(it does). 

Add an an mp4 file called 'vid.mp4' to the repo to run.

 Note: ```results = model.track(frame, persist=True, tracker="bytetrack.yaml, verbose=False")``` to stop printing to console 
 

# The two .py files
bytesort.py: just runs the tracking algo against the custom dataset

plot_over_time.py: draws a line to show movement of objects over time

# Next steps
Run the tracker using custom trained yolov8 model