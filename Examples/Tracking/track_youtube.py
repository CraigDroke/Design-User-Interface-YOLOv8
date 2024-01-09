## Runs but can't be shown on lambda, if run on local machine a new window is opened to show the tracking
## Is effective, but has no output. Maybe use for interface?

from ultralytics import YOLO

# Load an official or custom model
model = YOLO('yolov8n.pt')  # Load an official Detect model
model = YOLO('yolov8n-seg.pt')  # Load an official Segment model
model = YOLO('yolov8n-pose.pt')  # Load an official Pose model

# Perform tracking with the model
## LNwODJXcvt4 is 6 mins, it runs at real time so the full length
# results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)  # Tracking with default tracker
# results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")  # Tracking with ByteTrack tracker, shows in new window
# results = model.track(source="https://youtu.be/LNwODJXcvt4", show=False, tracker="bytetrack.yaml")  # Tracking with ByteTrack tracker, does not show
results = model.track(source="https://www.youtube.com/watch?v=BZP1rYjoBgI&ab_channel=UndoTube", show=True, tracker="bytetrack.yaml")  # Shorter video, 30s
print(results)