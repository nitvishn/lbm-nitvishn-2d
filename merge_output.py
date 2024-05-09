import cv2
import os
import re 

date = 'may_8'
# Directory containing frames
input_dir = 'output/frames/'
# Output video file
output_file = f'{date}/merged.mov'
# Frame rate
fps = 24

# Get the list of all frames, of the form "ink_framenum_time_float.png"
frames = os.listdir(input_dir)
# Sort the frames by frame number
frames.sort(key=lambda x: int(re.search(r'\d+', x).group()))
# Get the first frame to get the dimensions
frame = cv2.imread(input_dir + frames[0])
height, width, _ = frame.shape
# Initialize the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Create the video writer
out = cv2.VideoWriter(output_file, cv2.CAP_FFMPEG, fourcc, fps, (width, height))

# Write each frame to the video
for frame in frames:
    img = cv2.imread(input_dir + frame)
    out.write(img)
    
# Release the video writer

out.release()

print(f"Video saved to {output_file}")