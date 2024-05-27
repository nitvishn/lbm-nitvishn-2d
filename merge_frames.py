import os
import subprocess

def merge_frames(frames_directory, output_filename, frame_rate):
    # Check if the frames directory exists
    if not os.path.isdir(frames_directory):
        print(f"Directory {frames_directory} does not exist.")
        exit(1)

    # Construct the FFmpeg command
    ffmpeg_command = [
        "ffmpeg",
        "-framerate", str(frame_rate),
        "-i", os.path.join(frames_directory, "frame_%d.png"),
        "-c:v", "libx264",
        "-b:v", "100M",  # Set the bitrate to 100 Mbps
        "-pix_fmt", "yuv420p",
        output_filename
    ]

    # Execute the FFmpeg command
    try:
        subprocess.run(ffmpeg_command, check=True)
        print(f"Video saved as {output_filename}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed with error: {e}")

