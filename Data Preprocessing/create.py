import os
import urllib.parse
import pandas as pd

# Directory containing segmented audio files
segments_dir = "output_segments"

# List all .wav files in the directory
wav_files = [f for f in os.listdir(segments_dir) if f.endswith(".wav")]

data_rows = []

for f in wav_files:
    # f is something like: segment_1_%E4%BD%A0%E5%A5%BD.wav
    # Split by '_' to separate segment index from encoded caption
    # Format: segment_{idx}_{encoded_subtitle}.wav
    parts = f.split('_', 2)  # Split into 3 parts: "segment", idx, and encoded+subtitle.wav
    if len(parts) < 3:
        # If the file doesn't follow the expected pattern, skip it or handle differently
        continue

    # parts[0] = "segment"
    # parts[1] = {idx}, e.g. "1"
    # parts[2] = {encoded_subtitle}.wav
    encoded_subtitle_with_ext = parts[2]
    # Remove the ".wav" extension
    encoded_subtitle = encoded_subtitle_with_ext[:-4]

    # Decode the subtitle
    subtitle_text = urllib.parse.unquote(encoded_subtitle)

    # Construct full path
    full_path = os.path.abspath(os.path.join(segments_dir, f))

    # Add to our data rows
    data_rows.append({
        'audio': full_path,
        'caption': subtitle_text
    })

# Create a DataFrame and save as CSV
df = pd.DataFrame(data_rows, columns=['audio', 'caption'])
df.to_csv('audioldm_dataset.csv', index=False, encoding='utf-8')
print("audioldm_dataset.csv created successfully!")

