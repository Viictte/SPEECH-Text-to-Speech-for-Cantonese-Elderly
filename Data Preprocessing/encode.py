import os
import csv
import shutil
import re
import hashlib

def collect_all_segments(
    seg_rule1_folder: str,
    seg_rule2_folder: str,
    seg_rule1_augmented_folder: str,
    final_folder: str
):
    """
    1. Reads 'segments.csv' from seg_rule1, seg_rule2,
       and 'aug_segments.csv' from seg_rule1_augmented.
    2. Copies all .wav files to `final_folder`.
    3. Renames files to ASCII-friendly names (avoiding Chinese chars, etc.).
    4. Writes a single CSV: [new_path, caption].
       - If there's a collision in `final_folder`, we rename again to avoid overwriting.
    """

    os.makedirs(final_folder, exist_ok=True)

    # Build the final CSV path
    final_csv = os.path.join(final_folder, "combined_segments.csv")

    # We'll store (old_path -> caption) from each CSV
    # Then copy them to final_folder, creating a new ASCII-safe name.
    entries = []

    # Helper to load a CSV with columns [path, caption]
    def load_csv(csv_path: str):
        result = []
        if not os.path.isfile(csv_path):
            return result
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return result
            for row in reader:
                if len(row) < 2:
                    continue
                audio_path, caption = row
                result.append((audio_path, caption))
        return result

    # 1) Load seg_rule1/segments.csv
    sr1_csv = os.path.join(seg_rule1_folder, "segments.csv")
    seg_rule1_entries = load_csv(sr1_csv)

    # 2) Load seg_rule2/segments.csv
    sr2_csv = os.path.join(seg_rule2_folder, "segments.csv")
    seg_rule2_entries = load_csv(sr2_csv)

    # 3) Load seg_rule1_augmented/aug_segments.csv
    sr1_aug_csv = os.path.join(seg_rule1_augmented_folder, "aug_segments.csv")
    sr1_aug_entries = load_csv(sr1_aug_csv)

    # Combine them all
    entries.extend(seg_rule1_entries)
    entries.extend(seg_rule2_entries)
    entries.extend(sr1_aug_entries)

    print(f"Loaded {len(entries)} total entries from seg_rule1, seg_rule2, and seg_rule1_augmented.")

    # 4) Copy & rename all .wav files to final_folder; write final CSV
    with open(final_csv, mode="w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["path", "caption"])

        for old_path, caption in entries:
            if not os.path.isfile(old_path):
                print(f"Warning: audio file not found: {old_path}")
                continue

            original_name = os.path.basename(old_path)
            # Convert to an ASCII-safe filename
            ascii_name = safe_filename(original_name)
            new_fullpath = os.path.join(final_folder, ascii_name)
            # Resolve collisions
            new_fullpath = resolve_collision(new_fullpath)

            # Copy the file
            shutil.copyfile(old_path, new_fullpath)

            # Write row => [new_fullpath, caption]
            writer.writerow([new_fullpath, caption])

    print(f"\nDone! Copied and renamed files to: {final_folder}")
    print(f"Final CSV => {final_csv}")

def safe_filename(filename: str) -> str:
    """
    Generate an ASCII-friendly filename from 'filename'.
    1) Removes or replaces non-ASCII chars.
    2) Appends a short hash for uniqueness if needed.
    3) Preserves the file extension.

    Example:
       "Speaker3_中文.wav" => "Speaker3__686564a1.wav"
    """
    base, ext = os.path.splitext(filename)
    # 1) Keep only alphanumeric + some delimiters, replace others with '_'
    base_ascii = re.sub(r"[^a-zA-Z0-9._-]+", "_", base)
    if not base_ascii:
        base_ascii = "file"

    # 2) For uniqueness, append a short hash of the original base
    #    This ensures each new name is unique even if base_ascii is short
    h = hashlib.md5(base.encode("utf-8")).hexdigest()[:8]
    base_ascii = f"{base_ascii}_{h}"

    return f"{base_ascii}{ext}"

def resolve_collision(target_path: str) -> str:
    """
    If 'target_path' already exists, we insert an increment before the extension.
    e.g. "/folder/file.wav" => "/folder/file_1.wav", "/folder/file_2.wav", etc.
    """
    if not os.path.exists(target_path):
        return target_path

    base, ext = os.path.splitext(target_path)
    i = 1
    while True:
        new_path = f"{base}_{i}{ext}"
        if not os.path.exists(new_path):
            return new_path
        i += 1


if __name__ == "__main__":
    # Example usage: adjust these paths to your environment

    seg_rule1_folder = "/home/victor/AudioLDM-training-finetuning/data/All_Cleaned/segmentation_results/seg_rule1"
    seg_rule2_folder = "/home/victor/AudioLDM-training-finetuning/data/All_Cleaned/segmentation_results/seg_rule2"
    seg_rule1_augmented_folder = os.path.join(seg_rule1_folder, "seg_rule1_augmented")

    final_folder = "/home/victor/AudioLDM-training-finetuning/data/clean_augmented"

    collect_all_segments(
        seg_rule1_folder=seg_rule1_folder,
        seg_rule2_folder=seg_rule2_folder,
        seg_rule1_augmented_folder=seg_rule1_augmented_folder,
        final_folder=final_folder
    )

