import os
import csv


def create_wav_metadata_csv(folder_path, output_csv):
    metadata = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):

            parts = file_name.replace(".wav", "").split("_")
            if len(parts) == 8:
                metadata.append({
                    "file_path": os.path.join(folder_path, file_name),
                    "bpm": parts[0].replace("bpm", ""),
                    "genre": parts[1],
                    "subgenre": parts[2],
                    "category": parts[3],
                    "loop_length": parts[4],
                    "unique_id": parts[5],
                    "timestamp": parts[6],
                })
            else:
                print(f"Skipping file with unexpected format: {file_name}")

    with open(output_csv, mode="w", newline="") as csv_file:
        fieldnames = ["file_path", "bpm", "genre", "subgenre", "category",
                      "loop_length", "unique_id", "timestamp"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)

    print(f"Metadata CSV file created: {output_csv}")


create_wav_metadata_csv("data/wrld_smb_drm_8br_id_001_wav",
                        "data/wrld_smb_drm_8br_id_001_wav.csv")
