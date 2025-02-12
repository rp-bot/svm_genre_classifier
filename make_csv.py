import os
import numpy as np
import pandas as pd


patch_banks_folders = [
    # ("data/PatchBanks/edm_hse_id_001-004_wav", "house", 0),
    # ("data/PatchBanks/edm_tr8_drm_id_001-0013_wav", "tr 808", 1),
    ("data/edm_tr9_drm_id_001", "tr-909", 0),
    ("data/hh_lfbb_lps_mid_001-009", "lofi-boom-bap", 1),
    ("data/pop_rok_drm_id_001_wav", "pop rock", 2),
    # ("data/rtro_drm_id_001", "retro", 5),
    # ("data/wrld_lp_id_001", "latin percussion", 6),
    ("data/wrld_smb_drm_8br_id_001_wav", "samba", 3),
]


def create_wav_metadata_csv(folder_path, output_csv):
    metadata = []

    for file_name in os.listdir(folder_path):
        parts = file_name.replace(".wav", "").split("_")
        if len(parts) >= 4:
            metadata.append(
                {
                    "file_path": os.path.join(folder_path, file_name),
                    "bpm": parts[0].replace("bpm", ""),
                    "genre": parts[1],
                    "subgenre": parts[2],
                    "category": parts[3],
                    # "loop_length": parts[4],
                    # "unique_id": parts[5],
                    # "timestamp": parts[6],
                }
            )
        else:
            print(f"Skipping file with unexpected format: {file_name}")

    metadata_df = pd.DataFrame(metadata)
    indices = np.linspace(0, len(metadata_df) - 1, 1100, dtype=int)
    metadata_df = metadata_df.iloc[indices]
    metadata_df.to_csv(output_csv, index=False)

    print(f"Metadata CSV file created: {output_csv}")


def derive_annotations():

    metadata = []

    for folder, class_label, class_id in patch_banks_folders:
        files = sorted(os.listdir(folder))

        print(class_label)
        print(len(files))
        print("")
        # num_files = len(files)

        # indices = np.linspace(0, num_files - 1, 1100, dtype=int)
        # selected_files = files[indices]

        for file_name in files:
            parts = file_name.replace(".wav", "").split("_")
            metadata.append(
                {
                    "file_path": os.path.join(folder, file_name),
                    "class": class_label,
                    "class_id": class_id,
                }
            )

    return metadata


# create_wav_metadata_csv("data/wrld_smb_drm_8br_id_001_wav",
#                         "data/wrld_smb_drm_8br_id_001_wav.csv")


# create_wav_metadata_csv("data/pop_rok_drm_id_001_wav",
#                         "data/pop_rok_drm_id_001_wav.csv")
if __name__ == "__main__":

    for folder, _, _ in patch_banks_folders:
        create_wav_metadata_csv(folder, f"{folder}.csv")
    # dataset_dir = "data"
    # output_csv_file = "data/annotations.csv"
    # metadata = derive_annotations()
    # metadata_df = pd.DataFrame(metadata)
    # metadata_df.to_csv(output_csv_file)
