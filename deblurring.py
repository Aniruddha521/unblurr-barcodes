import os
import shutil
import subprocess
import warnings

warnings.filterwarnings("ignore")

CLICKED_DATASET = "clicked_dataset"
OUTPUT_FOLDER = "deblurred_stack_images"
WEIGHTS_PATH = "models/fpn_inception.h5"

if not os.path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)

sharp_images = os.listdir(CLICKED_DATASET)
stack_folders = []
for contains in sharp_images:
    src_path = os.path.join(CLICKED_DATASET, contains)
    if contains.lower().endswith(('.png', '.jpg', '.jpeg')):
        dest_path = os.path.join(OUTPUT_FOLDER, contains)
        shutil.copy(src_path, dest_path)
    else:
        stack_folders.append(src_path)

for folder in stack_folders:
    input_path = folder
    output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(folder))
    os.makedirs(output_path, exist_ok=True)

    subprocess.run([
        "python", "DeblurGANv2/prediction.py", 
        "--input_dir", input_path, 
        "--output_dir", output_path, 
        "--weights", WEIGHTS_PATH
    ])
    print(f"Done for folder! {folder}")

print("Deblurring completed for all folders!")

