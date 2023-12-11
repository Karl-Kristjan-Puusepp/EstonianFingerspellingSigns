import os
import random
import shutil

root_folder_path = "oneHandedGesturesCroppedReduced"
def reduce_folder_size(folder_path, target_size=50):
    try:
        # Get the list of files in the folder
        files = os.listdir(folder_path)

        # Check if the folder has more than the target size
        if len(files) > target_size:
            # Calculate the number of files to delete
            files_to_delete = len(files) - target_size

            # Randomly select files to delete
            files_to_delete = random.sample(files, files_to_delete)

            # Delete the selected files
            for file_name in files_to_delete:
                file_path = os.path.join(folder_path, file_name)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

    except Exception as e:
        print(f"Error reducing folder size: {e}")

def reduce_subfolders_size(root_folder, target_size=50):
    try:
        # Iterate over subfolders in the root folder
        for subdir, dirs, files in os.walk(root_folder):
            reduce_folder_size(subdir, target_size)

    except Exception as e:
        print(f"Error reducing subfolders size: {e}")
reduce_subfolders_size(root_folder_path)

