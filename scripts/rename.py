import os

def rename_files(folder_path):
    all_files = []
    for foldername, subfolders, filenames in os.walk(folder_path):
        for filename in filenames:
            all_files.append(os.path.join(foldername, filename))

    for subdirectory_path in set(os.path.dirname(file) for file in all_files):
        subdirectory_files = [file for file in all_files if os.path.dirname(file) == subdirectory_path]
        subdirectory_files.sort()

        for i, old_filepath in enumerate(subdirectory_files, start=1):
            subdirectory_name = os.path.basename(subdirectory_path)
            new_filename = ""
            if subdirectory_name == "D":
                new_filename = f'{subdirectory_name}_{(i):03d}.jpg'
            elif subdirectory_name == "H":
                new_filename = f'{subdirectory_name}_{(i):03d}.jpg'
            else:
                new_filename = f'{subdirectory_name}_{(i):03d}.jpg'

            new_filepath = os.path.join(os.path.dirname(old_filepath), new_filename)

            os.rename(old_filepath, new_filepath)
            print(f'Renamed: {old_filepath} -> {new_filepath}')

if __name__ == "__main__":
    folder_path = '../data/fix2'

    rename_files(folder_path)
