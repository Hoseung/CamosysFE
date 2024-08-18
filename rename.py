import os

def rename_files_in_directory(directory):
    # Iterate over all the files in the specified directory
    for filename in os.listdir(directory):
        # Check if the file name has a three-digit integer with an extension
        if len(filename.split('.')[0]) == 3 and filename.split('.')[0].isdigit():
            # Extract the file's base name and extension
            base, ext = os.path.splitext(filename)
            # Create the new file name with a four-digit integer
            new_base = base.zfill(4)  # Pad the base with a leading zero if necessary
            new_filename = f"{new_base}{ext}"
            # Generate the full old and new file paths
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            # Rename the file
            os.rename(old_file, new_file)
            print(f"Renamed: {filename} -> {new_filename}")

if __name__ == "__main__":
    # Replace 'your_directory_path' with the path to your directory
    directory_path = './test10'
    rename_files_in_directory(directory_path)

