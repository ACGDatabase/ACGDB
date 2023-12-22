import os
import shutil
import re
import datetime
source_folder = "/path/to/files/waiting/to be move"
destination_folder = "/path/to/folders/where/your/sub folder are"
ignore_folders = ()
known_existing = ()
# Get list of files in source folder
files = os.listdir(source_folder)
#abstraction moving
def my_move(file, source_folder, folder_path):
    #detect whether the folder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    try:
        shutil.move(os.path.join(source_folder, file), folder_path)
    except shutil.Error as e:
        print(f"Error occurred while moving {file} to {folder_path}: {e}")
        print("Here are two files' info for you to choose:")
        file1 = os.path.join(source_folder, file)
        file2 = os.path.join(folder_path, file)
        print(f"File 1 (Waiting to move): {file1}")
        print(f"Size: {os.path.getsize(file1)}")
        print(f"Last modified: {datetime.datetime.fromtimestamp(os.path.getmtime(file1))}")
        print(f"File 2 (Already in dest): {file2}")
        print(f"Size: {os.path.getsize(file2)}")
        print(f"Last modified: {datetime.datetime.fromtimestamp(os.path.getmtime(file2))}")
        choice = input("Please enter the file number to keep, or type 3 to enter a new name for new file:")
        if choice == "1":
            os.remove(file2)
            shutil.move(file1, folder_path) 
        elif choice == "2":
            return
        elif choice == "3":
            new_name = input("Please enter the new name:")
            shutil.move(file1, os.path.join(folder_path, new_name))
for file in files:
    # cut first 8 characters
    proc_file = file[8:]
    club_name = re.search(r"\[(.*?)\]", proc_file).group(1)
    print("Found club name:", club_name," in file:", file)
    # Search for matching folder in destination folder (excluding source folder)
    matching_folders = []
    name = []
    for folders in os.listdir(destination_folder):
        #if is a file, skip
        if os.path.isfile(os.path.join(destination_folder, folders)):
            continue
        temp_root = os.path.join(destination_folder, folders)
        print("Checking folder:", temp_root, "for", club_name, "in", file, "...")
        if temp_root == source_folder:
            continue
        if any(temp_root.startswith(x) for x in ignore_folders):
            continue
        for root, dirs, files in os.walk(temp_root):
            record_root = root
            for dir in dirs:
                if club_name.lower() in dir.lower():
                    matching_folders.append(os.path.join(root, dir))
                    name.append(dir)
            for file_loop in files:
                proc_ori_file = file_loop[8:]
                match = re.search(r"\[(.*?)\]", proc_ori_file)
                if match:
                    ori_club_name = match.group(1)
                    if club_name.lower() == ori_club_name.lower():
                        if record_root not in matching_folders:
                            #append folder
                            matching_folders.append(record_root)
                            #add file to name
                            name.append(file_loop)
    # Move file to matching folder
    if len(matching_folders) == 0:
        print(f"No matching folder found for {file}")
        # folder_path = input("Please enter the folder path:")
        # #if nothing is entered, skip
        # if folder_path == "":
        #     continue
        
        #4 choices:
        #1. create new folder with club name
        #2. create new folder with custom name
        #3. choose existing folder
        #4. skip
        #print instructions
        print("Search with Google: https://www.google.com/search?q="+club_name.replace(" ","+"))
        print("1. Create new folder \"",os.path.join(destination_folder,club_name),"\"")
        print("2. Create new folder with custom name")
        print("3. Choose existing folder")
        print("4. Skip")
        if "steam" in file.lower():
            print("5. Guessed from steam, use:"+known_existing[0])
        choice = int(input("Please enter the folder number:"))
        if choice == 1:
            folder_path = os.path.join(destination_folder,club_name)
            os.mkdir(folder_path)
        elif choice == 2:
            folder_path = input("Please enter the folder path:")
            os.mkdir(folder_path)
        elif choice == 3:
            if len(known_existing) == 0:
                print("No known existing folders.")
                folder_path = input("Please enter the folder path:")
            else:
                print("Choose an known existing folders, or manually type it in:")
                for i in range(len(known_existing)):
                    print(f"{i+1}. {known_existing[i]}")
                choice = input("Please enter the folder number / other paths:")
                #if choice is int
                if choice.isdigit():
                    folder_path = known_existing[int(choice)-1]
                else:
                    folder_path = choice
        elif choice == 4:
            continue
        elif choice == 5:
            folder_path = known_existing[0]
        else:
            raise ValueError("Invalid choice")
        my_move(file, source_folder, folder_path)
    elif len(matching_folders) == 1:
        folder_path = matching_folders[0]
        print(f"Moving {file} to {folder_path} because {name[0]} found")
        my_move(file, source_folder, folder_path)
    else:
        print(f"Multiple matching folders found for {file}:")
        for i, folder in enumerate(matching_folders):
            print(f"{i+1}. {folder} because {name[i]} found")
        choice = int(input("Please enter the folder number:"))
        if choice < 1 or choice > len(matching_folders):
            print("Invalid choice")
            choice = int(input("Please enter the folder number:"))
            continue
        folder_path = matching_folders[choice-1]
        print(f"Moving {file} to {folder_path}")
        my_move(file, source_folder, folder_path)
