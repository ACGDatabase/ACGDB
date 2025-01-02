import subprocess
import sys
import os
from tqdm import tqdm

def read_links(file_path):
    """Read and return all non-empty lines from the links file."""
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def write_links(file_path, links):
    """Write the remaining links back to the links file."""
    with open(file_path, 'w') as f:
        for link in links:
            f.write(f"{link}\n")

def import_link(link, destination):
    """
    Import a single MEGA link to mega drive using mega-import.
    Returns True if import is successful, False otherwise.
    """
    try:
        # Call mega-import with the link and destination directory
        result = subprocess.run(
            ['mega-import', link, destination],
            text=True
        )
        if result.returncode == 0:
            return True
        else:
            print(f"Error importing {link}:\n{result.stderr}")
            return False
    except Exception as e:
        print(f"Exception occurred while importing {link}: {e}")
        return False

def main():
    links_file = 'links.txt'  # Path to your links.txt
    destination_dir = '/asmr'  # Destination directory

    # Check if links file exists
    if not os.path.isfile(links_file):
        print(f"Links file '{links_file}' does not exist.")
        sys.exit(1)
    # Read all links
    links = read_links(links_file)

    if not links:
        print("No links to process.")
        sys.exit(0)

    # Logins to MEGA
    subprocess.run(['mega-login', 'EMAIL', 'PASSWORD'], text=True)

    # Initialize tqdm progress bar
    with tqdm(total=len(links), desc="Downloading files") as pbar:
        for index, link in enumerate(links[:]):  # Iterate over a copy of the list
            print(f"Starting download: {link}")
            success = import_link(link, destination_dir)
            if success:
                # Remove the imported link from the list
                links.remove(link)
                # Write the updated list back to the file
                write_links(links_file, links)
                pbar.update(1)
                print(f"Successfully imported and removed: {link}")
            else:
                print(f"Aborting due to download failure: {link}")
                sys.exit(1)  # Exit the script with error

    print("All files imported successfully.")

if __name__ == "__main__":
    main()
