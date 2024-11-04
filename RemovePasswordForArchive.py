import os
import subprocess
import re
import shutil
from typing import List, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Optional: Uncomment the following lines to use logging instead of print statements
# import logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def is_password_protected(filepath: str, dummy_password: str = "worilepython") -> bool:
    """
    Check if the archive is password protected by attempting to list its contents
    using a dummy password.

    Args:
        filepath (str): Path to the archive file.
        dummy_password (str): A dummy password to test protection.

    Returns:
        bool: True if the archive is password protected, False otherwise.
    """
    try:
        subprocess.run(
            ["7z", "l", f"-p{dummy_password}", filepath],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return False
    except subprocess.CalledProcessError:
        return True


def get_codec() -> str:
    """
    Get the appropriate codec based on the operating system.

    Returns:
        str: 'gbk' for Windows, 'utf-8' otherwise.
    """
    return "gbk" if os.name == "nt" else "utf-8"


def handle_remove_readonly(func, path, exc):
    """
    Handle "access denied" errors when deleting files/folders by changing permissions.

    Args:
        func: The function that raised the exception.
        path (str): Path to the file/folder.
        exc: The exception that was raised.
    """
    os.chmod(path, 0o777)
    func(path)


def extract_outpaths(output: bytes, codec: str) -> List[str]:
    """
    Extract the 'Path' values from the archive listing output.

    Args:
        output (bytes): Output from the subprocess command.
        codec (str): The codec to decode the output.

    Returns:
        List[str]: List of paths extracted from the archive.
    """
    outpaths = []
    decoded_output = output.decode(codec).splitlines()
    for line in decoded_output:
        if line.startswith("Path = "):
            path = line.split(" = ", 1)[1]
            if codec == "gbk":
                if '\\' not in path:
                    outpaths.append(path)
            else:
                if '/' not in path:
                    outpaths.append(path)
    return outpaths


def execute_subprocess(command: List[str], **kwargs) -> subprocess.CompletedProcess:
    """
    Execute a subprocess command with error handling.

    Args:
        command (List[str]): The command to execute.
        **kwargs: Additional arguments for subprocess.run.

    Returns:
        subprocess.CompletedProcess: The result of the subprocess execution.

    Raises:
        subprocess.CalledProcessError: If the subprocess fails.
    """
    return subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)


def decompress_archive(
    decompress_cmd: List[str],
    extract_method: Callable[[List[str]], subprocess.CompletedProcess]
):
    """
    Decompress the archive using the provided extraction method.

    Args:
        decompress_cmd (List[str]): The command to decompress the archive.
        extract_method (Callable): The method to execute the decompress command.

    Raises:
        subprocess.CalledProcessError: If decompression fails.
    """
    extract_method(decompress_cmd)


def recompress_archive(
    archive_path: str,
    outpaths: List[str],
    recompress_cmd: List[str]
):
    """
    Recompress the extracted files into a RAR archive.

    Args:
        archive_path (str): Path to the new RAR archive.
        outpaths (List[str]): List of file/folder paths to include in the archive.
        recompress_cmd (List[str]): The command to recompress the archive.

    Raises:
        subprocess.CalledProcessError: If recompression fails.
    """
    # Convert outpaths to absolute paths
    outpath_abs = [os.path.abspath(path) for path in outpaths]
    print(f"Running command: {' '.join(recompress_cmd)} {archive_path} {' '.join(outpath_abs)}")
    # logging.info(f"Running command: {' '.join(recompress_cmd)} {archive_path} {' '.join(outpath_abs)}")
    subprocess.run(
        recompress_cmd + [archive_path] + outpath_abs,
        check=True
    )
    print(f"{archive_path} re-compressed successfully!")
    # logging.info(f"{archive_path} re-compressed successfully!")


def clean_extracted_files(outpaths: List[str]):
    """
    Remove extracted files and directories.

    Args:
        outpaths (List[str]): List of file/folder paths to remove.
    """
    for path in outpaths:
        if os.path.isdir(path):
            shutil.rmtree(path, onerror=handle_remove_readonly)
        else:
            os.remove(path)


class ArchiveHandler:
    """
    A class to handle different types of archives, encapsulating their specific behaviors.
    """

    def __init__(self, filepath: str, dirpath: str, filename: str, codec: str):
        self.filepath = filepath
        self.dirpath = dirpath
        self.filename = filename
        self.codec = codec

    def try_passwords(
        self,
        password_list: List[str],
        is_split: bool = False
    ) -> bool:
        """
        Attempt to decrypt the archive using a list of passwords.

        Args:
            password_list (List[str]): List of passwords to try.
            is_split (bool): Indicates if the archive is split.

        Returns:
            bool: True if decryption was successful, False otherwise.
        """
        for password in password_list:
            print(f"Trying password '{password}' for '{self.filepath}'")
            # logging.info(f"Trying password '{password}' for '{self.filepath}'")
            try:
                # Attempt to list the archive contents with the password
                list_cmd = self.get_list_command(password, is_split)
                result = execute_subprocess(list_cmd)
                outpaths = extract_outpaths(result.stdout, self.codec)

                # Decompress the archive
                decompress_cmd = self.get_decompress_command(password, is_split)
                self.decompress_archive(decompress_cmd)

                print(f"Password found for '{self.filepath}': {password}")
                # logging.info(f"Password found for '{self.filepath}': {password}")

                # Recompress without password
                outpaths = [os.path.join(self.dirpath, p) for p in outpaths]
                if self.codec == "gbk":
                    outpaths = [p.rstrip('\r') for p in outpaths]

                self.recompress_archive(outpaths)

                # Clean up extracted files
                clean_extracted_files(outpaths)

                return True
            except subprocess.CalledProcessError:
                print("Wrong password, trying next...")
                # logging.warning("Wrong password, trying next...")
        return False

    def get_list_command(self, password: str, is_split: bool) -> List[str]:
        """
        Get the command to list archive contents.

        Args:
            password (str): The password to use.
            is_split (bool): Indicates if the archive is split.

        Returns:
            List[str]: The command to list archive contents.
        """
        raise NotImplementedError

    def get_decompress_command(self, password: str, is_split: bool) -> List[str]:
        """
        Get the command to decompress the archive.

        Args:
            password (str): The password to use.
            is_split (bool): Indicates if the archive is split.

        Returns:
            List[str]: The command to decompress the archive.
        """
        raise NotImplementedError

    def get_recompress_command(self) -> List[str]:
        """
        Get the command to recompress the archive.

        Returns:
            List[str]: The command to recompress the archive.
        """
        return ["rar", "a", "-ep1"]

    def decompress_archive(self, decompress_cmd: List[str]):
        """
        Decompress the archive using the provided command.

        Args:
            decompress_cmd (List[str]): The command to decompress the archive.

        Raises:
            subprocess.CalledProcessError: If decompression fails.
        """
        execute_subprocess(decompress_cmd)

    def recompress_archive(self, outpaths: List[str]):
        """
        Recompress the extracted files into a RAR archive.

        Args:
            outpaths (List[str]): List of file/folder paths to include in the archive.
        """
        recompress_cmd = self.get_recompress_command()
        recompress_archive(self.filepath, outpaths, recompress_cmd)


class RARHandler(ArchiveHandler):
    """
    Handler for RAR archives.
    """

    def get_list_command(self, password: str, is_split: bool) -> List[str]:
        return ["7z", "l", f"-p{password}", self.filepath]

    def get_decompress_command(self, password: str, is_split: bool) -> List[str]:
        return ["7z", "x", f"-o{self.dirpath}", f"-p{password}", self.filepath]


class SplitRARHandler(ArchiveHandler):
    """
    Handler for split RAR archives.
    """

    def __init__(self, filepath: str, dirpath: str, filename: str, codec: str, split_files: List[str]):
        super().__init__(filepath, dirpath, filename, codec)
        self.split_files = split_files

    def get_list_command(self, password: str, is_split: bool) -> List[str]:
        return ["7z", "l", f"-p{password}", self.filepath]

    def get_decompress_command(self, password: str, is_split: bool) -> List[str]:
        return ["7z", "x", f"-o{self.dirpath}", f"-p{password}", self.filepath]

    def recompress_archive(self, outpaths: List[str]):
        # Remove split files after successful recompression
        for split_file in self.split_files:
            os.remove(split_file)
        super().recompress_archive(outpaths)


class StandardArchiveHandler(ArchiveHandler):
    """
    Handler for standard ZIP and 7z archives.
    """

    def get_list_command(self, password: str, is_split: bool) -> List[str]:
        return ["7z", "l", f"-p{password}", self.filepath]

    def get_decompress_command(self, password: str, is_split: bool) -> List[str]:
        return ["7z", "x", f"-o{self.dirpath}", f"-p{password}", self.filepath]


class SplitStandardArchiveHandler(ArchiveHandler):
    """
    Handler for split ZIP and 7z archives.
    """

    def __init__(self, filepath: str, dirpath: str, filename: str, codec: str, split_files: List[str]):
        super().__init__(filepath, dirpath, filename, codec)
        self.split_files = split_files

    def get_list_command(self, password: str, is_split: bool) -> List[str]:
        return ["7z", "l", f"-p{password}"] + self.split_files

    def get_decompress_command(self, password: str, is_split: bool) -> List[str]:
        return ["7z", "x", f"-o{self.dirpath}", f"-p{password}"] + self.split_files

    def recompress_archive(self, outpaths: List[str]):
        # Remove split files after successful recompression
        for split_file in self.split_files:
            os.remove(split_file)
        super().recompress_archive(outpaths)


def determine_archive_handler(
    file_path: str,
    dirpath: str,
    filename: str,
    codec: str
) -> Tuple[Optional[ArchiveHandler], Optional[List[str]]]:
    """
    Determine the appropriate archive handler based on the file type.

    Args:
        file_path (str): Path to the archive file.
        dirpath (str): Directory path where the file is located.
        filename (str): Name of the file.
        codec (str): Codec used for decoding subprocess output.

    Returns:
        Tuple[Optional[ArchiveHandler], Optional[List[str]]]: The handler instance and list of split files if applicable.
    """
    # Check for split RAR files (e.g., file.part1.rar, file.part2.rar)
    split_rar_match = re.match(r"(.+)\.part(\d+)\.rar$", filename, re.IGNORECASE)
    if split_rar_match:
        base_name = split_rar_match.group(1)
        part_num = int(split_rar_match.group(2))
        if part_num == 1:
            # Gather all split RAR parts
            split_files = sorted([
                f for f in os.listdir(dirpath)
                if re.match(rf"{re.escape(base_name)}\.part\d+\.rar$", f, re.IGNORECASE)
            ])
            split_paths = [os.path.join(dirpath, f) for f in split_files]
            if split_files:
                return RARHandler(file_path, dirpath, filename, codec), split_paths

    # Check for split ZIP or 7z files (e.g., file.zip.001, file.7z.001)
    split_standard_match = re.match(r"(.+)\.(zip|7z)\.(\d+)$", filename, re.IGNORECASE)
    if split_standard_match:
        base_name, ext, split_num = split_standard_match.groups()
        split_num = int(split_num)
        if split_num == 1:
            # Gather all split ZIP/7z parts
            split_files = sorted([
                f for f in os.listdir(dirpath)
                if re.match(rf"{re.escape(base_name)}\.{re.escape(ext)}\.\d+$", f, re.IGNORECASE)
            ])
            split_paths = [os.path.join(dirpath, f) for f in split_files]
            if split_files:
                return StandardArchiveHandler(file_path, dirpath, filename, codec), split_paths

    # Handle standard RAR files
    if filename.lower().endswith(".rar"):
        return RARHandler(file_path, dirpath, filename, codec), None

    # Handle standard ZIP and 7z files
    if filename.lower().endswith(('.zip', '.7z')):
        return StandardArchiveHandler(file_path, dirpath, filename, codec), None

    return None, None


def remove_password_from_archive(
    handler: ArchiveHandler,
    split_files: Optional[List[str]],
    password_list: List[str]
) -> bool:
    """
    Attempt to remove the password from an archive using the provided handler.

    Args:
        handler (ArchiveHandler): The archive handler instance.
        split_files (Optional[List[str]]): List of split file paths if applicable.
        password_list (List[str]): List of passwords to try.

    Returns:
        bool: True if the password was successfully removed, False otherwise.
    """
    if isinstance(handler, RARHandler):
        return handler.try_passwords(password_list)
    elif isinstance(handler, SplitRARHandler):
        return handler.try_passwords(password_list, is_split=True)
    elif isinstance(handler, StandardArchiveHandler):
        return handler.try_passwords(password_list)
    elif isinstance(handler, SplitStandardArchiveHandler):
        return handler.try_passwords(password_list, is_split=True)
    else:
        return False


def process_archive_file(
    file_info: Tuple[str, str, str],
    codec: str,
    password_list: List[str],
    processed_files: set,
    lock: threading.Lock
):
    """
    Process a single archive file or a set of split archive files.

    Args:
        file_info (Tuple[str, str, str]): A tuple containing (dirpath, filename, file_path).
        codec (str): Codec used for decoding subprocess output.
        password_list (List[str]): List of passwords to try.
        processed_files (set): A thread-safe set to track already processed files.
        lock (threading.Lock): A lock to synchronize access to the processed_files set.
    """
    dirpath, filename, file_path = file_info

    with lock:
        if file_path in processed_files:
            return  # Already processed as part of a split archive

    handler, split_files = determine_archive_handler(file_path, dirpath, filename, codec)
    if handler:
        if split_files:
            # Mark all split files as processed
            with lock:
                for split_file in split_files:
                    processed_files.add(split_file)
            # Use appropriate handler for split archives
            if isinstance(handler, RARHandler):
                split_handler = SplitRARHandler(file_path, dirpath, filename, codec, split_files)
            else:
                split_handler = SplitStandardArchiveHandler(file_path, dirpath, filename, codec, split_files)
            success = remove_password_from_archive(split_handler, split_files, password_list)
        else:
            success = remove_password_from_archive(handler, None, password_list)

        if success:
            print(f"Successfully removed password from '{file_path}'.")
            # logging.info(f"Successfully removed password from '{file_path}'.")
        else:
            print(f"Failed to remove password from '{file_path}'.")
            # logging.warning(f"Failed to remove password from '{file_path}'.")
    else:
        print(f"Unsupported file type or already processed: '{file_path}'.")
        # logging.info(f"Unsupported file type or already processed: '{file_path}'.")


def remove_rar_password(directory: str, password_list: List[str], max_workers: int = 4):
    """
    Traverse the directory and attempt to remove passwords from RAR, ZIP, and 7z archives.

    Args:
        directory (str): The root directory to start processing.
        password_list (List[str]): List of passwords to try.
        max_workers (int): The maximum number of threads to use for parallel processing.
    """
    codec = get_codec()
    processed_files = set()
    lock = threading.Lock()

    # Collect all archive files first to avoid redundant os.walk during parallel processing
    archive_files = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            archive_files.append((dirpath, filename, file_path))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all archive files to the executor
        futures = [
            executor.submit(process_archive_file, file_info, codec, password_list, processed_files, lock)
            for file_info in archive_files
        ]

        # Optionally, monitor the progress
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing archive: {e}")
                # logging.error(f"Error processing archive: {e}")


if __name__ == "__main__":
    PASSWORD_LIST = [
        "免费分享倒卖死妈",
        "xlxb1001",
        "lawliet@south2022",
        "木偶",
        "yhsxsx1月",
        "yhsxsx2月",
        "yhsxsx3月",
        "yhsxsx4月",
        "yhsxsx5月",
        "yhsxsx6月",
        "yhsxsx7月",
        "yhsxsx8月",
        "yhsxsx9月",
        "yhsxsx10月",
        "yhsxsx11月",
        "yhsxsx12月",
        "yhsxsx"
    ]
    TARGET_DIRECTORY = "D:\\Downloads\\Galgames"
    MAX_WORKERS = 4  # Adjust based on your CPU and I/O capabilities

    remove_rar_password(TARGET_DIRECTORY, PASSWORD_LIST, max_workers=MAX_WORKERS)
