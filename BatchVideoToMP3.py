import os
import subprocess
import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
import json
import re
import mimetypes # Needed for guessing image MIME type
from typing import List, Tuple, Optional, Dict, Any, Iterator, Union

# Third-party imports
try:
    from mutagen.mp3 import MP3
    from mutagen.id3 import ID3, TPE1, TIT2, TDRC, COMM, APIC, Encoding, ID3NoHeaderError # Added APIC
except ImportError:
    print("Error: 'mutagen' library not found. Please install it: pip install mutagen", file=sys.stderr)
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Error: 'tqdm' library not found. Please install it: pip install tqdm", file=sys.stderr)
    # Provide a dummy tqdm class if not found, so the script can still run without progress bars
    class tqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable
            self.total = kwargs.get('total', None)
            if self.iterable is not None and self.total is None:
                try:
                    self.total = len(self.iterable)
                except (TypeError, AttributeError):
                    self.total = None

        def __iter__(self):
            return iter(self.iterable)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, n=1):
            pass

        def set_description(self, desc):
            pass

        def set_postfix_str(self, s):
            pass

        @staticmethod
        def write(s, file=sys.stdout):
            print(s, file=file)

# --- Configuration ---
SUPPORTED_EXTENSIONS: Tuple[str, ...] = ('.mp4', '.ts', '.mkv', '.avi', '.mov', '.wmv', '.flv')
FILENAME_METADATA_REGEX: re.Pattern = re.compile(
    r"^\[([^\]]+)\]\[(\d{4}-\d{2}-\d{2})\]\[(.+?)\](?:\[(\d{4}-\d{2}-\d{2})\])?$"
)
# Standard ID3 picture type for Cover (front)
ID3_PIC_TYPE_COVER_FRONT = 3
TEMP_SUFFIX = ".tmp" # Define the temporary suffix
# Specific FFmpeg error messages indicating no audio stream
NO_AUDIO_STREAM_ERRORS = (
    "Stream map '0:a:0' matches no streams", # Common when using -map 0:a:0?
    "Output file #0 does not contain any stream", # Can happen if -map isn't used but still no audio
    "does not contain any stream" # More general check
)
# List of folder names (case-insensitive) to completely ignore during scanning
IGNORE_FOLDERS: List[str] = [
    "[MissWarmJ]",
    # Add more folder names here as needed, e.g., "backup", "[Old Stuff]"
]


# --- Helper Types ---
FFmpegResult = Tuple[bool, Optional[str]] # (success, error_message)
# Added 'skipped_no_audio' to status types
ConversionResult = Tuple[str, Path, Any, str, str] # (status, input_path, output_or_error, action, metadata_status)

# --- Core Logic Functions ---

def find_executable(name: str) -> Optional[str]:
    """Checks if an executable is accessible in the system PATH."""
    path = shutil.which(name)
    if path:
        print(f"{name} found in PATH: {path}")
        return path
    else:
        tqdm.write(f"ERROR: {name} not found in system PATH.", file=sys.stderr)
        tqdm.write(f"Please install FFmpeg (which includes {name}) and ensure it's added to your PATH.", file=sys.stderr)
        tqdm.write("Download from: https://ffmpeg.org/download.html", file=sys.stderr)
        return None

def get_audio_codec(video_path: Path, ffprobe_path: str) -> Optional[str]:
    """Uses ffprobe to determine the codec of the first audio stream."""
    command: List[str] = [
        ffprobe_path,
        '-v', 'error', '-select_streams', 'a:0',
        '-show_entries', 'stream=codec_name', '-of', 'json', str(video_path)
    ]
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8', errors='replace')
        data = json.loads(process.stdout)
        if data and 'streams' in data and data['streams']:
            return data['streams'][0].get('codec_name')
        # If ffprobe runs successfully but finds no streams, return specific indicator
        return "no_audio_stream_found_by_ffprobe"
    except FileNotFoundError:
         tqdm.write(f"\nERROR: ffprobe executable not found at '{ffprobe_path}'. Cannot check audio codec.", file=sys.stderr)
         return None
    except subprocess.CalledProcessError as e:
        # Check if ffprobe failed because there were no streams selected
        if any(err_msg in e.stderr for err_msg in NO_AUDIO_STREAM_ERRORS) or "could not find codec parameters" in e.stderr:
             # This indicates successful run but no audio stream
             return "no_audio_stream_found_by_ffprobe"
        tqdm.write(f"Warning: Could not get audio codec for {video_path.name} (will attempt conversion). FFprobe Error: {e.stderr}", file=sys.stderr)
        return None # Indicate error occurred, distinct from no audio found
    except (json.JSONDecodeError, Exception) as e:
        tqdm.write(f"Warning: Could not get audio codec for {video_path.name} (will attempt conversion). Error: {e}", file=sys.stderr)
        return None

def extract_metadata_from_filename(filename_stem: str) -> Optional[Dict[str, str]]:
    """Parses the filename stem using regex to extract metadata."""
    match = FILENAME_METADATA_REGEX.match(filename_stem)
    if match:
        return {
            'artist': match.group(1).strip(),
            'date': match.group(2).strip(),
            'title': match.group(3).strip(),
            'original_filename': filename_stem
        }
    return None

def add_metadata_to_mp3(
    mp3_path: Path, # Path to the MP3 file (will be the temp file during processing)
    metadata: Dict[str, str],
    image_data: Optional[bytes],
    image_mime: Optional[str]
) -> bool:
    """
    Adds ID3 metadata tags (text and optional album art) to the MP3 file.

    Args:
        mp3_path: Path to the MP3 file (likely the temporary file).
        metadata: Dictionary with 'artist', 'date', 'title'.
        image_data: Raw bytes of the album art image, or None.
        image_mime: MIME type of the album art image, or None.

    Returns:
        True if successful, False otherwise.
    """
    try:
        try:
             audio = MP3(mp3_path, ID3=ID3)
        except ID3NoHeaderError:
             audio = MP3(mp3_path)
             # If no ID3 header, try adding one
             try:
                 audio.add_tags()
             except Exception as add_tags_err:
                 tqdm.write(f"Warning: Could not add ID3 tag structure to {mp3_path.name}. Metadata might not be saved. Error: {add_tags_err}", file=sys.stderr)
                 # Attempt to proceed without tags if structure couldn't be added
                 if audio.tags is None: # If tags are still None, fail
                     raise ValueError(f"Failed to initialize ID3 tags for {mp3_path.name}") from add_tags_err

        # Ensure tags attribute exists after attempting to add them
        if audio.tags is None:
             audio.tags = ID3() # Create an empty ID3 object if still missing

        # --- Add/Update Text Metadata ---
        if metadata.get('artist'):
            audio.tags.add(TPE1(encoding=Encoding.UTF8, text=metadata['artist']))
        if metadata.get('title'):
             audio.tags.add(TIT2(encoding=Encoding.UTF8, text=metadata['title']))
        if metadata.get('date'):
             # Validate date format slightly before adding (basic YYYY-MM-DD check)
             date_str = metadata['date']
             if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
                 audio.tags.add(TDRC(encoding=Encoding.UTF8, text=date_str))
             else:
                 tqdm.write(f"Warning: Skipping invalid date format '{date_str}' for {mp3_path.name}. Expected YYYY-MM-DD.", file=sys.stderr)


        comment_text = f"Original Filename Stem: {metadata.get('original_filename', mp3_path.stem.replace(TEMP_SUFFIX,''))}" # Clean temp suffix for comment
        # Remove existing comments before adding new one
        audio.tags.delall('COMM')
        audio.tags.add(COMM(encoding=Encoding.UTF8, lang='eng', desc='Converted Info', text=comment_text))

        # --- Add/Update Album Art ---
        # Remove existing cover art first to avoid duplicates
        audio.tags.delall('APIC')
        if image_data and image_mime:
            apic = APIC(
                encoding=Encoding.UTF8, # Encoding for the description text
                mime=image_mime,        # Image mime type
                type=ID3_PIC_TYPE_COVER_FRONT, # 3: Cover (front)
                desc='Cover',           # Description
                data=image_data         # Image data as bytes
            )
            audio.tags.add(apic)

        # Save changes using ID3v2.3, removing ID3v1 tags
        audio.save(v1=0, v2_version=3)
        return True

    except Exception as e:
        tqdm.write(f"Error: Failed to add metadata/art to {mp3_path.name}. Reason: {e}", file=sys.stderr)
        return False

def build_ffmpeg_command(
    video_path: Path,
    temp_output_path: Path,
    ffmpeg_path: str,
    original_audio_codec: Optional[str],
    vbr_quality: int # Changed from bitrate_k to vbr_quality
) -> Tuple[List[str], str]:
    """Constructs the appropriate ffmpeg command list using VBR quality."""
    common_opts: List[str] = [
        ffmpeg_path,
        '-y', # Always allow ffmpeg to overwrite the temp file
        '-i', str(video_path),
        '-vn', # Disable video recording
        '-loglevel', 'error', # Only show errors
        '-hide_banner',
        '-map_metadata', '-1', # Strip existing metadata
        '-map', '0:a:0?', # Map first audio stream optionally
    ]

    action: str
    command: List[str]

    if original_audio_codec == 'mp3':
        # Copy MP3 stream if ffprobe identified it as mp3
        command = common_opts + ['-codec:a', 'copy', '-f','mp3', str(temp_output_path)]
        action = "copied"
    else:
        # Convert to MP3 using quality-based VBR
        command = common_opts + [
            '-codec:a', 'libmp3lame',
            '-q:a', str(vbr_quality), # Use -q:a for VBR quality
            '-ar', '44100', # Common sample rate
            '-ac', '2',     # Stereo
            '-f', 'mp3',    # Explicitly set output container format
            str(temp_output_path)
        ]
        # Updated action string
        action = f"converted (VBR Q{vbr_quality})"

    return command, action

def run_ffmpeg(command: List[str]) -> FFmpegResult:
    """Executes the ffmpeg command and returns success status and error message."""
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=False, encoding='utf-8', errors='replace')
        if process.returncode == 0:
            return True, None
        else:
            error_msg = f"FFmpeg error (code {process.returncode}):\n--- FFMPEG STDERR ---\n{process.stderr.strip()}"
            # Check for specific "no audio stream" errors
            is_no_audio_error = any(err_msg in process.stderr for err_msg in NO_AUDIO_STREAM_ERRORS)
            if is_no_audio_error:
                # Append a more user-friendly reason if possible
                 error_msg += "\n(Likely reason: No audio stream found in the input video)"
            return False, error_msg
    except Exception as e:
        return False, f"Failed to run ffmpeg command: {e}"

def verify_output(output_path: Path) -> bool:
    """Checks if the output file exists and is not empty."""
    return output_path.exists() and output_path.stat().st_size > 0

def cleanup_temp_file(temp_output_path: Path):
    """Attempts to remove the temporary output file."""
    if temp_output_path and temp_output_path.exists() and temp_output_path.name.endswith(TEMP_SUFFIX):
        try:
            temp_output_path.unlink()
        except OSError as e:
            tqdm.write(f"Warning: Could not remove temporary file {temp_output_path}: {e}", file=sys.stderr)

def handle_metadata_tagging(
    add_metadata_flag: bool,
    video_path: Path,
    temp_mp3_path: Path, # Operates on the temporary MP3 file
    image_data: Optional[bytes],
    image_mime: Optional[str]
) -> str:
    """Handles metadata extraction and tagging on the temp file, returning the status."""
    if not add_metadata_flag and not (image_data and image_mime):
        return 'not_attempted'
    if not temp_mp3_path.exists():
        tqdm.write(f"Error: Temp file {temp_mp3_path.name} not found before metadata tagging.", file=sys.stderr)
        return 'failed'

    filename_stem = video_path.stem
    metadata = extract_metadata_from_filename(filename_stem) or {}

    if not metadata and not (image_data and image_mime):
         return 'skipped'

    if not metadata and add_metadata_flag:
         if not any(part.lower() == ignored.lower() for part in video_path.parts for ignored in IGNORE_FOLDERS):
            tqdm.write(f"Info: No metadata pattern matched for '{filename_stem}', skipping text tagging.")

    if add_metadata_to_mp3(temp_mp3_path, metadata, image_data, image_mime):
        if metadata or (image_data and image_mime): # Check if any tagging was actually done
             return 'added'
        else:
             return 'skipped' # No text match and no art provided
    else:
        return 'failed'


# --- Main Worker Function ---

def convert_single_video(
    video_path: Path,
    source_base: Path,
    output_base: Path,
    vbr_quality: int, # Changed from bitrate_k
    ffmpeg_path: str,
    ffprobe_path: str,
    overwrite: bool,
    add_metadata_flag: bool,
    album_art_data: Optional[bytes],
    album_art_mime: Optional[str]
) -> ConversionResult:
    """Orchestrates the conversion and tagging process for a single video file using VBR."""
    output_path: Optional[Path] = None
    temp_output_path: Optional[Path] = None
    metadata_status: str = 'not_attempted'
    action: str = 'failed' # Default action state

    try:
        # --- Path Setup ---
        relative_path = video_path.relative_to(source_base)
        output_path = output_base / relative_path.with_suffix('.mp3')
        temp_output_path = output_path.with_suffix(output_path.suffix + TEMP_SUFFIX)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Pre-flight Check (Final Destination) ---
        if not overwrite and output_path.exists():
             return 'skipped', video_path, output_path, 'skipped', 'not_attempted'

        # --- Determine Audio Codec ---
        original_audio_codec = get_audio_codec(video_path, ffprobe_path)
        if original_audio_codec == "no_audio_stream_found_by_ffprobe":
             return 'skipped_no_audio', video_path, "No audio stream detected by ffprobe", 'skipped', 'not_attempted'

        # --- Build and Run FFmpeg Command (to Temp File) ---
        command, action = build_ffmpeg_command(
            video_path, temp_output_path, ffmpeg_path, original_audio_codec, vbr_quality # Pass vbr_quality
        )
        success, ffmpeg_error = run_ffmpeg(command)

        # --- Handle FFmpeg Result ---
        if not success:
            cleanup_temp_file(temp_output_path)
            if ffmpeg_error and any(err_msg in ffmpeg_error for err_msg in NO_AUDIO_STREAM_ERRORS):
                return 'skipped_no_audio', video_path, "No audio stream found by FFmpeg", action, metadata_status
            else:
                return 'failed', video_path, ffmpeg_error, action, metadata_status

        # --- Verify Temporary Output ---
        if not verify_output(temp_output_path):
            error_message = f"FFmpeg OK, but temp output is invalid/empty: {temp_output_path.name}"
            cleanup_temp_file(temp_output_path)
            return 'failed', video_path, error_message, action, metadata_status

        # --- Handle Metadata Tagging (on Temp File) ---
        if add_metadata_flag or (album_art_data and album_art_mime):
            metadata_status = handle_metadata_tagging(
                add_metadata_flag, video_path, temp_output_path, album_art_data, album_art_mime
            )
            if metadata_status == 'failed':
                error_message = f"Metadata tagging failed for {temp_output_path.name}"
                cleanup_temp_file(temp_output_path)
                return 'failed', video_path, error_message, action, metadata_status
        else:
             metadata_status = 'not_attempted'

        # --- Final Rename ---
        try:
            if output_path.exists():
                if overwrite:
                    try:
                        output_path.unlink()
                    except OSError as e:
                         error_message = f"Cannot overwrite existing file {output_path.name}: {e}"
                         cleanup_temp_file(temp_output_path)
                         return 'failed', video_path, error_message, action, metadata_status
                else:
                    error_message = f"Final output file {output_path.name} appeared unexpectedly before rename (and overwrite is False)."
                    cleanup_temp_file(temp_output_path)
                    return 'skipped', video_path, error_message, action, metadata_status

            temp_output_path.rename(output_path)

            if not verify_output(output_path):
                 error_message = f"Rename successful, but final file is invalid/empty: {output_path.name}"
                 cleanup_temp_file(output_path)
                 return 'failed', video_path, error_message, action, metadata_status

            # --- Success ---
            return 'success', video_path, output_path, action, metadata_status

        except OSError as e:
            error_msg = f"Failed to rename temp file {temp_output_path.name} to {output_path.name}: {e}"
            cleanup_temp_file(temp_output_path)
            if overwrite and output_path.exists():
                 cleanup_temp_file(output_path)
            return 'failed', video_path, error_msg, action, 'failed'

    except Exception as e:
        if temp_output_path: cleanup_temp_file(temp_output_path)
        final_meta_status = metadata_status if metadata_status in ['failed', 'added', 'skipped'] else 'failed'
        import traceback
        error_msg = f"Unexpected Python error processing {video_path.name}: {e}\n{traceback.format_exc()}"
        return 'failed', video_path, error_msg, action, final_meta_status


# --- File Discovery and Filtering ---

def find_video_files(source_folder: Path) -> List[Path]:
    """Finds all supported video files recursively, skipping ignored folders."""
    video_files: List[Path] = []
    print(f"Scanning for video files in: {source_folder}")
    if IGNORE_FOLDERS:
        ignore_folders_lower = {f.lower() for f in IGNORE_FOLDERS}
        print(f"Ignoring folders (case-insensitive): {', '.join(IGNORE_FOLDERS)}")
    else:
        ignore_folders_lower = set()

    try:
        all_files_gen = source_folder.rglob("*")
        try:
            # Use tqdm for discovery if possible
            all_files_list = list(tqdm(all_files_gen, desc="Discovering files", unit="file", leave=False, ncols=100, miniters=100, mininterval=0.1))
        except (TypeError, ImportError): # Fallback if tqdm discovery fails or not installed
            all_files_list = list(all_files_gen)
            print("Discovering files...") # Simple message if no tqdm

        print(f"Filtering {len(all_files_list)} items (ignoring specified folders)...")
        ignored_count = 0
        # Use tqdm for filtering if possible
        filtering_iterable = tqdm(all_files_list, desc="Filtering videos", unit="item", leave=False, ncols=100) if 'tqdm' in sys.modules else all_files_list

        for file_path in filtering_iterable:
            if file_path.is_file():
                should_ignore = False
                if ignore_folders_lower:
                    try:
                        relative_path = file_path.relative_to(source_folder)
                        for part in relative_path.parts[:-1]:
                            if part.lower() in ignore_folders_lower:
                                should_ignore = True
                                break
                    except ValueError:
                         if 'tqdm' in sys.modules:
                             tqdm.write(f"Warning: Could not get relative path for {file_path}, skipping ignore check.", file=sys.stderr)
                         else:
                             print(f"Warning: Could not get relative path for {file_path}, skipping ignore check.", file=sys.stderr)


                if should_ignore:
                    ignored_count += 1
                    continue

                if file_path.suffix.lower() in SUPPORTED_EXTENSIONS and not file_path.name.endswith(TEMP_SUFFIX):
                    video_files.append(file_path)

    except Exception as e:
        print(f"\nError during file scan: {e}", file=sys.stderr)

    if ignored_count > 0:
        print(f"(Skipped {ignored_count} files found within ignored folders)")

    print(f"Found {len(video_files)} potential video files to process.")
    return video_files


def filter_existing_files(
    all_video_files: List[Path],
    source_path: Path,
    output_base: Path,
    overwrite: bool
) -> Tuple[List[Path], int]:
    """Filters out videos whose corresponding FINAL MP3 output already exists."""
    files_to_process: List[Path] = []
    skipped_count: int = 0
    if not overwrite:
        print("Checking for existing final output files (.mp3) to skip...")
        checking_iterable = tqdm(all_video_files, desc="Pre-checking", unit="file", leave=False, ncols=100) if 'tqdm' in sys.modules else all_video_files
        for video_path in checking_iterable:
            try:
                relative_path = video_path.relative_to(source_path)
                potential_output_path = output_base / relative_path.with_suffix('.mp3')
                if potential_output_path.exists():
                    skipped_count += 1
                else:
                    files_to_process.append(video_path)
            except ValueError:
                 log_func = tqdm.write if 'tqdm' in sys.modules else print
                 log_func(f"Warning: Skipping file due to path issue: {video_path}", file=sys.stderr)
                 skipped_count += 1 # Treat as skipped

        return files_to_process, skipped_count
    else:
        return all_video_files, 0

# --- Argument Parsing and Validation ---

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert video files to MP3 using VBR quality, optionally adding metadata and album art. Uses temporary files for safety. Skips videos with no audio stream and ignores specified folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("source_folder", help="Path to the folder containing video files (scanned recursively).")
    parser.add_argument("output_folder", help="Path to the folder where MP3 files will be saved.")
    # Removed bitrate, added vbr-quality
    parser.add_argument("-q", "--vbr-quality", type=int, default=3, choices=range(10), metavar="0-9",
                        help="VBR quality level for re-encoding (0=best, 9=worst). Lower numbers are higher quality/larger files.")
    parser.add_argument("-t", "--threads", type=int, default=os.cpu_count() or 1, help="Number of concurrent ffmpeg processes.")
    parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite existing FINAL MP3 files.")
    parser.add_argument("--no-metadata", action="store_true", help="Disable extracting/embedding text metadata from filenames.")
    parser.add_argument("-i","--album-art", type=str, default=None, metavar="IMAGE_PATH", help="Path to an image file (jpg/png) to embed as album art.")
    parser.add_argument("--ffmpeg", default=None, help="Optional: Explicit path to ffmpeg executable.")
    parser.add_argument("--ffprobe", default=None, help="Optional: Explicit path to ffprobe executable.")
    parser.add_argument("--ignore-folder", action='append', default=[], help="Add a folder name to ignore (case-insensitive). Can be used multiple times. Overrides internal list if used.")

    return parser.parse_args()

def validate_args_and_paths(args: argparse.Namespace) -> Tuple[Path, Path, int, int, bool]:
    """Validates arguments and paths, creates output directory."""
    source_path = Path(args.source_folder).resolve()
    output_path_base = Path(args.output_folder).resolve()

    if not source_path.is_dir():
        print(f"Error: Source folder '{args.source_folder}' not found or is not a directory.", file=sys.stderr)
        sys.exit(1)

    try:
        output_path_base.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create output directory '{output_path_base}': {e}", file=sys.stderr)
        sys.exit(1)

    threads = args.threads
    if threads <= 0:
        log_func = tqdm.write if 'tqdm' in sys.modules else print
        log_func(f"Warning: Invalid number of threads ({threads}). Using 1 thread.", file=sys.stderr)
        threads = 1

    # Get VBR quality (already validated by argparse choices)
    vbr_quality = args.vbr_quality

    add_metadata_flag = not args.no_metadata

    # Handle Ignore Folders Argument
    global IGNORE_FOLDERS
    if args.ignore_folder:
        IGNORE_FOLDERS = args.ignore_folder
        print(f"Using specified ignore folders: {', '.join(IGNORE_FOLDERS)}")

    # Return vbr_quality instead of bitrate
    return source_path, output_path_base, threads, vbr_quality, add_metadata_flag

def load_album_art(image_path_str: Optional[str]) -> Tuple[Optional[bytes], Optional[str]]:
    """Loads album art data and determines MIME type."""
    log_func = tqdm.write if 'tqdm' in sys.modules else print
    if not image_path_str:
        return None, None

    image_path = Path(image_path_str).resolve()
    if not image_path.is_file():
        log_func(f"Warning: Album art file not found: {image_path}. Skipping album art embedding.", file=sys.stderr)
        return None, None

    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()

        mime_type, _ = mimetypes.guess_type(str(image_path))

        supported_mimes = ('image/jpeg', 'image/png', 'image/gif')
        is_supported = False
        if mime_type:
            mime_type_lower = mime_type.lower()
            if mime_type_lower.startswith('image/'):
                 if mime_type_lower in supported_mimes:
                     is_supported = True
                 else:
                     log_func(f"Warning: Album art MIME type '{mime_type}' is not guaranteed to be supported by all players, but attempting embedding.", file=sys.stderr)
                     is_supported = True
            else:
                 log_func(f"Warning: Detected MIME type '{mime_type}' for album art file {image_path.name} is not an image type. Skipping album art.", file=sys.stderr)
                 return None, None
        else:
             log_func(f"Warning: Could not determine MIME type for album art file {image_path.name}. Skipping album art.", file=sys.stderr)
             return None, None

        if not is_supported:
            log_func(f"Warning: Album art MIME type '{mime_type}' not supported or identified correctly. Skipping album art.", file=sys.stderr)
            return None, None

        print(f"Album art loaded: {image_path} (Type: {mime_type})")
        return image_data, mime_type

    except IOError as e:
        log_func(f"Error reading album art file {image_path}: {e}. Skipping album art.", file=sys.stderr)
        return None, None
    except Exception as e:
        log_func(f"Unexpected error loading album art {image_path}: {e}. Skipping album art.", file=sys.stderr)
        return None, None

# --- Concurrent Processing ---

def process_files_concurrently(
    files_to_process: List[Path],
    source_path: Path,
    output_base: Path,
    vbr_quality: int, # Changed from bitrate_k
    ffmpeg_path: str,
    ffprobe_path: str,
    overwrite: bool,
    add_metadata_flag: bool,
    album_art_data: Optional[bytes],
    album_art_mime: Optional[str],
    max_workers: int
) -> Iterator[ConversionResult]:
    """Submits conversion tasks to a ThreadPoolExecutor and yields results."""
    log_func = tqdm.write if 'tqdm' in sys.modules else print # For logging inside the loop
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='Converter') as executor:
        futures = {
            executor.submit(
                convert_single_video,
                video_path, source_path, output_base, vbr_quality, # Pass vbr_quality
                ffmpeg_path, ffprobe_path, overwrite, add_metadata_flag,
                album_art_data, album_art_mime
            ): video_path
            for video_path in files_to_process
        }

        progress_bar_iter = futures
        if 'tqdm' in sys.modules:
             progress_bar_iter = tqdm(as_completed(futures), total=len(files_to_process), desc="Converting", unit="file", ncols=100, smoothing=0.05)

        for future in progress_bar_iter:
            video_path_orig = futures[future]
            rel_input_str = "Unknown"
            try:
                try:
                     rel_input = video_path_orig.relative_to(source_path)
                     rel_input_str = str(rel_input)
                except ValueError:
                     rel_input_str = str(video_path_orig) # Fallback

                result: ConversionResult = future.result()
                yield result

            except Exception as exc:
                 log_func(f"\nCRITICAL WORKER ERROR: {rel_input_str}\n  Reason: {exc}", file=sys.stderr)
                 import traceback
                 traceback.print_exc(file=sys.stderr)

                 try:
                      relative_path = video_path_orig.relative_to(source_path)
                      output_path = output_base / relative_path.with_suffix('.mp3')
                      temp_output_path = output_path.with_suffix(output_path.suffix + TEMP_SUFFIX)
                      cleanup_temp_file(temp_output_path)
                 except Exception as cleanup_err:
                      log_func(f"  Cleanup attempt failed during critical error handling: {cleanup_err}", file=sys.stderr)

                 yield 'failed', video_path_orig, f"Critical Worker Exception: {exc}", 'failed', 'failed'


# --- Summary Reporting ---

def print_summary(
    total_files_found: int,
    skipped_initially_count: int,
    skipped_no_audio_count: int,
    total_submitted_for_process: int,
    success_count: int,
    copied_count: int,
    converted_count: int,
    failed_count: int,
    skipped_during_process_count: int,
    metadata_added_count: int,
    metadata_skipped_count: int,
    metadata_failed_count: int,
    add_metadata_flag: bool,
    album_art_provided: bool,
    vbr_quality: Optional[int] # Add vbr_quality for info
):
    """Prints the final summary of the conversion process."""
    print("\n" + "=" * 30)
    print(" Processing Summary ".center(30, "="))
    print(f"Video Files Found (excl. ignored): {total_files_found}")
    print(f"Skipped (Output Existed):          {skipped_initially_count}")
    print(f"Skipped (No Audio Stream):         {skipped_no_audio_count}")
    print(f"----------------------------------")
    print(f"Files Submitted for Processing:    {total_submitted_for_process}")
    if vbr_quality is not None:
        print(f"  (Using VBR Quality: {vbr_quality} for re-encoding)")
    print(f"  Successfully Processed:          {success_count} ({copied_count} copied, {converted_count} converted)")
    if add_metadata_flag or album_art_provided:
        print(f"    Metadata Added/Updated:        {metadata_added_count}")
        print(f"    Metadata Skipped (No Match):   {metadata_skipped_count}")
        print(f"    Metadata Tagging Failed:       {metadata_failed_count}")
    print(f"  Skipped during processing:       {skipped_during_process_count}")
    print(f"  Failed (Error/Convert/Tag):      {failed_count}")
    print("=" * 30)

# --- Startup Cleanup ---
def initial_cleanup(output_base: Path):
    """Recursively removes leftover .mp3.tmp files from the output directory."""
    print("Performing startup cleanup of temporary files...")
    cleanup_count = 0
    try:
        tmp_files_gen = output_base.rglob(f"*{TEMP_SUFFIX}")
        # Make it a list to iterate with tqdm if available
        tmp_files = list(tmp_files_gen)
        if not tmp_files:
            print("No leftover temporary files found.")
            return

        cleanup_iterable = tmp_files
        if 'tqdm' in sys.modules:
            cleanup_iterable = tqdm(tmp_files, desc="Cleaning up", unit="file", leave=False, ncols=100)

        for tmp_file in cleanup_iterable:
            if tmp_file.is_file():
                try:
                    tmp_file.unlink()
                    cleanup_count += 1
                except OSError as e:
                    log_func = tqdm.write if 'tqdm' in sys.modules else print
                    log_func(f"Warning: Could not remove temporary file {tmp_file}: {e}", file=sys.stderr)
                except Exception as e:
                    log_func = tqdm.write if 'tqdm' in sys.modules else print
                    log_func(f"Warning: Error removing temp file {tmp_file}: {e}", file=sys.stderr)

    except Exception as e:
         log_func = tqdm.write if 'tqdm' in sys.modules else print
         log_func(f"Error during startup cleanup scan in {output_base}: {e}", file=sys.stderr)

    print(f"Startup cleanup complete. Removed {cleanup_count} temporary files.")
    print("-" * 30)


# --- Main Execution ---

def main():
    """Main script execution function."""
    args = parse_arguments()
    # Validation now returns vbr_quality instead of bitrate
    source_path, output_path_base, threads, vbr_quality, add_metadata_flag = validate_args_and_paths(args)

    # Add fallback logging if tqdm not installed
    log_func = tqdm.write if 'tqdm' in sys.modules else print

    # Initial Cleanup
    initial_cleanup(output_path_base)

    # Find Executables
    ffmpeg_executable = args.ffmpeg or find_executable("ffmpeg")
    ffprobe_executable = args.ffprobe or find_executable("ffprobe")
    if not ffmpeg_executable or not ffprobe_executable:
        sys.exit(1)
    print("-" * 30)

    # Load Album Art
    album_art_data, album_art_mime = load_album_art(args.album_art)
    album_art_provided = bool(album_art_data)
    print("-" * 30)

    # Find and Filter Files
    all_video_files = find_video_files(source_path)
    total_files_found = len(all_video_files)
    if total_files_found == 0:
        print("No video files found matching extensions (or all were in ignored folders):", ', '.join(SUPPORTED_EXTENSIONS))
        sys.exit(0)

    files_to_process, skipped_initially_count = filter_existing_files(
        all_video_files, source_path, output_path_base, args.overwrite
    )
    actual_submitted_count = len(files_to_process)

    if skipped_initially_count > 0:
        print(f"Skipped {skipped_initially_count} files as corresponding final MP3s already exist (use -o to overwrite).")

    if actual_submitted_count == 0:
        if skipped_initially_count > 0:
             print("No new files to process (all remaining found files already have existing outputs).")
        else:
             print("No files left to process after checking for existing output.")
        sys.exit(0)

    print("-" * 30)

    # Processing Information
    print(f"Starting processing for {actual_submitted_count} files using {threads} threads...")
    print(f"Source:        {source_path}")
    print(f"Output:        {output_path_base}")
    # Updated to show VBR quality
    print(f"VBR Quality:   {vbr_quality} (for re-encoding, 0=best, 9=worst)")
    print(f"Overwrite:     {'Yes' if args.overwrite else 'No'}")
    print(f"Add Text Meta: {'Yes' if add_metadata_flag else 'No'}")
    print(f"Add Album Art: {'Yes' if album_art_provided else 'No'}")
    if IGNORE_FOLDERS:
        print(f"Ignoring:      {', '.join(IGNORE_FOLDERS)}")
    print("-" * 30)

    # Initialize Counters
    success_count = copied_count = converted_count = failed_count = 0
    skipped_during_process_count = 0
    skipped_no_audio_count = 0
    metadata_added_count = metadata_skipped_count = metadata_failed_count = 0

    # Process Files
    results_iterator = process_files_concurrently(
        files_to_process, source_path, output_path_base, vbr_quality, # Pass vbr_quality
        ffmpeg_executable, ffprobe_executable, args.overwrite, add_metadata_flag,
        album_art_data, album_art_mime,
        threads
    )

    for status, input_path, output_or_error, action, meta_status in results_iterator:
        rel_input_display = "Unknown Path"
        if isinstance(input_path, Path):
            try:
                rel_input_display = str(input_path.relative_to(source_path))
            except ValueError:
                rel_input_display = str(input_path)

        if status == 'success':
            success_count += 1
            if action == 'copied': copied_count += 1
            # Check if action indicates conversion (contains 'converted')
            elif 'converted' in action: converted_count += 1

            if meta_status == 'added': metadata_added_count += 1
            elif meta_status == 'skipped': metadata_skipped_count += 1

        elif status == 'failed':
            failed_count += 1
            log_func(f"\nFAILED : {rel_input_display} (Action attempted: {action})\n  Reason: {output_or_error}", file=sys.stderr)
            if meta_status == 'failed':
                 metadata_failed_count += 1

        elif status == 'skipped':
            skipped_during_process_count += 1
            log_func(f"INFO: Skipped during processing: {rel_input_display} - Reason: {output_or_error}", file=sys.stderr)

        elif status == 'skipped_no_audio':
            skipped_no_audio_count += 1
            log_func(f"INFO: Skipped (No Audio): {rel_input_display} - Reason: {output_or_error}", file=sys.stdout)

    # Print Summary - Pass vbr_quality for display
    print_summary(
        total_files_found,
        skipped_initially_count,
        skipped_no_audio_count,
        actual_submitted_count,
        success_count, copied_count, converted_count, failed_count,
        skipped_during_process_count,
        metadata_added_count, metadata_skipped_count, metadata_failed_count,
        add_metadata_flag, album_art_provided,
        vbr_quality # Pass the quality level used
    )

    # Determine Exit Code
    final_error_count = failed_count
    exit_code = 0
    if final_error_count > 0:
        print(f"\nWARNING: {final_error_count} files encountered errors during processing. Check logs above.", file=sys.stderr)
        exit_code = 1
    elif total_files_found == 0:
        print("\nNo processable video files found in the specified source (considering ignored folders).")
    elif actual_submitted_count == 0:
         if skipped_initially_count > 0:
              print("\nProcessing complete. All found files were skipped because output already existed.")
         else:
              print("\nProcessing complete. No files were submitted for conversion.")
    elif success_count == 0 and skipped_no_audio_count > 0 and failed_count == 0 and skipped_during_process_count == 0:
        print("\nProcessing complete. All submitted files were skipped due to lacking audio streams.")
    else:
        print("\nAll tasks completed.")

    sys.exit(exit_code)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user (Ctrl+C). Cleaning up may be needed on next run.", file=sys.stderr)
        sys.exit(130)
