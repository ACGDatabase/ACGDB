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

# --- Helper Types ---
FFmpegResult = Tuple[bool, Optional[str]] # (success, error_message)
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
        return None
    except FileNotFoundError:
         tqdm.write(f"\nERROR: ffprobe executable not found at '{ffprobe_path}'. Cannot check audio codec.", file=sys.stderr)
         return None
    except (subprocess.CalledProcessError, json.JSONDecodeError, Exception) as e:
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
    temp_output_path: Path, # Takes the temporary path now
    ffmpeg_path: str,
    original_audio_codec: Optional[str],
    bitrate_k: int,
    overwrite: bool # Overwrite applies to the *final* destination logic, -y ensures ffmpeg can overwrite the temp if needed
) -> Tuple[List[str], str]:
    """Constructs the appropriate ffmpeg command list and determines the action."""
    common_opts: List[str] = [
        ffmpeg_path,
        '-y', # Always allow ffmpeg to overwrite the temp file if it somehow exists
        '-i', str(video_path),
        '-vn',
        '-loglevel', 'error',
        '-hide_banner',
        # Strip existing metadata from video - we'll add fresh tags later
        '-map_metadata', '-1',
        # map only the audio stream to the output
        '-map', '0:a:0?', # Map first audio stream, '?' makes it optional if no audio exists
    ]

    action: str
    command: List[str]

    if original_audio_codec == 'mp3':
        # Use the temp path for output. -f mp3 isn't strictly needed for copy
        # as FFmpeg usually handles container correctly when copying MP3 stream.
        command = common_opts + ['-codec:a', 'copy', str(temp_output_path)]
        action = "copied"
    else:
        # Use the temp path for output AND explicitly set format to mp3
        command = common_opts + [
            '-codec:a', 'libmp3lame',
            '-ab', f'{bitrate_k}k',
            '-ar', '44100', # Common sample rate
            '-ac', '2',     # Stereo
            '-f', 'mp3',    # Explicitly set output container format
            str(temp_output_path)
        ]
        action = "converted"

    return command, action

def run_ffmpeg(command: List[str]) -> FFmpegResult:
    """Executes the ffmpeg command and returns success status and error message."""
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=False, encoding='utf-8', errors='replace')
        if process.returncode == 0:
            return True, None
        else:
            # Removed the specific check for '-n' error message as we always use '-y' now.
            error_msg = f"FFmpeg error (code {process.returncode}):\n--- FFMPEG STDERR ---\n{process.stderr.strip()}"
            # Check if it failed because no audio stream was found with -map 0:a:0?
            if "Stream map '0:a:0' matches no streams" in process.stderr:
                error_msg += "\n(Likely reason: No audio stream found in the input video)"
            return False, error_msg
    except Exception as e:
        return False, f"Failed to run ffmpeg command: {e}"

def verify_output(output_path: Path) -> bool:
    """Checks if the output file exists and is not empty."""
    # Verifies the temp path or the final path after rename
    return output_path.exists() and output_path.stat().st_size > 0

def cleanup_temp_file(temp_output_path: Path):
    """Attempts to remove the temporary output file."""
    if temp_output_path and temp_output_path.exists() and temp_output_path.name.endswith(TEMP_SUFFIX):
        try:
            temp_output_path.unlink()
            # tqdm.write(f"Cleaned up temporary file: {temp_output_path.name}", file=sys.stderr) # Optional: more verbose logging
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
    if not add_metadata_flag and not (image_data and image_mime): # Skip if no text meta AND no image
        return 'not_attempted'
    if not temp_mp3_path.exists(): # Ensure temp file exists before tagging
        tqdm.write(f"Error: Temp file {temp_mp3_path.name} not found before metadata tagging.", file=sys.stderr)
        return 'failed'

    filename_stem = video_path.stem
    metadata = extract_metadata_from_filename(filename_stem) or {} # Use empty dict if no match

    if not metadata and not (image_data and image_mime):
         # This case is already covered by the first check, but keep for clarity
         return 'skipped' # No text match and no art provided

    if not metadata and add_metadata_flag: # Log only if text meta was expected but not found
         tqdm.write(f"Info: No metadata pattern matched for '{filename_stem}', skipping text tagging.")

    # Attempt to add metadata (text and/or art)
    if add_metadata_to_mp3(temp_mp3_path, metadata, image_data, image_mime):
        # Check which parts were actually added if we need finer grain status
        if metadata and (image_data and image_mime):
            return 'added' # Both attempted and succeeded (or just text/just art if only one provided)
        elif metadata:
             return 'added' # Only text attempted and succeeded
        elif image_data and image_mime:
             return 'added' # Only art attempted and succeeded
        else:
             return 'skipped' # Should not happen if initial checks are correct
    else:
        return 'failed' # Tagging failed


# --- Main Worker Function ---

def convert_single_video(
    video_path: Path,
    source_base: Path,
    output_base: Path,
    bitrate_k: int,
    ffmpeg_path: str,
    ffprobe_path: str,
    overwrite: bool,
    add_metadata_flag: bool,
    album_art_data: Optional[bytes],
    album_art_mime: Optional[str]
) -> ConversionResult:
    """Orchestrates the conversion and tagging process for a single video file using a temporary file."""
    output_path: Optional[Path] = None
    temp_output_path: Optional[Path] = None
    metadata_status: str = 'not_attempted'
    action: str = 'failed' # Default action state

    try:
        # --- Path Setup ---
        relative_path = video_path.relative_to(source_base)
        output_path = output_base / relative_path.with_suffix('.mp3')
        # Create the temporary path
        temp_output_path = output_path.with_suffix(output_path.suffix + TEMP_SUFFIX)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Pre-flight Check (Final Destination) ---
        if not overwrite and output_path.exists():
             # This check should ideally be caught by filter_existing_files, but double-check
             return 'skipped', video_path, output_path, 'skipped', 'not_attempted'


        # --- Determine Audio Codec ---
        original_audio_codec = get_audio_codec(video_path, ffprobe_path)

        # --- Build and Run FFmpeg Command (to Temp File) ---
        command, action = build_ffmpeg_command(
            video_path, temp_output_path, ffmpeg_path, original_audio_codec, bitrate_k, overwrite
        )
        success, ffmpeg_error = run_ffmpeg(command)

        if not success:
            cleanup_temp_file(temp_output_path)
            return 'failed', video_path, ffmpeg_error, action, metadata_status

        # --- Verify Temporary Output ---
        if not verify_output(temp_output_path):
            error_message = f"FFmpeg OK, but temp output is invalid/empty: {temp_output_path.name}"
            cleanup_temp_file(temp_output_path)
            return 'failed', video_path, error_message, action, metadata_status

        # --- Handle Metadata Tagging (on Temp File) ---
        # Only attempt if flag is set OR album art is provided
        if add_metadata_flag or (album_art_data and album_art_mime):
            metadata_status = handle_metadata_tagging(
                add_metadata_flag, video_path, temp_output_path, album_art_data, album_art_mime
            )
            if metadata_status == 'failed':
                # Decide if metadata failure should prevent the rename (treat as overall failure)
                # Current decision: Yes, metadata failure means the whole process failed for this file.
                error_message = f"Metadata tagging failed for {temp_output_path.name}"
                cleanup_temp_file(temp_output_path)
                return 'failed', video_path, error_message, action, metadata_status
        else:
             metadata_status = 'not_attempted' # Explicitly set if no tagging was done


        # --- Final Rename ---
        try:
            # Double-check final destination right before rename, respecting overwrite flag
            if output_path.exists():
                if overwrite:
                    try:
                        output_path.unlink() # Remove final destination if overwriting
                    except OSError as e:
                         error_message = f"Cannot overwrite existing file {output_path.name}: {e}"
                         cleanup_temp_file(temp_output_path)
                         return 'failed', video_path, error_message, action, metadata_status
                else:
                    # This should not happen if initial checks worked, but handle defensively
                    error_message = f"Final output file {output_path.name} appeared unexpectedly before rename (and overwrite is False)."
                    cleanup_temp_file(temp_output_path)
                    # Treat as skipped because the final file exists and we shouldn't overwrite
                    return 'skipped', video_path, error_message, action, metadata_status

            # Perform the rename
            temp_output_path.rename(output_path)

            # --- Final Verification (Optional but Recommended) ---
            if not verify_output(output_path):
                 error_message = f"Rename appeared successful, but final file is invalid/empty: {output_path.name}"
                 # Don't cleanup temp here as it's already renamed (or failed rename)
                 # output_path might exist but be empty, try cleaning it
                 cleanup_temp_file(output_path) # Try cleaning the potentially bad final file
                 return 'failed', video_path, error_message, action, metadata_status

            # --- Success ---
            # Status 'success' refers to the conversion/copy *and* rename being successful.
            # Metadata status reflects the outcome of the tagging step.
            return 'success', video_path, output_path, action, metadata_status

        except OSError as e:
            error_msg = f"Failed to rename temp file {temp_output_path.name} to {output_path.name}: {e}"
            cleanup_temp_file(temp_output_path) # Clean up the temp file
            # If rename failed, try to clean up potential partial final file if overwrite was true
            if overwrite and output_path.exists():
                 cleanup_temp_file(output_path)
            return 'failed', video_path, error_msg, action, 'failed' # Metadata status becomes failed as rename failed

    except Exception as e:
        error_msg = f"Unexpected Python error processing {video_path.name}: {e}"
        # Ensure cleanup happens even with unexpected errors
        if temp_output_path: cleanup_temp_file(temp_output_path)
        # Determine final meta status - if tagging was attempted and failed before this error, keep 'failed'
        final_meta_status = metadata_status if metadata_status in ['failed', 'added', 'skipped'] else 'failed'
        return 'failed', video_path, error_msg, action, final_meta_status


# --- File Discovery and Filtering ---

def find_video_files(source_folder: Path) -> List[Path]:
    """Finds all supported video files recursively."""
    video_files: List[Path] = []
    print(f"Scanning for video files in: {source_folder}")
    try:
        # Efficiently find all files first, then filter
        all_files_gen = source_folder.rglob("*")
        # Wrap generator with tqdm if list conversion is too slow for large dirs
        # Convert to list for stable progress bar, handle potential large directories
        try:
            all_files_list = list(tqdm(all_files_gen, desc="Discovering files", unit="file", leave=False, ncols=100, miniters=1000)) # Adjust miniters
        except TypeError: # Handle cases where len() is not supported directly
            all_files_list = list(all_files_gen)
            print("Found a large number of files, discovery progress bar may be inaccurate.")


        for file_path in tqdm(all_files_list, desc="Filtering videos", unit="file", leave=False, ncols=100):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                # Basic check to avoid adding our own temp files if they somehow end up in source
                if not file_path.name.endswith(TEMP_SUFFIX):
                    video_files.append(file_path)
    except Exception as e:
        print(f"\nError during file scan: {e}", file=sys.stderr)
    print(f"Found {len(video_files)} potential video files.")
    return video_files

def filter_existing_files(
    all_video_files: List[Path],
    source_path: Path,
    output_base: Path,
    overwrite: bool
) -> Tuple[List[Path], int]:
    """
    Filters out videos whose corresponding FINAL MP3 output already exists.
    Does NOT skip based on the presence of .tmp files.
    """
    files_to_process: List[Path] = []
    skipped_count: int = 0
    if not overwrite:
        print("Checking for existing final output files (.mp3) to skip...")
        for video_path in tqdm(all_video_files, desc="Pre-checking", unit="file", leave=False, ncols=100):
            try:
                relative_path = video_path.relative_to(source_path)
                # Check ONLY for the final .mp3 file
                potential_output_path = output_base / relative_path.with_suffix('.mp3')
                if potential_output_path.exists():
                    skipped_count += 1
                else:
                    files_to_process.append(video_path)
            except ValueError:
                 tqdm.write(f"Warning: Skipping file due to path issue: {video_path}", file=sys.stderr)
                 skipped_count += 1 # Treat as skipped

        return files_to_process, skipped_count
    else:
        # If overwriting, process all files found
        return all_video_files, 0

# --- Argument Parsing and Validation ---

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert video files to MP3, optionally adding metadata and album art. Uses temporary files for safety.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("source_folder", help="Path to the folder containing video files (scanned recursively).")
    parser.add_argument("output_folder", help="Path to the folder where MP3 files will be saved.")
    parser.add_argument("-b", "--bitrate", type=int, default=192, help="Audio bitrate (kbps) for re-encoding (if needed).")
    parser.add_argument("-t", "--threads", type=int, default=os.cpu_count() or 1, help="Number of concurrent ffmpeg processes.")
    parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite existing FINAL MP3 files.")
    parser.add_argument("--no-metadata", action="store_true", help="Disable extracting/embedding text metadata from filenames.")
    parser.add_argument("-i","--album-art", type=str, default=None, metavar="IMAGE_PATH", help="Path to an image file (jpg/png) to embed as album art.")
    parser.add_argument("--ffmpeg", default=None, help="Optional: Explicit path to ffmpeg executable.")
    parser.add_argument("--ffprobe", default=None, help="Optional: Explicit path to ffprobe executable.")
    return parser.parse_args()

def validate_args_and_paths(args: argparse.Namespace) -> Tuple[Path, Path, int, int, bool]:
    """Validates arguments and paths, creates output directory."""
    source_path = Path(args.source_folder).resolve()
    output_path_base = Path(args.output_folder).resolve()

    if not source_path.is_dir():
        print(f"Error: Source folder '{args.source_folder}' not found or is not a directory.", file=sys.stderr)
        sys.exit(1)

    # No need to check if output is a file, mkdir will handle it or fail.
    # if output_path_base.exists() and not output_path_base.is_dir():
    #      print(f"Error: Output path '{args.output_folder}' exists but is not a directory.", file=sys.stderr)
    #      sys.exit(1)

    try:
        output_path_base.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create output directory '{output_path_base}': {e}", file=sys.stderr)
        sys.exit(1)

    threads = args.threads
    if threads <= 0:
        tqdm.write(f"Warning: Invalid number of threads ({threads}). Using 1 thread.", file=sys.stderr)
        threads = 1

    bitrate = args.bitrate
    if bitrate <= 0:
         tqdm.write(f"Warning: Invalid bitrate ({bitrate}). Using 192k.", file=sys.stderr)
         bitrate = 192

    add_metadata_flag = not args.no_metadata

    return source_path, output_path_base, threads, bitrate, add_metadata_flag

def load_album_art(image_path_str: Optional[str]) -> Tuple[Optional[bytes], Optional[str]]:
    """Loads album art data and determines MIME type."""
    if not image_path_str:
        return None, None

    image_path = Path(image_path_str).resolve() # Resolve the path
    if not image_path.is_file():
        tqdm.write(f"Warning: Album art file not found: {image_path}. Skipping album art embedding.", file=sys.stderr)
        return None, None

    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()

        mime_type, _ = mimetypes.guess_type(str(image_path)) # Use str() for guess_type
        if not mime_type or not mime_type.startswith('image/'):
            supported_mimes = ('image/jpeg', 'image/png', 'image/gif') # Add more if needed by mutagen/ID3
            if mime_type not in supported_mimes:
                tqdm.write(f"Warning: Unsupported image MIME type '{mime_type}' for {image_path.name}. Supported: {supported_mimes}. Skipping album art.", file=sys.stderr)
                return None, None
            else:
                 tqdm.write(f"Warning: Could not confidently determine image MIME type for {image_path.name}, but attempting with '{mime_type}'.", file=sys.stderr)
                 # Allow common types even if guess is uncertain, mutagen might handle it.

        print(f"Album art loaded: {image_path} (Type: {mime_type})")
        return image_data, mime_type

    except IOError as e:
        tqdm.write(f"Error reading album art file {image_path}: {e}. Skipping album art.", file=sys.stderr)
        return None, None
    except Exception as e:
        tqdm.write(f"Unexpected error loading album art {image_path}: {e}. Skipping album art.", file=sys.stderr)
        return None, None

# --- Concurrent Processing ---

def process_files_concurrently(
    files_to_process: List[Path],
    source_path: Path,
    output_base: Path,
    bitrate_k: int,
    ffmpeg_path: str,
    ffprobe_path: str,
    overwrite: bool,
    add_metadata_flag: bool,
    album_art_data: Optional[bytes],
    album_art_mime: Optional[str],
    max_workers: int
) -> Iterator[ConversionResult]:
    """Submits conversion tasks to a ThreadPoolExecutor and yields results."""
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='Converter') as executor:
        futures = {
            executor.submit(
                convert_single_video,
                video_path, source_path, output_base, bitrate_k,
                ffmpeg_path, ffprobe_path, overwrite, add_metadata_flag,
                album_art_data, album_art_mime
            ): video_path
            for video_path in files_to_process
        }

        # Use tqdm directly on as_completed for progress
        for future in tqdm(as_completed(futures), total=len(files_to_process), desc="Converting", unit="file", ncols=100):
            video_path_orig = futures[future]
            rel_input_str = "Unknown"
            try:
                rel_input = video_path_orig.relative_to(source_path)
                rel_input_str = str(rel_input)
                # Update postfix dynamically - use short name
                # pbar.set_postfix_str(f"{rel_input.name[:30]}...", refresh=False) # tqdm handles refresh with iterators

                result: ConversionResult = future.result()
                yield result

            except Exception as exc:
                 # This catches exceptions *within* the future processing itself (less likely with try/except in worker)
                 # Or exceptions during future.result() call.
                 tqdm.write(f"\nCRITICAL WORKER ERROR: {rel_input_str}\n  Reason: {exc}", file=sys.stderr)
                 # Try to find associated temp file and clean it up if possible (best effort)
                 try:
                      relative_path = video_path_orig.relative_to(source_path)
                      output_path = output_base / relative_path.with_suffix('.mp3')
                      temp_output_path = output_path.with_suffix(output_path.suffix + TEMP_SUFFIX)
                      cleanup_temp_file(temp_output_path)
                 except Exception as cleanup_err:
                      tqdm.write(f"  Cleanup attempt failed during critical error handling: {cleanup_err}", file=sys.stderr)

                 yield 'failed', video_path_orig, f"Critical Worker Exception: {exc}", 'failed', 'failed'


# --- Summary Reporting ---

def print_summary(
    total_files_found: int,
    skipped_initially_count: int, # Files skipped because final .mp3 existed
    total_attempted: int,         # Files actually sent to process_files_concurrently
    success_count: int,
    copied_count: int,
    converted_count: int,
    failed_count: int,            # Files failed during conversion/rename/internal error
    skipped_during_process_count: int, # Files skipped by the worker (e.g., output existed just before rename)
    metadata_added_count: int,
    metadata_skipped_count: int,  # No pattern matched for text meta
    metadata_failed_count: int,   # Error during tagging process
    add_metadata_flag: bool,
    album_art_provided: bool
):
    """Prints the final summary of the conversion process."""
    print("\n" + "=" * 30)
    print(" Processing Summary ".center(30, "="))
    print(f"Total Video Files Found:      {total_files_found}")
    print(f"Skipped (Output Existed):     {skipped_initially_count}")
    print(f"-----------------------------")
    print(f"Files Submitted for Process:  {total_attempted}")
    print(f"  Successfully Processed:     {success_count} ({copied_count} copied, {converted_count} converted)")
    # Display metadata stats only if it was relevant
    if add_metadata_flag or album_art_provided:
        print(f"    Metadata Added/Updated:   {metadata_added_count}")
        print(f"    Metadata Skipped (No Match):{metadata_skipped_count}")
        print(f"    Metadata Tagging Failed:  {metadata_failed_count}")
    print(f"  Skipped during processing:  {skipped_during_process_count}") # e.g., race condition where output appeared
    print(f"  Failed (Error/Convert/Tag): {failed_count}")
    print("=" * 30)

# --- Startup Cleanup ---
def initial_cleanup(output_base: Path):
    """Recursively removes leftover .mp3.tmp files from the output directory."""
    print("Performing startup cleanup of temporary files...")
    cleanup_count = 0
    try:
        # Use rglob to find all matching files recursively
        tmp_files = list(output_base.rglob(f"*{TEMP_SUFFIX}"))
        if not tmp_files:
            print("No leftover temporary files found.")
            return

        for tmp_file in tqdm(tmp_files, desc="Cleaning up", unit="file", leave=False, ncols=100):
            if tmp_file.is_file(): # Ensure it's a file before deleting
                try:
                    tmp_file.unlink()
                    # Use tqdm.write for logging within the loop if needed, but might be too verbose
                    # tqdm.write(f"Removed leftover temporary file: {tmp_file.relative_to(output_base)}")
                    cleanup_count += 1
                except OSError as e:
                    tqdm.write(f"Warning: Could not remove temporary file {tmp_file}: {e}", file=sys.stderr)
                except Exception as e: # Catch broader exceptions during unlink
                    tqdm.write(f"Warning: Error removing temp file {tmp_file}: {e}", file=sys.stderr)

    except Exception as e:
         # Catch errors during the rglob scan itself
         tqdm.write(f"Error during startup cleanup scan in {output_base}: {e}", file=sys.stderr)

    print(f"Startup cleanup complete. Removed {cleanup_count} temporary files.")
    print("-" * 30)


# --- Main Execution ---

def main():
    """Main script execution function."""
    args = parse_arguments()
    source_path, output_path_base, threads, bitrate, add_metadata_flag = validate_args_and_paths(args)

    # --- Initial Cleanup ---
    initial_cleanup(output_path_base)

    # --- Find Executables ---
    ffmpeg_executable = args.ffmpeg or find_executable("ffmpeg")
    ffprobe_executable = args.ffprobe or find_executable("ffprobe")
    if not ffmpeg_executable or not ffprobe_executable:
        sys.exit(1)
    print("-" * 30)

    # --- Load Album Art (do this once) ---
    album_art_data, album_art_mime = load_album_art(args.album_art)
    album_art_provided = bool(album_art_data) # Flag if art was successfully loaded
    print("-" * 30)

    # --- Find and Filter Files ---
    all_video_files = find_video_files(source_path)
    total_files_found = len(all_video_files)
    if total_files_found == 0:
        print("No video files found matching extensions:", ', '.join(SUPPORTED_EXTENSIONS))
        sys.exit(0)

    files_to_process, skipped_initially_count = filter_existing_files(
        all_video_files, source_path, output_path_base, args.overwrite
    )
    total_to_process = len(files_to_process) # Number of files we will attempt

    if skipped_initially_count > 0:
        print(f"Skipped {skipped_initially_count} files as corresponding final MP3s already exist (use -o to overwrite).")
    if total_to_process == 0:
        print("No new files to process.")
        # If some files were skipped, exit code 0 is okay. If no files found at all, also okay.
        sys.exit(0)
    print("-" * 30)

    # --- Processing Information ---
    print(f"Starting processing for {total_to_process} files using {threads} threads...")
    print(f"Source:      {source_path}")
    print(f"Output:      {output_path_base}")
    print(f"Bitrate:     {bitrate} kbps (for re-encoding)")
    print(f"Overwrite:   {'Yes' if args.overwrite else 'No'}")
    print(f"Add Text Meta:{'Yes' if add_metadata_flag else 'No'}")
    print(f"Add Album Art:{'Yes' if album_art_provided else 'No'}")
    print("-" * 30)

    # --- Initialize Counters ---
    success_count = copied_count = converted_count = failed_count = 0
    skipped_during_process_count = 0
    metadata_added_count = metadata_skipped_count = metadata_failed_count = 0

    # --- Process Files ---
    results_iterator = process_files_concurrently(
        files_to_process, source_path, output_path_base, bitrate,
        ffmpeg_executable, ffprobe_executable, args.overwrite, add_metadata_flag,
        album_art_data, album_art_mime,
        threads
    )

    for status, input_path, output_or_error, action, meta_status in results_iterator:
        # Determine display name safely
        rel_input_display = "Unknown Path"
        if isinstance(input_path, Path):
            try:
                rel_input_display = str(input_path.relative_to(source_path))
            except ValueError:
                rel_input_display = str(input_path) # Fallback

        # Update counters based on results
        if status == 'success':
            success_count += 1
            if action == 'copied': copied_count += 1
            elif action == 'converted': converted_count += 1

            # Count metadata success based on its specific status
            if meta_status == 'added': metadata_added_count += 1
            elif meta_status == 'skipped': metadata_skipped_count += 1
            # 'not_attempted' doesn't increment any metadata counter here
            # 'failed' metadata status leads to overall 'failed' status below

        elif status == 'failed':
            failed_count += 1
            # Log the failure reason using tqdm.write to avoid messing up progress bar
            tqdm.write(f"\nFAILED : {rel_input_display}\n  Reason: {output_or_error}", file=sys.stderr)
            # If the failure was *specifically* due to failed metadata tagging, increment that counter too
            if meta_status == 'failed':
                 metadata_failed_count += 1

        elif status == 'skipped':
            skipped_during_process_count += 1
            # Optionally log these skips if they are unexpected
            # tqdm.write(f"INFO: Skipped during processing: {rel_input_display} - Reason: {output_or_error}", file=sys.stderr)


    # --- Print Summary ---
    print_summary(
        total_files_found, skipped_initially_count, total_to_process,
        success_count, copied_count, converted_count, failed_count,
        skipped_during_process_count,
        metadata_added_count, metadata_skipped_count, metadata_failed_count,
        add_metadata_flag, album_art_provided
    )

    # --- Determine Exit Code ---
    # Consider both conversion/file errors and metadata tagging failures as reasons for non-zero exit code
    final_error_count = failed_count # Includes metadata failures that caused overall failure

    if final_error_count > 0:
        print(f"\nWARNING: {final_error_count} files encountered errors during processing. Check logs above.", file=sys.stderr)
        sys.exit(1)
    elif skipped_initially_count > 0 and total_to_process == 0:
         print("\nAll matching files already had existing MP3s.")
         sys.exit(0)
    elif total_files_found == 0:
        # Already handled earlier, but double check
        print("\nNo video files found to process.")
        sys.exit(0)
    else:
        print("\nAll tasks completed successfully.")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user (Ctrl+C). Cleaning up may be needed on next run.", file=sys.stderr)
        # Exit code indicating interruption
        sys.exit(130)
