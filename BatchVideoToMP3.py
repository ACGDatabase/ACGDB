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
    sys.exit(1)


# --- Configuration ---
SUPPORTED_EXTENSIONS: Tuple[str, ...] = ('.mp4', '.ts', '.mkv', '.avi', '.mov', '.wmv', '.flv')
FILENAME_METADATA_REGEX: re.Pattern = re.compile(
    r"^\[([^\]]+)\]\[(\d{4}-\d{2}-\d{2})\]\[(.+?)\](?:\[(\d{4}-\d{2}-\d{2})\])?$"
)
# Standard ID3 picture type for Cover (front)
ID3_PIC_TYPE_COVER_FRONT = 3

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
    mp3_path: Path,
    metadata: Dict[str, str],
    image_data: Optional[bytes],
    image_mime: Optional[str]
) -> bool:
    """
    Adds ID3 metadata tags (text and optional album art) to the MP3 file.

    Args:
        mp3_path: Path to the MP3 file.
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
             audio.add_tags() # Add new ID3 frame if none exists

        if audio.tags is None: # Should be handled by add_tags, but double-check
             audio.add_tags()

        # --- Add/Update Text Metadata ---
        if metadata.get('artist'):
            audio.tags.add(TPE1(encoding=Encoding.UTF8, text=metadata['artist']))
        if metadata.get('title'):
             audio.tags.add(TIT2(encoding=Encoding.UTF8, text=metadata['title']))
        if metadata.get('date'):
             audio.tags.add(TDRC(encoding=Encoding.UTF8, text=metadata['date']))

        comment_text = f"Original Filename Stem: {metadata.get('original_filename', mp3_path.stem)}"
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
            # print(f"Debug: Added APIC frame to {mp3_path.name}")

        # Save changes using ID3v2.3, removing ID3v1 tags
        audio.save(v1=0, v2_version=3)
        return True

    except Exception as e:
        tqdm.write(f"Error: Failed to add metadata/art to {mp3_path.name}. Reason: {e}", file=sys.stderr)
        return False

def build_ffmpeg_command(
    video_path: Path,
    output_path: Path,
    ffmpeg_path: str,
    original_audio_codec: Optional[str],
    bitrate_k: int,
    overwrite: bool
) -> Tuple[List[str], str]:
    """Constructs the appropriate ffmpeg command list and determines the action."""
    common_opts: List[str] = [
        ffmpeg_path,
        '-y' if overwrite else '-n',
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
        command = common_opts + ['-codec:a', 'copy', str(output_path)]
        action = "copied"
    else:
        command = common_opts + [
            '-codec:a', 'libmp3lame',
            '-ab', f'{bitrate_k}k',
            '-ar', '44100',
            '-ac', '2',
            str(output_path)
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
            if "-n" in command and process.returncode == 1 and "already exists. Exiting." in process.stderr:
                 error_msg = "Output file already exists (Confirmed by FFmpeg -n)"
                 return False, error_msg
            else:
                error_msg = f"FFmpeg error (code {process.returncode}):\n--- FFMPEG STDERR ---\n{process.stderr.strip()}"
                # Check if it failed because no audio stream was found with -map 0:a:0?
                if "Stream map '0:a:0' matches no streams" in process.stderr:
                    error_msg += "\n(Likely reason: No audio stream found in the input video)"
                return False, error_msg
    except Exception as e:
        return False, f"Failed to run ffmpeg command: {e}"

def verify_output(output_path: Path) -> bool:
    """Checks if the output file exists and is not empty."""
    return output_path.exists() and output_path.stat().st_size > 0

def cleanup_failed_output(output_path: Path):
    """Attempts to remove a potentially corrupted output file."""
    if output_path and output_path.exists():
        try:
            output_path.unlink()
        except OSError as e:
            tqdm.write(f"Warning: Could not remove partial file {output_path}: {e}", file=sys.stderr)

def handle_metadata_tagging(
    add_metadata_flag: bool,
    video_path: Path,
    output_path: Path,
    image_data: Optional[bytes],
    image_mime: Optional[str]
) -> str:
    """Handles metadata extraction and tagging, returning the status."""
    if not add_metadata_flag:
        return 'not_attempted'

    filename_stem = video_path.stem
    metadata = extract_metadata_from_filename(filename_stem)

    if not metadata:
        tqdm.write(f"Info: No metadata pattern matched for '{filename_stem}', skipping text tagging.")
        # If no text metadata found, still try adding album art if provided
        if image_data and image_mime:
             if add_metadata_to_mp3(output_path, {}, image_data, image_mime): # Pass empty dict for text meta
                 return 'added' # Consider it 'added' if art was successful
             else:
                 return 'failed' # Art tagging failed
        else:
             return 'skipped' # No text meta matched, no art provided

    # Text metadata found, now add text and art
    if add_metadata_to_mp3(output_path, metadata, image_data, image_mime):
        return 'added'
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
    album_art_data: Optional[bytes], # Pass album art data
    album_art_mime: Optional[str]   # Pass album art MIME type
) -> ConversionResult:
    """Orchestrates the conversion and tagging process for a single video file."""
    output_path: Optional[Path] = None
    metadata_status: str = 'not_attempted'
    action: str = 'failed' # Default action state

    try:
        relative_path = video_path.relative_to(source_base)
        output_path = output_base / relative_path.with_suffix('.mp3')
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        if not overwrite and output_path.exists():
             err_msg = f"Output file {output_path.name} already exists (Internal Pre-Check)"
             tqdm.write(f"Warning: {err_msg} - processing should have been skipped earlier.", file=sys.stderr)
             return 'failed', video_path, err_msg, action, metadata_status

        original_audio_codec = get_audio_codec(video_path, ffprobe_path)
        command, action = build_ffmpeg_command(
            video_path, output_path, ffmpeg_path, original_audio_codec, bitrate_k, overwrite
        )
        success, ffmpeg_error = run_ffmpeg(command)

        if not success:
            cleanup_failed_output(output_path)
            return 'failed', video_path, ffmpeg_error, action, metadata_status

        if not verify_output(output_path):
            error_message = f"FFmpeg OK, but output is invalid/empty: {output_path}"
            cleanup_failed_output(output_path)
            return 'failed', video_path, error_message, action, metadata_status

        metadata_status = handle_metadata_tagging(
            add_metadata_flag, video_path, output_path, album_art_data, album_art_mime
        )

        # If metadata tagging failed, report success for conversion but note the meta failure
        # The status 'success' here refers to the audio conversion part.
        return 'success', video_path, output_path, action, metadata_status

    except Exception as e:
        error_msg = f"Python error processing {video_path.name}: {e}"
        if output_path: cleanup_failed_output(output_path)
        final_meta_status = metadata_status if metadata_status != 'not_attempted' else 'failed'
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
        all_files_list = list(tqdm(all_files_gen, desc="Discovering files", unit="file", leave=False, ncols=100))

        for file_path in tqdm(all_files_list, desc="Filtering videos", unit="file", leave=False, ncols=100):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
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
    """Filters out videos whose corresponding MP3 output already exists."""
    files_to_process: List[Path] = []
    skipped_count: int = 0
    if not overwrite:
        print("Checking for existing output files to skip...")
        for video_path in tqdm(all_video_files, desc="Pre-checking", unit="file", leave=False, ncols=100):
            try:
                relative_path = video_path.relative_to(source_path)
                potential_output_path = output_base / relative_path.with_suffix('.mp3')
                if potential_output_path.exists():
                    skipped_count += 1
                else:
                    files_to_process.append(video_path)
            except ValueError:
                 # This can happen if a file path somehow doesn't start with source_path
                 # Should be rare with rglob but handle defensively.
                 tqdm.write(f"Warning: Skipping file due to path issue: {video_path}", file=sys.stderr)
                 skipped_count += 1 # Treat as skipped

        return files_to_process, skipped_count
    else:
        # If overwriting, process all files
        return all_video_files, 0

# --- Argument Parsing and Validation ---

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert video files to MP3, optionally adding metadata and album art.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("source_folder", help="Path to the folder containing video files (scanned recursively).")
    parser.add_argument("output_folder", help="Path to the folder where MP3 files will be saved.")
    parser.add_argument("-b", "--bitrate", type=int, default=192, help="Audio bitrate (kbps) for re-encoding (if needed).")
    parser.add_argument("-t", "--threads", type=int, default=os.cpu_count() or 1, help="Number of concurrent ffmpeg processes.")
    parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite existing MP3 files.")
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

    if output_path_base.exists() and not output_path_base.is_dir():
         print(f"Error: Output path '{args.output_folder}' exists but is not a directory.", file=sys.stderr)
         sys.exit(1)

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

    image_path = Path(image_path_str)
    if not image_path.is_file():
        tqdm.write(f"Warning: Album art file not found: {image_path}. Skipping album art embedding.", file=sys.stderr)
        return None, None

    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()

        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith('image/'):
            tqdm.write(f"Warning: Could not determine valid image MIME type for {image_path}. Skipping album art.", file=sys.stderr)
            return None, None

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
    album_art_data: Optional[bytes], # Pass loaded data
    album_art_mime: Optional[str],   # Pass loaded mime type
    max_workers: int
) -> Iterator[ConversionResult]:
    """Submits conversion tasks to a ThreadPoolExecutor and yields results."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                convert_single_video,
                video_path, source_path, output_base, bitrate_k,
                ffmpeg_path, ffprobe_path, overwrite, add_metadata_flag,
                album_art_data, album_art_mime # Pass art info to worker
            ): video_path
            for video_path in files_to_process
        }

        pbar = tqdm(as_completed(futures), total=len(files_to_process), desc="Converting", unit="file", ncols=100)
        for future in pbar:
            video_path_orig = futures[future]
            rel_input_str = "Unknown"
            try:
                rel_input = video_path_orig.relative_to(source_path)
                rel_input_str = str(rel_input)
                pbar.set_postfix_str(f"{rel_input.name[:30]}...", refresh=True)

                result: ConversionResult = future.result()
                yield result

            except Exception as exc:
                 tqdm.write(f"FAILED (Future Exception): {rel_input_str}\n  Reason: {exc}", file=sys.stderr)
                 yield 'failed', video_path_orig, f"Future Exception: {exc}", 'failed', 'failed'

# --- Summary Reporting ---

def print_summary(
    total_files_found: int,
    skipped_initially_count: int,
    total_to_process: int,
    success_count: int,
    copied_count: int,
    converted_count: int,
    failed_count: int,
    metadata_added_count: int,
    metadata_skipped_count: int,
    metadata_failed_count: int,
    add_metadata_flag: bool,
    album_art_provided: bool
):
    """Prints the final summary of the conversion process."""
    print("-" * 30)
    print("Conversion Complete.")
    print(f"  Total Files Found:        {total_files_found}")
    print(f"  Skipped (Already Exist):  {skipped_initially_count}")
    print(f"-----------------------------")
    print(f"  Files Attempted:          {total_to_process}")
    print(f"  Successfully Processed:   {success_count} ({copied_count} copied, {converted_count} converted)")
    if add_metadata_flag or album_art_provided: # Show meta stats if either text meta or art was attempted
        print(f"    Metadata Added/Updated:   {metadata_added_count}")
        print(f"    Metadata Skipped (No Match):{metadata_skipped_count}") # Only relevant for text meta
        print(f"    Metadata Failed (Error):  {metadata_failed_count}")
    print(f"  Failed (Conversion/Error):{failed_count}")
    print("-" * 30)

# --- Main Execution ---

def main():
    """Main script execution function."""
    args = parse_arguments()
    source_path, output_path_base, threads, bitrate, add_metadata_flag = validate_args_and_paths(args)

    # Find executables
    ffmpeg_executable = args.ffmpeg or find_executable("ffmpeg")
    ffprobe_executable = args.ffprobe or find_executable("ffprobe")
    if not ffmpeg_executable or not ffprobe_executable:
        sys.exit(1)
    print("-" * 30)

    # Load Album Art (do this once)
    album_art_data, album_art_mime = load_album_art(args.album_art)
    album_art_provided = bool(album_art_data) # Flag if art was successfully loaded
    print("-" * 30)


    # Find and filter files
    all_video_files = find_video_files(source_path)
    total_files_found = len(all_video_files)
    if total_files_found == 0:
        print("No video files found matching extensions:", ', '.join(SUPPORTED_EXTENSIONS))
        sys.exit(0)

    files_to_process, skipped_initially_count = filter_existing_files(
        all_video_files, source_path, output_path_base, args.overwrite
    )
    total_to_process = len(files_to_process)

    if skipped_initially_count > 0:
        print(f"Skipping {skipped_initially_count} files as corresponding MP3s already exist.")
    if total_to_process == 0:
        print("No new files to process.")
        sys.exit(0)
    print("-" * 30)

    # Processing Information
    print(f"Starting conversion for {total_to_process} files with {threads} threads...")
    print(f"Source: {source_path}")
    print(f"Output: {output_path_base}")
    print(f"Re-encode Bitrate: {bitrate} kbps")
    print(f"Overwrite: {'Yes' if args.overwrite else 'No'}")
    print(f"Add Text Metadata: {'Yes' if add_metadata_flag else 'No'}")
    print(f"Add Album Art: {'Yes' if album_art_provided else 'No'}")
    print("-" * 30)


    # Initialize counters
    success_count = copied_count = converted_count = failed_count = 0
    metadata_added_count = metadata_skipped_count = metadata_failed_count = 0

    # Process files
    results_iterator = process_files_concurrently(
        files_to_process, source_path, output_path_base, bitrate,
        ffmpeg_executable, ffprobe_executable, args.overwrite, add_metadata_flag,
        album_art_data, album_art_mime, # Pass loaded art data/mime
        threads
    )

    for status, input_path, output_or_error, action, meta_status in results_iterator:
        # Make sure input_path is Path object before calling relative_to
        rel_input_display = "Unknown Path"
        if isinstance(input_path, Path):
            try:
                rel_input_display = str(input_path.relative_to(source_path))
            except ValueError:
                rel_input_display = str(input_path) # Fallback to absolute path if relative fails


        if status == 'success':
            success_count += 1
            if action == 'copied': copied_count += 1
            elif action == 'converted': converted_count += 1

            # Metadata counts are updated based on the combined text/art tagging result
            if meta_status == 'added': metadata_added_count += 1
            elif meta_status == 'skipped': metadata_skipped_count += 1
            elif meta_status == 'failed': metadata_failed_count += 1

        elif status == 'failed':
            failed_count += 1
            tqdm.write(f"FAILED : {rel_input_display}\n  Reason: {output_or_error}", file=sys.stderr)

    # Print Summary
    print_summary(
        total_files_found, skipped_initially_count, total_to_process,
        success_count, copied_count, converted_count, failed_count,
        metadata_added_count, metadata_skipped_count, metadata_failed_count,
        add_metadata_flag, album_art_provided # Pass flag to summary
    )

    # Determine Exit Code
    final_failed_count = failed_count + metadata_failed_count # Treat metadata errors as failures for exit code

    if final_failed_count > 0:
        print(f"WARNING: {final_failed_count} files encountered errors during conversion or metadata tagging. Check logs.", file=sys.stderr)
        sys.exit(1)
    else:
        print("All tasks completed successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
