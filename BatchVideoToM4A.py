#!/usr/bin/env python3
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
    # Import necessary mutagen modules for MP4 (M4A)
    from mutagen.mp4 import MP4, MP4Cover, MP4Tags, MutagenError
except ImportError:
    print("Error: 'mutagen' library not found. Please install it: pip install mutagen", file=sys.stderr)
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Error: 'tqdm' library not found. Please install it: pip install tqdm", file=sys.stderr)
    # Provide a dummy tqdm class if not found
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
OUTPUT_EXTENSION: str = '.m4a'
TARGET_CONTAINER_FORMAT: str = 'mp4' # M4A is an MP4 container

# --- Codecs ---
# Codecs that can be directly copied into an M4A/MP4 container
COPYABLE_CODECS: Tuple[str, ...] = ('aac', 'mp3')
# Target codec for conversion when the original cannot be copied
TARGET_AUDIO_CODEC_CONVERT: str = 'libfdk_aac' # High-quality AAC encoder
# Note about libfdk_aac requirement
LIBFDK_AAC_NOTE = f"NOTE: Conversion requires FFmpeg to be compiled with '--enable-libfdk-aac'. If not available, FFmpeg will error during conversion. Streams with codecs {COPYABLE_CODECS} will be copied directly."

# --- Filename Parsing ---
FILENAME_METADATA_REGEX: re.Pattern = re.compile(
    r"^\[([^\]]+)\]\[(\d{4}-\d{2}-\d{2})\]\[(.+?)\](?:\[(\d{4}-\d{2}-\d{2})\])?$"
)
# MP4 picture type constants
MP4_COVER_FORMAT_JPEG = 13
MP4_COVER_FORMAT_PNG = 14

# --- Temporary Files & Error Handling ---
TEMP_SUFFIX = ".tmp_audio_conv" # Use a more specific suffix
NO_AUDIO_STREAM_ERRORS = (
    "Stream map '0:a:0' matches no streams",
    "Output file #0 does not contain any stream",
    "does not contain any stream",
    "could not find codec parameters"
)
IGNORE_FOLDERS: List[str] = [
    "[MissWarmJ]",
    # Add more folder names here as needed
]

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
        tqdm.write(f"\nERROR: {name} not found in system PATH.", file=sys.stderr)
        tqdm.write(f"Please install FFmpeg (which includes {name}) and ensure it's added to your PATH.", file=sys.stderr)
        tqdm.write("Download from: https://ffmpeg.org/download.html", file=sys.stderr)
        if name == "ffmpeg":
            tqdm.write(LIBFDK_AAC_NOTE, file=sys.stderr)
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
            codec_name = data['streams'][0].get('codec_name')
            if not codec_name:
                return "unknown_codec_in_stream" # Codec name missing but stream exists

            codec_name_lower = codec_name.lower()
            # Standardize common AAC variations
            if 'aac' in codec_name_lower:
                return 'aac'
            # Standardize MP3 variations (less common, but just in case)
            if codec_name_lower in ['mp3', 'mp3float', 'mp3adu', 'mp3on4']: # Check common ffprobe names
                return 'mp3'
            # Return other codecs as detected
            return codec_name_lower
        return "no_audio_stream_found_by_ffprobe"
    except FileNotFoundError:
        tqdm.write(f"\nERROR: ffprobe executable not found at '{ffprobe_path}'. Cannot check audio codec.", file=sys.stderr)
        return None
    except subprocess.CalledProcessError as e:
        stderr_lower = e.stderr.lower()
        if any(err in stderr_lower for err in NO_AUDIO_STREAM_ERRORS):
            return "no_audio_stream_found_by_ffprobe"
        tqdm.write(f"Warning: Could not get audio codec for {video_path.name} (will attempt conversion). FFprobe Error: {e.stderr}", file=sys.stderr)
        return None # Treat as needing conversion if ffprobe fails unexpectedly
    except (json.JSONDecodeError, KeyError, IndexError, Exception) as e:
        tqdm.write(f"Warning: Could not parse ffprobe output for {video_path.name} (will attempt conversion). Error: {e}", file=sys.stderr)
        return None # Treat as needing conversion

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

def get_mp4_cover_format(mime_type: str) -> Optional[int]:
    """Maps MIME type to MP4Cover format constant."""
    mime_type = mime_type.lower()
    if mime_type == 'image/jpeg':
        return MP4_COVER_FORMAT_JPEG
    elif mime_type == 'image/png':
        return MP4_COVER_FORMAT_PNG
    else:
        return None

def add_metadata_to_m4a(
    m4a_path: Path,
    metadata: Dict[str, str],
    image_data: Optional[bytes],
    image_mime: Optional[str]
) -> bool:
    """Adds MP4 metadata tags (text and optional album art) to the M4A file."""
    try:
        audio = MP4(m4a_path)
        if audio.tags is None:
            audio.tags = MP4Tags()

        # Add Text Metadata
        if metadata.get('artist'):
            audio.tags['©ART'] = [metadata['artist']]
        if metadata.get('title'):
            audio.tags['©nam'] = [metadata['title']]
        if metadata.get('date'):
            date_str = metadata['date']
            if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
                audio.tags['©day'] = [date_str]
            else:
                tqdm.write(f"Warning: Skipping invalid date format '{date_str}' for {m4a_path.name}. Expected YYYY-MM-DD.", file=sys.stderr)

        comment_text = "本音频由ACGDB转换格式并编码。欢迎访问 https://acgdb.de/ 获取更多ACGDB资源。\n\nThis audio was converted and encoded by ACGDB. Visit https://acgdb.de/ for more ACGDB resources."
        audio.tags['©cmt'] = [comment_text]

        # Add Album Art
        if image_data and image_mime:
            cover_format = get_mp4_cover_format(image_mime)
            if cover_format:
                mp4_cover = MP4Cover(image_data, imageformat=cover_format)
                audio.tags['covr'] = [mp4_cover]
            else:
                tqdm.write(f"Warning: Unsupported image MIME type '{image_mime}' for {m4a_path.name}. Skipping art.", file=sys.stderr)
                if 'covr' in audio.tags: del audio.tags['covr']
        else:
            if 'covr' in audio.tags: del audio.tags['covr']

        audio.save()
        return True
    except (MutagenError, Exception) as e:
        tqdm.write(f"Error: Failed to add metadata/art to {m4a_path.name}. Reason: {e}", file=sys.stderr)
        # Attempt to clean up potentially corrupted tags if saving failed
        try:
            # Re-open and save with minimal changes or default tags? Might be risky.
            # Safer: just report failure.
            pass
        except Exception as e_cleanup:
            tqdm.write(f"  Additional error during metadata error handling: {e_cleanup}", file=sys.stderr)
        return False


def build_ffmpeg_command(
    video_path: Path,
    temp_output_path: Path,
    ffmpeg_path: str,
    original_audio_codec: Optional[str],
    vbr_quality: int,
    output_format: str = TARGET_CONTAINER_FORMAT
) -> Tuple[List[str], str]:
    """Constructs the ffmpeg command to copy or convert the audio stream."""
    common_opts: List[str] = [
        ffmpeg_path,
        '-y', # Overwrite temp file
        '-i', str(video_path),
        '-vn', # No video
        '-loglevel', 'error',
        '-hide_banner',
        '-map_metadata', '-1', # Strip original metadata
        '-map', '0:a:0?', # Map first audio stream, optional
    ]

    action: str
    command: List[str]

    # Decide whether to copy or convert
    if original_audio_codec and original_audio_codec in COPYABLE_CODECS:
        # Copy AAC or MP3 stream
        command = common_opts + [
            '-codec:a', 'copy',
            '-f', output_format, # Enforce MP4 container for M4A
            str(temp_output_path)
        ]
        action = f"copied ({original_audio_codec.upper()})"
    else:
        # Convert to AAC using libfdk_aac
        codec_to_convert = original_audio_codec if original_audio_codec else "unknown"
        command = common_opts + [
            '-codec:a', TARGET_AUDIO_CODEC_CONVERT,
            '-vbr', str(vbr_quality),
            '-ar', '44100', # Consider making this configurable or adaptive?
            '-ac', '2',
            '-f', output_format,
            str(temp_output_path)
        ]
        action = f"converted ({codec_to_convert} -> AAC VBR {vbr_quality})"

    return command, action

def run_ffmpeg(command: List[str]) -> FFmpegResult:
    """Executes the ffmpeg command."""
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=False, encoding='utf-8', errors='replace')
        if process.returncode == 0:
            return True, None
        else:
            error_msg = f"FFmpeg error (code {process.returncode}):\n--- FFMPEG STDERR ---\n{process.stderr.strip()}"
            stderr_lower = process.stderr.lower()
            is_no_audio_error = any(err_msg.lower() in stderr_lower for err_msg in NO_AUDIO_STREAM_ERRORS)

            if TARGET_AUDIO_CODEC_CONVERT in command and "Unknown encoder 'libfdk_aac'" in process.stderr:
                error_msg += f"\n\n>>> CRITICAL: FFmpeg not compiled with {TARGET_AUDIO_CODEC_CONVERT}! <<<\n{LIBFDK_AAC_NOTE}"
            elif is_no_audio_error:
                error_msg += "\n(Likely reason: No audio stream found in the input video)"
            # Check if copying failed due to incompatibility (less common for AAC/MP3 in MP4, but possible)
            elif 'copy' in command and ('codec not supported in mp4' in stderr_lower or 'incompatible stream' in stderr_lower):
                error_msg += f"\n(Likely reason: Original codec '{command[command.index('-codec:a')+1]}' is not directly copyable to MP4 container despite initial check.)"

            return False, error_msg
    except FileNotFoundError:
        return False, f"Failed to run ffmpeg: Executable not found at '{command[0]}'. {LIBFDK_AAC_NOTE if TARGET_AUDIO_CODEC_CONVERT in command else ''}"
    except Exception as e:
        return False, f"Failed to run ffmpeg command: {e}"

def verify_output(output_path: Path) -> bool:
    """Checks if the output file exists and is not empty."""
    return output_path.exists() and output_path.stat().st_size > 0

def cleanup_temp_file(temp_output_path: Path):
    """Attempts to remove the temporary output file."""
    if temp_output_path and temp_output_path.exists() and temp_output_path.name.endswith(OUTPUT_EXTENSION + TEMP_SUFFIX):
        try:
            temp_output_path.unlink()
            # tqdm.write(f"Debug: Cleaned up temp file {temp_output_path}", file=sys.stderr) # Optional debug msg
        except OSError as e:
            tqdm.write(f"Warning: Could not remove temporary file {temp_output_path}: {e}", file=sys.stderr)

def handle_metadata_tagging(
    add_metadata_flag: bool,
    video_path: Path,
    temp_audio_path: Path,
    image_data: Optional[bytes],
    image_mime: Optional[str]
) -> str:
    """Handles metadata extraction and tagging on the temp file."""
    if not add_metadata_flag and not (image_data and image_mime):
        return 'not_attempted'
    if not temp_audio_path.exists():
        tqdm.write(f"Error: Temp file {temp_audio_path.name} not found before metadata tagging.", file=sys.stderr)
        return 'failed'

    filename_stem = video_path.stem
    metadata = extract_metadata_from_filename(filename_stem) or {}

    if not metadata and not (image_data and image_mime):
        return 'skipped'

    if not metadata and add_metadata_flag:
        # Only warn if not in an ignored folder to reduce noise
        if not any(part.lower() in {f.lower() for f in IGNORE_FOLDERS} for part in video_path.parts):
            tqdm.write(f"Info: No metadata pattern matched for '{filename_stem}', skipping text tagging.")

    if add_metadata_to_m4a(temp_audio_path, metadata, image_data, image_mime):
        if metadata or (image_data and image_mime):
            return 'added'
        else:
            return 'skipped' # Nothing was actually added
    else:
        return 'failed'

# --- Main Worker Function ---

def convert_single_video(
    video_path: Path,
    source_base: Path,
    output_base: Path,
    vbr_quality: int,
    ffmpeg_path: str,
    ffprobe_path: str,
    overwrite: bool,
    add_metadata_flag: bool,
    album_art_data: Optional[bytes],
    album_art_mime: Optional[str]
) -> ConversionResult:
    """Orchestrates the conversion/copy and tagging process for a single video file."""
    output_path: Optional[Path] = None
    temp_output_path: Optional[Path] = None
    metadata_status: str = 'not_attempted'
    action: str = 'pending' # Initial action state

    try:
        # Path Setup
        relative_path = video_path.relative_to(source_base)
        output_path = output_base / relative_path.with_suffix(OUTPUT_EXTENSION)
        # Ensure temp suffix is correct
        temp_output_path = output_path.with_name(output_path.name + TEMP_SUFFIX)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Pre-flight Check (Final Destination)
        if not overwrite and output_path.exists():
            return 'skipped', video_path, output_path, 'skipped (exists)', 'not_attempted'

        # Determine Audio Codec
        original_audio_codec = get_audio_codec(video_path, ffprobe_path)
        if original_audio_codec == "no_audio_stream_found_by_ffprobe":
            return 'skipped_no_audio', video_path, "No audio stream detected by ffprobe", 'skipped (no audio)', 'not_attempted'
        elif original_audio_codec is None: # FFprobe failed unexpectedly
            tqdm.write(f"Warning: Proceeding with conversion attempt for {video_path.name} despite ffprobe error.", file=sys.stderr)
            # Fall through to attempt conversion

        # Build and Run FFmpeg Command (to Temp File)
        command, action = build_ffmpeg_command(
            video_path, temp_output_path, ffmpeg_path, original_audio_codec, vbr_quality
        )
        success, ffmpeg_error = run_ffmpeg(command)

        # Handle FFmpeg Result
        if not success:
            cleanup_temp_file(temp_output_path)
            if ffmpeg_error and any(err_msg.lower() in ffmpeg_error.lower() for err_msg in NO_AUDIO_STREAM_ERRORS):
                return 'skipped_no_audio', video_path, "No audio stream found by FFmpeg", action, metadata_status
            elif ffmpeg_error and "Unknown encoder 'libfdk_aac'" in ffmpeg_error:
                return 'failed', video_path, f"FFmpeg Configuration Error: {ffmpeg_error}", action, metadata_status
            else:
                # Other FFmpeg errors (could include copy failures)
                return 'failed', video_path, ffmpeg_error or "Unknown FFmpeg error", action, metadata_status

        # Verify Temporary Output
        if not verify_output(temp_output_path):
            error_message = f"FFmpeg OK, but temp output is invalid/empty: {temp_output_path.name}"
            cleanup_temp_file(temp_output_path)
            return 'failed', video_path, error_message, action, metadata_status

        # Handle Metadata Tagging (on Temp File)
        if add_metadata_flag or (album_art_data and album_art_mime):
            metadata_status = handle_metadata_tagging(
                add_metadata_flag, video_path, temp_output_path, album_art_data, album_art_mime
            )
            if metadata_status == 'failed':
                error_message = f"Metadata tagging failed for {temp_output_path.name}"
                cleanup_temp_file(temp_output_path)
                return 'failed', video_path, error_message, action, metadata_status # Tagging failure is a processing failure
        else:
            metadata_status = 'not_attempted'

        # Final Rename
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
                    # This case should have been caught earlier, but double-check
                    error_message = f"Final output file {output_path.name} exists, and overwrite is False. Skipping."
                    cleanup_temp_file(temp_output_path)
                    return 'skipped', video_path, error_message, action, metadata_status

            # Perform the rename
            shutil.move(str(temp_output_path), str(output_path)) # Use shutil.move for cross-filesystem safety

            # Final verification after rename
            if not verify_output(output_path):
                error_message = f"Rename seemed OK, but final file is invalid/empty: {output_path.name}"
                # Try to clean up the bad final file if possible
                if output_path.exists():
                    try: output_path.unlink()
                    except OSError: pass
                return 'failed', video_path, error_message, action, metadata_status

            return 'success', video_path, output_path, action, metadata_status

        except OSError as e:
            error_msg = f"Failed to move temp file {temp_output_path.name} to {output_path.name}: {e}"
            cleanup_temp_file(temp_output_path) # Clean up source
            # If overwrite was true, the target might exist partially, attempt cleanup
            if overwrite and output_path.exists():
                try: output_path.unlink()
                except OSError: pass
            return 'failed', video_path, error_msg, action, 'failed' # Metadata irrelevant if rename fails

    except Exception as e:
        # General exception handler
        if temp_output_path: cleanup_temp_file(temp_output_path)
        final_meta_status = metadata_status if metadata_status in ['failed', 'added', 'skipped'] else 'failed'
        import traceback
        error_msg = f"Unexpected Python error processing {video_path.name}: {e}\n{traceback.format_exc()}"
        return 'failed', video_path, error_msg, action if action != 'pending' else 'failed', final_meta_status


# --- File Discovery and Filtering ---

def find_video_files(source_folder: Path) -> List[Path]:
    """Finds all supported video files recursively, skipping ignored folders."""
    video_files: List[Path] = []
    print(f"Scanning for video files in: {source_folder}")
    if IGNORE_FOLDERS:
        # Prepare lowercase version for efficient checking
        ignore_folders_lower = {f.lower() for f in IGNORE_FOLDERS}
        print(f"Ignoring folders (case-insensitive): {', '.join(IGNORE_FOLDERS)}")
    else:
        ignore_folders_lower = set()

    try:
        # Efficiently find all files first, then filter
        all_files_gen = source_folder.rglob("*")
        try:
            all_files_list = list(tqdm(all_files_gen, desc="Discovering files", unit="file", leave=False, ncols=100, miniters=100, mininterval=0.1))
        except TypeError:
            all_files_list = list(all_files_gen)
            print("Found a large number of files, discovery progress bar may be inaccurate.")

        print(f"Filtering videos (ignoring specified folders)...")
        ignored_count = 0
        for file_path in tqdm(all_files_list, desc="Filtering videos", unit="file", leave=False, ncols=100):
            if file_path.is_file():
                # --- Ignore Folder Check ---
                should_ignore = False
                if ignore_folders_lower:
                    try:
                        relative_path = file_path.relative_to(source_folder)
                        # Check if any directory component in the relative path is in the ignore list
                        for part in relative_path.parts[:-1]: # Check only directory parts, not the filename itself
                            if part.lower() in ignore_folders_lower:
                                should_ignore = True
                                break
                    except ValueError:
                        # Should not happen if file_path is within source_folder from rglob
                        tqdm.write(f"Warning: Could not get relative path for {file_path}, skipping ignore check.", file=sys.stderr)

                if should_ignore:
                    ignored_count += 1
                    continue # Skip this file

                # --- Standard Checks (Extension, Temp Suffix) ---
                if file_path.suffix.lower() in SUPPORTED_EXTENSIONS and not file_path.name.endswith(TEMP_SUFFIX):
                    video_files.append(file_path)

    except Exception as e:
        print(f"\nError during file scan: {e}", file=sys.stderr)

    # Report ignored count only if some were ignored, keep it separate from final stats
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
    """Filters out videos whose corresponding FINAL M4A output already exists."""
    if overwrite:
        return all_video_files, 0 # No filtering needed if overwriting

    files_to_process: List[Path] = []
    skipped_count: int = 0
    print(f"Checking for existing final output files ({OUTPUT_EXTENSION}) to skip...")
    for video_path in tqdm(all_video_files, desc="Pre-checking", unit="file", leave=False, ncols=100):
        try:
            relative_path = video_path.relative_to(source_path)
            potential_output_path = output_base / relative_path.with_suffix(OUTPUT_EXTENSION)
            if potential_output_path.exists() and potential_output_path.stat().st_size > 0: # Check size > 0 too
                skipped_count += 1
            else:
                files_to_process.append(video_path)
        except ValueError:
            tqdm.write(f"Warning: Skipping file due to path issue during pre-check: {video_path}", file=sys.stderr)
            skipped_count += 1 # Count as skipped

    return files_to_process, skipped_count

# --- Argument Parsing and Validation ---

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description=f"Convert video files to {OUTPUT_EXTENSION}. Copies existing AAC/MP3 streams, converts others to {TARGET_AUDIO_CODEC_CONVERT} AAC VBR. Optionally adds metadata and album art. Uses temporary files for safety. Skips videos with no audio stream and ignores specified folders. {LIBFDK_AAC_NOTE}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("source_folder", help="Path to the folder containing video files (scanned recursively).")
    parser.add_argument("output_folder", help=f"Path to the folder where {OUTPUT_EXTENSION} files will be saved.")
    parser.add_argument("-q", "--vbr-quality", type=int, default=4, choices=range(1, 6), metavar='[1-5]',
                        help=f"AAC VBR mode for {TARGET_AUDIO_CODEC_CONVERT} (1=low/small, 5=high/large). Used ONLY when re-encoding.")
    parser.add_argument("-t", "--threads", type=int, default=os.cpu_count() or 1, help="Number of concurrent ffmpeg processes.")
    parser.add_argument("-o", "--overwrite", action="store_true", help=f"Overwrite existing FINAL {OUTPUT_EXTENSION} files in the output folder.")
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
        tqdm.write(f"Warning: Invalid number of threads ({threads}). Using 1 thread.", file=sys.stderr)
        threads = 1

    vbr_quality = args.vbr_quality
    if not 1 <= vbr_quality <= 5:
        tqdm.write(f"Warning: Invalid AAC VBR quality ({vbr_quality}). Using default 4.", file=sys.stderr)
        vbr_quality = 4

    add_metadata_flag = not args.no_metadata

    global IGNORE_FOLDERS
    if args.ignore_folder:
        # Use user-provided list exclusively if given
        IGNORE_FOLDERS = args.ignore_folder
        print(f"Using specified ignore folders: {', '.join(IGNORE_FOLDERS)}")
    elif not IGNORE_FOLDERS:
        print("No default or user-specified folders to ignore.")


    return source_path, output_path_base, threads, vbr_quality, add_metadata_flag

def load_album_art(image_path_str: Optional[str]) -> Tuple[Optional[bytes], Optional[str]]:
    """Loads album art data and determines MIME type."""
    if not image_path_str:
        return None, None

    image_path = Path(image_path_str).resolve()
    if not image_path.is_file():
        tqdm.write(f"Warning: Album art file not found: {image_path}. Skipping album art embedding.", file=sys.stderr)
        return None, None

    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()

        mime_type, _ = mimetypes.guess_type(str(image_path))
        supported_mimes = ('image/jpeg', 'image/png')

        if mime_type and mime_type.lower() in supported_mimes:
            print(f"Album art loaded: {image_path} (Type: {mime_type})")
            return image_data, mime_type
        else:
            guessed_type = f"'{mime_type}'" if mime_type else "undetermined"
            tqdm.write(f"Warning: Album art file {image_path.name} has unsupported or {guessed_type} MIME type for M4A/MP4 covers (only JPEG/PNG allowed). Skipping album art.", file=sys.stderr)
            return None, None

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
    vbr_quality: int,
    ffmpeg_path: str,
    ffprobe_path: str,
    overwrite: bool,
    add_metadata_flag: bool,
    album_art_data: Optional[bytes],
    album_art_mime: Optional[str],
    max_workers: int
) -> Iterator[ConversionResult]:
    """Submits tasks to a ThreadPoolExecutor and yields results."""
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='AudioWorker') as executor:
        futures = {
            executor.submit(
                convert_single_video,
                video_path, source_path, output_base, vbr_quality,
                ffmpeg_path, ffprobe_path, overwrite, add_metadata_flag,
                album_art_data, album_art_mime
            ): video_path
            for video_path in files_to_process
        }

        # Use tqdm for progress visualization
        progress_bar = tqdm(as_completed(futures), total=len(files_to_process), desc="Processing", unit="file", ncols=100, smoothing=0.05)
        for future in progress_bar:
            video_path_orig = futures[future]
            rel_input_str = "Unknown"
            try:
                # Get relative path for better logging, fallback to full path
                try:
                    rel_input = video_path_orig.relative_to(source_path)
                    rel_input_str = str(rel_input)
                except ValueError:
                    rel_input_str = str(video_path_orig)

                result: ConversionResult = future.result()
                # Optionally update progress bar postfix with last action
                # progress_bar.set_postfix_str(f"{result[0]} ({result[3]})", refresh=True)
                yield result

            except Exception as exc:
                # Critical error within a worker task itself
                tqdm.write(f"\nCRITICAL WORKER ERROR processing: {rel_input_str}\n  Reason: {exc}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)

                # Attempt to cleanup potential temp file for this failed task
                try:
                    relative_path = video_path_orig.relative_to(source_path)
                    output_path = output_base / relative_path.with_suffix(OUTPUT_EXTENSION)
                    temp_output_path = output_path.with_name(output_path.name + TEMP_SUFFIX)
                    cleanup_temp_file(temp_output_path)
                except Exception as cleanup_err:
                    tqdm.write(f"  Cleanup attempt failed during critical error handling for {rel_input_str}: {cleanup_err}", file=sys.stderr)

                # Yield a failure result
                yield 'failed', video_path_orig, f"Critical Worker Exception: {exc}", 'failed', 'failed'


# --- Summary Reporting ---

def print_summary(
    total_files_found: int,
    skipped_initially_count: int,
    skipped_no_audio_count: int,
    total_submitted_for_process: int,
    success_count: int,
    copied_aac_count: int, # Specific counts
    copied_mp3_count: int, # Specific counts
    converted_count: int,
    failed_count: int,
    skipped_during_process_count: int,
    metadata_added_count: int,
    metadata_skipped_count: int,
    metadata_failed_count: int,
    add_metadata_flag: bool,
    album_art_provided: bool
):
    """Prints the final summary of the conversion process."""
    print("\n" + "=" * 35)
    print(" Processing Summary ".center(35, "="))
    print(f"Video Files Found (excl. ignored): {total_files_found}")
    print(f"Skipped (Output {OUTPUT_EXTENSION} Existed):   {skipped_initially_count}")
    print(f"Skipped (No Audio Stream):         {skipped_no_audio_count}")
    print(f"-----------------------------------")
    print(f"Files Submitted for Processing:    {total_submitted_for_process}")
    copied_total = copied_aac_count + copied_mp3_count
    print(f"  Successfully Processed ({OUTPUT_EXTENSION}):    {success_count}")
    print(f"    Actions: {copied_aac_count} copied AAC, {copied_mp3_count} copied MP3, {converted_count} converted to AAC")
    if add_metadata_flag or album_art_provided:
        print(f"    Metadata Added/Updated:        {metadata_added_count}")
        print(f"    Metadata Skipped (No Match):   {metadata_skipped_count}")
        print(f"    Metadata Tagging Failed:       {metadata_failed_count}")
    print(f"  Skipped during processing:       {skipped_during_process_count}") # e.g., overwrite=False and file appeared
    print(f"  Failed (Error/Convert/Tag):      {failed_count}")
    print("=" * 35)

# --- Startup Cleanup ---
def initial_cleanup(output_base: Path):
    """Recursively removes leftover temporary files from the output directory."""
    temp_pattern = f"*{OUTPUT_EXTENSION}{TEMP_SUFFIX}" # e.g., *.m4a.tmp_audio_conv
    print(f"Performing startup cleanup of temporary files ({temp_pattern}) in {output_base}...")
    cleanup_count = 0
    try:
        # Use rglob which is recursive
        tmp_files = list(output_base.rglob(temp_pattern))
        if not tmp_files:
            print("No leftover temporary files found.")
            return

        for tmp_file in tqdm(tmp_files, desc="Cleaning up temp files", unit="file", leave=False, ncols=100):
            if tmp_file.is_file():
                try:
                    tmp_file.unlink()
                    cleanup_count += 1
                except OSError as e:
                    tqdm.write(f"Warning: Could not remove temporary file {tmp_file}: {e}", file=sys.stderr)
                except Exception as e:
                    tqdm.write(f"Warning: Error removing temp file {tmp_file}: {e}", file=sys.stderr)

    except Exception as e:
        # Error during the scan itself
        tqdm.write(f"Error during startup cleanup scan in {output_base}: {e}", file=sys.stderr)

    print(f"Startup cleanup complete. Removed {cleanup_count} temporary files.")
    print("-" * 30)


# --- Main Execution ---

def main():
    """Main script execution function."""
    args = parse_arguments()
    source_path, output_path_base, threads, vbr_quality, add_metadata_flag = validate_args_and_paths(args)

    initial_cleanup(output_path_base) # Cleanup before finding executables

    ffmpeg_executable = args.ffmpeg or find_executable("ffmpeg")
    ffprobe_executable = args.ffprobe or find_executable("ffprobe")
    if not ffmpeg_executable or not ffprobe_executable:
        sys.exit(1)
    print("-" * 30)
    print(LIBFDK_AAC_NOTE) # Remind about potential conversion requirement
    print("-" * 30)

    album_art_data, album_art_mime = load_album_art(args.album_art)
    album_art_provided = bool(album_art_data)
    print("-" * 30)

    all_video_files = find_video_files(source_path)
    total_files_found = len(all_video_files)
    if total_files_found == 0:
        print("No video files found matching extensions or criteria.", ', '.join(SUPPORTED_EXTENSIONS))
        sys.exit(0)

    files_to_process, skipped_initially_count = filter_existing_files(
        all_video_files, source_path, output_path_base, args.overwrite
    )
    actual_submitted_count = len(files_to_process)

    if skipped_initially_count > 0:
        print(f"Skipped {skipped_initially_count} files as corresponding final {OUTPUT_EXTENSION} files already exist (use -o to overwrite).")

    if actual_submitted_count == 0:
        print(f"No new files to process (found {total_files_found}, but all skipped or already exist as {OUTPUT_EXTENSION}).")
        sys.exit(0)

    print("-" * 30)
    print(f"Starting processing for {actual_submitted_count} files -> {OUTPUT_EXTENSION} using {threads} threads...")
    print(f"Source:      {source_path}")
    print(f"Output:      {output_path_base}")
    print(f"Copy Codecs: {COPYABLE_CODECS}")
    print(f"Convert To:  {TARGET_AUDIO_CODEC_CONVERT} VBR {vbr_quality} (when needed)")
    print(f"Overwrite:   {'Yes' if args.overwrite else 'No'}")
    print(f"Add Text Meta:{'Yes' if add_metadata_flag else 'No'}")
    print(f"Add Album Art:{'Yes' if album_art_provided else 'No'}")
    if IGNORE_FOLDERS:
        print(f"Ignoring:    {', '.join(IGNORE_FOLDERS)}")
    print("-" * 30)

    # Initialize counters
    success_count = copied_aac_count = copied_mp3_count = converted_count = failed_count = 0
    skipped_during_process_count = 0
    skipped_no_audio_count = 0
    metadata_added_count = metadata_skipped_count = metadata_failed_count = 0
    # Store detailed failure reasons if needed later
    failure_details = []

    results_iterator = process_files_concurrently(
        files_to_process, source_path, output_path_base, vbr_quality,
        ffmpeg_executable, ffprobe_executable, args.overwrite, add_metadata_flag,
        album_art_data, album_art_mime,
        threads
    )

    # Process results as they complete
    for status, input_path, output_or_error, action, meta_status in results_iterator:
        rel_input_display = "Unknown Path"
        if isinstance(input_path, Path):
            try:
                rel_input_display = str(input_path.relative_to(source_path))
            except ValueError:
                rel_input_display = str(input_path)

        if status == 'success':
            success_count += 1
            # Increment specific copy/convert counts based on action string
            if 'copied (AAC)' in action:
                copied_aac_count += 1
            elif 'copied (MP3)' in action:
                copied_mp3_count += 1
            elif 'converted' in action:
                converted_count += 1
            # Handle metadata status
            if meta_status == 'added': metadata_added_count += 1
            elif meta_status == 'skipped': metadata_skipped_count += 1
            # Metadata failure during a successful ffmpeg run is counted here but also below if tagging caused overall failure
            elif meta_status == 'failed':
                metadata_failed_count +=1
                # Note: If handle_metadata_tagging returns 'failed', the overall status in convert_single_video becomes 'failed'
                # So this case might only be reached if logic changes. Let's keep it for robustness.

        elif status == 'failed':
            failed_count += 1
            tqdm.write(f"\nFAILED : {rel_input_display} (Action: {action})\n  Reason: {output_or_error}", file=sys.stderr)
            failure_details.append((rel_input_display, action, output_or_error))
            if meta_status == 'failed': # If metadata tagging was the point of failure
                metadata_failed_count += 1

        elif status == 'skipped': # Skipped during processing (e.g., existed check inside worker)
            skipped_during_process_count += 1
            # Don't print noise unless the reason is unusual
            if 'exists' not in str(output_or_error):
                tqdm.write(f"INFO: Skipped during processing: {rel_input_display} - Reason: {output_or_error}", file=sys.stderr)

        elif status == 'skipped_no_audio':
            skipped_no_audio_count += 1
            # Optionally print these, or just summarize
            # tqdm.write(f"INFO: Skipped (No Audio): {rel_input_display} - Reason: {output_or_error}", file=sys.stdout)

    # Final Summary
    print_summary(
        total_files_found,
        skipped_initially_count,
        skipped_no_audio_count,
        actual_submitted_count,
        success_count, copied_aac_count, copied_mp3_count, converted_count, failed_count,
        skipped_during_process_count,
        metadata_added_count, metadata_skipped_count, metadata_failed_count,
        add_metadata_flag, album_art_provided
    )

    # Exit Status
    if failed_count > 0:
        print(f"\nWARNING: {failed_count} files encountered errors during processing. Check logs above.", file=sys.stderr)
        # Check specifically for the libfdk_aac missing error among failures
        if any("Unknown encoder 'libfdk_aac'" in str(details[2]) for details in failure_details):
            print(f">>> It seems FFmpeg might not be compiled with {TARGET_AUDIO_CODEC_CONVERT}. {LIBFDK_AAC_NOTE} <<<", file=sys.stderr)
        sys.exit(1)
    elif total_files_found == 0:
        print("\nNo processable video files found in the specified source (considering ignored folders).")
        sys.exit(0)
    elif actual_submitted_count == 0:
        print(f"\nProcessing complete. All found files were skipped or already existed as {OUTPUT_EXTENSION}.")
        sys.exit(0)
    elif success_count == 0 and skipped_no_audio_count > 0 and failed_count == 0 and skipped_during_process_count == 0:
        print("\nProcessing complete. All submitted files were skipped due to lacking audio streams.")
        sys.exit(0)
    else:
        # Includes cases where some succeeded, some were skipped (no audio)
        print("\nAll tasks completed.")
        sys.exit(0)

def close_tqdm_bars() -> None:
    """Close any live tqdm instances and restore the cursor."""
    try:
        from tqdm import tqdm as _tqdm_mod
        for bar in list(_tqdm_mod._instances):
            try:
                bar.close()
            except Exception:
                pass
        sys.stderr.write("\033[?25h")      # show cursor
        sys.stderr.flush()
    except Exception:
        pass  # tqdm not imported or other issue

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        close_tqdm_bars()
        print("\n\nProcess interrupted by user (Ctrl+C). Temporary files may remain. Run again to clean up.", file=sys.stderr)
        sys.exit(130) # Standard exit code for SIGINT
