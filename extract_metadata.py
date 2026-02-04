#!/usr/bin/env python3
"""
Photo Metadata Extractor

Extracts comprehensive metadata from images using:
- exiftool for EXIF/XMP/IPTC metadata
- Existing JSON/XMP sidecar files
- Vision AI model (qwen2.5vl) for image descriptions

Creates a JSON sidecar file for each image with all extracted metadata
and an AI-generated description.
"""

import argparse
import json
import base64
import requests
import io
import subprocess
import time
from pathlib import Path
from PIL import Image

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5vl:7b"
DEFAULT_MAX_WIDTH = 512


class ProgressTracker:
    """Track progress and calculate ETA."""

    def __init__(self, total: int):
        self.total = total
        self.processed = 0
        self.skipped = 0
        self.failed = 0
        self.start_time = time.time()
        self.last_report_time = self.start_time
        self.last_report_count = 0

    def update(self, success: bool, skipped: bool = False):
        self.processed += 1
        if skipped:
            self.skipped += 1
        elif not success:
            self.failed += 1

    def get_stats(self) -> dict:
        elapsed = time.time() - self.start_time
        rate = self.processed / elapsed if elapsed > 0 else 0
        remaining = self.total - self.processed
        eta_seconds = remaining / rate if rate > 0 else 0

        return {
            "processed": self.processed,
            "total": self.total,
            "percent": (self.processed / self.total * 100) if self.total > 0 else 0,
            "skipped": self.skipped,
            "failed": self.failed,
            "elapsed_seconds": elapsed,
            "rate_per_minute": rate * 60,
            "eta_seconds": eta_seconds,
        }

    def format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def should_report(self, interval: int = 50) -> bool:
        """Check if we should print a progress report."""
        return self.processed % interval == 0 or self.processed == self.total

    def print_progress(self):
        """Print current progress with rate and ETA."""
        stats = self.get_stats()
        eta_str = self.format_time(stats["eta_seconds"])
        elapsed_str = self.format_time(stats["elapsed_seconds"])

        print(f"\n{'='*60}", flush=True)
        print(f"PROGRESS: {stats['processed']:,}/{stats['total']:,} ({stats['percent']:.1f}%)", flush=True)
        print(f"  Rate: {stats['rate_per_minute']:.1f} img/min | Elapsed: {elapsed_str} | ETA: {eta_str}", flush=True)
        print(f"  Skipped: {stats['skipped']:,} | Failed: {stats['failed']:,}", flush=True)
        print(f"{'='*60}\n", flush=True)


def run_exiftool(image_path: str) -> dict:
    """Extract all metadata using exiftool."""
    try:
        result = subprocess.run(
            ['exiftool', '-json', '-a', '-u', '-g1', str(image_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            if data and len(data) > 0:
                return data[0]
    except Exception as e:
        print(f"  exiftool error: {e}", flush=True)
    return {}


def get_sidecar_content(image_path: Path) -> tuple[dict, str]:
    """Get content from JSON or XMP sidecar files. Returns (dict, text_for_prompt)."""
    sidecar_data = {}
    prompt_text = []

    # Check for JSON sidecar (various naming patterns)
    for pattern in [
        image_path.with_suffix('.json'),
        Path(str(image_path) + '.json'),
        image_path.with_name(image_path.stem + '_metadata.json'),
    ]:
        if pattern.exists():
            try:
                with open(pattern, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    sidecar_data['json_sidecar'] = json.loads(content)
                    prompt_text.append(f"JSON metadata: {content[:1500]}")
            except:
                pass
            break

    # Check for XMP sidecar
    for pattern in [
        image_path.with_suffix('.xmp'),
        Path(str(image_path) + '.xmp'),
    ]:
        if pattern.exists():
            try:
                with open(pattern, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    sidecar_data['xmp_sidecar'] = content
                    prompt_text.append(f"XMP metadata: {content[:1500]}")
            except:
                pass
            break

    return sidecar_data, "\n".join(prompt_text)


def load_and_resize_image(image_path: str, max_width: int) -> str:
    """Load image, resize if needed, return base64."""
    with Image.open(image_path) as img:
        if img.mode in ('RGBA', 'P', 'LA', 'L'):
            img = img.convert('RGB')

        if img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_description(image_path: str, sidecar_text: str, ollama_url: str, model: str, max_width: int) -> str:
    """Get 3-sentence description from vision model."""
    try:
        image_data = load_and_resize_image(image_path, max_width)
    except Exception as e:
        return f"ERROR: Could not load image - {type(e).__name__}: {e}"

    if sidecar_text:
        prompt = f"""Here is some metadata about this image:
{sidecar_text}

Based on the image and metadata, describe the main subject of this image in exactly 3 sentences. Be descriptive and specific."""
    else:
        prompt = "Describe the main subject of this image in exactly 3 sentences. Be descriptive and specific."

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_data],
        "stream": False
    }

    try:
        response = requests.post(ollama_url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json().get("response", "").strip()
        return result
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def process_image(image_path: Path, ollama_url: str, model: str, max_width: int, skip_existing: bool) -> bool:
    """Process a single image and create metadata sidecar."""
    output_path = image_path.parent / f"{image_path.name}_metadata.json"

    if skip_existing and output_path.exists():
        return None  # Return None to indicate skipped

    print(f"Processing: {image_path.name}", flush=True)

    metadata = {
        "source_file": str(image_path),
        "filename": image_path.name,
    }

    # Get exiftool metadata
    print(f"  Extracting EXIF...", flush=True)
    exif_data = run_exiftool(image_path)
    if exif_data:
        metadata["exiftool"] = exif_data

    # Get sidecar content
    print(f"  Checking sidecars...", flush=True)
    sidecar_data, sidecar_text = get_sidecar_content(image_path)
    if sidecar_data:
        metadata["sidecars"] = sidecar_data

    # Get AI description
    print(f"  Generating description...", flush=True)
    description = get_description(str(image_path), sidecar_text, ollama_url, model, max_width)
    metadata["ai_description"] = description

    # Write output
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
            f.write("\n")
        print(f"  Saved: {output_path.name}", flush=True)
        return True
    except Exception as e:
        print(f"  ERROR writing: {e}", flush=True)
        return False


def find_images(base_path: str) -> list[Path]:
    """Find all image files recursively."""
    base = Path(base_path)
    extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff', '.tif', '.cr2', '.nef', '.arw', '.dng'}

    all_images = []
    for ext in extensions:
        all_images.extend(base.rglob(f"*{ext}"))
        all_images.extend(base.rglob(f"*{ext.upper()}"))

    return list(set(all_images))


def main():
    parser = argparse.ArgumentParser(
        description='Extract metadata from images and create JSON sidecar files with AI descriptions.'
    )
    parser.add_argument('directory', help='Directory to process (recursive)')
    parser.add_argument('--model', default=DEFAULT_MODEL, help=f'Ollama model to use (default: {DEFAULT_MODEL})')
    parser.add_argument('--ollama-url', default=DEFAULT_OLLAMA_URL, help=f'Ollama API URL (default: {DEFAULT_OLLAMA_URL})')
    parser.add_argument('--max-width', type=int, default=DEFAULT_MAX_WIDTH, help=f'Max image width for AI (default: {DEFAULT_MAX_WIDTH})')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing metadata files')

    args = parser.parse_args()

    if not Path(args.directory).exists():
        print(f"Error: Directory not found: {args.directory}")
        return 1

    images = find_images(args.directory)
    total = len(images)
    print(f"Found {total:,} images to process\n", flush=True)

    if not images:
        print("No images found.")
        return 0

    tracker = ProgressTracker(total)

    for i, img in enumerate(sorted(images), 1):
        print(f"[{i}/{total}] ", end="", flush=True)
        result = process_image(img, args.ollama_url, args.model, args.max_width, not args.overwrite)

        if result is None:
            # Skipped
            print(f"Skipping (exists): {img.name}", flush=True)
            tracker.update(success=True, skipped=True)
        elif result:
            tracker.update(success=True)
        else:
            tracker.update(success=False)

        # Print progress every 50 images
        if tracker.should_report(50):
            tracker.print_progress()

    # Final summary
    stats = tracker.get_stats()
    print(f"\n{'='*60}", flush=True)
    print(f"COMPLETE!", flush=True)
    print(f"  Processed: {stats['processed']:,} images in {tracker.format_time(stats['elapsed_seconds'])}", flush=True)
    print(f"  Success: {stats['processed'] - stats['failed'] - stats['skipped']:,} | Skipped: {stats['skipped']:,} | Failed: {stats['failed']:,}", flush=True)
    print(f"  Average rate: {stats['rate_per_minute']:.1f} img/min", flush=True)
    print(f"{'='*60}", flush=True)

    return 0 if stats['failed'] == 0 else 1


if __name__ == "__main__":
    exit(main())
