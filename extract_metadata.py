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
from pathlib import Path
from PIL import Image

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5vl:7b"
DEFAULT_MAX_WIDTH = 512


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
        print(f"Skipping (exists): {image_path.name}", flush=True)
        return True

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
    print(f"Found {len(images)} images to process\n", flush=True)

    if not images:
        print("No images found.")
        return 0

    success = 0
    failed = 0

    for i, img in enumerate(sorted(images), 1):
        print(f"\n[{i}/{len(images)}] ", end="", flush=True)
        if process_image(img, args.ollama_url, args.model, args.max_width, not args.overwrite):
            success += 1
        else:
            failed += 1

    print(f"\n\nDone! Success: {success}, Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
