# Photo Metadata Extractor

Extracts comprehensive metadata from images and creates JSON sidecar files with AI-generated descriptions.

## Features

- **EXIF/XMP/IPTC extraction** via exiftool
- **Existing sidecar support** - reads JSON and XMP sidecars for additional context
- **AI descriptions** - generates 3-sentence descriptions using local vision models (Ollama)
- **Incremental processing** - skips already-processed files by default
- **RAW support** - handles CR2, NEF, ARW, DNG and other raw formats

## Requirements

- Python 3.8+
- [exiftool](https://exiftool.org/) installed and in PATH
- [Ollama](https://ollama.ai/) running locally with a vision model

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Basic usage
python extract_metadata.py /path/to/photos

# Use a different model
python extract_metadata.py /path/to/photos --model qwen2.5vl:32b

# Overwrite existing metadata files
python extract_metadata.py /path/to/photos --overwrite

# Custom Ollama URL
python extract_metadata.py /path/to/photos --ollama-url http://192.168.1.100:11434/api/generate
```

## Output

For each image `photo.jpg`, creates `photo.jpg_metadata.json` containing:

```json
{
  "source_file": "/path/to/photo.jpg",
  "filename": "photo.jpg",
  "exiftool": {
    "ExifIFD": {
      "Make": "Canon",
      "Model": "Canon EOS 7D",
      "ExposureTime": "1/640",
      "FNumber": 4.0,
      "ISO": 160,
      ...
    },
    ...
  },
  "sidecars": {
    "xmp_sidecar": "..."
  },
  "ai_description": "The image shows a scenic landscape at sunset..."
}
```

## Performance

- ~22 images/minute with qwen2.5vl:7b on RTX GPU
- ~2.8 seconds per image average
- Scales linearly with image count

## License

MIT
