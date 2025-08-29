#!/usr/bin/env python3
"""
Batch transcription & translation with WhisperX.

This script is optimized for robustness and accuracy, providing high-quality,
well-formatted subtitles suitable for video production.
"""

from __future__ import annotations

import os
import sys
import signal
import argparse
import logging
from pathlib import Path
from datetime import datetime
import shutil
import subprocess
from typing import Any, List

from tqdm import tqdm
import whisperx
import torch

# ------------------------------ Defaults -------------------------------------
DEFAULT_REPORTS_DIR = Path('./Data/Reports')
DEFAULT_BACKUP_ROOT = Path('./Data/Transcriptions/')
DURATION_THRESHOLD_MIN = 1
SUPPORTED_EXTENSIONS = {'.mp4', '.mov', '.mkv', '.avi', '.mxf', '.wav'}

# ------------------------------ FFprobe helpers ------------------------------
def get_media_duration(path: Path) -> float:
    try:
        out = subprocess.check_output([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', str(path)
        ], stderr=subprocess.STDOUT)
        return float(out)
    except Exception:
        return 0.0

def has_audio_stream(path: Path) -> bool:
    try:
        out = subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'a',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1', str(path)
        ], stderr=subprocess.STDOUT)
        return bool(out.strip())
    except Exception:
        return False

# ------------------------------ Main Transcriber class -----------------------
class Transcriber:
    def __init__(
        self,
        device: str,
        model_size: str,
        language: str,
        precision: str,
        asr_options: dict,
        vad_options: dict | None,
        align_model_name: str | None,
    ):
        logging.info('Loading WhisperX model (device=%s, precision=%s)…', device, precision)
        self.device = device
        self.language = language
        self.align_model_name = align_model_name
        
        self.model = whisperx.load_model(
            model_size, device, compute_type=precision, asr_options=asr_options, vad_options=vad_options
        )
        
        self.align_model = None
        self.align_metadata = None

    def _load_alignment_model(self):
        """Loads the alignment model only when needed."""
        if self.align_model is None:
            logging.info("Loading alignment model for language: %s", self.language)
            try:
                self.align_model, self.align_metadata = whisperx.load_align_model(
                    language_code=self.language, device=self.device, model_name=self.align_model_name
                )
            except ValueError:
                logging.warning('No alignment model for %s → skipping alignment', self.language)

    def transcribe(self, path: Path, task: str) -> dict | None:
        """Transcribes and aligns a single media file."""
        logging.info('» %s', path)
        if not has_audio_stream(path):
            logging.warning('No audio stream found – skipping.')
            return None

        audio = whisperx.load_audio(str(path))

        # 1. Transcribe or Translate
        logging.info("Step 1: Performing task '%s'...", task)
        result = self.model.transcribe(audio, language=self.language, task=task)
        
        # 2. Align
        self._load_alignment_model()
        if self.align_model:
            logging.info("Step 2: Aligning segments...")
            result = whisperx.align(result["segments"], self.align_model, self.align_metadata, audio, self.device)
        else:
            logging.info("Step 2: Skipped alignment (no model available).")
            
        return result

# ------------------------------ Utility functions ----------------------------
def list_files(root: Path) -> List[Path]:
    return [p for p in root.rglob('*') if p.suffix.lower() in SUPPORTED_EXTENSIONS]

def write_report(processed_files: List[Path], start: datetime, end: datetime, reports_dir: Path) -> None:
    reports_dir.mkdir(exist_ok=True, parents=True)
    report_path = reports_dir / f'report_{datetime.now():%Y-%m-%d_%H-%M-%S}.txt'
    with report_path.open('w', encoding='utf-8') as f:
        f.write('========================================\n')
        f.write('Transcription Report\n')
        f.write(f'Start Time: {start:%Y-%m-%d %H:%M:%S}\n')
        f.write(f'End Time:   {end:%Y-%m-%d %H:%M:%S}\n')
        f.write(f'Total Duration: {end - start}\n')
        f.write('========================================\n\n')
        if processed_files:
            f.write(f'Files Processed ({len(processed_files)}):\n')
            f.writelines(f'  - {p}\n' for p in processed_files)
        else:
            f.write('No new files were processed.\n')
    logging.info("Report saved to %s", report_path)

# ------------------------------ Runner helpers -------------------------------
def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

def _sigint_handler(sig, frame):
    logging.warning("Ctrl+C detected. Shutting down gracefully.")
    raise KeyboardInterrupt

# ------------------------------ Main entry -----------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description='Robust Batch WhisperX Transcriber')
    
    # I/O Arguments
    parser.add_argument('--root', type=Path, required=True, help='Root folder containing media files')
    parser.add_argument('--reports_dir', type=Path, default=DEFAULT_REPORTS_DIR, help='Directory to save reports')
    parser.add_argument('--backup_dir', type=Path, default=DEFAULT_BACKUP_ROOT, help='Directory to back up transcriptions')

    # Model & Hardware Arguments
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'], help='Device for computation')
    parser.add_argument('--model', default='large-v3', help='Whisper model size')
    parser.add_argument('--precision', default='int8', choices=['float32', 'float16', 'bfloat16', 'int8'], help='Model compute precision')
    
    # Language & Task Arguments
    parser.add_argument('--language', required=True, help='Language code of the audio (e.g., he, ru, ar)')
    parser.add_argument('--task', default='transcribe', choices=['transcribe', 'translate'], help='Task to perform')

    # Accuracy & Decoding Arguments
    parser.add_argument('--beam_size', type=int, default=5, help='Number of beams in beam search')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for sampling')
    parser.add_argument('--vad_filter', action='store_true', help='Enable VAD filtering')

    # Alignment & Subtitle Arguments
    parser.add_argument('--align_model', type=str, default=None, help='Manually specify alignment model')
    parser.add_argument('--max_line_width', type=int, default=42, help='Max characters per subtitle line')
    
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable detailed debug logging')
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    signal.signal(signal.SIGINT, _sigint_handler)
    
    # Automatic Alignment Model Selection
    if args.align_model is None:
        RECOMMENDED_ALIGN_MODELS = {
            "he": "imvladikon/wav2vec2-xls-r-300m-hebrew",
            "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
            "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
        }
        if args.language in RECOMMENDED_ALIGN_MODELS:
            args.align_model = RECOMMENDED_ALIGN_MODELS[args.language]
            logging.info(f"Using recommended alignment model for '{args.language}': {args.align_model}")

    # Prepare options dictionaries
    asr_options = {"beam_size": args.beam_size, "temperature": args.temperature}
    vad_options = {"vad_onset": 0.500, "vad_offset": 0.363} if args.vad_filter else None

    transcriber = Transcriber(
        device=args.device, model_size=args.model, language=args.language,
        precision=args.precision, asr_options=asr_options, vad_options=vad_options,
        align_model_name=args.align_model
    )
    
    files_to_process = list_files(args.root)
    processed_files = []
    start_time = datetime.now()

    try:
        for p in tqdm(files_to_process, desc='Processing files', unit='file'):
            try:
                # Output to "Transcription" subfolder, same as original script
                output_dir = p.parent / 'Transcription'
                output_dir.mkdir(parents=True, exist_ok=True)
                srt_path = output_dir / f'{p.stem}.srt'
                txt_path = output_dir / f'{p.stem}.txt'
                
                if srt_path.exists():
                    logging.debug("Output for '%s' already exists, skipping.", p.name)
                    continue

                result = transcriber.transcribe(p, args.task)
                if result and result.get('segments'):
                    # Write SRT using whisperx utility for good formatting
                    with open(srt_path, "w", encoding="utf-8") as f:
                        whisperx.utils.write_srt(result["segments"], file=f, max_line_width=args.max_line_width)
                    
                    # Write plain text file
                    with open(txt_path, "w", encoding="utf-8") as f:
                        for segment in result["segments"]:
                            f.write(segment['text'].strip() + "\n")
                    
                    logging.info("Outputs saved to %s", output_dir)
                    processed_files.append(p)
                else:
                    logging.warning("No segments transcribed for %s.", p.name)
            except Exception as e:
                logging.exception("Error processing file %s: %s", p, e)
    except KeyboardInterrupt:
        logging.info("Processing interrupted.")
    finally:
        end_time = datetime.now()
        if processed_files:
            write_report(processed_files, start_time, end_time, args.reports_dir)

if __name__ == '__main__':
    main()
