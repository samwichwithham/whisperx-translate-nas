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
# Suggest making these configurable via CLI arguments for better portability
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
        beam_size: int,
        temperature: float,
        best_of: int,
        no_speech_threshold: float,
        vad_filter: bool,
        align_model: str | None,
    ):
        logging.info('Loading WhisperX model (device=%s, precision=%s)…', device, precision)
        self.device = device
        self.language = language
        self.align_model_name = align_model
        
        # VAD options for filtering out short/inaudible speech segments
        self.vad_options = {
            "vad_onset": 0.500,
            "vad_offset": 0.363,
        }
        self.vad_filter = vad_filter

        # ASR options for decoding, focused on accuracy
        self.asr_options = {
            "beam_size": beam_size,
            "best_of": best_of,
            "temperature": temperature,
            "no_speech_threshold": no_speech_threshold,
            "suppress_tokens": [-1],
            "condition_on_previous_text": True,
        }

        self.model = whisperx.load_model(
            model_size,
            device=device,
            compute_type=precision,
            asr_options=self.asr_options,
            vad_options=self.vad_options if vad_filter else None,
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
                logging.warning('No alignment model available for %s → skipping alignment', self.language)

    def transcribe(self, path: Path) -> dict:
        """Transcribes and aligns a single media file."""
        logging.info('» %s', path)
        if not has_audio_stream(path):
            logging.warning('No audio stream found – skipping file.')
            return None

        audio = whisperx.load_audio(str(path))

        # 1. Transcribe
        logging.info("Step 1: Transcribing...")
        result = self.model.transcribe(audio, language=self.language, task="transcribe")
        
        # 2. Align
        self._load_alignment_model()
        if self.align_model:
            logging.info("Step 2: Aligning subtitles...")
            result = whisperx.align(
                result["segments"], self.align_model, self.align_metadata, audio, self.device
            )
        else:
            logging.info("Step 2: Skipped alignment (no model available).")

        # 3. Translate if required
        if 'to-translate' in str(path).lower():
            logging.info("Step 3: Translating to English...")
            result = whisperx.translate(audio, self.model, result, lang_codes=[self.language])

        return result

# ------------------------------ Utility functions ----------------------------
def list_files(root: Path) -> List[Path]:
    return [p for p in root.rglob('*') if p.suffix.lower() in SUPPORTED_EXTENSIONS]

def write_report(processed_files: List[Path], start: datetime, end: datetime, reports_dir: Path) -> None:
    reports_dir.mkdir(exist_ok=True)
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
    
    # --- I/O Arguments ---
    parser.add_argument('--root', type=Path, required=True, help='Root folder containing media files')
    parser.add_argument('--reports_dir', type=Path, default=DEFAULT_REPORTS_DIR, help='Directory to save reports')
    parser.add_argument('--backup_dir', type=Path, default=DEFAULT_BACKUP_ROOT, help='Directory to back up transcriptions')

    # --- Model & Hardware Arguments ---
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'], help='Device for computation')
    parser.add_argument('--model', default='large-v3', help='Whisper model size (e.g., large-v3)')
    parser.add_argument('--precision', default='int8', choices=['float32', 'float16', 'bfloat16', 'int8'], help='Model compute precision')
    
    # --- Language & Task Arguments ---
    parser.add_argument('--language', required=True, help='Language code of the audio (e.g., he, ru, ar)')

    # --- Accuracy & Decoding Arguments ---
    parser.add_argument('--beam_size', type=int, default=5, help='Number of beams in beam search')
    parser.add_argument('--best_of', type=int, default=5, help='Number of candidates to consider')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for sampling. 0 for deterministic.')
    parser.add_argument('--no_speech_threshold', type=float, default=0.6, help='Threshold for detecting no speech')
    parser.add_argument('--vad_filter', action='store_true', help='Enable VAD filtering to remove short speech segments')

    # --- Alignment & Subtitle Arguments ---
    parser.add_argument('--align_model', type=str, default=None, help='Manually specify a wav2vec2 alignment model')
    parser.add_argument('--max_line_width', type=int, default=42, help='Maximum number of characters per subtitle line')
    parser.add_argument('--max_line_count', type=int, default=2, help='Maximum number of lines per subtitle card')

    # --- General Arguments ---
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable detailed debug logging')
    
    args = parser.parse_args()

    setup_logging(args.verbose)
    signal.signal(signal.SIGINT, _sigint_handler)
    
    # Ensure output directories exist
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    args.backup_dir.mkdir(parents=True, exist_ok=True)

    transcriber = Transcriber(
        device=args.device,
        model_size=args.model,
        language=args.language,
        precision=args.precision,
        beam_size=args.beam_size,
        temperature=args.temperature,
        best_of=args.best_of,
        no_speech_threshold=args.no_speech_threshold,
        vad_filter=args.vad_filter,
        align_model=args.align_model,
    )
    
    files_to_process = list_files(args.root)
    processed_files = []
    start_time = datetime.now()

    try:
        for p in tqdm(files_to_process, desc='Processing files', unit='file'):
            out_dir = p.parent / 'Transcription'
            out_dir.mkdir(exist_ok=True)
            srt_path = out_dir / f'{p.stem}.srt'

            if srt_path.exists():
                logging.debug("'%s' already transcribed, skipping.", p.name)
                continue

            if get_media_duration(p) / 60 < DURATION_THRESHOLD_MIN:
                logging.debug("'%s' is shorter than the minimum duration, skipping.", p.name)
                continue

            try:
                result = transcriber.transcribe(p)
                if result and result['segments']:
                    # Use the robust SRT writer from whisperx
                    whisperx.utils.write_srt(
                        result,
                        str(srt_path),
                        max_line_width=args.max_line_width,
                        max_line_count=args.max_line_count
                    )
                    logging.info("Subtitle saved to %s", srt_path)
                    
                    # Backup the SRT file
                    backup_subdir = args.backup_dir / p.parent.relative_to(args.root) / 'Transcription'
                    backup_subdir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(srt_path, backup_subdir / srt_path.name)
                    
                    processed_files.append(p)
                else:
                    logging.warning("No segments transcribed for %s.", p.name)
            except Exception as e:
                logging.exception("Error processing file %s: %s", p, e)

    except KeyboardInterrupt:
        logging.info("Processing interrupted by user.")
    finally:
        end_time = datetime.now()
        write_report(processed_files, start_time, end_time, args.reports_dir)

if __name__ == '__main__':
    main()
