#!/usr/bin/env python3
"""
Batch transcription & translation with WhisperX on a QNAP NAS (TBS‑h574TX).

Key features
------------
* **High‑quality transcription** – manual language selection, large‑v3 by default.
* **Safe alignment** – skips languages without an align model.
* **MPS backend** – supports Apple Silicon GPUs.
* **Configurable CLI** – root, device, model, language, precision, decoding.
* **Robust logging and graceful Ctrl‑C**.
* **Thread control** – honours `OMP_NUM_THREADS`; sets Torch threads accordingly.
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
import torch  # for thread control

# ------------------------------ Defaults -------------------------------------
FOOTAGE_ROOT = Path('/data/footage/')
REPORTS_DIR = Path('/data/Reports/')
BACKUP_ROOT = Path('/data/Transcriptions/')
DURATION_THRESHOLD_MIN = 1
SUPPORTED_EXTENSIONS = {'.mp4', '.mov', '.mkv', '.avi', '.mxf', '.wav'}

# FFprobe helpers
# -----------------------------------------------------------------------------

def get_media_duration(path: Path) -> float:
    try:
        out = subprocess.check_output([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', str(path)
        ], stderr=subprocess.STDOUT)
        return float(out)
    except Exception:  # pragma: no cover
        return 0.0

def has_audio_stream(path: Path) -> bool:
    try:
        out = subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'a',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1', str(path)
        ], stderr=subprocess.STDOUT)
        return bool(out.strip())
    except Exception:  # pragma: no cover
        return False

# -----------------------------------------------------------------------------
# Small text utilities
# -----------------------------------------------------------------------------

def _timestamp(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02}:{m:02}:{s:06.3f}".replace('.', ',')

def export_srt(segments, path: Path) -> None:
    with path.open('w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n{_timestamp(seg['start'])} --> "
                    f"{_timestamp(seg['end'])}\n{seg['text'].strip()}\n\n")

def dedup_lines(txt_path: Path) -> None:
    if not txt_path.exists():
        return
    seen_last = ''
    new_lines = []
    for ln in txt_path.read_text(encoding='utf-8').splitlines():
        if ln.strip() != seen_last:
            new_lines.append(ln)
            seen_last = ln.strip()
    txt_path.write_text('\n'.join(new_lines), encoding='utf-8')

def clean_srt_repeats(srt_path: Path) -> None:
    if not srt_path.exists():
        return
    with srt_path.open('r', encoding='utf-8') as f:
        lines = f.readlines()
    out, last = [], ''
    i = 0
    while i < len(lines):
        ln = lines[i]
        st = ln.strip()
        if st.isdigit() or '-->' in st or st == '':
            out.append(ln)
        else:
            if st != last:
                out.append(ln)
                last = st
        i += 1
    srt_path.write_text(''.join(out), encoding='utf-8')

# -----------------------------------------------------------------------------
# Alignment helper (skip if no model)
# -----------------------------------------------------------------------------

def safe_load_align(lang: str, device: str):
    try:
        return whisperx.load_align_model(language_code=lang, device=device)
    except ValueError:
        logging.info('No align model for %s → skip', lang)
        return None, None

# -----------------------------------------------------------------------------
# Main Transcriber class
# -----------------------------------------------------------------------------

class Transcriber:
    def __init__(
        self,
        device: str,
        model_size: str = 'large-v3',
        language: str = 'en',
        precision: str = 'float32',
        beam_size: int = 5,
        best_of: int = 5,
        temperature: float = 0.0,
        chunk_size: int = 30,
        condition_on_prev_text: bool = True,
    ):
        logging.info('Loading WhisperX model (device=%s, precision=%s)…', device, precision)
        self.device = device
        self.language = language
        self.beam_size = beam_size
        self.best_of = best_of
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.condition_on_prev_text = condition_on_prev_text
        self.align_cache: dict[str, tuple[Any, Any]] = {}
        self.model_main = whisperx.load_model(
            model_size,
            device=device,
            compute_type=precision,
            vad_method='pyannote',
        )

    # ------------------------- per‑file processing -------------------------
    def transcribe(self, path: Path) -> None:
        logging.info('» %s', path)
        if not has_audio_stream(path):
            logging.warning('No audio – skip')
            return

        out_dir = path.parent / 'Transcription'
        out_dir.mkdir(exist_ok=True)
        txt_out = out_dir / f'{path.stem}.txt'
        if txt_out.exists():
            logging.debug('Already done')
            return

        if 'to-translate' not in str(path).lower():
            if get_media_duration(path) / 60 < DURATION_THRESHOLD_MIN:
                logging.debug('Short clip – skip')
                return

        lang = self.language
        logging.info('Language preset: %s', lang)

        with torch.inference_mode():
            res = self.model_main.transcribe(
                str(path),
                language=lang,
                task='translate',
                beam_size=self.beam_size,
                best_of=self.best_of,
                temperature=self.temperature,
                chunk_size=self.chunk_size,
                condition_on_prev_text=self.condition_on_prev_text,
            )

        if lang not in self.align_cache:
            self.align_cache[lang] = safe_load_align(lang, self.device)
        model_align, meta = self.align_cache[lang]
        segments = res['segments']
        if model_align:
            segments = whisperx.align(segments, model_align, meta, str(path),
                                      device=self.device, return_char_alignments=False)['segments']

        # -------- write outputs --------
        txt_out.write_text('\n'.join(seg['text'].strip() for seg in segments), encoding='utf-8')
        srt_path = out_dir / f'{path.stem}.srt'
        export_srt(segments, srt_path)
        dedup_lines(txt_out)
        clean_srt_repeats(srt_path)

        # backup
        rel = path.parent.relative_to(FOOTAGE_ROOT)
        backup_dir = BACKUP_ROOT / rel / 'Transcription'
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(txt_out, backup_dir / txt_out.name)

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def list_files(root: Path) -> List[Path]:
    return [p for p in root.rglob('*') if p.suffix.lower() in SUPPORTED_EXTENSIONS]

def write_report(file_list: List[Path], start: datetime, end: datetime) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    report_path = REPORTS_DIR / f'{datetime.now():%d.%m.%y}.txt'
    with report_path.open('a', encoding='utf-8') as f:
        f.write('========================================\n')
        f.write('Transcription Report\n')
        f.write(f'Start: {start:%Y-%m-%d %H:%M:%S}\nEnd:   {end:%Y-%m-%d %H:%M:%S}\n')
        if file_list:
            f.write('Files processed:\n')
            f.writelines(f'  - {p}\n' for p in file_list)
        else:
            f.write('No files processed.\n')
        f.write('========================================\n\n')

# -----------------------------------------------------------------------------
# Runner helpers
# -----------------------------------------------------------------------------

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

def _sigint_handler(sig, frame):  # pragma: no cover
    raise KeyboardInterrupt

# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Batch WhisperX transcriber for QNAP')
    parser.add_argument('--root', type=Path, default=FOOTAGE_ROOT, help='Footage root folder')
    default_device = (
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()
        else 'cpu'
    )
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default=default_device)
    parser.add_argument('--model', default='large-v3', help='Whisper model size')
    parser.add_argument('--language', required=True, help='Language code (e.g. en, fr)')
    parser.add_argument('--precision', choices=['float32', 'float16', 'bfloat16'], help='Model compute precision')
    parser.add_argument('--beam-size', type=int, default=5, help='Decoding beam size')
    parser.add_argument('--best-of', type=int, default=5, help='Number of candidates explored')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--chunk-size', type=int, default=30, help='Audio chunk size in seconds')
    parser.add_argument('--no-condition-on-prev-text', dest='condition_on_prev_text', action='store_false',
                        help='Disable conditioning on previous text')
    parser.set_defaults(condition_on_prev_text=True)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    # logging
    setup_logging(args.verbose)

    # thread pool tuning – honour env or guess
    num_threads = int(os.getenv('OMP_NUM_THREADS', max(1, os.cpu_count() // 2)))
    torch.set_num_threads(num_threads)
    logging.info('Torch threads=%d', num_threads)

    signal.signal(signal.SIGINT, _sigint_handler)

    start = datetime.now()

    precision = args.precision or ('float16' if args.device in ('cuda', 'mps') else 'float32')
    transcriber = Transcriber(
        device=args.device,
        model_size=args.model,
        language=args.language,
        precision=precision,
        beam_size=args.beam_size,
        best_of=args.best_of,
        temperature=args.temperature,
        chunk_size=args.chunk_size,
        condition_on_prev_text=args.condition_on_prev_text,
    )
    files = list_files(args.root)

    for p in tqdm(files, desc='Transcribing', unit='file'):
        try:
            transcriber.transcribe(p)
        except Exception as e:  # noqa: BLE001
            logging.exception('Error with %s: %s', p, e)

    end = datetime.now()
    write_report(files, start, end)


if __name__ == '__main__':
    main()
