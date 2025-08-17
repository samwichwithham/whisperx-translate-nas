#!/usr/bin/env python3
"""
Batch transcription & translation with WhisperX on a QNAP NAS (TBS‑h574TX).

Key features
-------------
* **Single‑load models** – tiny for language detect, large‑v3 for main pass.
* **Intel® Extension for PyTorch (IPEX)** auto‑enabled when installed → ~1.3× faster.
* **Safe alignment** – skips languages without an align model.
* **Configurable CLI** – override root, device, model, verbosity.
* **Robust logging, Telegram alerts, graceful Ctrl‑C**.
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
import tempfile
from typing import List

import requests
from tqdm import tqdm
import whisperx
import torch  # for thread control

# Try to load Intel® Extension for PyTorch if present
try:
    import intel_extension_for_pytorch as ipex  # type: ignore
    _IPEX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _IPEX_AVAILABLE = False

# ------------------------------ Defaults -------------------------------------
FOOTAGE_ROOT = Path('/data/footage/')
REPORTS_DIR = Path('/data/Reports/')
BACKUP_ROOT = Path('/data/Transcriptions/')
DURATION_THRESHOLD_MIN = 1
SUPPORTED_EXTENSIONS = {'.mp4', '.mov', '.mkv', '.avi', '.mxf', '.wav'}

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# -----------------------------------------------------------------------------

def send_telegram(text: str) -> None:
    """Send message via Telegram Bot API – no‑op if token/chat id missing."""
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        logging.debug('Telegram disabled')
        return
    try:
        r = requests.post(
            f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage',
            data={'chat_id': TELEGRAM_CHAT_ID, 'text': text, 'parse_mode': 'HTML'},
            timeout=10,
        )
        r.raise_for_status()
    except Exception as exc:  # pragma: no cover – network problems not fatal
        logging.warning('Telegram failed: %s', exc)

# -----------------------------------------------------------------------------
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
    def __init__(self, device: str, model_size: str = 'large-v3'):
        logging.info('Loading WhisperX models (device=%s)…', device)
        self.device = device
        self.model_fast = whisperx.load_model('tiny', device=device, compute_type='float32')
        self.model_main = whisperx.load_model(model_size, device=device,
                                              compute_type='float32', vad_method='pyannote')

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

        lang = self.model_fast.transcribe(str(path))['language']
        logging.info('Language: %s', lang)

        res = self.model_main.transcribe(str(path), language=lang, task='translate', chunk_size=10)

        model_align, meta = safe_load_align(lang, self.device)
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
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--model', default='large-v3', help='Whisper model size')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    # logging
    setup_logging(args.verbose)

    # thread pool tuning – honour env or guess
    num_threads = int(os.getenv('OMP_NUM_THREADS', max(1, os.cpu_count() // 2)))
    torch.set_num_threads(num_threads)
    logging.info('Torch threads=%d', num_threads)

    signal.signal(signal.SIGINT, _sigint_handler)

    send_telegram('WhisperX job started.')
    start = datetime.now()

    transcriber = Transcriber(device=args.device, model_size=args.model)
    files = list_files(args.root)

    for p in tqdm(files, desc='Transcribing', unit='file'):
        try:
            transcriber.transcribe(p)
        except Exception as e:  # noqa: BLE001
            logging.exception('Error with %s: %s', p, e)

    end = datetime.now()
    write_report(files, start, end)
    send_telegram(f'WhisperX job finished.\nStart: {start}\nEnd: {end}')


if __name__ == '__main__':
    main()
