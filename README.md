# whisperx-translate-nas

Batch transcription & translation tool powered by WhisperX.

Optimised for Apple Silicon (MPS) but works on CPU and CUDA as well.

<<<<<<< HEAD
## Setup (PDM)

1. Install [PDM](https://pdm.fming.dev):
   ```bash
   pip install pdm
   ```
2. Install dependencies:
   ```bash
   pdm install
   ```

=======
>>>>>>> main
## Usage

Provide the language manually and pick the desired model:

```bash
<<<<<<< HEAD
pdm run python whisperx_batch_transcriber.py \
=======
python whisperx_batch_transcriber.py \
>>>>>>> main
    --device mps \
    --model large-v3 \
    --language en \
    --precision float16
```

Additional flags allow fine‑tuning transcription quality:

* `--beam-size` / `--best-of` – decoding search parameters
* `--temperature` – sampling temperature (0 for deterministic)
* `--chunk-size` – audio chunk size in seconds
* `--no-condition-on-prev-text` – disable context conditioning

Run `python whisperx_batch_transcriber.py -h` for full options.
