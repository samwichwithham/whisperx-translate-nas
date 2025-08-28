# whisperx-translate-nas

Batch transcription & translation tool powered by WhisperX.

Optimised for Apple Silicon (MPS) but works on CPU and CUDA as well.

## Usage

Provide the language manually and pick the desired model:

```bash
python whisperx_batch_transcriber.py \
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
