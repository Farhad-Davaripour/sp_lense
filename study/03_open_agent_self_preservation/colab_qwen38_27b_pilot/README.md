# Qwen3.8-27B quantized Colab pilot

This is a separate hardware and unchanged-model inference check. It does not
fine-tune a model, evaluate self-preservation, or change the frozen Research 3
studies. Run the versioned
[`SP_Lense_Qwen38_27B_Quantized_Colab_Pilot.ipynb`](SP_Lense_Qwen38_27B_Quantized_Colab_Pilot.ipynb)
in Colab. The browser-run copy is [in the owner's Colab account](https://colab.research.google.com/drive/1aNMNJxX6SxmqIn-GFlJcIjZkXEytL3Z-).

## Pinned inputs

- Free Google Colab T4 runtime: 14.56 GiB visible VRAM and 12.67 GB system RAM
  in the observed session.
- `llama.cpp` official CUDA 12.8 Ubuntu release `b11266`.
- `unsloth/Qwen3.8-27B-GGUF` at revision
  `4ca720788d1e01f1bff70c033e0d0028fd02e502`, file
  `Qwen3.8-27B-UD-IQ3_S.gguf` (12,040,883,104 bytes; published LFS SHA-256
  `d847e2c1e4aa276e4b7b8e9ad7628050e61e165d49ab995407bc36677a6f3864`).
- A short, fictional inventory arithmetic prompt. No model-generated output is
  executed as code or given tools.

The [ready-made Unsloth Qwen3.8-27B conversational notebook](https://github.com/unslothai/notebooks/blob/main/nb/Qwen3.8_%2827B%29-Conversational.ipynb)
is a useful starting point for future fine-tuning. It explicitly requires two
T4 GPUs, and [Unsloth's requirement table](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements)
lists 22 GB as the minimum for 27B 4-bit fine-tuning. This free Colab session
has one T4 with 14.56 GiB. The pilot therefore checks inference with a 3-bit
model, rather than claiming to run the fine-tuning notebook.

## Result

The public 3-bit file downloaded and loaded successfully on the free T4. A
fictional inventory prompt asked for 13 + 3 blue pens; the model returned
`16`. With `llama.cpp` `b11266`, 99 requested GPU layers, 512-token context,
24 maximum generated tokens, and a single-turn setting, the process exited
successfully in 138.9 seconds. The CLI reported 1.2 prompt tokens/second and
0.3 generated tokens/second. The model process used approximately 11,059 MiB
of GPU memory during generation.

The initial run with 48 requested GPU layers and a 1,024-token context timed
out after its 300-second subprocess limit. That unsuccessful attempt is retained
in the browser-run notebook. The bounded full-GPU retry above is the completed
smoke result. This is a hardware feasibility check, not a meaningful model
benchmark or evidence about self-preservation. In particular, the 3-bit model
has different numerical behavior from the original BF16 checkpoint.

This free T4 has too little memory for the prepared 27B QLoRA notebook. No
fine-tuning was attempted, no user files or Google Drive folders were mounted,
and no model-generated output was executed.
