# Egyptian National ID OCR Pipeline

An end-to-end OCR pipeline for extracting structured data (Full Name, Address,
14-digit National ID Number) from images of Egyptian National ID cards —
built to handle low-quality photos, uneven lighting, and tilted/perspective-skewed shots.

> **Status:** Work in progress. This README will be filled in with full
> setup instructions, before/after preprocessing images, and metrics as the
> project develops. See commit history for build progress.

## Why synthetic data?

Real Egyptian National ID images contain sensitive PII and using scraped or
real samples would raise both privacy and legal concerns. Instead, this
project generates **synthetic ID cards** with a programmatically-created
template, randomized (fake) Arabic names/addresses/ID numbers, and known
ground truth — which also enables proper CER/WER evaluation, something that's
hard to do credibly without labeled real data anyway.

## Project structure (evolving)

```
src/
  generator/        # synthetic ID card generation
  preprocessing/     # OpenCV: perspective correction, denoising, binarization
  ocr/                # PaddleOCR detection + recognition wrappers
  postprocessing/    # regex validation, Arabic text cleaning, numeral normalization
  api/                # FastAPI app
data/
  synthetic_clean/    # generated "ideal" ID images + ground truth
  synthetic_degraded/ # same IDs after augmentation (blur/rotate/warp/noise)
tests/
notebooks/
```

More to come.
