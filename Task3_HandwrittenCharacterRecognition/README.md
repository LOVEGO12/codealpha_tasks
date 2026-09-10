# Handwritten Character Recognition

A complete pipeline for recognizing handwritten digits and characters using
CNNs, with a built-in extension path to full word/sentence recognition via
a CRNN (CNN + BiLSTM + CTC).

## Project layout

```
handwriting_recognition/
├── requirements.txt
├── model.py           # CNN architecture (digits + EMNIST characters)
├── train.py            # Training loop for MNIST / EMNIST
├── infer.py             # Load a trained model, predict on a single image
├── crnn_model.py    # CRNN for word/sentence-level sequence recognition
├── crnn_train.py     # Training loop skeleton for CRNN + CTC loss
└── README.md
```

## Quick start

```bash
pip install -r requirements.txt

# Train on MNIST (10 digit classes)
python train.py --dataset mnist --epochs 10

# Train on EMNIST (47 balanced character classes: digits + upper/lowercase)
python train.py --dataset emnist --emnist-split balanced --epochs 15

# Run inference on a single image of one handwritten character
python infer.py --checkpoint checkpoints/emnist_best.pt --image my_char.png --dataset emnist
```

## Why this design

**CNN for single characters.** A LeNet/VGG-style CNN with batch norm and
dropout is the standard, well-proven approach for MNIST/EMNIST. It gets
~99.3-99.6% on MNIST and ~88-91% on EMNIST-balanced without heavy tuning.

**EMNIST over MNIST when you need real characters.** MNIST only has
digits 0-9. EMNIST (Extended MNIST) reuses the same 28x28 format but adds
uppercase and lowercase letters, so it's a drop-in upgrade — same input
shape, same training loop, just a different label set. The `balanced`
split (47 classes) merges visually-ambiguous case pairs (e.g. 'O'/'o')
which is what most character-recognition systems use.

**CRNN for sequences.** A single-character CNN can't read a whole word —
it doesn't know where one character ends and the next begins, and it
needs a fixed-size input for a fixed-size output. The CRNN pattern
(CNN feature extractor → sequence of column-features → bidirectional
LSTM → CTC loss) solves this: it slides a "reading window" over the
image and lets CTC figure out the alignment between image columns and
output characters. This is the same backbone architecture used in
production OCR systems (e.g. CRNN, the basis of many handwriting/OCR
engines).

## Extending further

- Swap EMNIST for your own scanned handwriting samples (same 28x28
  grayscale format) via a custom `Dataset` class in `train.py`.
- For full-page recognition, add a text-line segmentation step
  (e.g. connected components or a detection model) before feeding
  lines into the CRNN.
- Beam-search CTC decoding (vs. greedy) + a character-level language
  model typically adds a few more points of accuracy on real
  handwriting.
