#!/usr/bin/env python3
"""Train a BPE tokenizer on all parsed book text for the Student TTS model."""

import json
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers


def main():
    book_dir = Path("book_parsed")
    if not book_dir.exists():
        raise ValueError("No book_parsed/ directory found. Run convert_book.py first.")

    # Collect all text
    texts = []
    for json_file in sorted(book_dir.glob("*.json")):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        for chapter_data in data.values():
            text = chapter_data.get("text", "")
            if text:
                texts.append(text)

    if not texts:
        raise ValueError("No text found in book_parsed/*.json")

    print(f"[Tokenizer] Training on {len(texts)} chapters...")

    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=8000,
        special_tokens=["<pad>", "<unk>", "<eos>", "<s>"],
        min_frequency=2,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Set padding and truncation
    tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
    tokenizer.enable_truncation(max_length=512)

    out_path = Path("models/tokenizer.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))

    vocab_size = len(tokenizer.get_vocab())
    print(f"[Tokenizer] Saved to {out_path} | Vocab size: {vocab_size}")

    # Quick test
    test = "Hello world. This is a test."
    encoded = tokenizer.encode(test)
    print(f"[Tokenizer] Test: '{test}' -> {len(encoded.ids)} tokens")


if __name__ == "__main__":
    main()
