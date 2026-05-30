# Chapter 1 Master Voice — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make Chapter 1 (Preface) sound perfectly human, then generalize to the full book via Ouroboros.

**Architecture:** Fix inference pipeline → extract Chapter 1 → emotional parsing → generate conditioned training pairs → train Faraday → Ouroboros refinement → full book generation.

**Tech Stack:** PyTorch, SpeechT5, custom emotional conditioning, Faraday diffusion, Ouroboros self-training loop.

---

## The Problem

Current voice is robotic because:
1. Speaker embedding projection layers are **randomly initialized** on every load
2. Faraday is barely trained (epoch 1, step 700 / ~30,000 needed)
3. Aether LSTM is untrained (809K params, old architecture)
4. Parsed text has PDF encoding garbage (� characters)
5. DSP limiter crushes dynamics
6. Text chunks exceed SpeechT5's 600-token limit, causing crashes

## The Fix Strategy

### Phase 1: Clean Inference Pipeline
- Use SpeechT5's **native trained speaker embedding** (not random projection)
- Clean text encoding garbage
- Token-aware chunking (max 500 tokens)
- Disable Faraday/Aether until retrained
- Gentle DSP (just loudness normalize, no brickwall)

### Phase 2: Emotional Parsing
Parse Preface text for:
- **Punctuation cues:** ! → excitement, ? → curiosity, ... → contemplation
- **Sentence length:** short = urgent/impact, long = flowing/calm
- **Dialogue tags:** "said", "shouted", "whispered"
- **Emotional keywords:** fear, love, anger, wonder

Map each chunk to an emotion vector that modulates the speaker embedding.

### Phase 3: Chapter 1 Training Data
Generate ~100 synthetic pairs from Preface text only:
- Corrupt SpeechT5 output → enhanced target
- Emotion vector conditions both input and target
- Focus all training energy on one chapter's style

### Phase 4: Faraday Training (100 epochs on Chapter 1)
Train Faraday specifically on Preface pairs:
- Batch size 1, grad accum 8
- 100 epochs = ~2-3 hours on RTX 4060
- This overfits Faraday to Chapter 1's style — intentional

### Phase 5: Ouroboros Generalization
Use the Chapter 1-trained Faraday to generate better synthetic data from ALL chapters, then retrain. The snake eats its tail.

### Phase 6: Full Book Generation
Generate complete audiobook with trained pipeline.

---

## Task 1: Fix Clean Inference Pipeline

**Files:**
- Create: `demo_tts_clean.py`
- Modify: `generate_audiobook.py` (to use clean pipeline)

**Steps:**
- [ ] Verify `demo_tts_clean.py` uses SpeechT5 native speaker embedding
- [ ] Verify token-aware chunking stays under 500 tokens
- [ ] Verify text cleaning removes � and other garbage
- [ ] Test generate one clean audio sample

## Task 2: Emotional Parser

**Files:**
- Create: `emotion_parser.py`

**Steps:**
- [ ] Parse Preface text into chunks
- [ ] Extract emotion vectors per chunk
- [ ] Map emotions to speaker embedding perturbations

## Task 3: Chapter 1 Training Pairs

**Files:**
- Create: `generate_chapter1_pairs.py`

**Steps:**
- [ ] Load Preface text
- [ ] Generate 100 synthetic (corrupted, enhanced) pairs
- [ ] Attach emotion vectors to each pair
- [ ] Save to `data/chapter1_pairs/`

## Task 4: Train Faraday on Chapter 1

**Files:**
- Modify: `training/train_faraday_supervised.py`

**Steps:**
- [ ] Point trainer at `data/chapter1_pairs/`
- [ ] Train 100 epochs
- [ ] Save best checkpoint to `checkpoints/faraday_chapter1/`

## Task 5: Ouroboros Pass

**Files:**
- Modify: `ouroboros_trainer.py`

**Steps:**
- [ ] Use Chapter 1-trained Faraday as teacher
- [ ] Generate enhanced pairs from ALL chapters
- [ ] Retrain Faraday on full book data

## Task 6: Generate Full Audiobook

**Files:**
- Modify: `generate_audiobook.py`

**Steps:**
- [ ] Load trained Faraday
- [ ] Enable emotion modulation
- [ ] Generate all chapters
- [ ] Combine into single FLAC
