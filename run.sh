#!/bin/bash
# DemonTTS Audiobook Generator Launcher
# Install deps then launch GUI

pip install -q customtkinter pypdf pdfplumber torch torchaudio numpy soundfile pygame tokenizers transformers pydub
python gui.py
