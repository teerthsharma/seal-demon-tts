"""DemonTTS Audiobook Generator GUI.

Dark glassmorphic theme with neon cyan + magenta accents.
"""

import json
import os
import sys
import threading
import time
import tkinter as tk
import tkinter.simpledialog as simpledialog
from datetime import timedelta
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Optional

import customtkinter as ctk
import numpy as np
import pygame
import soundfile as sf
import torch

from demo_tts import DemonTTS
from pdf_parser import PDFParser

# ---------------------------------------------------------------------------
# Theme / styling
# ---------------------------------------------------------------------------
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

COLOR_BG = "#0d0d0d"
COLOR_CARD = "#151515"
COLOR_CYAN = "#00f0ff"
COLOR_MAGENTA = "#ff00a0"
COLOR_TEXT = "#e0e0e0"
COLOR_TEXT_DIM = "#888888"
COLOR_BORDER = "#2a2a2a"

FONT_FAMILY = "Segoe UI"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_eta(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


# ---------------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------------


class DemonGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("DemonTTS Audiobook Generator")
        self.geometry("1200x800")
        self.configure(fg_color=COLOR_BG)

        self.tts: Optional[DemonTTS] = None
        self.parser = PDFParser()
        self.chapters: Dict[str, Dict] = {}
        self.selected_chapters: set = set()
        self.current_wav: Optional[np.ndarray] = None
        self.current_sr = 24_000
        self.generation_cancelled = False
        self.generation_thread: Optional[threading.Thread] = None

        # Audio player state
        pygame.mixer.init(frequency=24_000, channels=1)
        self._audio_path: Optional[str] = None
        self._audio_start_time = 0.0
        self._audio_paused = False

        self._build_ui()
        self._load_voices_into_combo()
        self._init_tts_bg()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Main grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---- Left sidebar ------------------------------------------------
        sidebar = ctk.CTkFrame(self, width=280, fg_color=COLOR_CARD, corner_radius=12)
        sidebar.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")
        sidebar.grid_propagate(False)

        lbl_title = ctk.CTkLabel(
            sidebar,
            text="DemonTTS",
            font=ctk.CTkFont(family=FONT_FAMILY, size=24, weight="bold"),
            text_color=COLOR_CYAN,
        )
        lbl_title.pack(pady=(18, 4), padx=16, anchor="w")

        lbl_sub = ctk.CTkLabel(
            sidebar,
            text="ElevenLabs-grade Audiobooks",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            text_color=COLOR_TEXT_DIM,
        )
        lbl_sub.pack(pady=(0, 16), padx=16, anchor="w")

        # --- Book folder ---
        ctk.CTkLabel(sidebar, text="Book Folder", font=self._font(13, True), text_color=COLOR_TEXT).pack(
            padx=16, pady=(8, 2), anchor="w"
        )
        self.entry_folder = ctk.CTkEntry(
            sidebar, placeholder_text="./book", font=self._font(12), fg_color=COLOR_BG
        )
        self.entry_folder.pack(padx=16, pady=2, fill="x")
        self.entry_folder.insert(0, "./book")

        btn_browse = ctk.CTkButton(
            sidebar,
            text="Browse…",
            command=self._on_browse_folder,
            font=self._font(12),
            fg_color=COLOR_BORDER,
            hover_color="#3a3a3a",
            text_color=COLOR_TEXT,
            height=32,
        )
        btn_browse.pack(padx=16, pady=(2, 8), fill="x")

        # --- Parse button ---
        btn_parse = ctk.CTkButton(
            sidebar,
            text="Parse PDFs",
            command=self._on_parse,
            font=self._font(13, True),
            fg_color=COLOR_CYAN,
            hover_color="#00c8d6",
            text_color="black",
            height=36,
        )
        btn_parse.pack(padx=16, pady=8, fill="x")

        sep1 = ctk.CTkFrame(sidebar, height=2, fg_color=COLOR_BORDER)
        sep1.pack(padx=16, pady=10, fill="x")

        # --- Voice selector ---
        ctk.CTkLabel(sidebar, text="Voice", font=self._font(13, True), text_color=COLOR_TEXT).pack(
            padx=16, pady=(4, 2), anchor="w"
        )
        self.combo_voice = ctk.CTkComboBox(
            sidebar,
            values=[],
            font=self._font(12),
            fg_color=COLOR_BG,
            dropdown_fg_color=COLOR_BG,
            button_color=COLOR_BORDER,
        )
        self.combo_voice.pack(padx=16, pady=2, fill="x")

        btn_add_voice = ctk.CTkButton(
            sidebar,
            text="+ Add Voice from Audio",
            command=self._on_add_voice,
            font=self._font(11),
            fg_color=COLOR_BORDER,
            hover_color="#3a3a3a",
            text_color=COLOR_TEXT,
            height=28,
        )
        btn_add_voice.pack(padx=16, pady=(4, 8), fill="x")

        # --- Font selector (preview text) ---
        ctk.CTkLabel(
            sidebar, text="Preview Font", font=self._font(13, True), text_color=COLOR_TEXT
        ).pack(padx=16, pady=(4, 2), anchor="w")
        self.combo_font = ctk.CTkComboBox(
            sidebar,
            values=["Segoe UI", "Consolas", "Georgia", "Times New Roman", "Arial"],
            font=self._font(12),
            fg_color=COLOR_BG,
            command=self._on_font_change,
        )
        self.combo_font.pack(padx=16, pady=2, fill="x")
        self.combo_font.set("Segoe UI")

        sep2 = ctk.CTkFrame(sidebar, height=2, fg_color=COLOR_BORDER)
        sep2.pack(padx=16, pady=10, fill="x")

        # --- Generate button ---
        btn_gen = ctk.CTkButton(
            sidebar,
            text="Generate Audiobook",
            command=self._on_generate,
            font=self._font(14, True),
            fg_color=COLOR_MAGENTA,
            hover_color="#d60088",
            text_color="white",
            height=44,
        )
        btn_gen.pack(padx=16, pady=8, fill="x")

        btn_cancel = ctk.CTkButton(
            sidebar,
            text="Cancel",
            command=self._on_cancel,
            font=self._font(12),
            fg_color=COLOR_BORDER,
            hover_color="#3a3a3a",
            text_color=COLOR_TEXT,
            height=32,
        )
        btn_cancel.pack(padx=16, pady=(0, 8), fill="x")

        # --- Progress ---
        self.progress = ctk.CTkProgressBar(sidebar, height=14, corner_radius=7)
        self.progress.pack(padx=16, pady=(4, 2), fill="x")
        self.progress.set(0)

        self.lbl_status = ctk.CTkLabel(
            sidebar, text="Ready", font=self._font(11), text_color=COLOR_TEXT_DIM
        )
        self.lbl_status.pack(padx=16, pady=(0, 12), anchor="w")

        # ---- Right content area -----------------------------------------
        content = ctk.CTkFrame(self, fg_color=COLOR_CARD, corner_radius=12)
        content.grid(row=0, column=1, padx=(0, 12), pady=12, sticky="nsew")
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(1, weight=2)
        content.grid_rowconfigure(3, weight=1)

        # Header
        header = ctk.CTkLabel(
            content,
            text="Chapters",
            font=ctk.CTkFont(family=FONT_FAMILY, size=18, weight="bold"),
            text_color=COLOR_TEXT,
        )
        header.grid(row=0, column=0, padx=16, pady=(12, 4), sticky="w")

        # Chapter tree
        tree_frame = ctk.CTkFrame(content, fg_color=COLOR_BG, corner_radius=8)
        tree_frame.grid(row=1, column=0, padx=16, pady=4, sticky="nsew")
        tree_frame.grid_columnconfigure(0, weight=1)
        tree_frame.grid_rowconfigure(0, weight=1)

        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "Treeview",
            background=COLOR_BG,
            foreground=COLOR_TEXT,
            fieldbackground=COLOR_BG,
            borderwidth=0,
            font=(FONT_FAMILY, 11),
        )
        style.configure("Treeview.Heading", background=COLOR_CARD, foreground=COLOR_TEXT, font=(FONT_FAMILY, 11, "bold"))
        style.map("Treeview", background=[("selected", COLOR_BORDER)])

        self.tree = ttk.Treeview(tree_frame, columns=("chars",), show="headings", selectmode="extended")
        self.tree.heading("#0", text="Chapter")
        self.tree.heading("chars", text="Chars")
        self.tree.column("#0", width=300)
        self.tree.column("chars", width=80, anchor="e")
        self.tree.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=vsb.set)

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # Text preview
        ctk.CTkLabel(
            content, text="Preview", font=self._font(14, True), text_color=COLOR_TEXT
        ).grid(row=2, column=0, padx=16, pady=(12, 4), sticky="w")

        preview_frame = ctk.CTkFrame(content, fg_color=COLOR_BG, corner_radius=8)
        preview_frame.grid(row=3, column=0, padx=16, pady=4, sticky="nsew")
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_rowconfigure(0, weight=1)

        self.text_preview = tk.Text(
            preview_frame,
            wrap="word",
            bg=COLOR_BG,
            fg=COLOR_TEXT,
            font=(FONT_FAMILY, 12),
            padx=8,
            pady=8,
            relief="flat",
            highlightthickness=0,
        )
        self.text_preview.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        psb = ttk.Scrollbar(preview_frame, orient="vertical", command=self.text_preview.yview)
        psb.grid(row=0, column=1, sticky="ns")
        self.text_preview.configure(yscrollcommand=psb.set)

        # Audio player bar
        player = ctk.CTkFrame(content, fg_color=COLOR_BG, corner_radius=8, height=80)
        player.grid(row=4, column=0, padx=16, pady=(12, 16), sticky="ew")
        player.grid_columnconfigure(1, weight=1)

        self.btn_play = ctk.CTkButton(
            player,
            text="▶",
            width=40,
            height=40,
            font=self._font(16),
            fg_color=COLOR_CYAN,
            hover_color="#00c8d6",
            text_color="black",
            command=self._on_play,
        )
        self.btn_play.grid(row=0, column=0, padx=(12, 6), pady=12)

        self.btn_pause = ctk.CTkButton(
            player,
            text="⏸",
            width=40,
            height=40,
            font=self._font(16),
            fg_color=COLOR_BORDER,
            hover_color="#3a3a3a",
            text_color=COLOR_TEXT,
            command=self._on_pause,
        )
        self.btn_pause.grid(row=0, column=1, padx=6, pady=12, sticky="w")

        self.slider_seek = ctk.CTkSlider(
            player,
            from_=0,
            to=100,
            number_of_steps=100,
            fg_color=COLOR_BORDER,
            progress_color=COLOR_CYAN,
            button_color=COLOR_CYAN,
            command=self._on_seek,
        )
        self.slider_seek.grid(row=0, column=2, padx=12, pady=12, sticky="ew")

        self.lbl_time = ctk.CTkLabel(
            player, text="0:00 / 0:00", font=self._font(11), text_color=COLOR_TEXT_DIM
        )
        self.lbl_time.grid(row=0, column=3, padx=(0, 12), pady=12)

        self.slider_volume = ctk.CTkSlider(
            player,
            from_=0,
            to=100,
            number_of_steps=100,
            width=100,
            fg_color=COLOR_BORDER,
            progress_color=COLOR_MAGENTA,
            button_color=COLOR_MAGENTA,
            command=self._on_volume,
        )
        self.slider_volume.set(80)
        self.slider_volume.grid(row=0, column=4, padx=(0, 12), pady=12)
        self._on_volume(80)

        # Periodic UI updater
        self._poll_id = self.after(500, self._poll_ui)

    def _font(self, size: int, bold: bool = False):
        return ctk.CTkFont(family=FONT_FAMILY, size=size, weight=("bold" if bold else "normal"))

    # ------------------------------------------------------------------
    # Background init
    # ------------------------------------------------------------------

    def _init_tts_bg(self):
        def load():
            try:
                self.tts = DemonTTS()
                self.after(0, lambda: self.lbl_status.configure(text="Models loaded"))
            except Exception as exc:
                self.after(0, lambda: self.lbl_status.configure(text=f"Model load error: {exc}"))

        threading.Thread(target=load, daemon=True).start()
        self.lbl_status.configure(text="Loading models…")

    # ------------------------------------------------------------------
    # Voices
    # ------------------------------------------------------------------

    def _load_voices_into_combo(self):
        voices_path = Path("./models/voices.json")
        if voices_path.exists():
            data = json.loads(voices_path.read_text())
            names = list(data.keys())
        else:
            names = []
        self.combo_voice.configure(values=names)
        if names:
            self.combo_voice.set(names[0])

    def _on_add_voice(self):
        path = filedialog.askopenfilename(
            title="Select a 3-second voice sample",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg")],
        )
        if not path:
            return
        name = simpledialog.askstring("Voice Name", "Enter a name for this voice:")
        if not name:
            return
        if self.tts is None:
            messagebox.showerror("Error", "TTS models are still loading. Please wait.")
            return
        try:
            emb = self.tts.clone_voice(path)
            self.tts.voices[name] = emb
            self.tts.save_voices()
            self._load_voices_into_combo()
            self.combo_voice.set(name)
            messagebox.showinfo("Success", f"Voice '{name}' added.")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    # ------------------------------------------------------------------
    # Folder / Parse
    # ------------------------------------------------------------------

    def _on_browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.entry_folder.delete(0, "end")
            self.entry_folder.insert(0, folder)

    def _on_parse(self):
        folder = self.entry_folder.get().strip()
        if not folder:
            messagebox.showwarning("Warning", "Please select a book folder first.")
            return
        self.lbl_status.configure(text="Parsing PDFs…")
        self.progress.set(0.3)
        self.tree.delete(*self.tree.get_children())
        self.chapters.clear()

        def parse():
            try:
                result = self.parser.parse_folder(folder)
                self.after(0, lambda: self._populate_chapters(result))
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("Parse Error", str(exc)))
                self.after(0, lambda: self.lbl_status.configure(text="Parse failed"))
                self.after(0, lambda: self.progress.set(0))

        threading.Thread(target=parse, daemon=True).start()

    def _populate_chapters(self, result: Dict[str, Dict]):
        self.chapters = result
        for key, ch in result.items():
            chars = len(ch.get("text", ""))
            self.tree.insert("", "end", iid=key, text=key, values=(chars,))
        self.lbl_status.configure(text=f"Parsed {len(result)} chapters")
        self.progress.set(1.0)

    # ------------------------------------------------------------------
    # Tree / Preview
    # ------------------------------------------------------------------

    def _on_tree_select(self, _event=None):
        sel = self.tree.selection()
        if not sel:
            return
        key = sel[0]
        text = self.chapters.get(key, {}).get("text", "")
        self.text_preview.delete("1.0", "end")
        self.text_preview.insert("1.0", text[:4000])  # limit preview

    def _on_font_change(self, font_name: str):
        self.text_preview.configure(font=(font_name, 12))

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _on_generate(self):
        if not self.chapters:
            messagebox.showwarning("Warning", "No chapters parsed yet.")
            return
        if self.tts is None:
            messagebox.showerror("Error", "TTS models are still loading.")
            return

        sel = self.tree.selection()
        if not sel:
            sel = list(self.chapters.keys())
        else:
            sel = list(sel)

        voice = self.combo_voice.get()
        out_dir = Path("./audiobook")
        out_dir.mkdir(parents=True, exist_ok=True)

        self.generation_cancelled = False
        self.progress.set(0)
        self.lbl_status.configure(text="Generating…")

        def gen():
            total = len(sel)
            chapter_wavs = []
            start_time = time.time()

            for idx, key in enumerate(sel, 1):
                if self.generation_cancelled:
                    self.after(0, lambda: self.lbl_status.configure(text="Cancelled"))
                    return

                text = self.chapters[key]["text"]
                self.after(0, lambda k=key: self.lbl_status.configure(text=f"Synthesizing: {k[:40]}…"))

                try:
                    wav = self.tts.synthesize(text, voice_id=voice)
                    # Save per-chapter
                    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in key)[:60]
                    fpath = out_dir / f"{safe_name}.flac"
                    self.tts.save_wav(wav, str(fpath))
                    chapter_wavs.append(wav)
                except Exception as exc:
                    self.after(0, lambda e=exc: messagebox.showerror("Synthesis Error", str(e)))
                    continue

                # Progress + ETA
                frac = idx / total
                elapsed = time.time() - start_time
                eta = (elapsed / idx) * (total - idx) if idx > 0 else 0
                self.after(0, lambda f=frac, e=eta: (
                    self.progress.set(f),
                    self.lbl_status.configure(text=f"Progress {int(f*100)}%  ETA {_fmt_eta(e)}"),
                ))

            if not chapter_wavs:
                self.after(0, lambda: self.lbl_status.configure(text="No audio generated."))
                return

            # Combined audiobook with 0.5s silence
            silence = np.zeros(int(0.5 * 24_000), dtype=np.float32)
            combined = []
            for w in chapter_wavs:
                combined.append(w)
                combined.append(silence)
            if combined:
                combined.pop()  # remove trailing silence
            full_wav = np.concatenate(combined)
            full_path = out_dir / "full_audiobook.flac"
            sf.write(str(full_path), full_wav, 24_000)

            self.after(0, lambda: (
                self.lbl_status.configure(text=f"Done! Saved to {out_dir}"),
                self.progress.set(1.0),
                messagebox.showinfo("Success", f"Audiobook saved to\n{out_dir}"),
            ))

        self.generation_thread = threading.Thread(target=gen, daemon=True)
        self.generation_thread.start()

    def _on_cancel(self):
        self.generation_cancelled = True
        self.lbl_status.configure(text="Cancelling…")

    # ------------------------------------------------------------------
    # Audio player
    # ------------------------------------------------------------------

    def _on_play(self):
        sel = self.tree.selection()
        if not sel:
            return
        key = sel[0]
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in key)[:60]
        fpath = Path("./audiobook") / f"{safe_name}.flac"
        if not fpath.exists():
            # Try to load the last generated preview if any
            if self.current_wav is not None:
                self._play_array(self.current_wav)
            return
        self._play_file(str(fpath))

    def _on_pause(self):
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.pause()
            self._audio_paused = True
        elif self._audio_paused:
            pygame.mixer.music.unpause()
            self._audio_paused = False

    def _on_seek(self, value: float):
        # Pygame mixer seek is limited; replay from offset
        if self._audio_path and Path(self._audio_path).exists():
            total = self._get_duration(self._audio_path)
            pos = (value / 100.0) * total
            pygame.mixer.music.play(start=pos)

    def _on_volume(self, value: float):
        pygame.mixer.music.set_volume(value / 100.0)

    def _play_file(self, path: str):
        self._audio_path = path
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        self._audio_paused = False
        self._audio_start_time = time.time()

    def _play_array(self, wav: np.ndarray, sr: int = 24_000):
        tmp = "_preview.wav"
        sf.write(tmp, wav, sr)
        self._play_file(tmp)

    @staticmethod
    def _get_duration(path: str) -> float:
        try:
            info = sf.info(path)
            return info.duration
        except Exception:
            return 0.0

    def _poll_ui(self):
        # Update time label
        if self._audio_path and Path(self._audio_path).exists():
            total = self._get_duration(self._audio_path)
            if pygame.mixer.music.get_busy() and not self._audio_paused:
                pos = pygame.mixer.music.get_pos() / 1000.0
                self.lbl_time.configure(text=f"{int(pos)//60}:{int(pos)%60:02d} / {int(total)//60}:{int(total)%60:02d}")
                if total > 0:
                    self.slider_seek.set((pos / total) * 100)
            elif not pygame.mixer.music.get_busy() and not self._audio_paused:
                self.lbl_time.configure(text="0:00 / 0:00")
                self.slider_seek.set(0)
        self._poll_id = self.after(500, self._poll_ui)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main():
    app = DemonGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
