#!/usr/bin/env python3
"""
Batch audiobook generation for cloud compute.
Processes entire libraries in parallel across multiple GPUs.
"""
import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
import torch.multiprocessing as mp

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from demo_tts import DemonTTS
from pdf_parser import PDFParser


def process_book(args_tuple):
    """Worker function for parallel book processing."""
    book_path, output_dir, voice_id, device_id = args_tuple
    
    # Set GPU for this worker
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
    
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    
    try:
        # Initialize TTS (each worker gets its own instance)
        tts = DemonTTS(device=device)
        
        # Parse PDF
        parser = PDFParser()
        book_data = parser.parse_pdf(book_path)
        
        book_name = Path(book_path).stem
        book_output = Path(output_dir) / book_name
        book_output.mkdir(parents=True, exist_ok=True)
        
        # Get voice embedding
        speaker_emb = None
        if voice_id and voice_id in tts.voices:
            speaker_emb = torch.tensor(tts.voices[voice_id]).to(device)
        
        # Process each chapter
        total_chars = 0
        start_time = time.time()
        chapter_files = []
        
        for i, chapter in enumerate(book_data.get("chapters", [])):
            text = chapter.get("text", "").strip()
            if not text:
                continue
            
            total_chars += len(text)
            safe_title = "".join(c for c in chapter.get("title", f"ch{i:03d}") 
                                if c.isalnum() or c in (' ', '-', '_')).rstrip()
            
            out_path = book_output / f"{i:03d}_{safe_title}.flac"
            
            # Synthesize
            wav = tts.synthesize(text, speaker_emb=speaker_emb)
            tts.save_audio(wav, str(out_path))
            chapter_files.append(str(out_path))
            
            # Progress
            elapsed = time.time() - start_time
            chars_per_sec = total_chars / max(elapsed, 0.001)
            print(f"[{book_name}] Ch {i+1}/{len(book_data.get('chapters', []))} "
                  f"done | {chars_per_sec:.0f} chars/sec | device=cuda:{device_id}")
        
        # Combine into single audiobook
        if chapter_files:
            combined_path = book_output / "full_audiobook.flac"
            tts.combine_chapters(chapter_files, str(combined_path))
        
        elapsed = time.time() - start_time
        return {
            "book": book_name,
            "status": "success",
            "chapters": len(chapter_files),
            "chars": total_chars,
            "time_sec": elapsed,
            "rtf": elapsed / max(total_chars / 1000, 0.001),
            "output_dir": str(book_output),
            "device": device_id,
        }
    
    except Exception as e:
        return {
            "book": Path(book_path).stem,
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "device": device_id,
        }


def main():
    parser = argparse.ArgumentParser(description="Batch audiobook generation")
    parser.add_argument("--book_dir", required=True, help="Directory containing PDF books")
    parser.add_argument("--output_dir", required=True, help="Output directory for audiobooks")
    parser.add_argument("--voices", default="./voices.json", help="voices.json path")
    parser.add_argument("--voice", default=None, help="Voice ID to use")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default = num GPUs)")
    parser.add_argument("--gpu_ids", type=str, default=None, help="Comma-separated GPU IDs to use (default = all)")
    args = parser.parse_args()
    
    # Find books
    book_dir = Path(args.book_dir)
    books = sorted(book_dir.glob("*.pdf"))
    print(f"📚 Found {len(books)} books in {book_dir}")
    
    if not books:
        print("No PDFs found!")
        return
    
    # Determine GPU allocation
    if torch.cuda.is_available():
        if args.gpu_ids:
            gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
        else:
            gpu_ids = list(range(torch.cuda.device_count()))
    else:
        gpu_ids = [0]
    
    num_workers = args.workers or len(gpu_ids)
    print(f"🖥️  GPUs: {gpu_ids} | Workers: {num_workers}")
    
    # Build task list with round-robin GPU assignment
    tasks = []
    for i, book in enumerate(books):
        gpu = gpu_ids[i % len(gpu_ids)]
        tasks.append((str(book), args.output_dir, args.voice, gpu))
    
    # Run parallel processing
    mp.set_start_method("spawn", force=True)
    results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_book, task): task for task in tasks}
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            
            if result["status"] == "success":
                print(f"✅ {result['book']}: {result['chapters']} chapters, "
                      f"{result['chars']} chars in {result['time_sec']:.1f}s "
                      f"(RTF={result['rtf']:.3f})")
            else:
                print(f"❌ {result['book']}: FAILED — {result['error']}")
    
    # Summary
    successes = [r for r in results if r["status"] == "success"]
    failures = [r for r in results if r["status"] == "error"]
    
    print("\n" + "="*60)
    print(f"📊 BATCH COMPLETE: {len(successes)}/{len(results)} books")
    if successes:
        total_time = sum(r["time_sec"] for r in successes)
        total_chars = sum(r["chars"] for r in successes)
        print(f"   Total time: {total_time/60:.1f} min")
        print(f"   Total chars: {total_chars:,}")
        print(f"   Avg RTF: {sum(r['rtf'] for r in successes)/len(successes):.3f}")
    if failures:
        print(f"\n❌ Failures:")
        for r in failures:
            print(f"   - {r['book']}: {r['error']}")
    print("="*60)
    
    # Write summary JSON
    summary_path = Path(args.output_dir) / "batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📝 Summary written to {summary_path}")


if __name__ == "__main__":
    main()
