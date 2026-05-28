use anyhow::Result;
use clap::{Parser, Subcommand};
use demon_pipeline::{audio_io, DemonPipeline};
use std::time::Instant;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser)]
#[command(name = "demon-tts")]
#[command(about = "ElevenLabs-killing TTS inference pipeline")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Synthesize audio from text
    Synthesize {
        #[arg(short, long)]
        text: String,
        #[arg(short, long)]
        speaker: Option<String>,
        #[arg(short, long, default_value = "output.wav")]
        output: String,
    },
    /// Benchmark RTF over N synthetic chapters
    Benchmark {
        #[arg(short, long, default_value = "800")]
        chapters: usize,
    },
}

fn main() -> Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let cli = Cli::parse();

    match cli.command {
        Commands::Synthesize { text, speaker, output } => {
            let mut pipeline = load_pipeline()?;
            let t0 = Instant::now();
            let samples = pipeline.synthesize(&text, speaker.as_deref())?;
            let elapsed = t0.elapsed().as_secs_f32();
            let audio_duration = samples.len() as f32 / 24_000.0;
            let rtf = elapsed / audio_duration;

            audio_io::write_wav(&output, &samples, 24_000)?;
            info!(
                "Synthesized {}s audio in {:.2}s (RTF={:.3}) -> {}",
                audio_duration, elapsed, rtf, output
            );
        }
        Commands::Benchmark { chapters } => {
            run_benchmark(chapters)?;
        }
    }

    Ok(())
}

fn load_pipeline() -> Result<DemonPipeline> {
    let base = std::env::var("DEMON_MODELS").unwrap_or_else(|_| "./checkpoints".to_string());
    DemonPipeline::new(
        &format!("{}/student/export/student_int8.onnx", base),
        &format!("{}/faraday/export/faraday.onnx", base),
        &format!("{}/student/export/vocoder.onnx", base),
        &format!("{}/aether/export/aether.onnx", base),
        &format!("{}/tokenizer/vocab.json", base),
        &format!("{}/tokenizer/merges.txt", base),
    )
}

fn run_benchmark(chapters: usize) -> Result<()> {
    let mut pipeline = load_pipeline()?;
    let dummy_text = "The quick brown fox jumps over the lazy dog. ".repeat(50);
    let mut total_rtf = 0.0f32;
    let mut total_ttfb = 0.0f32;

    info!("Starting benchmark: {} chapters", chapters);
    for i in 0..chapters {
        let t0 = Instant::now();
        let samples = pipeline.synthesize(&dummy_text, None)?;
        let elapsed = t0.elapsed().as_secs_f32();
        let audio_dur = samples.len() as f32 / 24_000.0;
        let rtf = elapsed / audio_dur;
        total_rtf += rtf;
        total_ttfb += elapsed; // simplified TTFB ≈ total for short text

        if (i + 1) % 100 == 0 {
            info!(
                "Progress: {}/{} | Avg RTF: {:.3} | Avg TTFB: {:.3}s",
                i + 1,
                chapters,
                total_rtf / (i + 1) as f32,
                total_ttfb / (i + 1) as f32
            );
        }
    }

    let avg_rtf = total_rtf / chapters as f32;
    let avg_ttfb = total_ttfb / chapters as f32;
    info!("Benchmark complete. Avg RTF={:.3}, Avg TTFB={:.3}s", avg_rtf, avg_ttfb);

    if avg_rtf > 0.5 {
        eprintln!("WARNING: RTF target 0.5 exceeded. Optimize kernels.");
    }

    Ok(())
}
