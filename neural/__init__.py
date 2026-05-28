from .student import StudentTTS
from .speaker_encoder import SpeakerEncoder

__all__ = ["StudentTTS", "SpeakerEncoder"]

# Optional training-only imports
try:
    from .distillation_wrapper import DistillationTrainer
    from .vocoder import HiFiGenerator
    __all__.extend(["DistillationTrainer", "HiFiGenerator"])
except Exception:
    pass
