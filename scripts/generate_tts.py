from TTS.api import TTS
import torch
import os
import sys

# ✅ Allow PyTorch 2.6+ to unpickle XTTS safely
from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.models.xtts import XttsArgs

torch.serialization.add_safe_globals([
    XttsConfig,
    XttsAudioConfig,
    BaseDatasetConfig,
    XttsArgs,
])

# ✅ Paths
sample_dir = "/app/voice_data/samples"
output_path = "/app/output/tiggy.wav"
text = "Hey Lance, Tiggy is alive and speaking from your Mac."

# 🧠 Collect speaker samples
speaker_wavs = [
    os.path.join(sample_dir, f)
    for f in os.listdir(sample_dir)
    if f.endswith(".wav")
]

# ❌ Handle missing voice data
if not speaker_wavs:
    print(f"❌ No .wav samples found in {sample_dir}")
    sys.exit(1)

print(f"✅ Using {len(speaker_wavs)} voice sample(s):")
for path in speaker_wavs:
    print(f"  - {path}")

# 🔊 Load model and synthesize
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
tts.tts_to_file(
    text=text,
    speaker_wav=speaker_wavs,
    language="en",
    file_path=output_path
)

print(f"✅ Done. Output written to: {output_path}")
