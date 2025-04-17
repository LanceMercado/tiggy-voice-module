import torch
import numpy as np
from TTS.api import TTS
from torch.serialization import add_safe_globals
from TTS.tts.models.xtts import XttsArgs
from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig
from pathlib import Path
import sys

# Allow necessary globals for safe deserialization (PyTorch >= 2.6)
add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

def main():
    # Load XTTS model (speaker embedding support)
    try:
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    except Exception as e:
        print(f"⚠️ Failed to load TTS model: {e}", file=sys.stderr)
        sys.exit(1)

    # Input/output paths
    sample_dir = Path("/app/voice_data/samples")
    output_path = Path("/app/voice_data/tiggy_embed.pth")

    if not sample_dir.exists():
        print(f"❌ Sample directory not found: {sample_dir}", file=sys.stderr)
        sys.exit(1)

    wav_files = sorted(sample_dir.glob("*.wav"))
    if not wav_files:
        print("❌ No .wav files found in sample directory.", file=sys.stderr)
        sys.exit(1)

    embeddings = []
    print(f"✅ Found {len(wav_files)} sample(s):")
    for wav_file in wav_files:
        print(f"  - {wav_file.name}")
        try:
            # Extract speaker embedding via the XTTS model's conditioning latents
            latents = tts.synthesizer.tts_model.get_conditioning_latents(audio_path=str(wav_file))
            if isinstance(latents, dict):
                emb_np = latents.get("d_vector")
            elif isinstance(latents, tuple) and len(latents) >= 2:
                # latents tuple: (gpt_cond_latent, speaker_embedding)
                emb_np = latents[1]
            else:
                raise ValueError("Unexpected latents format")
            emb_tensor = torch.from_numpy(np.array(emb_np))
            embeddings.append(emb_tensor)
        except Exception as e:
            print(f"⚠️ Failed to process {wav_file.name}: {e}", file=sys.stderr)

    if not embeddings:
        print("❌ No embeddings were successfully computed.", file=sys.stderr)
        sys.exit(1)

    # Average embeddings and save
    avg_embedding = torch.mean(torch.stack(embeddings), dim=0)
    torch.save(avg_embedding, output_path)
    print(f"✅ Saved average voice embedding to: {output_path}")

if __name__ == "__main__":
    main()