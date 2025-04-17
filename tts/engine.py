
import subprocess

def synthesize(text, output_path="output/tiggy.wav"):
    cmd = [
        "tts",
        "--model_name", "tts_models/multilingual/multi-dataset/xtts_v2",
        "--text", text,
        "--speaker_embedding_path", "voice_data/tiggy_embed.pth",
        "--out_path", output_path
    ]
    subprocess.run(cmd, check=True)
