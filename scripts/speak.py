
import argparse
from tts.engine import synthesize

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--out", default="output/tiggy.wav", help="Output WAV file path")
    args = parser.parse_args()

    synthesize(args.text, output_path=args.out)
