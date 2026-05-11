import sys

from qwen3_tts_ov.cli import main


if __name__ == "__main__":
    main(["voice-design", *sys.argv[1:]])
