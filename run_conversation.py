#!/usr/bin/env python3
import os
# Disable Triton compilation (useful on macOS MPS)
os.environ["NO_TORCH_COMPILE"] = "1"

import argparse
import re
import sys

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment

def load_prompt_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor:
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = audio_tensor.squeeze(0)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
    )
    return audio_tensor

def prepare_prompt(text: str, speaker_id: int, audio_path: str, sample_rate: int) -> Segment:
    audio = load_prompt_audio(audio_path, sample_rate)
    return Segment(text=text, speaker=speaker_id, audio=audio)

# Download default speaker prompt WAVs
prompt_filepath_conversational_a = hf_hub_download(
    repo_id="sesame/csm-1b",
    filename="prompts/conversational_a.wav"
)
prompt_filepath_conversational_b = hf_hub_download(
    repo_id="sesame/csm-1b",
    filename="prompts/conversational_b.wav"
)

# Speaker style definitions (text + audio) pulled from sesame/csm-1b
SPEAKER_PROMPTS = {
    "conversational_a": {
        "text": (
            "like revising for an exam I'd have to try and like keep up the momentum because I'd "
            "start really early I'd be like okay I'm gonna start revising now and then like "
            "you're revising for ages and then I just like start losing steam I didn't do that "
            "for the exam we had recently to be fair that was a more of a last minute scenario "
            "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
            "sort of start the day with this not like a panic but like a"
        ),
        "audio": prompt_filepath_conversational_a,
    },
    "conversational_b": {
        "text": (
            "like a super Mario level. Like it's very like high detail. And like, once you get "
            "into the park, it just like, everything looks like a computer game and they have all "
            "these, like, you know, if, if there's like a, you know, like in a Mario game, they "
            "will have like a question block. And if you like, you know, punch it, a coin will "
            "come out. So like everyone, when they come into the park, they get like this little "
            "bracelet and then you can go punching question blocks around."
        ),
        "audio": prompt_filepath_conversational_b,
    },
}

def main():
    parser = argparse.ArgumentParser(
        description="Generate a conversation from a text file using CSM-1B TTS"
    )
    parser.add_argument(
        "input",
        help="Path to input text file. Lines should start with '0:' or '1:' for speaker ID."
    )
    parser.add_argument(
        "--style0",
        choices=list(SPEAKER_PROMPTS.keys()),
        default="conversational_a",
        help="Prompt style for speaker 0",
    )
    parser.add_argument(
        "--style1",
        choices=list(SPEAKER_PROMPTS.keys()),
        default="conversational_b",
        help="Prompt style for speaker 1",
    )
    parser.add_argument(
        "--output",
        default="full_conversation.wav",
        help="Output WAV file path",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Compute device",
    )
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Load TTS generator
    generator = load_csm_1b(device)

    # Prepare speaker prompt segments
    prompt_segments = []
    for speaker_id, style in enumerate([args.style0, args.style1]):
        info = SPEAKER_PROMPTS[style]
        seg = prepare_prompt(
            text=info["text"], speaker_id=speaker_id,
            audio_path=info["audio"], sample_rate=generator.sample_rate
        )
        prompt_segments.append(seg)

    # Read conversation lines from input
    conversation = []
    with open(args.input, "r") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^([01])\s*[:\-\s]\s*(.+)$", line)
            if not m:
                print(f"Skipping invalid line {lineno}: {line}")
                continue
            conversation.append({
                "speaker_id": int(m.group(1)),
                "text": m.group(2),
            })

    if not conversation:
        print("No valid utterances found in input file.")
        sys.exit(1)

    # Set max audio length (shorter on non-CUDA devices)
    max_audio_length_ms = 10000 if device == "cuda" else 2000

    # Generate each utterance
    generated = []
    for utt in conversation:
        sid = utt["speaker_id"]
        txt = utt["text"]
        print(f"Generating speaker {sid}: {txt}")
        audio = generator.generate(
            text=txt,
            speaker=sid,
            context=prompt_segments + generated,
            max_audio_length_ms=max_audio_length_ms,
        )
        generated.append(Segment(text=txt, speaker=sid, audio=audio))

    # Concatenate and save
    all_audio = torch.cat([seg.audio for seg in generated], dim=0)
    torchaudio.save(args.output, all_audio.unsqueeze(0).cpu(), generator.sample_rate)
    print(f"Saved conversation to {args.output}")

if __name__ == "__main__":
    main()