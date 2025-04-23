import os
# Disable torch.compile for moshi utilities
os.environ["NO_TORCH_COMPILE"] = "1"
import time
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment
from dataclasses import dataclass
import argparse

# Default prompts are available at https://hf.co/sesame/csm-1b
prompt_filepath_conversational_a = hf_hub_download(
    repo_id="sesame/csm-1b",
    filename="prompts/conversational_a.wav"
)
prompt_filepath_conversational_b = hf_hub_download(
    repo_id="sesame/csm-1b",
    filename="prompts/conversational_b.wav"
)

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
        "audio": prompt_filepath_conversational_a
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
        "audio": prompt_filepath_conversational_b
    }
}

def load_prompt_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor:
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = audio_tensor.squeeze(0)
    # Resample is lazy so we can always call it
    audio_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
    )
    return audio_tensor

def prepare_prompt(text: str, speaker: int, audio_path: str, sample_rate: int) -> Segment:
    audio_tensor = load_prompt_audio(audio_path, sample_rate)
    return Segment(text=text, speaker=speaker, audio=audio_tensor)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate CSM conversation from a conversation text file")
    parser.add_argument("--speaker1-id", type=int, default=0, help="ID to assign to speaker1 in the conversation")
    parser.add_argument("--speaker2-id", type=int, default=1, help="ID to assign to speaker2 in the conversation")
    parser.add_argument("--conversation-file", type=str, default=os.path.join(os.path.dirname(__file__), "sample_conversation.txt"), help="Path to the conversation text file")
    parser.add_argument("--output-file", "-o", type=str, default="full_conversation.wav", help="Output WAV file path")
    args = parser.parse_args()

    # Select the best available device, skipping MPS due to float64 limitations
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    # GPU diagnostics
    if device.startswith("cuda"):
        print("Torch CUDA version:", torch.version.cuda)
        print("CUDA available:", torch.cuda.is_available())
        num_dev = torch.cuda.device_count()
        print("Number of CUDA devices:", num_dev)
        for i in range(num_dev):
            print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")

    # Load model
    generator = load_csm_1b(device)
    # Confirm model device and memory usage
    print(f"Model loaded on device: {generator.device}")
    if device.startswith("cuda"):
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    # Prepare prompts
    prompt_a = prepare_prompt(
        SPEAKER_PROMPTS["conversational_a"]["text"],
        0,
        SPEAKER_PROMPTS["conversational_a"]["audio"],
        generator.sample_rate
    )

    prompt_b = prepare_prompt(
        SPEAKER_PROMPTS["conversational_b"]["text"],
        1,
        SPEAKER_PROMPTS["conversational_b"]["audio"],
        generator.sample_rate
    )

    # Load conversation from text file
    conversation = []
    conv_filepath = args.conversation_file
    if not os.path.exists(conv_filepath):
        raise FileNotFoundError(f"Conversation file not found: {conv_filepath}")
    with open(conv_filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(f"Invalid line in {conv_filepath}: {line}")
            speaker, text = line.split(":", 1)
            speaker = speaker.strip().lower()
            text = text.strip()
            if speaker in ("speaker1", "0"):
                speaker_id = args.speaker1_id
            elif speaker in ("speaker2", "1"):
                speaker_id = args.speaker2_id
            else:
                raise ValueError(f"Unknown speaker '{speaker}' in {conv_filepath}")
            conversation.append({"text": text, "speaker_id": speaker_id})

    # Generate each utterance
    generated_segments = []
    prompt_segments = [prompt_a, prompt_b]

    for utterance in conversation:
        print(f"Generating: {utterance['text']}")
        start_time = time.time()
        audio_tensor = generator.generate(
            text=utterance['text'],
            speaker=utterance['speaker_id'],
            context=prompt_segments + generated_segments,
            max_audio_length_ms=10_000,
        )
        elapsed = time.time() - start_time
        print(f"TTS time for \"{utterance['text']}\": {elapsed:.3f} seconds")
        generated_segments.append(Segment(text=utterance['text'], speaker=utterance['speaker_id'], audio=audio_tensor))

    # Concatenate all generations
    all_audio = torch.cat([seg.audio for seg in generated_segments], dim=0)
    torchaudio.save(
        args.output_file,
        all_audio.unsqueeze(0).cpu(),
        generator.sample_rate
    )
    print(f"Successfully generated {args.output_file}")

if __name__ == "__main__":
    main() 