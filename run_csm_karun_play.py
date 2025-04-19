#!/usr/bin/env python3
"""
Wrapper script to generate a TTS conversation with CSM and play the resulting audio on speaker.
"""
import platform
import run_csm_karun
import subprocess
import shutil

def main():
    # Generate TTS conversation (creates full_conversation.wav)
    run_csm_karun.main()
    wav_path = "full_conversation.wav"
    system = platform.system()
    played = False
    # Windows: try winsound
    if system == "Windows":
        try:
            import winsound
            print(f"Playing audio via winsound: {wav_path}")
            winsound.PlaySound(wav_path, winsound.SND_FILENAME)
            played = True
        except ImportError:
            pass
    # Non-Windows or fallback: try simpleaudio
    if not played:
        try:
            import simpleaudio as sa
            print(f"Playing audio via simpleaudio: {wav_path}")
            wave_obj = sa.WaveObject.from_wave_file(wav_path)
            play_obj = wave_obj.play()
            play_obj.wait_done()
            played = True
        except Exception:
            pass
    # Fallback to system player (Darwin: afplay; Linux: aplay/paplay/play/ffplay)
    if not played:
        if system == "Darwin":
            players = ["afplay"]
        else:
            players = ["aplay", "paplay", "play", "ffplay"]
        for player in players:
            cmd = shutil.which(player)
            if cmd:
                # ffplay may be noisy, suppress extra output
                args = [player, wav_path]
                if player == "ffplay":
                    args = [player, "-nodisp", "-autoexit", wav_path]
                print(f"Playing audio via {player}: {wav_path}")
                try:
                    subprocess.run(args, check=True)
                    played = True
                except Exception:
                    pass
                break
    # Final fallback
    if not played:
        print(f"Audio generated at {wav_path}. Please play it manually.")

if __name__ == "__main__":
    main()