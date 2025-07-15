from playsound import playsound
from pathlib import Path

def play_alert_sound():
    try:
        sound_path = Path("alert.mp3").resolve()
        playsound(str(sound_path))
    except Exception as e:
        print(f"Sound playback failed: {e}")

if __name__ == "__main__":
    play_alert_sound()
