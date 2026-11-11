"""Text-to-speech so visually impaired / elderly users can hear the label."""
from __future__ import annotations


def speak_to_file(text: str, path: str = "label_summary.mp3", lang: str = "en") -> str:
    """Try gTTS (online, natural voice); fall back to pyttsx3 (offline)."""
    try:
        from gtts import gTTS
        gTTS(text=text, lang=lang).save(path)
        return path
    except Exception:
        import pyttsx3
        wav = path.rsplit(".", 1)[0] + ".wav"
        engine = pyttsx3.init()
        engine.save_to_file(text, wav)
        engine.runAndWait()
        return wav


def speak_now(text: str) -> None:
    """Speak straight through the speakers (offline)."""
    import pyttsx3
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
