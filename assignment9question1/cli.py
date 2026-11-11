"""Command-line use:  python cli.py samples/demo_label.png --speak --ask "when does it expire?" """
import argparse
import json

from medassist.assistant import MedicineLabelAssistant


def main():
    ap = argparse.ArgumentParser(description="Read a medicine label from an image.")
    ap.add_argument("image", nargs="?", help="Path to label image")
    ap.add_argument("--text", help="Skip OCR and analyse this text instead")
    ap.add_argument("--backend", default="auto", choices=["auto", "easyocr", "tesseract"])
    ap.add_argument("--gpu", dest="gpu", action="store_true", default=None, help="force GPU (EasyOCR)")
    ap.add_argument("--cpu", dest="gpu", action="store_false", help="force CPU")
    ap.add_argument("--db", help="Path to drug-name CSV (default data/drugs.csv or sample)")
    ap.add_argument("--json", action="store_true", help="Print structured JSON")
    ap.add_argument("--speak", action="store_true", help="Save spoken summary to summary.mp3")
    ap.add_argument("--ask", help="Ask a question about the label")
    a = ap.parse_args()

    bot = MedicineLabelAssistant(a.backend, a.db, a.gpu)
    if a.text:
        info = bot.read_text(a.text)
    elif a.image:
        info, res, _ = bot.read_image(a.image)
        print(f"[OCR: {res.backend}/{res.variant}, confidence {res.confidence:.0f}%]\n")
    else:
        ap.error("give an image path or --text")

    print(json.dumps(info.to_dict(), indent=2) if a.json else info.summary())
    if a.ask:
        print("\nQ:", a.ask, "\nA:", bot.ask(a.ask))
    if a.speak:
        from medassist.speech import speak_to_file
        print("Audio saved to", speak_to_file(info.summary(), "summary.mp3"))


if __name__ == "__main__":
    main()
