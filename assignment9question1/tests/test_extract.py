import datetime as dt
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from medassist.drugdb import DrugDB
from medassist.extract import extract, parse_date

DB = DrugDB()
TODAY = dt.date(2026, 11, 11)

LABEL = """DOLO 650
Paracetamol Tablets IP 650 mg
Take 1 tablet every 6 hours after food.
Do not exceed 4 tablets in 24 hours.
Batch No: DL2406A
MFG: 03/2025
EXP: 02/2027
MRP Rs. 30.91
Store below 30C. Protect from light and moisture.
Keep out of reach of children."""


def test_full_label():
    i = extract(LABEL, DB, TODAY)
    assert "paracetamol" in i.drug_names and "dolo" in i.drug_names
    assert "650 mg" in i.strengths
    assert i.form == "tablet"
    assert i.expiry_date == "2027-02-28" and i.expiry_status == "valid"
    assert i.manufacture_date == "2025-03-31"
    assert i.batch == "DL2406A" and i.mrp == "30.91"
    assert any("every 6 hours" in s for s in i.instructions)
    assert any("children" in s.lower() for s in i.warnings)


def test_expired():
    i = extract("Amoxicillin Capsules 500 mg\nEXP 08/2024", DB, TODAY)
    assert i.expiry_status == "expired" and "EXPIRED" in i.summary()


def test_ocr_typos_fuzzy_match():
    i = extract("Paracetarnol Tablets 500mg\nEXP: 11/27", DB, TODAY)
    assert "paracetamol" in i.drug_names
    assert i.expiry_date == "2027-11-30"


def test_date_formats():
    assert parse_date("15/08/2027") == dt.date(2027, 8, 15)
    assert parse_date("AUG 2027") == dt.date(2027, 8, 31)
    assert parse_date("2027-08") == dt.date(2027, 8, 31)
    assert parse_date("08.27") == dt.date(2027, 8, 31)
    assert parse_date("nonsense") is None


def test_question_answering():
    from medassist.assistant import MedicineLabelAssistant
    bot = MedicineLabelAssistant()
    bot.read_text(LABEL)
    assert "2027" in bot.ask("When does it expire?")
    assert "6 hours" in bot.ask("How do I take it?")
    assert "children" in bot.ask("Any warnings?").lower()
