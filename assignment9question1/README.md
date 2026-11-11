# Cognitive Assistant for Reading Medicine Bottle Labels

Pipeline: **Perceive** (image preprocessing + OCR) -> **Understand** (NLP: drug-name fuzzy
matching, dosage, expiry, batch, warnings) -> **Interact** (Q&A + text-to-speech).

```
photo -> OpenCV cleanup (CLAHE, denoise, deskew, threshold, invert, rotate)
      -> OCR (EasyOCR or Tesseract, best-of-N variants)
      -> Extractor (regex + fuzzy match vs drug DB) -> LabelInfo
      -> Streamlit UI / CLI / spoken summary / question answering
```

## Setup
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
# Tesseract binary (needed for the tesseract backend):
#   Ubuntu: sudo apt install tesseract-ocr    macOS: brew install tesseract
#   Windows: https://github.com/UB-Mannheim/tesseract/wiki  (add to PATH)
```
EasyOCR is optional (better on curved/noisy labels, downloads models on first run).

## Run
```bash
python scripts/make_demo_label.py                 # synthetic test label
python cli.py samples/demo_label.png --ask "when does it expire?" --speak
streamlit run app.py                              # full web UI with camera + voice
python -m pytest tests                            # unit tests
```

## Dataset
`data/drugs.csv` (1,804 generic + 132k Indian brand names) was built from the Kaggle
"A-Z Medicine Dataset of India" with `scripts/build_drug_list.py`. Generics are fuzzy-matched
(tolerates OCR typos); brands are exact-match only. If `drugs.csv` is deleted the code falls
back to `data/drugs_sample.csv` (~130 common drugs).
For real use, build a bigger drug-name list and test on real label images:

| Purpose | Source |
|---|---|
| Drug-name list (generic + brand) | openFDA drug label API: https://open.fda.gov/apis/drug/label/  -> `python scripts/fetch_openfda_names.py` |
| Drug-name list (Indian brands) | Kaggle: "A-Z Medicine Dataset of India" -> `python scripts/build_drug_list.py file.csv --column name` |
| Label / pack images | Kaggle: "Pharmaceutical Drugs and Vitamins Synthetic Images"; Roboflow Universe (search "medicine label OCR"); DailyMed https://dailymed.nlm.nih.gov |
| Handwritten prescriptions (extension) | Kaggle: "Illegible Medical Prescription Images Dataset" |

Evaluate: make a CSV `image,drug_name,expiry[,text]` for ~30-50 images, then
`python scripts/evaluate.py images/ truth.csv` (drug-name accuracy, expiry accuracy, CER).

## Bundled image dataset: `medicine_v1i_coco_plus`
`datasets/medicine_v1i_coco_plus/` (COCO format; train / valid / test) is the Roboflow Universe
"medicine" dataset (447 small Turkish pack photos, CC BY 4.0) **plus 2 Indian pack photos**
(Daplin Trio cream, Nor Co-Q10 softgels) added to `train`. Boxes cover the whole pack; the class is
the brand name. Text-level truth for the 2 added images is in `extra_ground_truth.csv`.
```bash
# OCR accuracy on the dataset (Tesseract; add --backend easyocr if installed)
python scripts/evaluate_coco.py datasets/medicine_v1i_coco_plus --split test
python scripts/evaluate_coco.py datasets/medicine_v1i_coco_plus --split test --crop
```
Baseline (Tesseract, original 45-image test split): brand found in raw OCR text 20%, extracted
name correct 8.9% - the original images are tiny (~290x300 px). The 2 added photos (1280 px) are
read well, so image resolution is the main factor. Results go to `<split>_results[_crop]_<backend>.csv`.

## Limitations
- OCR struggles with heavy glare, curved bottles, tiny print; use the editable-text box in the UI.
- Dates are read as dd/mm/yyyy; month-only dates are treated as the last day of that month.
- Not medical advice - always verify with a doctor or pharmacist.
