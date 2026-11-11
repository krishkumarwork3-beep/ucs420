"""Streamlit UI:  streamlit run app.py"""
import streamlit as st

from medassist.assistant import DISCLAIMER, MedicineLabelAssistant

st.set_page_config(page_title="Medicine Label Assistant", page_icon="💊", layout="wide")
st.title("💊 Cognitive Assistant for Reading Medicine Labels")
st.caption(DISCLAIMER)

with st.sidebar:
    backend = st.selectbox("OCR engine", ["auto", "easyocr", "tesseract"])
    use_tts = st.checkbox("Read result aloud", value=True)
    db_path = st.text_input("Drug-name CSV (optional)", value="")
    st.markdown("Tip: good light, no glare, hold the label flat and fill the frame.")


@st.cache_resource
def get_bot(b, p):
    return MedicineLabelAssistant(b, p or None)


bot = get_bot(backend, db_path)

c1, c2 = st.columns(2)
with c1:
    up = st.file_uploader("Upload a photo of the label", type=["jpg", "jpeg", "png", "webp"])
    cam = st.camera_input("...or take a picture")
src = up or cam

if src is not None:
    with c1:
        st.image(src, caption="Input", use_container_width=True)
    if st.button("Analyse label", type="primary") or "info" not in st.session_state:
        with st.spinner("Reading label..."):
            info, res, variants = bot.read_image(src.getvalue())
            st.session_state.update(info=info, ocr=res, variants=variants)

if "info" in st.session_state:
    info, res = st.session_state["info"], st.session_state["ocr"]
    bot.info = info
    with c2:
        st.subheader("Result")
        if info.expiry_status == "expired":
            st.error("⚠️ EXPIRED - do not use this medicine.")
        elif info.expiry_status == "expiring_soon":
            st.warning(f"Expires in {info.days_to_expiry} days.")
        elif info.expiry_status == "valid":
            st.success("Within expiry date.")
        st.markdown(f"**Medicine:** {', '.join(n.title() for n in info.drug_names) or '—'}"
                    f"  \n**Strength:** {', '.join(info.strengths) or '—'}"
                    f"  \n**Form:** {info.form or '—'}"
                    f"  \n**Expiry:** {info.expiry_date or '—'}"
                    f"  \n**Batch:** {info.batch or '—'}  \n**MRP:** {info.mrp or '—'}")
        if info.name_confidence == "low":
            st.info("Name match is uncertain - please verify against the bottle.")
        for title, items in (("Directions", info.instructions),
                             ("Warnings", info.warnings), ("Storage", info.storage)):
            if items:
                st.markdown(f"**{title}**")
                for it in items:
                    st.write("• " + it)
        st.progress(min(int(res.confidence), 100), text=f"OCR confidence {res.confidence:.0f}%")

        if use_tts:
            from medassist.speech import speak_to_file
            try:
                st.audio(speak_to_file(info.summary(), "summary.mp3"))
            except Exception as e:
                st.caption(f"Speech unavailable: {e}")

        st.subheader("Ask about this medicine")
        q = st.text_input("e.g. When does it expire? How do I take it? Any warnings?")
        if q:
            st.write(bot.ask(q))

    with st.expander("Raw OCR text (edit to fix OCR mistakes, then re-analyse)"):
        fixed = st.text_area("OCR text", info.raw_text, height=200)
        if st.button("Re-analyse text"):
            st.session_state["info"] = bot.read_text(fixed)
            st.rerun()
    with st.expander("Preprocessed images"):
        cols = st.columns(len(st.session_state["variants"]))
        for col, (k, v) in zip(cols, st.session_state["variants"].items()):
            col.image(v, caption=k, clamp=True)
