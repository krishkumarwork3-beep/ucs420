"""Offline test of the full pipeline (model -> smoothing -> state -> policy) using dataset images as a fake video stream."""
import numpy as np
from assistant import *
te = np.load('test.npz'); X, y = te['X'], te['y']; m = SVMModel(); rng = np.random.default_rng(0)
def stream(label, n=60):
    idx = rng.choice(np.where(y == CLASSES.index(label))[0], n, replace=False)
    st = EmotionState(history_sec=1e9); t = 0
    for i in idx: st.update(m.predict_proba(X[i]), t); t += 1     # 1 frame per second
    return st
for label in ['sad', 'angry', 'fear', 'neutral', 'happy']:
    st = stream(label); s = infer_user_state(st, window=60)
    print(f'{label:8s} -> frame-level top emotion: {CLASSES[int(st.ema.argmax())]:9s} user state: {s:11s} | action: {POLICY[s]["action"]}')
print('\nSample prompt:\n' + build_llm_prompt('I do not get recursion.', 'frustrated'))