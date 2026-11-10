"""Emotion-aware cognitive assistant: smoothing -> state tracking -> adaptive policy -> webcam demo.
Run:  python assistant.py            (webcam, uses emotion_hog_svm.joblib)
      python assistant.py --cnn emotion_cnn.pt   (after running train_cnn.py)
"""
import argparse, time, collections
import numpy as np, cv2

CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ---------- 1. Emotion models (both return a 7-way probability vector) ----------
class SVMModel:
    def __init__(self, path='emotion_hog_svm.joblib'):
        import joblib
        d = joblib.load(path); self.pca, self.clf = d['pca'], d['clf']
        self.hog = cv2.HOGDescriptor((48, 48), (16, 16), (8, 8), (8, 8), 9)
    def predict_proba(self, face48):
        f = self.hog.compute(cv2.equalizeHist(face48)).ravel()[None]
        s = self.clf.decision_function(self.pca.transform(f))[0]   # 7 scores, one per emotion
        e = np.exp(s - s.max())
        return e / e.sum()                                          # softmax -> pseudo-probabilities

class CNNModel:
    def __init__(self, path):
        import torch; from train_cnn import EmotionCNN
        self.torch = torch; self.m = EmotionCNN(); self.m.load_state_dict(torch.load(path, map_location='cpu')); self.m.eval()
    def predict_proba(self, face48):
        x = self.torch.tensor(face48, dtype=self.torch.float32)[None, None] / 255.
        with self.torch.no_grad(): return self.torch.softmax(self.m((x - 0.5) / 0.5), 1)[0].numpy()

# ---------- 2. Temporal smoothing + emotional state ----------
class EmotionState:
    """Exponential moving average of probabilities, plus a time-stamped history."""
    def __init__(self, alpha=0.2, history_sec=180):
        self.alpha, self.ema, self.hist = alpha, None, collections.deque()
        self.history_sec = history_sec
    def update(self, probs, t=None):
        t = time.time() if t is None else t
        self.ema = probs if self.ema is None else self.alpha * probs + (1 - self.alpha) * self.ema
        self.hist.append((t, int(self.ema.argmax())))
        while self.hist and t - self.hist[0][0] > self.history_sec: self.hist.popleft()
        return CLASSES[int(self.ema.argmax())]
    def share(self, labels, window_sec):
        """fraction of the last window_sec seconds spent in any of `labels`."""
        if not self.hist: return 0.0
        t_now = self.hist[-1][0]
        recent = [c for t, c in self.hist if t_now - t <= window_sec]
        return sum(CLASSES[c] in labels for c in recent) / max(len(recent), 1)

# ---------- 3. Cognitive state inference (FER has no 'confused'/'bored' class, so we derive them) ----------
def infer_user_state(st: EmotionState, window=60):
    if st.share({'angry', 'sad', 'disgust'}, window) > 0.5: return 'frustrated'
    if st.share({'fear', 'surprise'}, window) > 0.4:       return 'confused'
    if st.share({'neutral'}, window) > 0.85:               return 'disengaged'
    if st.share({'happy'}, window) > 0.4:                  return 'engaged'
    return 'neutral'

POLICY = {
    'frustrated': dict(tone='calm, empathetic', action='simplify the explanation, acknowledge the difficulty, offer a short break'),
    'confused':   dict(tone='patient, clear',   action='explain step by step with one concrete example, then check understanding'),
    'disengaged': dict(tone='energetic',        action='change the pace or format (quiz, example, challenge) and ask what they want next'),
    'engaged':    dict(tone='encouraging',      action='keep momentum and raise the difficulty slightly'),
    'neutral':    dict(tone='friendly',         action='answer normally and concisely'),
}

def build_llm_prompt(user_msg, user_state):
    p = POLICY[user_state]
    return (f"You are a helpful cognitive assistant. The user appears {user_state} (inferred from facial expression; "
            f"this is a soft signal, never state it as fact). Use a {p['tone']} tone and {p['action']}.\n\nUser: {user_msg}")

# ---------- 4. Face detection + webcam loop ----------
def run_webcam(model):
    face_det = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap, st, last = cv2.VideoCapture(0), EmotionState(), 'neutral'
    print("Press 'q' to quit, 'p' to print the adaptive LLM prompt for a sample question.")
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_det.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
        if len(faces):
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])          # largest face only
            face = cv2.resize(gray[y:y + h, x:x + w], (48, 48))
            emo = st.update(model.predict_proba(face)); last = infer_user_state(st)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{emo} | state: {last}', (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
        cv2.imshow('Emotion-aware assistant (frames are never saved)', frame)   # privacy: nothing written to disk
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        if k == ord('p'): print(build_llm_prompt('Explain gradient descent.', last))
    cap.release(); cv2.destroyAllWindows()

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--cnn'); a = ap.parse_args()
    run_webcam(CNNModel(a.cnn) if a.cnn else SVMModel())