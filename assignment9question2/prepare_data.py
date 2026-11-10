import os, numpy as np, cv2
CLASSES=['angry','disgust','fear','happy','neutral','sad','surprise']
def load(split):
    X,y=[],[]
    for i,c in enumerate(CLASSES):
        d=f'data/{split}/{c}'
        for f in sorted(os.listdir(d)):
            im=cv2.imread(f'{d}/{f}',0)
            X.append(im); y.append(i)
    return np.array(X,dtype=np.uint8),np.array(y)
if __name__=='__main__':
    for s in ['train','test']:
        X,y=load(s); np.savez_compressed(f'{s}.npz',X=X,y=y); print(s,X.shape,np.bincount(y))