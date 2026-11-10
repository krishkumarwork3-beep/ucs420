import numpy as np, cv2, time, joblib
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
CLASSES=['angry','disgust','fear','happy','neutral','sad','surprise']
hog=cv2.HOGDescriptor((48,48),(16,16),(8,8),(8,8),9)
def feats(X):
    out=[]
    for im in X:
        im=cv2.equalizeHist(im)
        out.append(hog.compute(im).ravel())
    return np.array(out,dtype=np.float32)
tr=np.load('train.npz'); te=np.load('test.npz')
t=time.time(); Xtr=feats(tr['X']); Xte=feats(te['X']); print('hog',Xtr.shape,round(time.time()-t,1),'s')
ytr,yte=tr['y'],te['y']
t=time.time(); pca=PCA(150,random_state=0).fit(Xtr); Ptr=pca.transform(Xtr); Pte=pca.transform(Xte); print('pca',round(time.time()-t,1),'s')
t=time.time(); clf=SVC(C=5,kernel='rbf',gamma='scale',class_weight='balanced').fit(Ptr,ytr); print('svc fit',round(time.time()-t,1),'s')
p=clf.predict(Pte)
print('acc',accuracy_score(yte,p),'macroF1',f1_score(yte,p,average='macro'))
print(classification_report(yte,p,target_names=CLASSES,digits=3))
cm=confusion_matrix(yte,p); np.save('cm.npy',cm)
joblib.dump({'pca':pca,'clf':clf,'classes':CLASSES},'emotion_hog_svm.joblib',compress=3)