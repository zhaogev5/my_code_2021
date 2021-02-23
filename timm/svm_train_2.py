import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
# ***use seaborn plotting style defaults
import seaborn as sns; sns.set()
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from PIL import Image
from sklearn.metrics import confusion_matrix
#********************* KEY IMPORT OF THIS LECTURE********************************
from sklearn import svm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import numpy as np

def return_shuzu(path_base):
    # path_base = 'train_new/inf'
    path_bath_list = os.listdir(path_base)
    L = len(path_bath_list)
    label = []
    count = 0
    for i in path_bath_list:
        exact_path = os.path.join(path_base,str(i))
        exact_path_dict = os.listdir(exact_path)
        for j in exact_path_dict:
            exact_img = os.path.join(exact_path,j)
            im = Image.open(exact_img)
            im = im.resize((128,128))
            img1 = np.array(im)
            img1 = img1.reshape(-1)
            img1 = img1[:, np.newaxis]
            if count == 0:
                X = img1
                label.append(int(i))
            else:
                X = np.concatenate((X,img1),1)
                label.append(int(i))
            count = count + 1
    return X, label

def pca_svm(train_f, train_l, val_f, val_l):
    n_comp = 30
    pca = PCA(n_comp, whiten = True)
    X = train_f.T  #这里进行了转置 从49152,235 到235,49152
    print(X.shape)
    Y = val_f.T
    pca.fit(X)
    Xtrain_proj = pca.transform(X)
    print(Xtrain_proj.shape)
    Xval_proj = pca.transform(Y)
    # print(Xtrain_proj.shape)
    clf = svm.SVC(gamma=0.02, C=1000.) 
# apply SVM to training data and draw boundaries.
    clf.fit(Xtrain_proj, train_l)
    ypred = clf.predict(Xval_proj)
    # print(val_l)
    # print(ypred)
    cm = confusion_matrix(val_l, ypred)
    print(cm)
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    print(P,R)  
    correct = np.sum(val_l == ypred)
    print(correct/len(val_l)*100)
    print(type(cm))
    kp = kappa(cm)
    return kp, cm

def kappa(confusion_matrix):
    pe_rows = np.sum(confusion_matrix, axis=0)
    pe_cols = np.sum(confusion_matrix, axis=1)
    sum_total = np.sum(pe_cols)
    pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
    po = np.trace(confusion_matrix) / float(sum_total)
    return (po - pe) / (1 - pe)

def plot_confusion_matrix(c_m, labels_name, title):
    print(type(labels_name))
    c_m = c_m.astype('float') / c_m.sum(axis=1)[:, np.newaxis]
    # print(cm.shape)    # 归一化
    # plt.figure()
    # cm =np.around(cm,decimals=2)
    plt.imshow(c_m,interpolation='nearest',cmap=plt.cm.Blues,extent='left')   # 在特定的窗口上显示图像
    # plt.title(title)    # 图像标题
    # # plt.colorbar()
    # num_local = np.array(range(len(labels_name)))    
    # plt.xticks(num_local, labels_name)    # 将标签印在x轴坐标上
    # plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    # plt.ylabel('True label')    
    # plt.xlabel('Predicted label')
    # for first_index in range(len(cm)):    #第几行
    #     for second_index in range(len(cm[first_index])):    #第几列
    #         plt.text(first_index-0.25, second_index, str(round(float(cm[second_index][first_index]),2)), fontsize=8)
    # plt.imshow(cm, interpolation='nearest',cmap=plt.cm.Blues) 
    # plt.show()
    plt.savefig(f'svm_{title}.png', format='png')

if __name__ == "__main__":

    kinds = ['Armored','Ground','Plane','Tank','Truck','Ship','Harbor'] #'All',
    kinds = ['All'] #'All',
    for kind in kinds:
        train_base = f'datasets/{kind}/train/RGB'
        path_bath_list = os.listdir(train_base)
        list_int =[int(a) for a in path_bath_list]
        train_f, train_l = return_shuzu(train_base)
        print(train_f.shape,len(train_l))
        val_base = f'datasets/{kind}/val/RGB'
        val_f, val_l = return_shuzu(val_base)
        print(val_f.shape,len(val_l))
        kp,cm = pca_svm(train_f, train_l, val_f, val_l)  #计算pca_svm
        plot_confusion_matrix(cm, list_int, kind)  #画混淆矩阵
        with open('svm_结果.txt','a+') as f:
            f.write(f"{kind}:{kp}\n")
# vis 40准确率  inf 29准确率