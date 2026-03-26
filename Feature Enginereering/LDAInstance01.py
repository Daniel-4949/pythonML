from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ds = load_wine()
df = pd.DataFrame(ds.data, columns=ds.feature_names)
# print(df.head())
# print(df.info())
#total feature :13
# print(ds.DESCR)

X=df.values
y=ds.target#在原始資料集裡面，df只將資料欄位提取出來，不包含target
# target通常是一個一維陣列，每個元素對應一個樣本的類別標籤。

# print(y.unique())0,1,2
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=100)

# print(X_train.shape)    (142, 13)
# print(X_test.shape)     (36, 13)
# print(y_train.shape)    (142,)
# print(y_test.shape)     (36,)

# 縮放
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# LDA特徵萃取，目的是降維
lda = LDA(n_components=2)#(將13維特徵降為2維)
X_train_lda = lda.fit_transform(X_train_std,y_train)
X_test_lda = lda.transform(X_test_std)

# 訓練模型，邏輯回歸
clf = LogisticRegression()

clf.fit(X_train_lda,y_train)
y_pred = clf.predict(X_test_lda)
# print(f'{accuracy_score(y_pred, y_test) *100:.2f}%')97.22%

# 中間小結:
# Wine 資料集是線性可分的：不同品種的酒在化學成分上有很強的規律性。
# LDA 是正確的選擇：它成功捕捉到了類別間的差異。
# 特徵萃取的魔力：原本要看 13 個指標才能分出品種，現在你只需要看 LDA 生成的 2 個「超級指標」就夠了。

# 補充:
# LDA不總是優於PCA，視情況而定
# 當類別數據少、沒有標籤、資料分布不符合常態分佈時，LDA可能不是最佳選擇。
# 目的是「資料壓縮」或「探索規律」？  PCA。
# 目的是「預測/分類」且「有標籤」？LDA。
# 特徵數極多（如 1000+）且樣本少？
#  先用 PCA 降到 50 維（去噪），再用 LDA 降到 C-1 維（增強分類感）。C為總最大維度
# PCA、Kernel_PCA、LDA是降維三劍客

def plot_decision_regions(X,y,classifier,resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 拓寬邊界
    x1_min,x1_max= X[:,0].min()-1,X[:,0].max()+1
    x2_min,x2_max= X[:,1].min()-1,X[:,1].max()+1
    # 方法meshgrid會將一維數列轉換為二維矩陣
    x_x1,x_x2= np.meshgrid(np.arange(x1_min,x1_max,resolution),
                            np.arange(x2_min,x2_max,resolution))
    # 此時x_x1、x_x2都是2維矩陣，象徵全部的x,y。
    # 座標上需要一組x,y才會出現一個點，因此x_x1、x_x2的矩陣維度是相同的。
    # 如x_x1有兩個、x_x2有三個，那樣np.meshgrid就會將其各自複製成2*3的矩陣

    # 邏輯回歸模型只收樣本清單，以[[1, 10], [2, 10], [3, 10], ...] 這種標準格式。
    # ravel()方法會將多維陣列轉換為一維陣列
    # np.array()將轉換成二維陣列，原本兩個一維假設是[1,2,3]、[4,5,6]，變為[[1,2,3],
    #                                                                  [4,5,6]]
    # 變成2*3，二維矩陣後.T方法轉換維度，即row、column互換成3*2

    Z = classifier.predict(np.array([x_x1.ravel(),x_x2.ravel()]).T)
    Z = Z.reshape(x_x1.shape)#變回矩陣，即標籤label。標註顏色分類
    plt.contourf(x_x1,x_x2,Z,alpha=0.4,cmap=cmap)#它負責畫出決策邊界（Decision Regions）。
    plt.xlim(x_x1.min(),x_x1.max())
    plt.ylim(x_x2.min(),x_x2.max())
    
        #enumerate(np.unique(y))會將標籤去重，並給予序號
        # 跟Z標籤不同，這裡是原始資料的分類(有三類酒)
    for idx,cl in enumerate(np.unique(y)): 
        plt.scatter(
                x=X[y==cl, 0],
                y=X[y==cl, 1],
                alpha=0.6,
                color=cmap(idx),
                marker=markers[idx],
                label=cl
        )

plot_decision_regions(X_test_lda,y_test,classifier=clf)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('decision_regions.png', dpi=300)
plt.show()