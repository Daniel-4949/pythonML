from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap

ds = datasets.load_wine()
df = pd.DataFrame(ds.data, columns= ds.feature_names)

X = df.values
y = ds.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=100)

scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# PCA
# print(X.shape)      (178, 13)
# print(X.T.shape)    (13, 178)
# np.cov需要column * row，所以需要轉置
# 計算協方差矩陣，先示範未縮放狀態
# cov_mat= np.cov(X.T)
# print(cov_mat)
# pp = pd.DataFrame(cov_mat)
# print(pp)用DataFrame資料表查看，為整份資料的變異數數據
# 對角線 (Diagonal)：從左上到右下的那條線。這裡的值是純粹的變異數 (Variance)。這就是「資訊量」。
# 非對角線 (Off-diagonal)：其他空格。這裡的值是共變異數 (Covariance)。
# 它在衡量「特徵 A 增加時，特徵 B 是跟著增加還是減少？」。



# 計算特徵值(eigenvalue)及對應的特徵向量(eigenvector)
# eigen_vals,eigen_vecs= np.linalg.eig(cov_mat)
# 方法作用：在13x13 的變異數地圖中尋找「數據的主軸」
# 物理意義：
# 特徵值 (Eigenvalues)：代表這根直徑的長度。長度越長，代表這個方向含有的「變異量（資訊）」越多。
# 特徵向量 (Eigenvectors)：代表這根直徑的方向。它告訴你數據是往哪個角度「長」最開的。
# eigen_vals:一維向量 (13,)
# eigen_vecs:二維矩陣 (13,13)，這裡的必須「直著看」，每一列 (Column) 才是一個完整的特徵向量。

# 降維公式:X_new = X_std * W
# X_std: 已縮放的數據矩陣 (178, 13)
# W: 特徵向量矩陣 (13, 2)，這裡的 2 代表我們要降到 2 維

# PCA函數:
# no: 降維到的維度
def PCA_numpy(X,X_test,no):
    cov_mat = np.cov(X.T)
    eigen_vals,eigen_vecs= np.linalg.eig(cov_mat)
    # 合併特徵向量及特徵值
    eigen_pairs = [(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]

   # 針對特徵值降冪排序
    eigen_pairs.sort(key=lambda x: x[0], reverse=True)

    w = eigen_pairs[0][1][:,np.newaxis]

    for i in range(1,no):
        w = np.hstack((w,eigen_pairs[i][1][:,np.newaxis]))
# no是我們要降到的維度數量，簡單說就是有no個columns，每個column就是一個主軸
# hstack: 水平合併，將特徵向量矩陣拼在一起(水平)，即增加的是**「欄 (Column)」**，
# 也就是增加新的維度（PC1, PC2...）。
    # 矩陣相乘
    # w為特徵投影矩陣，裡面是混合比例
    return X.dot(w),X_test.dot(w)
    # 點乘（Dot Product）」動作，在幾何學上就是**「投影（Projection）
    # X * w 的運算過程，本質上是在計算：「原始數據點在這些新主軸上的座標位置。」

    # 維度消去法（為什麼能降維？）
    # X_std：(142, 13)—— 有 142 個點，每個點有 13 個特徵。
    # w：(13, 2) —— 13 個舊特徵對應到 2 個新方向的比例。相乘後：(142, 13) * (13, 2) = (142, 2)
    # 中間的 13 被「消掉」了！ 剩下的 2 就是每個樣本在新空間（PC1, PC2 平面）上的新身分證字號。

X_train_pca,X_test_pca = PCA_numpy(X_train_std,X_test_std,2)

clf = LogisticRegression()
clf.fit(X_train_pca,y_train)

y_pred= clf.predict(X_test_pca)
# print(f'{accuracy_score(y_test,y_pred)*100:.2f}%')97.22%



# 決策邊界
def plot_decision_regions(X,y,classifier,resolution=0.02):
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        x1_min,x1_max = X[:,0].min()-1,X[:,0].max()+1
        x2_min,x2_max = X[:,1].min()-1,X[:,1].max()+1
        xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
                              np.arange(x2_min,x2_max,resolution))
        
        Z= classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
        Z= Z.reshape(xx1.shape)

        plt.contourf(xx1,xx2,Z,alpha=0.4,cmap=cmap)
        plt.xlim(xx1.min(),xx1.max())
        plt.ylim(xx2.min(),xx2.max())

        for idx,cl in enumerate(np.unique(y)):
            plt.scatter(
                x= X[y==cl,0],
                y=X[y==cl,1],
                alpha=0.6,
                color=cmap(idx),
                marker=markers[idx],
                label=cl
            )
plot_decision_regions(X_test_pca,y_test,clf)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='upper left')
plt.show()
