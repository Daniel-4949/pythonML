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
# print(df.head())5 rows x 13 columns

# 假設資料乾淨無異常缺失值
x=df.values
y= ds.target


# 資料分割
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

# 縮放
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# 計算 S_W, S_B 散佈矩陣
def calculate_SW_SB(X,y,label_count):
    mean_vecs=[]
    # 逐一將每一類酒(label)所有數據點的均值向量計算並存儲在 mean_vecs 中
    for label in range(label_count):
        mean_vecs.append(np.mean(X[y==label],axis=0))

    d = X.shape[1]# 特徵數量，shape=[ rows * cols ]
    S_W = np.zeros((d,d))# 初始化 S_W 為零矩陣，如同sum=0的初始化
    for label,mv in zip(range(label_count),mean_vecs): #同時取得中心向量和對應類別標籤
        # 計算該類別內部的協方差矩陣 (Covariance Matrix)。
        # 因為預設cov將特徵放在row，原始資料則是在column，所以要轉置
        # 實際意義：它在衡量「這一類的酒，自己內部的點分佈得有多散？」。
        class_scatter = np.cov(X[y==label].T)
        # S_W += class_scatter，範例可接受的方法，但是嚴謹的作法是加權總和

        # Cov1+Cov2+Cov3+...，
        # cov方法經把數據除以了(n_i - 1)來得到平均的混亂度。
        # 這代表你假設每個類別對於「總體混亂度」的貢獻是平等的，

        # 嚴謹的作法：加權總和 (Individual Class Scatters)
        # 計算方式：先算出每一類的 np.cov，再乘上該類的 (樣本數 - 1)，最後再加總。
        # 邏輯：樣本數多的類別，其內部的分佈情況對整體的代表性更高，因此權重較大。
        n_i = X[y==label].shape[0]
        # 這裡是還原這類數據的「原始混亂總量」。
        S_W += (n_i-1) * class_scatter
        print(f'S_W shape:{S_W.shape}')

        mean_overall = np.mean(X,axis=0)
        S_B=np.zeros((d,d))
        for i ,mean_vec in enumerate(mean_vecs):
            n=X[y==i].shape[0]#這是一個權重
            mean_vec = mean_vec.reshape(d, 1)  # make column vector
            # 將原本「扁平」的一維陣列（13,）轉向，變成直立的列向量Column Vector（13, 1）。
            # 為什麼要這樣做？：這是為了接下來的矩陣乘法。
            # 在線性代數中，(13*1) 乘以 (1*13) 才能得到一個 (13*13)的矩陣。
            mean_overall = mean_overall.reshape(d, 1)  # make column vector
            S_B += n*(mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
            print(f'S_B shape:{S_B.shape}')
            # (mean_vec - mean_overall)這是在算「該類別中心」與「總重心」之間的位移向量。
            # dot(...T) (外積運算)：一個列向量乘以它自己的轉置（行向量）。
            # 物理意義：這個運算會產生一個矩陣，(13*1)*(1*13)=(13*13)
            # 記錄了這 13 個特徵在「偏離中心」這件事上是如何共同貢獻的。
        return S_W,S_B
        #S_W：希望它縮小（讓同一類更緊湊）。S_B：希望它放大（讓類別中心更疏遠）。 


# 為什麼 LDA 需要S_B這份「總體混亂分布」？這是為了後續的**「去噪」或「標準化」**：
# LDA 接下來會拿 S_B（類別間的距離）去對比這個 S_W（類內的混亂）。
# 如果某個方向 S_W很大：
# 代表這個方向大家都很亂，就算類別中心在那裡分得很開（S_B 大），也不可靠。
# 如果某個方向 S_W 很小：
# 代表同一類的點都縮得很緊，只要類別中心在那裡有一點點距離，這就是一個絕佳的分類方向。

#LDA函數操作過程:
def LDA_numpy(X,X_test,y,label_count,no):
    S_W,S_B= calculate_SW_SB(X,y,label_count)
    # 計算特徵值(eigenvalue)及對應的特徵向量(eigenvector)
    eigen_val,eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    eigen_pair = [(np.abs(eigen_val[i]),eigen_vecs[:,i]) 
                  for i in range(len(eigen_vecs))]
    print('Eigenvalues in descending order:\n')
    for eigen_val in eigen_pair:
        print(eigen_val[0])

    eigen_pair.sort(key=lambda x:x[0],reverse=True)

    # 將剛才排序完、拿到的「選秀狀元」（最強特徵向量）從一個扁平的陣列，
    # 轉換成一個**「立體」的列向量**，準備用來構建最終的投影矩陣。
    w = eigen_pair[0][1][:,np.newaxis].real
    # 一維陣列 (13,)既不是行也不是列為了後續能把多個向量「橫向黏貼」成一個投影矩陣（例如 13*2），
    # 我們必須先把這個向量變成**「直立」的列向量**，形狀要變成 (13, 1)。
    # .real：只取實部在執行 np.linalg.eig（特徵分解）時，如果矩陣運算產生了極其微小的波動，
    # 數學上可能會出現帶有 +0.j（虛部為 0）的複數。.real 會把虛數部分丟掉，
    # 確保 w 矩陣裡全部都是乾淨的實數浮點數。

    for i in range(1,no):
        w = np.hstack((w,eigen_pair[i][1][:,np.newaxis].real))

    return X.dot(w),X_test.dot(w)

 # 取 2 個特徵
X_train_LDA,X_test_LDA =LDA_numpy(X_train_std,X_test_std,y_train,len(ds.target_names),2)

clf =LogisticRegression()

X_train_std = clf.fit(X_train_LDA,y_train)
y_pred = clf.predict(X_test_LDA)
print(f'{accuracy_score(y_test,y_pred)*100:.2f}%')

# 繪製決策邊界(Decision regions)
def plot_decision_regions(X,y,classifier,resolution=0.02):
    markers = ('s','x','o','^','v')
    color = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(color[:len(np.unique(y))])

    # plot the decision surface
    X1_min,X1_max = X[:,0].min()-1,X[:,0].max()+1
    X2_min,X2_max = X[:,1].min()-1,X[:,1].max()+1
    xx1,xx2 = np.meshgrid(np.arange(X1_min,X1_max,resolution),
                          np.arange(X2_min,X2_max,resolution))
    

    Z= classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z,alpha=0.4,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    for idx,cl in enumerate(np.unique(y)): 
        plt.scatter(
            x = X[y==cl,0],
            y=X[y==cl,1],
            alpha=0.6,
            color=cmap(idx),
            marker=markers[idx],
            label=cl
        )

plot_decision_regions(X_train_LDA,y_train,classifier=clf)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('decision_regions.png', dpi=300)
plt.show()
