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
df = pd.DataFrame(ds.data, columns=ds.feature_names)

X=df.values
y=ds.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=100)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

clf = LogisticRegression()

clf.fit(X_train_pca,y_train)
y_predict = clf.predict(X_test_pca)
print(f'{accuracy_score(y_test,y_predict)*100:.2f}%')

def plot_decision_regions(X,y,classifier,resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min,x1_max = X[:,0].min()-1,X[:,0].max()+1
    x2_min,x2_max = X[:,1].min()-1,X[:,1].max()+1
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
                          np.arange(x2_min,x2_max,resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z,alpha=0.4,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

      # plot class samples
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(
            x =X[y==cl,0],
            y=X[y==cl,1],
            alpha=0.6,
            color=cmap(idx),
            marker=markers[idx],
            label=cl
        )

plot_decision_regions(X_train_pca,y_train,classifier=clf)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.tight_layout
plt.show()

# 觀察每個特徵資訊量
pca2 = PCA()
X_train_pca2 = pca2.fit_transform(X_train_std)
# print(pca2.explained_variance_ratio_) 每個特徵的變異數
# np.sum(pca2.explained_variance_ratio_) 總共的變異數

plt.bar(range(1,14),pca2.explained_variance_ratio_)
plt.step(range(1,14),np.cumsum(pca2.explained_variance_ratio_))
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.axhline(0.8,color='red',linestyle='--')
plt.show()

# 設定可解釋變異下限
# 告訴 PCA：「請幫我保留 80% 的資訊量，至於要用幾個維度，你自己算。」
# 結果：PCA 算完發現前 5 個成分加起來剛好過 80%，所以 X_train_pca.shape 會變成 (142, 5)。
pca3 = PCA(0.8)
X_train_pca3= pca3.fit_transform(X_train_std)
print(X_train_pca3.shape)#(142, 5)
