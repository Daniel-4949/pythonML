from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA,KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import numpy as np
X,y = make_moons(n_samples=1000,noise=0.05,random_state= 0)

X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=0)

# 資料清洗是檢視數據，但這裡是檢視特徵
# _, (train_ax, test_ax) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10, 4))

# train_ax.scatter(X_train[:,0],X_train[:,1],c=y_train)
# train_ax.set_ylabel("Feature #1")
# train_ax.set_xlabel("Feature #0")
# train_ax.set_title("Training data")


# test_ax.scatter(X_test[:,0],X_test[:,1],c=y_test)
# test_ax.set_xlabel("Feature #0")
# test_ax.set_ylabel("Feature #1")
# test_ax.set_title("Testing data")

# plt.show()

pca = PCA(n_components=2)
# X_test_pca = pca.fit(X_train).transform(X_test)

# X_test_kpca = KernelPCA(n_components=None, kernel='rbf',gamma=10,
#                         fit_inverse_transform=True,alpha=0.1)
# X_test_kpca = X_test_kpca.fit(X_train).transform(X_test)


X_test_lda = LDA(n_components=1).fit(X_train, y_train).transform(X_test)

fig,(orig_data_ax, LDA_proj_ax) = plt.subplots(ncols=2,figsize = (10,4))


orig_data_ax.scatter(X_test[:,0],X_test[:,1],c=y_test)
orig_data_ax.set_ylabel("Feature #1")
orig_data_ax.set_xlabel("Feature #0")
orig_data_ax.set_title("Testing data")

# LDA_proj_ax.scatter(X_test_kpca[:,0],X_test_kpca[:,1],c=y_test) or pca
LDA_proj_ax.scatter(X_test_lda[:,0],np.zeros_like(X_test_lda[:, 0]),c=y_test)

LDA_proj_ax.set_ylabel('Feature#1')
LDA_proj_ax.set_xlabel('Feature#0')
LDA_proj_ax.set_title("Projection of testing data\n using PCA")

plt.show()



