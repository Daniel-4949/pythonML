import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA

X,y = make_circles(n_samples=1000,factor=0.3,noise=0.05,random_state=0)

X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=0)
#  畫出原始數據內外圓
# _, (train_ax,test_ax) = plt.subplots(ncols=2,sharex=True,sharey=True,figsize=(8,4)) 


# train_ax.scatter(X_train[:,0],X_train[:,1],c=y_train)
# train_ax.set_ylabel("Feature #1")
# train_ax.set_xlabel("Feature #0")
# train_ax.set_title("Training data")

# test_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
# test_ax.set_xlabel("Feature #0")
# _ = test_ax.set_title("Testing data")

# 試圖以PCA線性轉換，將原始數據降維以分離內圓的目標(特徵)值
# 無效。由於 PCA 僅能進行線性轉換（旋轉/縮放），
# pca = PCA(n_components=2)
# X_test_pca = pca.fit(X_train).transform(X_test)

# fig,(orig_data_ax, pca_proj_ax) = plt.subplots(
#     ncols=2,figsize=(8,4)
# )

# orig_data_ax.scatter(X_test[:,0],X_test[:,1],c=y_test)
# orig_data_ax.set_ylabel("Feature #1")
# orig_data_ax.set_xlabel("Feature #0")
# orig_data_ax.set_title("Testing data")

# pca_proj_ax.scatter(X_test_pca[:,0],X_test_pca[:,1], c=y_test)
# pca_proj_ax.set_ylabel("Principal component #1")
# pca_proj_ax.set_xlabel("Principal component #0")
# pca_proj_ax.set_title("Projection of testing data\n using PCA")

# plt.show()


kernel_pca= KernelPCA(n_components=None,kernel='rbf',gamma=6,
                       fit_inverse_transform=True,alpha=0.1)

X_test_kpca = kernel_pca.fit(X_train).transform(X_test)
fig, (orig_data_ax, kernel_pca_proj_ax) = plt.subplots(
    ncols=2, figsize=(10, 4)
)

# 透過kernelPCA成功將目標特徵分離
orig_data_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
orig_data_ax.set_ylabel("Feature #1")
orig_data_ax.set_xlabel("Feature #0")
orig_data_ax.set_title("Testing data")

kernel_pca_proj_ax.scatter(X_test_kpca[:, 0], X_test_kpca[:, 1], c=y_test)
kernel_pca_proj_ax.set_ylabel("Principal component #1")
kernel_pca_proj_ax.set_xlabel("Principal component #0")
_ = kernel_pca_proj_ax.set_title("Projection of testing data\n using KernelPCA")

# plt.show()

