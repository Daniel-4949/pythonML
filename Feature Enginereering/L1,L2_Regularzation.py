import numpy as np
import pandas as pd

# 權重
# w = np.array([-1,5,3,-9]).reshape(2,2)
# print(w)
# [[-1  5]
#  [ 3 -9]]

# Lambda = 0.5
# L1 = Lambda*np.sum(np.abs(w))
# print(L1)
# L2 = Lambda*np.sum(w**2)
# print(L2)
# 9.0  L1
# 58.0 L2
# lambda 是一個超參數（Hyperparameter），它代表你對「權重過大」這件事的處理。
# lambda=0 模型想把權重w 設多大就多大（為了迎合訓練資料，容易造成過擬合）。
# lambda愈大，模型為了讓「總損失」變小，會被迫把所有權重 w 縮得很小，甚至縮成 0。


# 訓練資料（Loss Function）在給予 w 獎勵：它希望 w 變大，好讓預測變準。
# 正規化項（Penalty Term）在給予 w懲罰：它因為 lambda 的存在，希望 w 變小。
# 最後模型會找到一個平衡點。
# 如果一個特徵真的很有用，它的「預測獎勵」會大於「lambda 懲罰」
# ，所以它能保留較大的權重。如果一個特徵只是雜訊，它的「預測獎勵」很小，禁不起 lambda 的抽稅，
# 權重就會被壓到接近 0（L2）或直接歸零（L1）。

