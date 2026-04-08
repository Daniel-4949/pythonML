from numpy import log
import numpy as np
import math

# Binary Cross Entropy, BCE 二元交叉熵
# 二元交叉損失函數

# 但因為 y（真實標籤）只能是 0 或 1，所以它其實是一個「二選一」的開關
# Loss = - [y*log(yhat) + (1 - y) * log(1 - yhat)]
def cross_entropy(y,yhat):
    loss = -1*(y*log(yhat) + (1-y)*log(1-yhat))
    total_loss = np.sum(loss)
    entropy = total_loss/y.size
    return entropy

# log(x) 在 x 位於 0 到 1 之間（也就是機率的範圍）時，其數值永遠是負的。
# 當真實標籤 y=0 時，代表我們希望預測值 yhat 越接近 0 越好。
# 當 y=0 時，左邊那項消失，只剩 -log(1-yhat)
# 當 y=1 時，右邊那項消失，只剩 -log(yhat)


# 均方誤差 (MSE) 損失函數
# Mean Squared Error
# MSE = 1/n * Σ (yhat-y)^2
# 用來衡量預測值與真實值之間的平均平方差
# 類似變異數
def MSE(y,yhat):
    sq_error = (yhat-y)**2
    sum_sq_error = np.sum(sq_error)
    mse = sum_sq_error/y.size
    return mse

# 平均絕對誤差(MAE) 損失函數
# Mean Absolute Error
# 1/n * Σ |yhat-y|
def MAE(y,yhat):
    abs_error = np.absolute(yhat-y)
    total_abs_erroe = np.sum(abs_error)
    mae = - total_abs_erroe/ y.size
    return mae


# 均方根誤差(RMSE) 損失函數
# Root Mean Squared Error
# sqrt(1/n * Σ (yhat-y)^2)
# 用來衡量預測值與真實值之間的平均平方根差
# 類似標準差
def RMSE(y,yhat):
    sq_error = (yhat-y)**2
    sum_sq_error = np.sum(sq_error)
    mse = sum_sq_error/y.size
    rmse = math.sqrt(mse)
    return rmse

# y = np.array([23.2, 29.3, 28.9, 21.1, 29.1, 25.7, 22.6])
# yhat = np.array([23.5, 28.9, 28.8, 20.8, 29.2, 26.2, 23.2])

# print("MSE", MSE(y, yhat)) #MSE 0.13857142857142843
# print("MAE",MAE(y, yhat)) #MAE 0.3285714285714282
# print("RMSE",RMSE(y, yhat)) #RMSE 0.3722518348798679
# 數學上RMSE > MAE永遠成立
# MAE是平均每步錯了約 0.33 單位，
# RMSE則是給予大誤差更高的權重（平方後再開根號）。
# 由RMSE與MAE來看，誤差小於1
# 誤差已經縮小到 1 以內，這時 MSE 的下降會變得非常緩慢（因為平方後數值更小了）
# 這表示你的模型已經進入了「精細調校期」


# 負對數似然 (Negative Log-Likelihood, 簡稱 NLL)。
# 在二元分類中，它也完全等於 二元交叉熵 (Binary Cross Entropy, BCE)。
def NLL_loss(y,y_predicted):
    y_predicted = np.clip(y_predicted, 1e-15 , 1-1e-15)

    likelihood_elements = (y *np.log(y_predicted)) + ((1 - y) * np.log(1 - y_predicted))
    avg_log_likelihood = np.sum(likelihood_elements) / y.size
    # # 直接回傳「負」的對數概似，這就是 Loss（正數）
    return -avg_log_likelihood

data = np.array([0.3 ,0.7 ,0.8 ,0.5 ,0.6 ,0.4])
data2 =[]
for i in data:
    data2.append(1 if i >0.5 else 0)
# print(data2)[0, 1, 1, 0, 1, 0]

yhat = np.array([0, 1, 1, 1, 1, 0])#預測類別
y = np.array([0, 1, 1, 0, 1, 0])#實際類別


lhl_loss = NLL_loss(y, yhat)

print("Loss",lhl_loss)#損失
# Loss 5.75659599872348

# Likelihood (似然)：數值越大，代表模型越準。
# Loss (損失)：數值越小，代表模型越準。

# 在機器學習的實務中，我們幾乎永遠都在透過「最小化 NLL」來達成「最大化 MLE」。

# 跟二元交叉熵一樣(邏輯開關):
# 當 y=0 時，左邊那項消失，只剩 -log(1-yhat)
# 當 y=1 時，右邊那項消失，只剩 -log(yhat)

# 為什麼要求 avg_log_likelihood（平均值）？
# 梯度穩定性 (Gradient Stability)：
# 如果你只算 sum（總和），當你的數據量從 10 筆增加到 1000 筆時，總誤差會暴增 100 倍
# 導致梯度不穩定，導致模型訓練時的穩定性問題。所以我們需要取平均值，將總誤差轉換為平均誤差

# 學習率 (Learning Rate) 的適配：
# 如果損失函數是平均值，你可以為不同的任務設定一套通用的學習率。
# 如果是總和，你必須根據資料量不斷手動調整學習率，這在工程上極度低效。

# MLE (Maximum Likelihood Estimation)最大概似估計
# MLE 的原始定義是**「所有樣本猜對機率的連乘(∏，就像Σ但是為乘法)」**。

# 不考慮電腦計算問題，MLE 的原始代碼:
# def MLE_original(y,y_predict):
#     probobilities = np.where(y==1, y_predict , 1-y_predict)
#     # MLE 的核心：連乘所有機率
#     likehood = np.prod(probobilities)
#     return likehood

# 當樣本數增加（例如 1000 個樣本），每個機率都是 0.8 時：
# 0.8^1000會趨近於 0.00000000...，電腦的浮點數會直接溢位成 0，導致無法計算。

# 解決方法：
# 1.取 Log (對數)：
# 我們對 MLE 的「連乘」取 log，利用 log(a*b) = log(a) + log(b)，
# 把地獄級的「連乘」變成天堂級的「連加」。這就是 avg_log_likelihood 的由來。

# 2.變號 (Negative)：
# MLE 的目標是最大化（爬到山頂），但電腦優化器（Optimizer）的設計習慣是最小化（走到谷底）。
# 所以我們加個負號，把「最大化概似」轉化成「最小化負對數概似」。