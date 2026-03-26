from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler


ds = load_wine()
df = pd.DataFrame(ds.data, columns=ds.feature_names)
# print(df.head())
# print(df.info)[178 rows x 13 columns]

X= df.values
y= ds.target

#分割資料
# 80% 訓練資料，20% 測試資料
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=50)


# 縮放
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)



# L1、L2正規化:
# 與 PCA/LDA 不同，PCA/LDA 是在進模型前先「變換空間」
# ，而 L1/L2 是在模型訓練的過程中「直接對權重施壓」。


# lr_L1 = LogisticRegression(penalty='l1',solver='saga',C=0.5,random_state=50)
# lr_L2 = LogisticRegression(penalty='l2',solver='saga',C=0.5,random_state=50)

# lr_L1.fit(X_train_std,y_train)
# lr_L2.fit(X_train_std,y_train)

# print(f'L1: {lr_L1.coef_}')
# print(" ")
# print(f'L2: {lr_L2.coef_}')
# 模型實際上為每一類紅酒都訓練了一組權重（即 3 *13 的矩陣）。
# 每一橫列代表該類別對 13 個特徵的看法。

# print(f'L1截距 {lr_L1.intercept_}')
# print(f'L2截距 {lr_L2.intercept_}')

# 截距的數學意義與直線截距b (y = ax +b)相似
# 計算方式：截距是在訓練過程中，與權重 w 同時被優化出來的。
# 它代表的是：
# 當所有特徵 x 經過標準化後都等於 0 時（即所有特徵都處於平均值)，該樣本屬於某一類的機率基礎。
# L1截距 [ 0.04103736  0.33739082 -0.37842818]
# L2截距 [ 0.056136    0.49947305 -0.55560906]

# 截距:
# 正值 = 代表在沒有任何強烈特徵證據下，數據「天生較傾向」被歸類為這一類。
# 負值 = 代表該類別的門檻較高，需要更強的特徵證據（w * x 的部分要夠大）才能跨過門檻被判定為此類。
# 雖然截距對預測很重要，但在分析特徵時我們通常忽略它，原因如下：
# 1.截距不針對任何特定特徵（如酒精或顏色），它只針對整個類別。
# 2.縮放的影響：因為你做了 StandardScaler，數據中心化在 0。這使得截距的數值變得更有參考價值
# （代表平均水準下的分類傾向)，但它依然無法告訴你「哪種成分」比較重要。


# y_pred_l1 = lr_L1.predict(X_test_std)
# y_pred_l2 = lr_L2.predict(X_test_std)

# score_l1 = accuracy_score(y_test,y_pred_l1)
# score_l2 = accuracy_score(y_test,y_pred_l2)

# print(f'L1 準確率: {score_l1}')
# print(f'L2 準確率: {score_l2}')
# L1 準確率: 0.9444444444444444
# L2 準確率: 1.0
# L1 因為砍掉很多特徵，準確度略低於 L2，但它的模型非常「輕量且好解釋」。

#------------------------------------------------------------------------------------- 

# 模型中的C: 正規化的強度的"倒數"，在SKLEARN中 C= 1/λ(Lambda)
# λ (Lambda)：是我們之前算的「稅率」或「懲罰力道」。
# C ("Inverse of Regularization Strength")：則是對模型的「信任度」或「寬容度」。
# 當 C 很大時（例如 1000）＝ 高度信任、低度懲罰 :
# 容易造成 過擬合 (Overfitting)。模型會把數據中的「雜訊」也當成真理來學習。

# 當 C 很小時（例如 0.01）＝ 極度懷疑、高度懲罰 :
# 模型被戴上了沉重的枷鎖。只要權重 w 稍微變大，懲罰就會高得讓模型受不了。
# ex:L1 => 更多特徵會被歸零。  L2 => 所有權重都會縮得非常小。


# 視覺化: 熱力圖heatmap
# fig,ax = plt.subplots(1,2,figsize=(15,8))
# sns.heatmap(lr_L1.coef_,annot=True,cmap='RdBu_r',ax=ax[0],
#             xticklabels=ds.feature_names)
# ax[0].set_title(f'L1 Regularization (C={lr_L1.C})')

# sns.heatmap(lr_L2.coef_,annot=True,cmap='RdBu_r',ax=ax[1],
#             xticklabels=ds.feature_names)
# ax[1].set_title(f'L2 Regularization (C={lr_L2.C})')
# plt.show()

# 實戰中標準的 C 值測試清單通常我們會直接橫跨 5 到 6 個數量級，
# 這樣才能看完整個模型的「一生」（從完全欠擬合到完全過擬合
# ex: c_range = np.logspace(-3, 2, 6) 方法產生10^-3 到 10^2 之間的 6 個數值
# 使用對數尺度（10 倍跳躍法），這樣才能有效率地測試不同的 C 值。
# 極強懲罰：0.001, 0.01 (用來找最強特徵)
# 中等懲罰：0.1, 1.0 (通常是預設值，追求平衡)
# 極弱懲罰：10, 100 (幾乎沒有正規化，看原始全貌)

# 驗證曲線 (Validation Curve)
# C_range = np.logspace(-3,2,6)
# lr = LogisticRegression(penalty='l1',solver='saga',max_iter=10000,random_state=0)
# # 計算驗證曲線 (這裡直接用全部的 X_train_std，cv=5 代表做 5 折交叉驗證)
# train_scores,test_scores = validation_curve(
#             estimator=lr,
#             X=X_train_std,
#             y=y_train,
#             param_name='C',
#             param_range=C_range,
#             cv=5
# )

# 計算平均分
# train_mean = np.mean(train_scores,axis=1)
# test_mean = np.mean(test_scores,axis=1)


# plt.figure(figsize=(8, 6))
# plt.plot(C_range, train_mean, color='blue', marker='o', label='Training accuracy')
# plt.plot(C_range, test_mean, color='green', marker='s', linestyle='--', label='Validation accuracy')

# plt.title('Validation Curve for Logistic Regression (L1)')
# plt.xlabel('Parameter C (Inverse of Regularization Strength)')
# plt.ylabel('Accuracy')
# plt.xscale('log') # 關鍵：橫軸設定為對數尺度
# plt.legend(loc='lower right')
# plt.grid()
# plt.show()

# L1正規化，param = C驗證曲線結果分析:
# 1. 10⁻³ 到 10⁻²：欠擬合（Underfitting）階段。訓練與驗證準確率都極低（約 38%~40%）。

# 2. 10⁻¹ 到 10⁰：甜蜜點（Sweet Spot。準確率斜率陡峭上升，在 C=0.1時，驗證準確率衝到了95%以上。

# 3. 10⁰ 到 10²：潛在過擬合（Overfitting）傾向。
# 訓練準確率（藍線）鎖死在 100% (1.0)，但驗證準確率（綠線）開始持平，甚至有極其細微的下降趨勢。

#------------------------------------------------------------------------------------- 

# 混淆矩陣，觀察C=1.0時模型有沒有因為 L1 的"精簡特徵"而誤判了哪種酒？
best_lr = LogisticRegression(penalty='l1',solver='saga',C=1.0,max_iter=10000,random_state=0)
best_lr.fit(X_train_std,y_train)

y_pred_best = best_lr.predict(X_test_std)

# 混淆矩陣
cm = confusion_matrix(y_test,y_pred_best)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=ds.target_names)

fig,ax = plt.subplots(figsize=(6,6))
disp.plot(cmap='Blues',ax=ax)
plt.title(f'Confusion Matrix (L1, C=1.0)')
plt.show()
