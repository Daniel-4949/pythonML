import numpy as np
import matplotlib.pyplot as plt


# 計算原始分數 (Logits):
# 即直線公式
# z = w * x + b
#輸入值 z，在神經網路中，這個 z 通常代表「加權總和」（所有"特徵 * 權重"後的累加結果）。 
z= np.arange(-7,7,0.1)
# print(len(z))140

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# Sigmoid 函數（也稱為 S 型函數或 Logistic 函數）。
# 數學上會將直線變成曲線，我們就不能只用∆y/∆x （直線斜率），
# 必須開始用 dy/dx（瞬時斜率）來監控每一點的變化

# sigma(z) = 1 / (1 + e^(-z))
# sigmoid'(z) = sigma(z) * (1 - sigma(z))
# 這代表你只要知道輸出值，就能瞬間算出梯度。
# 無論你的 z 有多大或多小，輸出的結果永遠會被壓縮在 0 到 1 之間。
# np.exp(x) 在 NumPy 中代表自然對數的底數 e（約等於 2.718）的 x 次方。
# 也就是數學上的 e^x。
# 當 z 很大時，e^(-z) 會趨近於 0。當 z 很小（很大的負數）時，e^(-z) 會趨近於 無限大。
# --------------------------------------------------------------------------------------

# 機率化 (Probabilistic Interpretation):
# 因為輸出範圍是 (0, 1)，我們可以直接把結果看作「機率」。
# ex:輸出 0.8 有 80% 的機率是 Class A。
# 它將一個無限範圍的數值，轉換成了一個有意義的決策信號。
# 在神經網絡中"單個神經元"其實就是 z = w * x + b 再套上這層 sigmoid(z)。
# 多層網路：就是一堆神經元的輸出再變成下一層的輸入。

# --------------------------------------------------------------------------------------


phi_z = sigmoid(z)
# plt.plot(z,phi_z)
# plt.axvline(0.0,color='r')
# plt.xlabel('z')
# plt.ylabel('phi(z)')

# plt.yticks([0.0,0.5,1.0])
# ax = plt.gca()  # get current axis獲取當前軸
# ax.yaxis.grid(True)
# plt.tight_layout()
# plt.show()
# --------------------------------------------------------------------------------------

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

# 核心邏輯是「比例分配」
# 如果cls =3，則 softmax 函數的輸出會是 3 個機率值，總和為 1。
# 使用sigmoid函數，則會獲得 3 個機率值，總和不為1。ex: [0.7,0.5,0.3] 

# 公式：softmax(x) = e^x / sum(e^x)
# 分子代表每個輸入值的指數化結果。
# 分母代表所有輸入值的指數化結果的總和。
# 每個項目的「得分能量」除以「總能量」，這就是歸一化（Normalization）。

# 數值爆炸問題（Overflow）:
# 數學上softmax(x) = e^x / sum(e^x)是完美的。
# 但在運算上若x=10000， e^10000 會超出 NumPy 的表示範圍，導致電腦當機（Overflow）。
# 通常會先減去 x 中的最大值，這不會改變比例，但能保證最大值只會是 e^0 = 1，絕不爆炸。
# 減去最大值後(平移)，指數函數的輸入項最大永遠只有 0。


# 解釋:
# 原始數值是 x_1, x_2, ..., x_n，最大值是 M。
# 我們把每個數都減去 M，得到新數值 x_i - M。
# 代入 Softmax 公式: softmax(x_i - M) = e^(x_i - M) / sum(e^(x_i - M))
# 因為e^(x_i - M) = e^x_i / e^M，在分數加減中抵銷了 e^M。
# 所以 softmax(x_i - M) = softmax(x_i)。
# 這說明減去最大值後，Softmax 函數的輸出不會改變比例。
# 當你減去最大值 M 後，所有的新數值會具備以下特性：
# 1. 最大值一定是 0：因為 M - M = 0。
# 2. 其餘數值一定是負數或 0: 因為 M 是最大的，而我們則將所有x_i減去M(平移)，所以所有x_i <= M，

def stable_softmax(x):
    exps = np.exp(x- np.max(x)) 
    return exps/sum(exps)


phi_z = softmax(z)
# phi_z = stable_softmax(z)

# plt.plot(z,phi_z)
# plt.axvline(0.0,color='r')
# plt.xlabel('z')
# plt.ylabel('phi(z)')

# plt.yticks([0.0,0.5,1.0])
# ax = plt.gca()  # get current axis獲取當前軸
# ax.yaxis.grid(True)
# plt.tight_layout()
# plt.show()


# 雙曲正切函數，Hyperbolic Tangent
def tanh(x):
    s1 = np.exp(x) - np.exp(-x)
    s2 = np.exp(x) + np.exp(-x)
    s = s1 / s2
    return s

# tanh'(x) = 2 * sigma(2x) - 1
# 也就是說，Tanh 就是把 Sigmoid 的高度變成兩倍，再往下挪 1 個單位。


phi_z = tanh(z)

# plt.plot(z,phi_z)
# plt.axvline(0.0 ,color='r')
# plt.xlabel('z')
# plt.ylabel('phi(z)')

# plt.yticks([0.0,0.5,1.0])
# ax = plt.gca()  # get current axis獲取當前軸
# ax.yaxis.grid(True)
# plt.tight_layout()
# plt.show()


# Rectified Linear Unit，修正線性單元
def relu(x):
    s = np.where(x <0 , 0 , x)
    return s

# f(x) = max(0, x)
# 邏輯：如果輸入 x 是正數，直接原樣輸出；如果是負數，通通變成 0
# Sigmoid/Tanh：當 x 很大時，曲線變得非常平坦，導數（斜率）接近 0。
# 這會讓模型在深層網路中「學不動」。
# ReLU：只要 x > 0，它的斜率永遠是 1。
# 這代表信號可以毫無損耗地傳遞到前面幾層，讓訓練深層網路成為可能。

# 計算速度極快:Sigmoid：需要算 e^x、除法，對電腦來說很累。
# ReLU：只需要一個 if (x > 0) 判斷。在處理幾百萬個參數時，這能節省巨大的運算時間。

# ReLU 會讓一部分神經元的輸出直接變成 0。這意味著在任何時刻，只有一部分神經元是在工作的。
# 這種「稀疏性」讓模型更具魯棒性（不易過擬合），也更像人類的大腦。

# ReLU 的缺點：Dying ReLU（神經元死亡）
# 如果一個神經元的輸入一直落在負數區，它的輸出永遠是 0，斜率也是 0。
# 結果：這個神經元就「死掉了」，梯度再也傳不回來，它永遠無法被更新。

# phi_z = relu(z)
# plt.plot(z, phi_z)
# plt.axvline(0.0, color='k')
# #plt.ylim(-0.1, 1.1)
# plt.xlabel('z')
# plt.ylabel('$\phi (z)$')

# # y axis ticks and gridline
# plt.yticks([0.0, 0.5, 1.0])
# ax = plt.gca()
# ax.yaxis.grid(True)

# plt.tight_layout()
# #plt.savefig('images/03_02.png', dpi=300)
# plt.show()



def leaky_relu(x,a):
    if x < 0 :return a*x
    else :return x
# 放大負數區間，因為斜率不再是 0
# 代表當模型預測錯誤且輸入為負時，誤差訊號（面積的變化）仍然可以透過這個微小的a 傳導回去。

# def elu(x,alpha):
#   if x < 0:
    #     return alpha*(np.exp(x)-1)
    # else:
    #     return x

# x ≥ 0 :跟 ReLU 一樣是直線，斜率為 1。
# x < 0 :它不是像 Leaky ReLU 那樣給一條直線，而是給了一條平滑的曲線。
# 優勢：在 $x=0$ 這個轉折點，ELU 比 ReLU 更「圓滑」。
# 在數學上，這意味著它在全區域都是可微的，這會讓梯度下降的過程更加穩定，減少抖動。 

def plot(px, py):
    plt.plot(px, py)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))
    plt.show()


a = 0.07
x =[]
dx = -20

while dx <= 20:
    x.append(dx)
    dx += 0.1

# px , py
px = [xv for xv in x]
py = [leaky_relu(xv,a) for xv in x]

plot(px, py)
plt.show()


