import torch
import matplotlib.pyplot as plt

def selu(x, alpha=1.67326, scale=1.0507):
    # 定义SELU激活函数
    pos = (x > 0).type(torch.float32)
    return scale * (pos * x + (1 - pos) * (alpha * (torch.exp(x) - 1)))

# 生成输入数据
x = torch.arange(-5., 5., 0.1)
y = selu(x)

# 绘制SELU激活函数曲线图
plt.plot(x.numpy(), y.numpy())
plt.title('SELU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.savefig("SELU.PNG")
plt.show()


