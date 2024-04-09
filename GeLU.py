import numpy as np
import matplotlib.pyplot as plt

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

# 生成输入数据
x = np.linspace(-3, 3, 1000)
y = gelu(x)

# 绘制曲线图
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='GeLU', color='blue')
plt.title('GeLU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.legend()
plt.savefig("GeLU.png")
plt.show()
