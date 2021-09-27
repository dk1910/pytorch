import torch as t
from matplotlib import pyplot as plt

# 生成训练数据，并可视化数据分布情况
t.manual_seed(100)
dtype=t.float
# 生成坐标数据，x为tensor，需要把x的形状转换为100x1
x=t.unsqueeze(t.linspace(-1,1,100),dim=1)
# 生成y坐标数据，y为tensor,形状100x1，另加上一些噪声
y=3*x.pow(2)+2+0.2*t.rand(x.shape)
# 画图，把tensor数据转换为numpy数据
plt.scatter(x.numpy(),y.numpy())
plt.show()

# 初始化权重
w=t.randn(1,1,dtype=dtype,requires_grad=True)
b=t.randn(1,1,dtype=dtype,requires_grad=True)
# 训练模型
lr=0.001
for ii in range(800):
    # 前向传播
    y_pred=x.pow(2).mm(w)+b
    loss=0.5*(y_pred-y)**2
    loss=loss.sum()

    # 自动计算梯度，梯度存放在grad属性中
    loss.backward()
    # 手动更新参数，需要用torch.no_grad(),使上下文环境中切断自动求导的计算
    with t.no_grad():
        w-=lr*w.grad
        b-=lr*b.grad
    # 梯度清零
    w.grad.zero_()
    b.grad.zero_()

# 可视化训练结果
# 可视化结果
plt.plot(x.numpy(), y_pred.detach().numpy(), 'r-', label='predict')
plt.scatter(x.numpy(), y.numpy(), color='blue', marker='o', label='true')
plt.xlim(-1, 1)
plt.ylim(2, 6)
plt.legend()
plt.show()

print(w, b)
