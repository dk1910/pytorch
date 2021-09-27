import torch
# 定义叶子节点张量x,形状1x2
x=torch.tensor([[2,3]],dtype=torch.float,requires_grad=True)
print(x)
#初始化Jacobian矩阵
J=torch.zeros(2,2)
# 初始化目标张量，形状为1*2
y=torch.zeros(1,2)
# 定义y与x的映射关系
# y1=x1**2+3*x2,y2=x2**2+2*x1
y[0,0]=x[0,0]**2+3*x[0,1]
y[0,1]=x[0,1]**2+2*x[0,0]

y.backward(torch.Tensor([[1,0]]),retain_graph=True)
J[0]=x.grad
print(J[0])

# 梯度是累加的，故需要对x的梯度清零
x.grad=torch.zeros_like(x.grad)

y.backward(torch.Tensor([[0,1]]))
J[1]=x.grad
print(J[1])