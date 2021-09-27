import torch
# 定义输入张量
x=torch.Tensor([2])
# 初始化权重参数w,偏移量b、并设置require_grad属性为TRUE
w=torch.randn(1,requires_grad=True)
b=torch.randn(1,requires_grad=True)

# 实现前向传播
y=torch.mul(w,x)
z=torch.add(y,b)


# 查看x,w,b页子节点的require_grad属性
print("x,w,b的require_grad属性分别为：{},{},{}".format(x.requires_grad,w.requires_grad,b.requires_grad))

# 查看叶子节点、非叶子节点的其他属性
print("y,z的require_grad属性分别为：{},{}".format(y.requires_grad,z.requires_grad))
print("x,w,b,y,z的是否为叶子节点：{},{},{},{},{}".format(x.is_leaf,w.is_leaf,b.is_leaf,b.is_leaf,y.is_leaf,z.is_leaf))

# 查看叶子节点的grad_fn属性
print("x,w,b的grad_fn属性：{},{},{}".format(x.grad_fn,w.grad_fn,b.grad_fn))

# 查看非叶子节点的grad_fn属性
print("y,z是否为叶子节点：{},{}".format(y.grad_fn,z.grad_fn))

# 基于z张量进行梯度反向传播，执行backward之后计算图会自动清空
z.backward()
# 如果需要多次使用backward,需要修改参数retain_graph为True，此时梯度是累加的
# z.backward(retain_graph=True)
# 查看叶子节点的梯度，x是叶子节点但它无须求导，故其梯度为None
print("参数w,b的梯度分别为:{},{},{}".format(w.grad,b.grad,x.grad))
# 参数w,b的梯度分别为：tensor([2.]),tensor([1.]),None

# 非叶子节点的梯度，执行backward之后，会自动清空
print("非叶子节点y,z的梯度分别为:{},{}".format(y.grad,z.grad))