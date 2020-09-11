import torch
import time

print(torch.__version__)  # 查看PyTorch版本
print(torch.cuda.is_available())  # 检查cuda是否可用


"""比较CPU和GPU的运算速度"""
# 随机生成数据
a = torch.randn(10000, 1000)
b = torch.randn(1000, 10000)

# 使用CPU进行运算
t0 = time.time()
c = torch.matmul(a, b)
t1 = time.time()
print(a.device, t1-t0, c.norm(2))  # norm()用于求范数，此处求的是矩阵的2范数(谱范数)

# CPU运算耗时
device = torch.device('cuda')  # 指明使用GPU
a = a.to(device)  # 将数据放入GPU
b = b.to(device)

# 包含初始化耗时的GPU运算
t0 = time.time()
c = torch.matmul(a, b)
t2 = time.time()
print(a.device, t2-t0, c.norm(2))

# 不包含初始化耗时的GPU运算
t0 = time.time()
c = torch.matmul(a, b)
t2 = time.time()
print(a.device, t2-t0, c.norm(2))