import torch
from torch import autograd


x = torch.tensor(1.)
a = torch.tensor(1., requires_grad=True)
b = torch.tensor(2., requires_grad=True)
c = torch.tensor(3., requires_grad=True)
y = a**2 * x + b*x + c

print("Before", a.grad, b.grad, c.grad)
grads = autograd.grad(y, [a, b, c])
print("After", grads[0], grads[1], grads[2])


