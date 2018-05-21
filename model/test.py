import torch
class C(object):
    def __init__(self):
        self.a = 1
    def add(self, b):
        self.a += b

class C1(C):
    def __init__(self):
        super(C1, self).__init__()
    def add(self, b, c):
        self.a = self.a + b + c

if __name__ == '__main__':
    a = torch.ones(3, requires_grad=True)
    b = a * 2
    c = b + b.detach()
    d = c.mean()
    d.backward()
    print(a.grad)



    x = torch.ones(3, requires_grad=True)
    y = x * 2
    z = y + y
    out = z.mean()
    out.backward()
    print(x.grad)


