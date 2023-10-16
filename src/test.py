import time

from ase import io
import torch
import torch.nn as nn
from ase.io.xyz import write_xyz
import op
# # 读取vasprun.xml文件并创建ASE Atoms对象
# atoms = io.read("data/Zn_Cu_vasprun.xml",index='0:1000')
#
# # 保存Atoms对象为XYZ文件
# io.write("structure.xyz", atoms)
torch.set_default_dtype(torch.float64)
t1 = time.time()
a = torch.rand(2000, 100, 3)
b = torch.randint(1, 2000, (2000, 100))
c = torch.zeros(2000, 3)
for i in range(2000):
    c[i] = c[i] + a[b == i].sum(dim=0)
t2 = time.time()

p = torch.zeros(2000, 3)
idx = b.view(-1).unsqueeze(-1).expand(-1, 3)
values = a.view(-1, 3)
p = p.scatter_add(0, idx, values).view(2000, 3)
t3 = time.time()
mask = (b.unsqueeze (0) == torch.arange(2000).unsqueeze (-1).unsqueeze (-1)).double()  # materialize large mask tensor
cB = torch.einsum('mij, ijn -> mn', mask, a)
t4 = time.time()
print ('torch.allclose (c, cB):', torch.allclose (c, cB),torch.allclose (c, p))

print(t4-t3,t3-t2,t2-t1)
class SANNet(nn.Module):
    def __init__(self):
        super(SANNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=200, out_features=1, bias=True)
        )
        self.net1 = nn.Parameter(torch.Tensor(1, 200))

    def forward(self, feat):
        feat.requires_grad_()
        # aa = torch.mul(self.net(feat),feat)
        aa = torch.randn(1, 200).requires_grad_()
        # t1 = time.time()
        m = torch.ones(1, 200)
        m = feat * aa
        return m

    @torch.jit.script
    def add(self, feat):
        return feat + 1


model = SANNet()

print(model)
x = torch.ones(1, 200)
y = model(x)
ms = torch.jit.script(model)
ms.save('temp.pt')
print("end successful")
ms = torch.jit.load('temp.pt')

# aa = ms(torch.ones(1,200000))
# print('ns',aa)
print("successful", t2-t1, t3-t2)
