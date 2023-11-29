# 建立一个通用的机器学习力场模型

>**为了让每个力场模型都可以通过同一种方式调用，所以写模式时需要注意一些规则。以下是我使用的方式，我将做以介绍！因为水平有限，可能存在一些未注意到的事项，希望大家可以不吝指出。对于下述方式如果大家有更好的实现方案，希望可以提出，共同商讨一个公用的标准！**

## 一.变量解释

```python
1.Rcut: float = 6.0, 建立最近邻表时所需要的最大截断半径
2.neighbor: int = 100, 建立最近邻表时最大邻居数
3.type_map: list[int] = [29,30] (表示该力场包含的元素类型为Cu和Zn), 体系中所有元素类型
4.Imagetype: torch.Tensor = [batch, natom],torch.int, 中心原子的原子类型，以POSCAR为例
#######################################################
1.0500000
     10.00000000        0.00000000        0.00000000
      0.00000000       10.00000000        0.00000000
      0.00000000        0.00000000       10.00000000
 Cu Zn
   1 2
Direct
      0.00000000        0.00000000        0.00000000
      0.20000000        0.00000000        0.00000000
      0.00000000        0.30000000        0.00000000
######################################################
该POSCAR的Imagetype将表示为
>>>Imagetype
>>>tensor([[29, 30, 30]])
5.neighbor_list: torch.Tensor = [batch, natom, neighbor],torch.int
中心原子所对应的最近邻原子序号，特别注意原子标号从 0 开始，
使用 -1 补足不够最大近邻数的部分
设最大近邻数 neighbor=5，则上述POSCAR的neighbor_list将表示为
>>>neighbor_list
   tensor([[[1, 2, -1, -1, -1],
            [0, 2, -1, -1, -1],
            [1, 0, -1, -1, -1]]])
6.neighbor_type: torch.Tensor = [batch, natom, neighbor],torch.int
中心原子所对应的最近邻原子类型，该表必须与neighbor_list对应，
使用 -1 补足不够最大近邻数的部分
>>>neighbor_type
   tensor([[[30, 30, -1, -1, -1],
            [29, 30, -1, -1, -1],
            [30, 29, -1, -1, -1]]])
7.ImagedR: torch.Tensor = [batch, natom, neighbor, 4], 4 ->[rij, xij, yij, zij], torch.float,见公式（1）
中心原子与其最近邻原子间[rij, xij, yij, zij]，该表必须与neighbor_list对应，
这里的 0.0 用作补足不够最大近邻数的部分
>>>ImagedR
  tensor([[[[2.00000,-2.00000, 0.00000, 0.00000],
            [3.00000, 0.00000,-3.00000, 0.00000],
            [0.00000, 0.00000, 0.00000, 0.00000],
            [0.00000, 0.00000, 0.00000, 0.00000],
            [0.00000, 0.00000, 0.00000, 0.00000]],
           [[2.00000, 2.00000, 0.00000, 0.00000],
            [3.60555, 2.00000,-3.00000, 0.00000],
            [0.00000, 0.00000, 0.00000, 0.00000],
            [0.00000, 0.00000, 0.00000, 0.00000],
            [0.00000, 0.00000, 0.00000, 0.00000]],
           [[3.60555, -2.00000,3.00000, 0.00000],
            [3.00000, 0.00000, 3.00000, 0.00000],
            [0.00000, 0.00000, 0.00000, 0.00000],
            [0.00000, 0.00000, 0.00000, 0.00000],
            [0.00000, 0.00000, 0.00000, 0.00000]]]])
8.nghost: int = 0, (如果你使用的中最近邻表原子序数，没有超过最大原子数natom，则设为0)，默认设0，但是在lammps中有用
```

$$\vec{r}_{ij}= \vec{r}_{i} - \vec{r}_{j}                 （1）$$
$${r}_{ij}= \|\vec{r}_{ij}\|                 （1）$$

## 二.模型建立

```python  
####################   part-1   #################### 
from typing import List, Optional, Dict
import torch
import torch.nn as nn

####################   part-2   #################### 
class MLP(nn.Module):
    #################   part-2-1   ################# 
    def __init__(self,
                 type_map: list[int],
                 Rcut: float,
                 neighbor: int,
                 config: dict
                 ):
        super(MLP, self).__init__()
        self.type_map = type_map
        self.Rcut = Rcut
        self.max_neighbor = neighbor

    #################   part-2-2   ################# 
    def forward(self,
                Imagetype: torch.Tensor,
                neighbor_list: torch.Tensor,
                neighbor_type: torch.Tensor,
                ImagedR: torch.Tensor,
                nghost: int):
        # Etot[batch, 1], Ei[batch, natom, 1], Force[batch, natom + nghost , 3]
        return Etot, Ei, Force

    #################   part-2-3   ################# 
    def body2fun(self,
                 Imagetype: torch.Tensor,
                 neighbor_list: torch.Tensor,
                 neighbor_type,
                 ImagedR: torch.Tensor,
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        return feat, dfeat

```

> **建立机器学习力场必须包含part-1，part-2-1，part-2-2中所有代码**
> 1 .对于part-1，建议只使用上述三个导入，可以导入torch的库，禁止导入python的其他库！
> 2 .对于part-2-1，必须全部包含！。自己需要的初始化参数应该整理为字典类型，在初始化操作中进行相应解析。
> 3 .对于part-2-2，forward参数不允许修改，只能传入上述五个变量，作者认为上述变量已经包含了所有原子结构信息，后续计算应该从上述参数展开。如果你认为这部分没有包含你所需要的信息，应当提出，大家共同完善这部分输入。必须（return Etot，Ei，Force），且其tensor shape应该符合代码中注释所示。对于vairal量，大家在计算出后暂时不要放入return，这部分正在努力。
> 4 .对于part-2-3部分可以根据自己是否计算feature，d_feature 选择性的使用。

## 使用lammps

```linux
1. 下载lammps，libtorch
2. 将Makefile.mpi中的libtorch路径替换为自己的路径
>>>cp Makefile.mpi lammps/src/MAKE
在lammps/src目录下
>>>mkdir MLFF
>>>cp pair_mlff.cpp pair_mlff.h MLFF
3. 编译
>>>make yes-mlff
>>>make mpi 
```

> 通过jit.script导出jitscript模型命名为mlff.pb
> 复制测试脚本，创建lammps的data.lmp文件
> ../lammps/src/lmp_mpi -in in.lmp

```lammps
clear
# 初始模拟系统设置
dimension         3                # 维数
boundary          p  p  p      # 设置边界条件
units             metal            # 设置单位
atom_style        atomic         # 定义原子类型
timestep          0.001

neighbor          5 bin
neigh_modify      every 1 delay 0 check yes

# 定义晶格
read_data         data.lmp
#力场设置
pair_style        mlff script.pt
pair_coeff        * * 29

#热力学输出
thermo_style	    custom step time temp pe etotal enthalpy press pxx pyy pzz vol lx ly lz 
thermo	          1

#温度初始化
velocity          all  create 300 4981299  dist gaussian

dump	            1 all custom 1 nvt.dump id type x y z fx fy fz
fix               1 all nvt temp 300 300 $(100.0*dt)
run               1000
```
