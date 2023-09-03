# 基于 Benders 分解的环形生产线平衡（Benders Decomposition for Circular Assembly Line Balancing）

本项目为 ["Min" 团队](https://competition.huaweicloud.com/information/1000041930/ranking) 针对 [苏州园区“华为云杯”2023人工智能应用创新大赛（创客）](https://competition.huaweicloud.com/information/1000041930/introduction) 开发的求解算法。赛题为 **“考虑任务拆分的环形生产线平衡问题（Circular Assembly Line Balancing Problem with Task Splitting, CALBP-TS）”**，是一种新的生产线平衡问题（Assembly Line Balancing Problem, ALBP）。详细的题目介绍可参见 [官方“赛题详情”](https://competition.huaweicloud.com/information/1000041930/circumstance)。题目目标为，在尽可能多算例上找到可行解的情况下，减小工人间节拍（工作量）的差异。

赛题的约束类型和问题规模使得几乎所有算例都难以直接通过求解整体的优化模型得到最优解，甚至可行解都难以找到。因此，本项目基于**组合 Benders 分解（Combinatorial Benders Decomposition, CBD）**和**基于逻辑的 Benders 分解（Logic-based Benders Decomposition, LBBD）**开发**精确算法（Exact algorithm）**。CBD 和 LBBD 均为经典 Benders 分解（Benders Decomposition，BD）的扩展和推广。这三者的思想均为将原始的优化问题分解为主问题（Master Problem）和子问题（Subprobelm），主问题向子问题提供部分决策变量值，子问题向主问题提供割平面（cut），从而最终证明问题不可行和找到最优解。经典的 BD 要求子问题为线性规划（Linear Programming, LP）问题，这大大限制了 BD 的适用范围。以赛题为例，显然无法仅用连续变量和线性约束完整刻画该问题。本项目整合了 CBD 的核心：组合 Benders 割（Combinatorial Benders Cut），和 LBBD 的核心：子问题可以为任何形式的优化问题，不严格要求子问题为 LP 或其他数学规划（Mathematical Programming）问题。例如设计的算法中子问题为**约束规划（Constraint Programming, CP）**和**启发式的不可行性证明**。关于 BD 及其变种的更多介绍可参见 [维基百科](https://en.wikipedia.org/wiki/Benders_decomposition) 和 Rahmaniani 等的综述文章。值得注意的是，BD 的使用范围并不局限于赛题的 ALBP，可适用于更广泛类型的优化问题。算法还采用了**局部分支（Local Branching）**以提升搜索有效上界（可行解）的能力。

项目设计了自顶向下和自底向上两种分解方法：

- （1）先确定工序到工人的分配，再确定工序到站位的分配（`UALBP_CB.py`）；

- （2）先确定工序到站位的分配，再确定工序到工人的分配（`UALBP_CB2.py`）。

两种分解策略分别适用于工序间依赖关系简单（拼接工序较少）和工序间依赖关系复杂（拼接工序较多）的算例。

**本项目理论上可找到以代理目标函数（如最小化最大节拍、最小化平均节拍、和最小化工人节拍的基尼系数等）为优化目标的算例最优解，或提供有效的上界（可行解），或证明问题不可行。实际计算中，在 40/60 以上公开算例上实现了这一目标。**

## 项目文件

```shell
.
├── README.md  # 本文件
├── requirements.txt  # 运行依赖
├── instances  # 算例文件夹
├── solutions  # 求解结果文件夹
├── config.py  # 运行参数
├── instance.py  # 读取和处理算例数据
├── UALB.py  # 基于单个 MIP 求解赛题问题（未实际参与运算，仅供参考）
├── UALB_CB.py  # 分解策略（1）
├── UALB_CB2.py  # 分解策略（2）
├── solution.py  # 验证和保存求解结果
└── utility.py  # 读取和保存 json 文件
```

求解程序的流程为 `instance.py` 从 `instances` 文件夹读取算例数据；`UALB_CB.py` 中 `Solver` 类继承 `instance.py` 的 `Instance` 类，并求解，`UALB_CB.py` 中 的 `Solver` 会调用 `UALB_CB2.py` 中的 `Solver`；求解完成后将求解结果实例化为 `solution.py` 中的 `Solution` 类；`Solution` 验证求解结果的正确性，并将结果按照要求的格式保存在 `solutions` 文件夹。

## 运行程序

### 配置 Python 环境

项目基于 `Python 3.7.0` 开发，依赖以下第三方软件包运行。

```
ply==3.11
pyomo==6.6.0
highspy==1.5.3
ortools==9.5.2237
networkx==2.6.3
toposort==1.10
```

其中 `ply` 为 `pyomo` 的依赖；`pyomo` 和开源数学规划求解器 [HiGHS](https://highs.dev/) 的 Python 接口 `highspy` 分别用于混合整数规划模型（Mixed-Integer Programming, MIP）的建模和求解，用于确定工序到工人的分配；`ortools` 的 [CP-SAT Solver](https://developers.google.com/optimization/cp/cp_solver) 用于 CP 的建模和求解，用于确定工序到站位的分配；`networkx` 和 `toposort` 用于根据工序的先后关系确定工序的拓扑序（即可行的工序执行顺序），其中 `networkx` 可生成单一的拓扑序，`toposort` 可生成集合形式的拓扑序，即所有拓扑序 。

可执行以下命令在 conda 下配置 Python 运行环境。

```shell
# 在 conda 创建新环境安装 Python 3.7.0，名称为 py37
conda create -n py37 python=3.7.0
# 激活安装的环境 py37
source activate py37
# 安装运行项目所需的第三方软件包
pip install -r requirements.txt
```

### 配置参数

项目在 `config.py` 文件提供了用户自定义运行时间等参数的功能。但请谨慎修改带 “\*” 的参数，除非已真正了解它的含义及其可能造成的影响。 **推荐所有参数使用默认值。**

````python
# config.py

# 算例名称列表，文件存储于 instances 文件夹
INSTANCES = ["instance-2.txt"]

# 程序运行参数，**推荐所有参数使用默认值**
PARAMETERS = {
    "OBJ_WEIGHT": (0, 0, 1),  # *目标函数权重，依次为最小节拍，平均节拍，和 0 目标函数（可行性问题）
    "MAX_SPLIT_TASK_NUM": None,  # *允许拆分的最大任务数量，None 代表根据赛题 API 中的 max_split_num 确定
    "UALB_CB_TIME_LIMIT": 70,  # 分解策略（1）整体允许的最大运行时间（秒）
    "UALB_CB2_TIME_LIMIT": 360,  # 分解策略（2）整体允许的最大运行时间（秒）
    "CP_TIME_LIMIT": 25,  # 子问题（工序到站位的分配）CP 求解允许的最大运行时间（秒）
    "LOCAL_BRANCHING_K_SET": range(1, 90, 4),  # *局部分支的临域范围
    "CHANGE_OBJ_ITER": 50,  # *CHANGE_OBJ_ITER 次迭代后改变目标函数
    "CHANGE_OBJ_TIME": 30,  # *CHANGE_OBJ_TIME 秒后改变目标函数，CHANGE_OBJ_TIME 和 CHANGE_OBJ_ITER 取先到达的
}
````

### 单个求解

```python
from UALB_CB import Solver
from utility import load_json, save_json

# 以 instance-2.txt 为例
instance = "instance-2.txt"

S = Solver(load_json(f"instances/{instance}"))
output_json, real_obj = S.run()
save_json(output_json, f"solutions/{instance}_result.txt")
```

### 批量求解

求解前请保证要求解的算例在  `instances` 文件夹下，且 `config.py` 中 `INSTANCES` 列表有要求解算例的文件名。

**推荐在创建的 `py37` 环境下直接运行：**

```shell
python UALB_CB.py
```

以下代码修改自 `UALB_CB.py` 的 `if __name__ == '__main__':`。

```python
import time

from config import INSTANCES
from UALB_CB import Solver
from utility import load_json, save_json

instance_li = INSTANCES

start_time = time.time()
real_objectives = {}
instance_count = 0
for instance in instance_li:
    instance_count += 1
    instance_start_time = time.time()
    print(f"[{instance_count}/{len(instance_li)}] Solving {instance}")

    real_obj = 10
    try:
        S = Solver(load_json(f"instances/{instance}"))
        output_json, real_obj = S.run()
        save_json(output_json, f"solutions/{instance}_result.txt")
    except Exception as e:
        print(e)

    real_objectives[instance] = real_obj
    print(f"Ins. Runtime    : {time.time() - instance_start_time} seconds")
    print()

print(f"Real objectives : {list(real_objectives.values())}")
print(f"Mean objective  : {sum(real_objectives.values()) / len(real_objectives)}")
print(f"Total Runtime   : {time.time() - start_time} seconds")
```

## 已知的问题

- 控制台中输出的内容未正确显示单个工序分配给多个工人和站位的情况，但不影响保存到 `solutions` 文件夹下的最终结果。

## 许可

本人保留对该项目的一切权利，包括但不限于将本项目用于商业目的；直接将本项目算法或代码用于公开发表的论文等。

## 致谢

感谢大赛组织机构、华为算法专家、和 “华为云 AI 小助手” 为本次比赛付出的劳动，以及其他参赛选手对赛题理解提供的帮助。

## 参考文献

1. Boysen, N., Schulze, P., & Scholl, A. (2022). Assembly line balancing: What happened in the last fifteen years? *European Journal of Operational Research*, *301*(3), 797–814. https://doi.org/10.1016/j.ejor.2021.11.043
2. Benders, J. F. (1962). Partitioning procedures for solving mixed-variables programming problems. *Numerische Mathematik*, *4*(1), 238–252. https://doi.org/10.1007/BF01386316
3. Codato, G., & Fischetti, M. (2006). Combinatorial Benders’ cuts for mixed-integer linear programming. *Operations Research*, *54*(4), 756–766. https://doi.org/10.1287/opre.1060.0286
4. Hooker, J. N., & Ottosson, G. (2003). Logic-based Benders decomposition. *Mathematical Programming*, *96*(1), 33–60. https://doi.org/10.1007/s10107-003-0375-9
5. Rahmaniani, R., Crainic, T. G., Gendreau, M., & Rei, W. (2017). The Benders decomposition algorithm: A literature review. *European Journal of Operational Research*, *259*(3), 801–817. https://doi.org/10.1016/j.ejor.2016.12.005
6. Fischetti, M., & Lodi, A. (2003). Local branching. *Mathematical Programming*, *98*(1), 23–47. https://doi.org/10.1007/s10107-003-0395-5
7. Rei, W., Cordeau, J.-F., Gendreau, M., & Soriano, P. (2009). Accelerating Benders Decomposition by Local Branching. *INFORMS Journal on Computing*, *21*(2), 333–345. https://doi.org/10.1287/ijoc.1080.0296
8. Tsang, M. Y., & Shehadeh, K. S. (2022). *Convex Fairness Measures: Theory and Optimization* (arXiv:2211.13427). arXiv. http://arxiv.org/abs/2211.13427