# coding:utf-8
# By Penghui Guo (https://guo.ph) for "苏州园区“华为云杯”2023人工智能应用创新大赛（创客）" 2023, all rights reserved.

SOLVERS = [
    "gurobi",
    "glpk",
    "cbc",
    "lp_solve",
    "scip"
]

INSTANCES = [
    "instance-2.txt",  # |P|=45, |S|=20, |W|=8, MC=5, COL=36000
    "instance-3.txt",  # |P|=45, |S|=20, |W|=8, MC=5, COL=36000
    "instance-4.txt",  # |P|=45, |S|=20, |W|=8, MC=5, COL=36000
    # # "instance-7.txt",  # |P|=28, |S|=24, |W|=14, MC=4, COL=37632, 🟥 INFEASIBLE without splitting
    # # "instance-8.txt",  # |P|=28, |S|=24, |W|=14, MC=4, COL=37632, 🟥 INFEASIBLE without splitting
    "instance-40.txt",  # |P|=45, |S|=20, |W|=9, MC=5, COL=40500
    "instance-41.txt",  # |P|=45, |S|=20, |W|=9, MC=5, COL=40500
    "instance-42.txt",  # |P|=45, |S|=20, |W|=9, MC=5, COL=40500
    # # "instance-9.txt",  # |P|=28, |S|=26, |W|=14, MC=4, COL=40768, 🟥 INFEASIBLE without splitting
    # # "instance-50.txt",  # |P|=50, |S|=20, |W|=9, MC=5, COL=45000, 🟥 INFEASIBLE without splitting
    # # "instance-54.txt",  # |P|=38, |S|=20, |W|=16, MC=4, COL=48640, 🟥 INFEASIBLE without splitting
    "instance-60.txt",  # |P|=35, |S|=24, |W|=16, MC=4, COL=53760
    "instance-39.txt",  # |P|=68, |S|=23, |W|=12, MC=3, COL=56304
    # # "instance-35.txt",  # |P|=36, |S|=25, |W|=16, MC=4, COL=57600, 🟥 INFEASIBLE without splitting
    # # "instance-36.txt",  # |P|=36, |S|=25, |W|=16, MC=4, COL=57600, 🟥 INFEASIBLE without splitting
    "instance-15.txt",  # |P|=45, |S|=23, |W|=14, MC=4, COL=57960
    "instance-10.txt",  # |P|=45, |S|=23, |W|=15, MC=4, COL=62100
    "instance-14.txt",  # |P|=45, |S|=23, |W|=15, MC=4, COL=62100
    "instance-16.txt",  # |P|=45, |S|=23, |W|=15, MC=4, COL=62100
    # # "instance-12.txt",  # |P|=39, |S|=24, |W|=17, MC=4, COL=63648, 🟥 INFEASIBLE without splitting
    "instance-6.txt",  # |P|=63, |S|=20, |W|=11, MC=5, COL=69300
    # # "instance-11.txt",  # |P|=39, |S|=27, |W|=17, MC=4, COL=71604, 🟥 INFEASIBLE without splitting
    "instance-13.txt",  # |P|=41, |S|=25, |W|=18, MC=4, COL=73800
    # "instance-5.txt",  # |P|=68, |S|=20, |W|=11, MC=5, COL=74800, 🟨 hard for both alchemy() and solve() -> 🟩 local branching
    # "instance-43.txt",  # |P|=80, |S|=20, |W|=10, MC=5, COL=80000, 🟨 hard for both alchemy() and solve() -> 🟩 local branching
    # "instance-18.txt",  # |P|=26, |S|=26, |W|=15, MC=8, COL=81120, 🟥 INFEASIBLE without splitting
    # "instance-38.txt",  # |P|=74, |S|=23, |W|=12, MC=4, COL=81696, 🟨 hard for both alchemy() and solve()
    "instance-46.txt",  # |P|=76, |S|=23, |W|=12, MC=4, COL=83904
    # "instance-37.txt",  # |P|=28, |S|=26, |W|=15, MC=8, COL=87360, 🟥 INFEASIBLE without splitting (workload)
    # "instance-32.txt",  # |P|=61, |S|=24, |W|=15, MC=4, COL=87840 🟨 hard for both alchemy() and solve()
    # "instance-29.txt",  # |P|=68, |S|=20, |W|=11, MC=6, COL=89760 🟨 hard for both alchemy() and solve()
    "instance-51.txt",  # |P|=51, |S|=20, |W|=11, MC=8, COL=89760
    "instance-48.txt",  # |P|=66, |S|=23, |W|=12, MC=5, COL=91080
    "instance-49.txt",  # |P|=66, |S|=23, |W|=12, MC=5, COL=91080
    # "instance-31.txt",  # |P|=60, |S|=24, |W|=16, MC=4, COL=92160 🟨 hard for both alchemy() and solve()
    # "instance-28.txt",  # |P|=68, |S|=23, |W|=13, MC=5, COL=101660 🟨 hard for both alchemy() and solve()
    # "instance-53.txt",  # |P|=56, |S|=24, |W|=16, MC=5, COL=107520 🟨 hard for solve()
    "instance-45.txt",  # |P|=78, |S|=23, |W|=12, MC=5, COL=107640
    "instance-26.txt",  # |P|=35, |S|=26, |W|=15, MC=8, COL=109200
    # "instance-56.txt",  # |P|=32, |S|=27, |W|=16, MC=8, COL=110592, 🟥 INFEASIBLE without splitting (workload)
    # "instance-1.txt",  # |P|=83, |S|=20, |W|=14, MC=5, COL=116200 🟨 hard for both alchemy() and solve()
    "instance-52.txt",  # |P|=48, |S|=24, |W|=15, MC=8, COL=138240
    "instance-57.txt",  # |P|=51, |S|=24, |W|=15, MC=8, COL=146880
    "instance-47.txt",  # |P|=71, |S|=23, |W|=12, MC=8, COL=156768
    "instance-44.txt",  # |P|=72, |S|=23, |W|=12, MC=8, COL=158976
    "instance-17.txt",  # |P|=67, |S|=32, |W|=19, MC=4, COL=162944
    "instance-27.txt",  # |P|=70, |S|=23, |W|=13, MC=8, COL=167440
    "instance-30.txt",  # |P|=55, |S|=24, |W|=16, MC=8, COL=168960
    # "instance-55.txt",  # |P|=69, |S|=24, |W|=16, MC=8, COL=211968 🟨 hard for both alchemy() and solve()
    "instance-33.txt",  # |P|=84, |S|=32, |W|=19, MC=5, COL=255360 🟨 hard for solve()
    "instance-24.txt",  # |P|=61, |S|=32, |W|=19, MC=8, COL=296704
    # "instance-20.txt",  # |P|=54, |S|=32, |W|=22, MC=8, COL=304128, 🟥 INFEASIBLE without splitting
    # "instance-21.txt",  # |P|=55, |S|=32, |W|=23, MC=8, COL=323840 🟨 hard for solve()
    # "instance-34.txt",  # |P|=58, |S|=32, |W|=23, MC=8, COL=341504 🟨 hard for both alchemy() and solve()
    "instance-22.txt",  # |P|=71, |S|=32, |W|=20, MC=8, COL=363520
    "instance-23.txt",  # |P|=81, |S|=32, |W|=20, MC=8, COL=414720
    # "instance-25.txt",  # |P|=104, |S|=32, |W|=24, MC=8, COL=638976 🟨 hard for both alchemy() and solve()
    # "instance-59.txt",  # |P|=56, |S|=55, |W|=26, MC=8, COL=640640, 🟥 INFEASIBLE without splitting
    "instance-19.txt",  # |P|=63, |S|=53, |W|=25, MC=8, COL=667800
    # "instance-58.txt",  # |P|=63, |S|=55, |W|=28, MC=8, COL=776160, 🟥 INFEASIBLE without splitting
]

PARAMETERS = {
    "SOLVER": SOLVERS[-1],
    "TIME_LIMIT": 60,
    "MIP_GAP": 0.01,
    "ALLOW_TASK_SPLITTING": True,
}

if __name__ == '__main__':
    pass
