# coding:utf-8
# By Penghui Guo (https://guo.ph) for "苏州园区“华为云杯”2023人工智能应用创新大赛（创客）" 2023, all rights reserved.


INSTANCES = [
    # # 🟩 Easy instances without splitting
    # "instance-2.txt",  # |P|=45, |S|=20, |W|=8, C=5, COL=36000
    # "instance-3.txt",  # |P|=45, |S|=20, |W|=8, C=5, COL=36000
    # "instance-4.txt",  # |P|=45, |S|=20, |W|=8, C=5, COL=36000
    # "instance-40.txt",  # |P|=45, |S|=20, |W|=9, C=5, COL=40500
    # "instance-41.txt",  # |P|=45, |S|=20, |W|=9, C=5, COL=40500
    # "instance-42.txt",  # |P|=45, |S|=20, |W|=9, C=5, COL=40500
    # "instance-60.txt",  # |P|=35, |S|=24, |W|=16, C=4, COL=53760
    # "instance-39.txt",  # |P|=68, |S|=23, |W|=12, C=3, COL=56304
    # "instance-15.txt",  # |P|=45, |S|=23, |W|=14, C=4, COL=57960
    # "instance-10.txt",  # |P|=45, |S|=23, |W|=15, C=4, COL=62100
    # "instance-14.txt",  # |P|=45, |S|=23, |W|=15, C=4, COL=62100
    # "instance-16.txt",  # |P|=45, |S|=23, |W|=15, C=4, COL=62100
    # "instance-13.txt",  # |P|=41, |S|=25, |W|=18, C=4, COL=73800
    # "instance-46.txt",  # |P|=76, |S|=23, |W|=12, C=4, COL=83904
    # "instance-51.txt",  # |P|=51, |S|=20, |W|=11, C=8, COL=89760
    # "instance-26.txt",  # |P|=35, |S|=26, |W|=15, C=8, COL=109200
    # "instance-52.txt",  # |P|=48, |S|=24, |W|=15, C=8, COL=138240
    # "instance-57.txt",  # |P|=51, |S|=24, |W|=15, C=8, COL=146880
    # "instance-47.txt",  # |P|=71, |S|=23, |W|=12, C=8, COL=156768
    # "instance-44.txt",  # |P|=72, |S|=23, |W|=12, C=8, COL=158976
    # "instance-17.txt",  # |P|=67, |S|=32, |W|=19, C=4, COL=162944
    # "instance-27.txt",  # |P|=70, |S|=23, |W|=13, C=8, COL=167440
    # "instance-30.txt",  # |P|=55, |S|=24, |W|=16, C=8, COL=168960
    # "instance-24.txt",  # |P|=61, |S|=32, |W|=19, C=8, COL=296704
    # "instance-22.txt",  # |P|=71, |S|=32, |W|=20, C=8, COL=363520
    # "instance-23.txt",  # |P|=81, |S|=32, |W|=20, C=8, COL=414720
    # "instance-43.txt",  # |P|=80, |S|=20, |W|=10, C=5, COL=80000C
    # "instance-21.txt",  # |P|=55, |S|=32, |W|=23, C=8, COL=323840C
    # "instance-34.txt",  # |P|=58, |S|=32, |W|=23, C=8, COL=341504C
    # "instance-48.txt",  # |P|=66, |S|=23, |W|=12, C=5, COL=91080, solved with obj max, k list(range(1, 90, 4))
    # "instance-49.txt",  # |P|=66, |S|=23, |W|=12, C=5, COL=91080, solved with obj max, k list(range(1, 90, 4))
    # "instance-45.txt",  # |P|=78, |S|=23, |W|=12, C=5, COL=107640, solved with obj max, k list(range(1, 90, 4))
    # "instance-6.txt",  # |P|=63, |S|=20, |W|=11, C=5, COL=69300, solved with obj 0, k list(range(1, 90, 4))
    # "instance-55.txt",  # |P|=69, |S|=24, |W|=16, C=8, COL=211968C, solved with obj 0, k list(range(1, 90, 4))
    # "instance-33.txt",  # |P|=84, |S|=32, |W|=19, C=5, COL=255360C, solved with obj 0, k list(range(1, 90, 4))

    # # TODO: 🟨 Hard instances without splitting
    # "instance-19.txt",  # |P|=63, |S|=53, |W|=25, C=8, COL=667800
    # "instance-5.txt",  # |P|=68, |S|=20, |W|=11, C=5, COL=74800C
    # "instance-38.txt",  # |P|=74, |S|=23, |W|=12, C=4, COL=81696C
    # "instance-32.txt",  # |P|=61, |S|=24, |W|=15, C=4, COL=87840C
    # "instance-29.txt",  # |P|=68, |S|=20, |W|=11, C=6, COL=89760C
    # "instance-31.txt",  # |P|=60, |S|=24, |W|=16, C=4, COL=92160C
    # "instance-28.txt",  # |P|=68, |S|=23, |W|=13, C=5, COL=101660C
    # "instance-53.txt",  # |P|=56, |S|=24, |W|=16, C=5, COL=107520C
    # "instance-1.txt",  # |P|=83, |S|=20, |W|=14, C=5, COL=116200C
    # "instance-25.txt",  # |P|=104, |S|=32, |W|=24, C=8, COL=638976C

    # # 🟥 Infeasible instances without splitting
    # "instance-7.txt",  # |P|=28, |S|=24, |W|=14, C=4, COL=37632C
    # "instance-8.txt",  # |P|=28, |S|=24, |W|=14, C=4, COL=37632C
    # "instance-9.txt",  # |P|=28, |S|=26, |W|=14, C=4, COL=40768C
    # "instance-35.txt",  # |P|=36, |S|=25, |W|=16, C=4, COL=57600C
    # "instance-36.txt",  # |P|=36, |S|=25, |W|=16, C=4, COL=57600C
    # "instance-12.txt",  # |P|=39, |S|=24, |W|=17, C=4, COL=63648C
    # "instance-11.txt",  # |P|=39, |S|=27, |W|=17, C=4, COL=71604C
    # "instance-18.txt",  # |P|=26, |S|=26, |W|=15, C=8, COL=81120C
    # "instance-37.txt",  # |P|=28, |S|=26, |W|=15, C=8, COL=87360C
    # "instance-56.txt",  # |P|=32, |S|=27, |W|=16, C=8, COL=110592C
    # "instance-20.txt",  # |P|=54, |S|=32, |W|=22, C=8, COL=304128C
    # "instance-59.txt",  # |P|=56, |S|=55, |W|=26, C=8, COL=640640C, hard
    # "instance-58.txt",  # |P|=63, |S|=55, |W|=28, C=8, COL=776160C, hard
    # "instance-50.txt",  # |P|=50, |S|=20, |W|=9, C=5, COL=45000C, infeasible inherently
    # "instance-54.txt",  # |P|=38, |S|=20, |W|=16, C=4, COL=48640C, infeasible (?)
]

SOLVERS = ["gurobi", "glpk", "cbc", "lp_solve", "scip"]

PARAMETERS = {
    "TOTAL_TIME_LIMIT": 360 * 2,
    "CP_TIME_LIMIT": 25,
    "LOCAL_BRANCHING_K_SET": list(range(1, 90, 4)),
}

if __name__ == '__main__':
    for i in sorted(INSTANCES):
        print(i)
