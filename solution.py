# coding:utf-8
# By Penghui Guo (https://guo.ph) for "苏州园区“华为云杯”2023人工智能应用创新大赛（创客）" 2023, all rights reserved.

from instance import Instance
from config import INSTANCES
from utility import load_json


class Solution(Instance):

    def __init__(self, instance_data, solution_data):
        super().__init__(instance_data)

        self.dispatch_results = solution_data["dispatch_results"]

        assert self.all_processes_are_assigned()
        assert self.all_workers_are_assigned()

        self.is_valid_process_order()

        self.obj = self.compute_real_objective()

    def all_processes_are_assigned(self):
        assigned_precesses = set()
        for a in self.dispatch_results:
            for b in a["operation_list"]:
                assigned_precesses.add(b["operation"])
        return assigned_precesses == set(a["operation"] for a in self.process_list)

    def all_workers_are_assigned(self):
        assigned_workers = set()
        for a in self.dispatch_results:
            worker = a["worker_code"]
            if worker:
                assigned_workers.add(worker)
        return assigned_workers == set(a["worker_code"] for a in self.worker_list)

    def is_valid_process_order(self):
        # TODO: how to detect cycle and revisit?
        pass

    def compute_revisit(self):
        # TODO: how to detect cycle and revisit?
        pass

    def compute_cycle(self):
        # TODO: how to detect cycle and revisit?
        pass

    def compute_workload(self):
        pass

    def compute_real_objective(self):
        # TODO: compute real objective
        pass


if __name__ == '__main__':
    instance_li = ["instance-50.txt"]
    # instance_li = INSTANCES[:10]

    for instance in instance_li:
        instance_data = load_json(f"instances/{instance}")
        solution_data = load_json(f"solutions/{instance}_result.txt")
        Sol = Solution(instance_data, solution_data)
