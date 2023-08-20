# coding:utf-8
# By Penghui Guo (https://guo.ph) for "苏州园区“华为云杯”2023人工智能应用创新大赛（创客）" 2023, all rights reserved.


import os
from itertools import product

from config import INSTANCES
from utility import load_json

import networkx as nx
from toposort import toposort


class Station(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Process(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Worker(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Machine(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Instance(object):

    def __init__(self, instance_data):
        self.instance_data = instance_data

        self.station_list = instance_data["station_list"]
        self.process_list = instance_data["process_list"]
        self.worker_list = instance_data["worker_list"]
        self.machine_list = instance_data["machine_list"]
        self.joint_operation_list = instance_data["joint_operation_list"]

        self.config_param = instance_data["config_param"]
        self.max_worker_per_oper = self.config_param["max_worker_per_oper"]
        self.max_station_per_worker = self.config_param["max_station_per_worker"]
        self.max_cycle_count = self.config_param["max_cycle_count"]
        self.max_revisited_station_count = self.config_param["max_revisited_station_count"]
        self.volatility_rate = self.config_param["volatility_rate"]
        self.volatility_weight = self.config_param["volatility_weight"]
        self.upph_weight = self.config_param["upph_weight"]
        self.max_machine_per_station = self.config_param["max_machine_per_station"]
        self.max_station_per_oper = self.config_param["max_station_per_oper"]
        self.max_split_num = self.config_param["max_split_num"]

        self._max_revisit_no_unmovable = 2

        self.station_list = sorted(self.station_list, key=lambda x: int(x["line_number"]))
        self.stations = {i + 1: Station(**s) for i, s in enumerate(self.station_list)}
        self.processes = {i + 1: Process(**p) for i, p in enumerate(self.process_list)}
        self.workers = {i + 1: Worker(**w) for i, w in enumerate(self.worker_list)}

        self.process_num = len(self.processes)
        self.station_num = len(self.stations)
        self.worker_num = len(self.workers)
        self.ncol = self.process_num * self.station_num * self.worker_num * self.max_cycle_count

        self.make_aux_machines()
        self.make_code_id_map()
        self.get_immediate_precedence()
        self.get_workers_capable_processes()
        self.get_processes_aux_machine()
        self.get_mono_aux_machines()
        self.get_stations_fixed_aux_machines()
        self.get_stations_with_unmovable_machine()

        self.task_tp_order = self.topological_ordering()
        self.task_tp_order_set = self.topological_ordering_set()

    def __str__(self):
        n = f"|P|={self.process_num}, |S|={self.station_num}, |W|={self.worker_num}, MC={self.max_cycle_count}, " \
            f"Ncol={self.ncol}"
        return n

    def topological_ordering(self):
        G = nx.DiGraph()
        G.add_nodes_from(list(self.processes.keys()))
        G.add_edges_from(self.immediate_precedence)
        topological_order = list(nx.topological_sort(G))
        return topological_order

    def topological_ordering_set(self):
        dep = dict()
        for p in self.processes:
            dep[p] = set()
            for p1, p2 in self.immediate_precedence:
                if p2 == p:
                    dep[p].add(p1)
        return list(toposort(dep))

    def make_aux_machines(self):
        self.aux_machine_list = []
        for p, process in self.processes.items():
            machine_type = process.machine_type
            machine_type_2 = process.machine_type_2
            aux_machine_type = f"{machine_type}_{machine_type_2}"
            for machine_info in self.machine_list:
                if machine_info["machine_type"] == machine_type:
                    aux_machine_info = {
                        "machine_type": aux_machine_type,
                        "is_mono": machine_info["is_mono"],
                        "is_movable": machine_info["is_movable"],
                        "is_machine_needed": machine_info["is_machine_needed"],
                    }
                    if aux_machine_info not in self.aux_machine_list:
                        self.aux_machine_list.append(aux_machine_info)
        self.aux_machines = {i: Machine(**m) for i, m in enumerate(self.aux_machine_list)}

    def make_code_id_map(self):
        self.station_code_to_id = {s.station_code: i for i, s in self.stations.items()}
        self.process_code_to_id = {p.operation: i for i, p in self.processes.items()}
        self.worker_code_to_id = {w.worker_code: i for i, w in self.workers.items()}
        self.machine_code_to_id = {m.machine_type: i for i, m in self.aux_machines.items()}

    def get_immediate_precedence(self):
        self.immediate_precedence = []

        # Get ordered parts' processes based on "operation_number"
        parts = sorted(list(set([p.part_code for p in self.processes.values()])))
        self.parts_processes = {part: None for part in parts}
        for part in parts:
            for process in self.processes.values():
                if process.part_code == part:
                    if self.parts_processes[part] is None:
                        self.parts_processes[part] = [process]
                    else:
                        self.parts_processes[part].append(process)
        for k, v in self.parts_processes.items():
            self.parts_processes[k] = sorted(v, key=lambda x: int(x.operation_number))

        # Immediate precedence based on "operation_number"
        for part, processes in self.parts_processes.items():
            for i, process in enumerate(processes):
                # print(part, process.operation_number, process.operation, self.process_code_to_id[process.operation])
                if i != 0:
                    a = self.process_code_to_id[processes[i - 1].operation]
                    b = self.process_code_to_id[process.operation]
                    self.immediate_precedence.append((a, b))

        # Immediate precedence based on "joint_operation_list"
        for d in self.joint_operation_list:
            a = self.process_code_to_id[self.parts_processes[d["part_code"]][-1].operation]
            b = self.process_code_to_id[d["joint_operation"]]
            self.immediate_precedence.append((a, b))

    def get_workers_capable_processes(self):
        self.workers_capble_pro = dict()
        self.workers_category_capable_pro = dict()
        for i, worker in self.workers.items():
            self.workers_capble_pro[i] = {
                self.process_code_to_id[p["operation_code"]] for p in worker.operation_skill_list}
            self.workers_category_capable_pro[i] = set()
            category_set = set(p["operation_category"] for p in worker.operation_category_skill_list)
            for process in self.processes.values():
                if process.operation_category in category_set:
                    self.workers_category_capable_pro[i].add(self.process_code_to_id[process.operation])

        self.skill_capable = {
            (w, p): 1 if p in self.workers_capble_pro[w] else 0
            for w, p in product(self.workers, self.processes)}
        self.category_capable = {
            (w, p): 1 if p in self.workers_category_capable_pro[w] else 0
            for w, p in product(self.workers, self.processes)}

        self.pros_have_capable_skill_workers = set()
        for p, w in product(self.processes, self.workers):
            if self.skill_capable[(w, p)] == 1:
                self.pros_have_capable_skill_workers.add(p)
        self.pros_have_no_capable_skill_workers = set(self.processes.keys()) - self.pros_have_capable_skill_workers

        self.pros_skill_capable_workers = {
            p: {w for w in self.workers if self.skill_capable[(w, p)] == 1} for p in self.processes}
        self.pros_category_capable_workers = {
            p: {w for w in self.workers if self.category_capable[(w, p)] == 1} for p in self.processes}

    def get_processes_aux_machine(self):
        self.processes_required_machine = dict()
        for p, process in self.processes.items():
            required_machine_type = f"{process.machine_type}_{process.machine_type_2}"
            for k, machine in self.aux_machines.items():
                if machine.machine_type == required_machine_type:
                    self.processes_required_machine[p] = k

    def get_stations_fixed_aux_machines(self):
        self.stations_fixed_machines = {s: [] for s in self.stations.keys()}
        for s, station in self.stations.items():
            if station.curr_machine_list:
                assert len(station.curr_machine_list) == 1
                for k, machine in self.aux_machines.items():
                    if machine.machine_type.split('_')[0] in [a["machine_type"] for a in station.curr_machine_list]:
                        if not machine.is_movable:
                            self.stations_fixed_machines[s].append(k)
                            assert len(self.stations_fixed_machines[s]) == 1
        self.stations_fixed_machine = {k: v[0] if v else None for k, v in self.stations_fixed_machines.items()}

        self.fixed_machines = set()
        for v in self.stations_fixed_machine.values():
            if v:
                self.fixed_machines.add(v)

    def get_mono_aux_machines(self):
        self.mono_aux_machines = [k for k, m in self.aux_machines.items() if m.is_mono]

    def get_stations_with_unmovable_machine(self):
        stations_with_unmovable_machine = dict()
        for s, station in self.stations.items():
            if station.curr_machine_list:
                for machine in station.curr_machine_list:
                    mt = machine["machine_type"]
                    for m, machine_ in self.aux_machines.items():
                        if mt == machine_.machine_type.split("_")[0] and not machine_.is_movable:
                            stations_with_unmovable_machine.update({s: station})
        self.stations_with_unmovable_machine = stations_with_unmovable_machine
        self.station_with_no_unmovable_machine = set(self.stations) - set(self.stations_with_unmovable_machine.keys())

    def _get_worker_capable_process(self, w):
        res = self.workers_capble_pro[w].union(self.workers_category_capable_pro[w])
        return res

    def _get_efficiency(self, w, p):
        # DEBUG: instance-50 is inherently infeasible without splitting?
        if p in self.workers_capble_pro[w]:
            for skill in self.workers[w].operation_skill_list:
                if skill["operation_code"] == self.processes[p].operation:
                    return skill["efficiency"]
        elif p in self.workers_category_capable_pro[w]:
            for category in self.workers[w].operation_category_skill_list:
                if category["operation_category"] == self.processes[p].operation_category:
                    return category["efficiency"]
        else:
            raise Exception(
                f"Process '{p}' in neither pros_have_capable_skill_workers nor pros_have_no_capable_skill_workers")


if __name__ == '__main__':
    instance_li = INSTANCES
    # instance_li = ["instance-50.txt"]

    for instance in instance_li:
        print("Loading instance:", instance)
        I = Instance(load_json(f"./instances/{instance}"))
        # operation_time_li = [p.standard_oper_time for p in I.processes.values()]
        # print(instance, min(operation_time_li), max(operation_time_li), sum(operation_time_li) / len(operation_time_li))

        # print("max_revisits =", I.max_revisited_station_count, "| max_cycle =", I.max_cycle_count)
        # print("mono_machines", I.mono_aux_machines)
        # stations_fixed_machines = {k: v for k, v in I.stations_fixed_machines.items() if v}
        # print("stations_fixed_machines", stations_fixed_machines)
        # processes_required_mono_machine = {k: v for k, v in I.processes_required_machine.items() if
        #                                    v in I.mono_aux_machines}
        # print("processes_required_mono_machine", processes_required_mono_machine)
        # print(I.immediate_precedence)
        # print(I.task_tp_order)
        # print(I.volatility_rate)
        # print()

        # print([v.is_machine_needed for k, v in I.aux_machines.items()])
        # print(I.max_worker_per_oper, I.max_split_num)

        # # Print skill & skill category efficiency
        # for w, p in product(I.workers, I.processes):
        #     skill_efficiency = None
        #     category_efficiency = None
        #     if p in I.workers_capble_pro[w]:
        #         for skill in I.workers[w].operation_skill_list:
        #             if skill["operation_code"] == I.processes[p].operation:
        #                 skill_efficiency = skill["efficiency"]
        #     elif p in I.workers_category_capable_pro[w]:
        #         for category in I.workers[w].operation_category_skill_list:
        #             if category["operation_category"] == I.processes[p].operation_category:
        #                 category_efficiency = category["efficiency"]
        #     if skill_efficiency or category_efficiency:
        #         print(w, p, skill_efficiency, category_efficiency)
        #         assert skill_efficiency is None or category_efficiency is None
        # print()

        # import networkx as nx
        # import matplotlib.pyplot as plt
        # G = nx.DiGraph()
        # G.add_nodes_from(list(I.processes.keys()))
        # G.add_edges_from(I.immediate_precedence)
        # pos = nx.planar_layout(G)
        # # pos = nx.spring_layout(G)
        # nx.draw_networkx(G, pos=pos, node_color='lightblue', node_size=200, font_size=10, edge_color='k', arrows=True)
        # ax = plt.gca()
        # ax.margins(0.0)
        # plt.axis("off")
        # plt.show()
        # topological_order = list(nx.topological_sort(G))
        # print("Topological Order:", topological_order)
        # print(I.immediate_precedence)
        # break

        print("task_tp_order_set", I.task_tp_order_set)
        print("task_tp_set", I.task_tp_order)
        print("max_worker_per_oper:", I.max_worker_per_oper)
        print("max_station_per_oper:", I.max_station_per_oper)
        print("max_split_num:", I.max_split_num)

        # print(I.upph_weight, I.volatility_weight)

        print()
