# coding:utf-8
# By Penghui Guo (https://guo.ph) for "苏州园区“华为云杯”2023人工智能应用创新大赛（创客）" 2023, all rights reserved.

from itertools import product

from instance import Instance
from config import INSTANCES
from utility import load_json


class Solution(Instance):

    def __init__(self, instance_data, worker_to_process, process_to_station, worker_to_station, cycle_num, split_task):
        super().__init__(instance_data)

        self.cycle_num = cycle_num
        self.worker_to_process = worker_to_process
        self.process_to_station = process_to_station
        self.worker_to_station = worker_to_station
        self.split_task = split_task

        self.is_feasible = self.valid_volatility_rate(worker_to_process)

        if self.split_task:
            (self.processes, self.immediate_precedence, self.worker_to_process, self.processes_required_machine,
             self.process_map) = self._make_dummy_process(worker_to_process)

        self.is_feasible = all([
            self.all_workers_are_assigned(),
            self.all_processes_are_assigned(),
            self.valid_station_worker_num(),
            self.valid_worker_station_num(),
            self.valid_process_order(),
            self.valid_revisit(),
            # self.valid_worker_capability(),
            self.valid_machine(),
        ])

        # if self.split_task:
        #     self.is_feasible = all([
        #         self.is_feasible,
        #         self.split_process_use_single_station(),
        #     ])

        self.write_solution()

    def valid_volatility_rate(self, worker_to_process):
        process_worker_num = {p: sum(worker_to_process[w, p] for w in self.workers) for p in self.processes}
        workloads = {w: sum(self.processes[p].standard_oper_time * worker_to_process[w, p]
                            / process_worker_num[p]
                            / self._get_efficiency(w, p) for p in self._get_worker_capable_process(w))
                     for w in self.workers}
        mean_workload = sum(workloads.values()) / len(workloads)
        for workload in workloads.values():
            assert workload <= mean_workload * (1 + self.volatility_rate), \
                f"Workload [{workload}] exceed [{mean_workload * self.volatility_rate}]."
            assert workload >= mean_workload * (1 - self.volatility_rate), \
                f"Workload [{workload}] less than [{mean_workload * self.volatility_rate}]."

    def all_workers_are_assigned(self):
        for w in self.workers:
            w_assigned_stations = [s for s in self.stations if self.worker_to_station[w, s] == 1]
            assert len(w_assigned_stations) >= 1, f"Worker [{w}] is not assigned to any station."
        return True

    def all_processes_are_assigned(self):
        for p in self.processes:
            p_assigned_stations = [
                s for p_, s in self.process_to_station if self.process_to_station[p_, s] == 1 and p_ == p]
            assert len(p_assigned_stations) >= 1, f"Process [{p}] is not assigned to any station."
            p_assigned_workers = [
                w for w, p_ in self.worker_to_process if self.worker_to_process[w, p_] == 1 and p_ == p]
            assert len(p_assigned_workers) >= 1, f"Process [{p}] is not assigned to any worker."
        return True

    def valid_station_worker_num(self):
        for s in self.stations:
            s_assigned_workers = [w for w in self.workers if self.worker_to_station[w, s] == 1]
            assert len(s_assigned_workers) <= 1, \
                f"Station [{s}] assigned workers [{len(s_assigned_workers)}] exceed [1]."
        return True

    # def valid_worker_capability(self):
    #     for w, p in self.worker_to_process:
    #         if self.worker_to_process[w, p] == 1:
    #             capable = False
    #             # if p in self.pros_skill_capable_workers:
    #             #     print(w, self.pros_skill_capable_workers[p])
    #             #     print(w in self.pros_skill_capable_workers[p])
    #             #     capable = w in self.pros_skill_capable_workers[p]
    #             # else:
    #             #     if p in self.pros_category_capable_workers:
    #             #         capable = w in self.pros_category_capable_workers[p]
    #             assert capable, f"Worker [{w}] is not capable of process [{p}]."
    #     return True

    def valid_worker_station_num(self):
        for w in self.workers:
            w_assigned_stations = [s for s in self.stations if self.worker_to_station[w, s] == 1]
            assert len(w_assigned_stations) <= self.max_station_per_worker, \
                f"Worker [{w}] assigned stations [{len(w_assigned_stations)}] exceed [{self.max_station_per_worker}]."
            assert len(w_assigned_stations) >= 1, \
                f"Worker [{w}] is not assigned to any station."
        return True

    def valid_process_order(self):
        process_station = {
            p: s for p, s in self.process_to_station if self.process_to_station[p, s] == 1}
        for p1, p2 in self.immediate_precedence:
            assert process_station[p1] <= process_station[p2], \
                f"Invalid process order: [{p1}]({process_station[p1]}) -> [{p2}]({process_station[p2]})."
        return True

    def valid_revisit(self):
        station_visit = {s: 0 for s in self.stations}
        for s, c in product(self.stations, range(self.cycle_num)):
            res = [p for p in self.processes if self.process_to_station[(p, s + c * self.station_num)] == 1]
            if res != []:
                station_visit[s] += 1

        station_revisit = {s: max(0, station_visit[s] - 1) for s in self.stations}
        for s in self.station_with_no_unmovable_machine:
            assert station_revisit[s] <= self._max_revisit_no_unmovable, \
                f"Station [{s}] revisit [{station_revisit[s]}] exceed [{self._max_revisit_no_unmovable}]."

        total_revisit = sum(station_revisit.values())
        assert total_revisit <= self.max_revisited_station_count, \
            f"Total station revisits [{total_revisit}] exceed [{self.max_revisited_station_count}]."
        return True

    def valid_machine(self):
        station_machines = {s: [] for s in self.stations}
        for s, c in product(self.stations, range(self.cycle_num)):
            res = [p for p in self.processes if self.process_to_station[(p, s + c * self.station_num)] == 1]
            if res != []:
                station_machines[s] += [
                    self.processes_required_machine[p] for p in res
                    if self.aux_machines[self.processes_required_machine[p]].is_machine_needed]
        station_machines = {s: set(station_machines[s]) for s in self.stations}

        for s in self.stations:
            assert len(station_machines[s]) <= self.max_machine_per_station, \
                f"Station [{s}] assigned machines [{len(station_machines[s])}] exceed [{self.max_machine_per_station}]."
        return True

    def split_process_use_single_station(self):
        # HINT: stronger than required (?)

        split_process = set(self.process_map.keys())
        split_process_station = {
            p: (s - 1) % self.station_num + 1
            for p, s in self.process_to_station if self.process_to_station[p, s] == 1 and p in split_process}
        process_station = {
            p: (s - 1) % self.station_num + 1
            for p, s in self.process_to_station if self.process_to_station[p, s] == 1}

        for p, s in split_process_station.items():
            for p_, s_ in process_station.items():
                if p != p_ and s == s_:
                    assert False, f"Split process [{p}] and process [{p_}] are assigned to the same station [{s}]."
        return True

    def write_solution(self):
        # TODO: correct (?), according to v25 -> NO
        self.process_station = {p: s for p, s in self.process_to_station if self.process_to_station[p, s] == 1}
        self.station_worker = {s: w for w, s in self.worker_to_station if self.worker_to_station[w, s] == 1}

        def __make_process_order():
            process_order = []
            for c in range(self.cycle_num):
                for s in self.stations:
                    station_id = s + c * self.station_num
                    for p in self.task_tp_order:
                        if self.process_station[p] == station_id:
                            process_order += [p]
            return process_order

        process_order = __make_process_order()

        # Initialize dispatch_results
        dispatch_results = []

        # Station worker
        for s in self.stations:
            station_result = dict()
            station_result["station_code"] = self.stations[s].station_code
            station_result["worker_code"] = ''
            station_result["operation_list"] = []
            if s in self.station_worker:
                station_result["worker_code"] = self.workers[self.station_worker[s]].worker_code
            dispatch_results.append(station_result)

        # Station operation
        process_operation_number = {p: -1 for p in self.processes}
        for p, c, s in product(process_order, range(self.cycle_num), self.stations):
            station_result = dispatch_results[s - 1]
            if self.process_to_station[p, s + c * self.station_num]:
                station_operation = {
                    "operation": self.processes[p].operation,
                    "operation_number": process_order.index(p) + 1,
                    # "station_id": s + c * self.station_num,
                }
                process_operation_number[p] = process_order.index(p) + 1
                station_result["operation_list"].append(station_operation)

        # Assign dummy process
        if self.split_task:
            for p, c, s in product(self.process_map, range(self.cycle_num), self.stations):
                if p not in process_order:
                    station_result = dispatch_results[s - 1]
                    if self.process_to_station[p, s + c * self.station_num]:
                        station_operation = {
                            "operation": self.processes[p].operation,
                            "operation_number": process_operation_number[self.process_map[p]],
                            # "station_id": s + c * self.station_num,
                        }
                        station_result["operation_list"].append(station_operation)

        # Clean empty station
        dispatch_results = [a for a in dispatch_results if not (a["operation_list"] == [] or a["worker_code"] == '')]

        return {"dispatch_results": dispatch_results}


if __name__ == '__main__':
    pass
