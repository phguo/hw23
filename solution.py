# coding:utf-8
# By Penghui Guo (https://guo.ph) for "苏州园区“华为云杯”2023人工智能应用创新大赛（创客）" 2023, all rights reserved.

from itertools import product

from instance import Instance
from config import INSTANCES
from utility import load_json


class Solution(Instance):

    def __init__(self, instance_data, worker_to_process, process_to_station, cycle_num):
        super().__init__(instance_data)
        self.cycle_num = cycle_num
        self.worker_to_process = worker_to_process
        self.process_to_station = process_to_station
        self.ext_stations = self.__make_ext_stations()
        self.worker_to_station = self.__calculate_worker_to_station()

        self.process_station = {p: s for p, s in self.process_to_station if self.process_to_station[p, s] == 1}
        self.station_worker = {s: w for w, s in self.worker_to_station if self.worker_to_station[w, s] == 1}
        self.process_order = self.__make_process_order()

        # a = self.all_processes_are_assigned()
        # b = self.all_workers_are_assigned()
        # c = self.valid_station_worker_num()
        # d = self.valid_worker_station_num()
        # e = self.valid_process_order()
        # f = self.valid_revisit()
        # self.is_feasible = a and b and c and d and e and f

        # self.write_solution()

    def __make_process_order(self):
        process_order = []
        for c in range(self.cycle_num):
            for s in self.stations:
                station_id = s + c * self.station_num
                for p in self.task_tp_order:
                    if self.process_station[p] == station_id:
                        process_order += [p]
        return process_order

    def __make_ext_stations(self):
        # Extend stations by duplication according to max_cycle_count
        dummy_stations = dict()
        for s in range(self.station_num + 1, self.cycle_num * self.station_num + 1):
            original_s = (s - 1) % self.station_num + 1
            dummy_stations.update({s: self.stations[original_s]})
        return {**self.stations, **dummy_stations}

    def __calculate_worker_to_station(self):
        worker_to_station = {(w, s): 0 for w in self.workers for s in self.stations}
        for p, w, s in product(self.processes, self.workers, self.stations):
            for c in range(self.cycle_num):
                if self.worker_to_process[w, p] * self.process_to_station[p, s + c * self.station_num] == 1:
                    worker_to_station[w, s] = 1
        return worker_to_station

    def all_workers_are_assigned(self):
        for w in self.workers:
            w_assigned_stations = [s for s in self.stations if self.worker_to_station[w, s] == 1]
            assert len(w_assigned_stations) >= 1, f"Worker [{w}] is not assigned to any station."
        return True

    def all_processes_are_assigned(self):
        for p in self.processes:
            p_assigned_stations = [s for p, s in self.process_to_station if self.process_to_station[p, s] == 1]
            assert len(p_assigned_stations) >= 1, f"Process [{p}] is not assigned to any station."
        return True

    def valid_station_worker_num(self):
        for s in self.stations:
            s_assigned_workers = [w for w in self.workers if self.worker_to_station[w, s] == 1]
            assert len(s_assigned_workers) <= 1, \
                f"Station [{s}] assigned workers [{len(s_assigned_workers)}] exceed [1]."
        return True

    def valid_worker_station_num(self):
        for w in self.workers:
            w_assigned_stations = [s for s in self.stations if self.worker_to_station[w, s] == 1]
            assert len(w_assigned_stations) <= self.max_station_per_worker, \
                f"Worker [{w}] assigned stations [{len(w_assigned_stations)}] exceed [{self.max_station_per_worker}]."
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
        for s in self.stations:
            for c in range(self.cycle_num):
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

    def write_solution(self):
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
        for p, c, s in product(self.process_order, range(self.cycle_num), self.stations):
            station_result = dispatch_results[s - 1]
            if self.process_to_station[p, s + c * self.station_num]:
                station_operation = {
                    "operation": self.processes[p].operation,
                    "operation_number": self.process_order.index(p) + 1,
                    # "station_id": s + c * self.station_num,
                }
                station_result["operation_list"].append(station_operation)

        # Clean empty station
        dispatch_results = [a for a in dispatch_results if a["worker_code"] != '']

        # from rich import print
        # print(dispatch_results)

        return {"dispatch_results": dispatch_results}


if __name__ == '__main__':
    pass
