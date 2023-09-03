# coding:utf-8
# By Penghui Guo (https://guo.ph) for "苏州园区“华为云杯”2023人工智能应用创新大赛（创客）" 2023. All rights reserved.

import time
from itertools import product

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
from ortools.sat.python import cp_model

from instance import Instance
from config import INSTANCES, PARAMETERS
from solution import Solution
from utility import load_json

optimizer = pyo.SolverFactory('appsi_highs')
optimizer.config.load_solution = False


class Solver(Instance):
    def __init__(self, instance_data):
        super().__init__(instance_data)

        self.no_need_split = None

    def make_master_problem(self, process_to_station):
        model = pyo.ConcreteModel()

        # model.worker_to_process = pyo.Set(initialize=product(self.workers, self.processes))
        model.assign_worker_to_process_vars = pyo.Var(
            list(product(self.workers, self.processes)), domain=pyo.Binary, initialize=0)
        model.assign_worker_to_station_vars = pyo.Var(
            list(product(self.workers, self.stations)), domain=pyo.Binary, initialize=0)

        # Link w-p with w-s
        model.linking_cons = pyo.ConstraintList()
        for w, s in product(self.workers, self.stations):
            model.linking_cons.add(
                expr=model.assign_worker_to_station_vars[w, s] <=
                     sum(model.assign_worker_to_process_vars[w, p] * process_to_station[p, s] for p in self.processes))
            model.linking_cons.add(
                expr=model.assign_worker_to_station_vars[w, s] >=
                     sum(model.assign_worker_to_process_vars[w, p] * process_to_station[p, s] for p in
                         self.processes) / self.process_num)

        # Maximum number of stations for each worker
        model.max_station_per_worker_cons = pyo.ConstraintList()
        for w in self.workers:
            model.max_station_per_worker_cons.add(
                expr=sum(model.assign_worker_to_station_vars[w, s] for s in self.stations) <=
                     self.max_station_per_worker)

        # Each worker must be assigned to at least one station
        model.assign_worker_to_station_b_cons = pyo.ConstraintList()
        for w in self.workers:
            model.assign_worker_to_station_b_cons.add(
                expr=sum(model.assign_worker_to_station_vars[w, s] for s in self.stations) >= 1)

        # Each station has no more than one worker
        model.station_worker_cons = pyo.ConstraintList()
        for s in self.stations:
            model.station_worker_cons.add(
                expr=sum(model.assign_worker_to_station_vars[w, s] for w in self.workers) <= 1)

        # Each process must be assigned to exactly one worker
        model.assign_worker_to_process_cons = pyo.Constraint(
            self.processes,
            rule=lambda m, p:
            sum(m.assign_worker_to_process_vars[w, p] for w in self.workers) == 1)

        model.max_worker_per_operation_cons = pyo.Constraint(
            self.processes,
            rule=lambda m, p:
            sum(m.assign_worker_to_process_vars[w, p] for w in self.workers) <= self.max_worker_per_oper)

        # Each worker must be assigned to at least one process
        model.assign_worker_to_process_b_cons = pyo.Constraint(
            self.workers,
            rule=lambda m, w:
            sum(m.assign_worker_to_process_vars[w, p] for p in self.processes) >= 1)

        # Skill & skill category constraints
        model.worker_skill_capable_cons = pyo.ConstraintList()
        for w, p in product(self.workers, self.processes):
            model.worker_skill_capable_cons.add(
                expr=model.assign_worker_to_process_vars[w, p] <=
                     max(self.skill_capable[(w, p)], self.category_capable[(w, p)]))

        # Must assign processes that have capable-skill workers to at least one of them
        model.worker_skill_capable_b_cons = pyo.ConstraintList()
        for p in self.pros_have_capable_skill_workers:
            model.worker_skill_capable_b_cons.add(
                expr=sum(
                    model.assign_worker_to_process_vars[w, p] for w in self.pros_skill_capable_workers[p]) == 1)
            for w in self.workers:
                if w not in self.pros_skill_capable_workers[p]:
                    model.worker_skill_capable_b_cons.add(expr=model.assign_worker_to_process_vars[w, p] == 0)

        # Fix worker to processes
        model.fix_worker_cons = pyo.ConstraintList()
        for p, process in self.processes.items():
            if process.fixed_worker_code:
                w = self.worker_code_to_id[process.fixed_worker_code]
                model.fix_worker_cons.add(expr=model.assign_worker_to_process_vars[w, p] == 1)
                for w_ in set(self.workers.keys()) - {w}:
                    model.fix_worker_cons.add(expr=model.assign_worker_to_process_vars[w_, p] == 0)

        """
        Objective
        """
        model.workload_vars = pyo.Var(self.workers, domain=pyo.NonNegativeReals, initialize=0)
        # Define workload without splitting process
        model.def_workload_cons = pyo.Constraint(
            self.workers,
            rule=lambda m, w:
            m.workload_vars[w] == sum(
                self.processes[p].standard_oper_time / self._get_efficiency(w, p)
                * m.assign_worker_to_process_vars[w, p] for p in self._get_worker_capable_process(w)))

        # Mean workload constraint
        model.mean_workload_var = pyo.Var(domain=pyo.NonNegativeReals, initialize=0)
        model.mean_workload_cons = pyo.Constraint(
            expr=model.mean_workload_var == sum(model.workload_vars[w] for w in self.workers) / self.worker_num)

        # Max workload constraint
        model.max_workload_var = pyo.Var(domain=pyo.NonNegativeReals, initialize=0)
        model.max_workload_cons = pyo.Constraint(
            self.workers, rule=lambda m, w: m.max_workload_var >= m.workload_vars[w])

        # # Workload volatility_rate constraint
        # model.workload_volatility_a_cons = pyo.Constraint(
        #     self.workers,
        #     rule=lambda m, w:
        #     m.workload_vars[w] <= m.mean_workload_var * (1 + self.volatility_rate))
        # model.workload_volatility_b_cons = pyo.Constraint(
        #     self.workers,
        #     rule=lambda m, w:
        #     m.workload_vars[w] >= m.mean_workload_var * (1 - self.volatility_rate))

        # Objective: feasibility checking
        model.objective = pyo.Objective(expr=0, sense=pyo.minimize)

        return model

    def make_cp_sub_problem(self, cycle_count=None):
        # Extend stations by duplication according to max_cycle_count
        if cycle_count is None:
            cycle_count = self.max_cycle_count
        dummy_stations = dict()
        for s in range(self.station_num + 1, cycle_count * self.station_num + 1):
            original_s = (s - 1) % self.station_num + 1
            dummy_stations.update({s: self.stations[original_s]})
        ext_stations = {**self.stations, **dummy_stations}
        self.ext_stations = ext_stations

        def __get_dummy_stations(s):
            return [a for a in ext_stations if (a - 1) % self.station_num + 1 == s]

        processes, immediate_precedence, processes_required_machine = (
            self.processes, self.immediate_precedence, self.processes_required_machine)

        cp = cp_model.CpModel()

        cp_process_to_station = {
            (p, s): cp.NewBoolVar('p_{}_s_{}'.format(p, s))
            for p, s in product(processes, ext_stations)}
        cp_aux_process_to_station = {
            (p, s): cp.NewBoolVar('p_{}_ss_{}'.format(p, s))
            for p, s in product(processes, self.stations)}
        cp_machine_to_station = {
            (m, s): cp.NewBoolVar('m_{}_s_{}'.format(m, s))
            for m, s in product(self.aux_machines, self.stations)}

        """
        Linking
        """
        # Link p-s with p-s_
        for p, s in product(processes, self.stations):
            for s_ in __get_dummy_stations(s):
                cp.AddImplication(cp_process_to_station[(p, s_)], cp_aux_process_to_station[(p, s)])
            cp.Add(cp_aux_process_to_station[(p, s)] <= sum(
                cp_process_to_station[(p, s_)] for s_ in __get_dummy_stations(s)))

        """
        Process
        """
        # Each process must be assigned to a station
        for p in processes:
            cp.Add(sum(cp_process_to_station[(p, s)] for s in ext_stations) == 1)

        # Precedence constraints
        for p1, p2 in immediate_precedence:
            cp.Add(sum(s * cp_process_to_station[(p1, s)] for s in ext_stations) <=
                   sum(s * cp_process_to_station[(p2, s)] for s in ext_stations))

        """
        Machine
        """
        # Required machines should be prepared at the station
        for p, s in product(processes, ext_stations):
            cp.Add(cp_process_to_station[(p, s)] <=
                   cp_machine_to_station[(processes_required_machine[p], (s - 1) % self.station_num + 1)])

        # Maximum number of machines in each station
        for s in self.stations:
            cp.Add(
                sum(cp_machine_to_station[(k, s)] for k, v in self.aux_machines.items() if v.is_machine_needed)
                <= self.max_machine_per_station)

        # Mono-machine constraint
        for s, k1 in product(self.stations, self.mono_aux_machines):
            for k in set(self.aux_machines.keys()) - {k1}:
                cp.Add(cp_machine_to_station[(k, s)] <= 1 - cp_machine_to_station[(k1, s)])

        # Fixed machines only appear in stations that require them
        for s, ms in self.stations_fixed_machines.items():
            for m in ms:
                cp.Add(cp_machine_to_station[(m, s)] == 1)
        for m, s in product(self.fixed_machines, self.stations):
            if m not in self.stations_fixed_machines[s]:
                cp.Add(cp_machine_to_station[(m, s)] == 0)

        """
        Station
        """
        # Fix station to processes
        for p, process in processes.items():
            if process.fixed_station_code:
                s = self.station_code_to_id[process.fixed_station_code]
                cp.Add(sum(cp_process_to_station[(p, s_)] for s_ in __get_dummy_stations(s)) == 1)
                for s_ in set(self.stations.keys()) - {s}:
                    for s__ in __get_dummy_stations(s_):
                        cp.Add(cp_process_to_station[(p, s__)] == 0)

        """
        Revisit
        """
        # Define variables for maximum revisit constraint
        cp_visit = {
            (s, c): cp.NewBoolVar(name=f"visit_{s}_{c}")
            for s, c in product(self.stations, range(cycle_count))}
        cp_revisit = {
            s: cp.NewIntVar(name=f"revisit_{s}", lb=0, ub=cycle_count)
            for s in self.stations}

        # Calculate visit_vars
        for s, c, p in product(self.stations, range(cycle_count), processes):
            cp.AddImplication(cp_process_to_station[(p, s + c * self.station_num)], cp_visit[(s, c)])

        # Calculate revisit_vars
        for s in self.stations:
            cp.AddMaxEquality(cp_revisit[s], [sum(cp_visit[(s, c)] for c in range(cycle_count)) - 1, 0])

        # Maximum revisit count (2) for each station without unmovable machine
        for s in self.station_with_no_unmovable_machine:
            cp.Add(cp_revisit[s] <= self._max_revisit_no_unmovable)

        # Maximum total revisit count constraint
        cp.Add(sum(cp_revisit[s] for s in self.stations) <= self.max_revisited_station_count)

        """
        Objective
        """
        cp.Minimize(0)

        return cp, cp_process_to_station, cp_aux_process_to_station

    def get_real_objective(self, master_model):
        workloads = [pyo.value(master_model.workload_vars[w]) for w in self.workers]
        mean_workload = sum(workloads) / self.worker_num
        std_dev = (sum([(x - mean_workload) ** 2 for x in workloads]) / self.worker_num) ** 0.5
        score1 = mean_workload / max(workloads)
        score2 = std_dev / mean_workload
        real_objective = (1 - score1) * self.upph_weight + score2 * self.volatility_weight
        return real_objective

    def solve(self, cp_time_limit, total_time_limit):
        f_s = lambda a, b, vars: 1 if solver.Value(vars[a, b]) > 0.5 else 0
        f_m = lambda w, p, vars: 1 if vars[w, p].value > 0.5 else 0

        sub, cp_process_to_station, cp_aux_process_to_station = self.make_cp_sub_problem()
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = False
        solver.parameters.max_time_in_seconds = cp_time_limit

        solver.Solve(sub)
        iter = 0

        start_time = time.time()

        process_to_station = None
        worker_to_process = None
        worker_to_station = None
        self.solved = False

        obj = 10

        while time.time() - start_time <= total_time_limit and not self.solved:

            process_to_station = {(p, s): f_s(p, s, cp_process_to_station) for p, s in cp_process_to_station}
            aux_process_to_station = {(p, s): 0 for p, s in product(self.processes, self.stations)}
            for p, s in cp_process_to_station:
                if process_to_station[(p, s)]:
                    aux_process_to_station[(p, (s - 1) % self.station_num + 1)] = process_to_station[(p, s)]
            ext_station_processes = {s: [p for p in self.processes if process_to_station[(p, s)] == 1] for s in
                                     self.ext_stations}
            station_processes = {s: [p for p in self.processes if aux_process_to_station[(p, s)] == 1] for s in
                                 self.stations}

            have_empty_intersection = False
            for s, ps in station_processes.items():
                li = [self.pros_skill_capable_workers[p]
                      if self.pros_skill_capable_workers[p] else self.pros_category_capable_workers[p] for p in ps]
                if li:
                    intersection = set.intersection(*li)
                    print(s, ps, end=' -> ')
                    # print(s, li, intersection)
                    if not intersection:
                        have_empty_intersection = True
                        ps_one = set()
                        for p in ps:
                            ps_one.add((p, s))
                        sub.Add(sum(1 - cp_aux_process_to_station[(p, s)] for p, s in ps_one) >= 1)

            if not have_empty_intersection:
                master = self.make_master_problem(aux_process_to_station)
                optimizer = pyo.SolverFactory('appsi_highs')
                optimizer.config.load_solution = False
                results = optimizer.solve(master, tee=False, load_solutions=False)
                print("Master is:", results.solver.termination_condition)

                if results.solver.termination_condition == TerminationCondition.optimal:
                    self.solved = True
                    master.solutions.load_from(results)
                    mean_workload = pyo.value(master.mean_workload_var)
                    for w in self.workers:
                        workload = pyo.value(master.workload_vars[w])
                        valid_workload_ub = workload <= mean_workload * (1 + self.volatility_rate)
                        valid_workload_lb = workload >= mean_workload * (1 - self.volatility_rate)
                        print(
                            w, workload, valid_workload_ub, valid_workload_lb,
                            mean_workload * (1 + self.volatility_rate), mean_workload * (1 - self.volatility_rate))
                        if not valid_workload_ub or not valid_workload_lb:
                            self.solved = False
                            worker_processes = [
                                p for p in self.processes if
                                pyo.value(master.assign_worker_to_process_vars[w, p]) >= 0.5]
                            ps_one = set(
                                (p, s) for p, s in product(worker_processes, self.stations)
                                if aux_process_to_station[(p, s)] >= 0.5)
                            ps_zero = set(
                                (p, s) for p, s in product(worker_processes, self.stations)
                                if aux_process_to_station[(p, s)] <= 0.5)
                            print("Workload feasibility cut added.")
                            sub.Add(sum(1 - cp_aux_process_to_station[(p, s)] for p, s in ps_one) +
                                    sum(cp_aux_process_to_station[(p, s)] for p, s in ps_zero) >= 1)

                    if self.solved:
                        process_to_station = {
                            (p, s): f_s(p, s, cp_process_to_station) for p, s in cp_process_to_station}
                        worker_to_process = {
                            (w, p): f_m(w, p, master.assign_worker_to_process_vars) for w, p in
                            master.assign_worker_to_process_vars}
                        worker_to_station = {
                            (w, s): f_m(w, s, master.assign_worker_to_station_vars) for w, s in
                            master.assign_worker_to_station_vars}
                        obj = self.get_real_objective(master)

                else:
                    print("Other feasibility cut added.")
                    one = set((p, s) for p, s in aux_process_to_station if aux_process_to_station[(p, s)] >= 0.5)
                    zero = set((p, s) for p, s in aux_process_to_station if aux_process_to_station[(p, s)] <= 0.5)
                    sub.Add(sum(1 - cp_aux_process_to_station[(p, s)] for p, s in one) +
                            sum(cp_aux_process_to_station[(p, s)] for p, s in zero) >= 1)

            solver.Solve(sub)
            print()
        return obj, worker_to_process, process_to_station, worker_to_station, self.max_cycle_count, None


if __name__ == '__main__':
    instance_li = INSTANCES

    start_time = time.time()
    real_objectives = {}
    instance_count = 0

    for instance in instance_li:
        instance_count += 1
        instance_start_time = time.time()
        print(f"[{instance_count}/{len(instance_li)}] Solving {instance}")

        S = Solver(load_json(f"instances/{instance}"))
        real_obj, worker_to_process, process_to_station, worker_to_station, cycle_num, process_map = (
            S.solve(cp_time_limit=PARAMETERS["CP_TIME_LIMIT"], total_time_limit=PARAMETERS["UALB_CB2_TIME_LIMIT"]))
        solution = Solution(
            S.instance_data,
            worker_to_process, process_to_station, worker_to_station, cycle_num, split_task=False)
        output_json = solution.write_solution()

        print(f"Ins. Runtime    : {time.time() - instance_start_time} seconds\n")

    print(f"Total Runtime   : {time.time() - start_time} seconds")
