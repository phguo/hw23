# coding:utf-8
# By Penghui Guo (https://guo.ph) for "苏州园区“华为云杯”2023人工智能应用创新大赛（创客）" 2023, all rights reserved.

import time
from itertools import product

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
from ortools.sat.python import cp_model

from instance import Instance
from config import INSTANCES
from solution import Solution
from utility import load_json


class Solver(Instance):
    def __init__(self, instance_data):
        super().__init__(instance_data)

    def make_master_problem(self, split_task=False, obj_weight=(1, 0, 0), shrink=(0, 0)):
        model = pyo.ConcreteModel()

        model.worker_to_process = pyo.Set(initialize=product(self.workers, self.processes))
        model.assign_worker_to_process_vars = pyo.Var(
            model.worker_to_process, domain=pyo.Binary, initialize=0)

        # Each process must be assigned to exactly one worker
        model.assign_worker_to_process_cons = pyo.Constraint(
            self.processes,
            rule=lambda m, p:
            sum(m.assign_worker_to_process_vars[w, p] for w in self.workers) == 1)

        # Maximum worker per operation
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
                     self.skill_capable[(w, p)] + self.category_capable[(w, p)])

        # Must assign processes that have capable-skill workers to at least one of them
        model.worker_skill_capable_b_cons = pyo.Constraint(
            self.pros_have_capable_skill_workers,
            rule=lambda m, p:
            sum(m.assign_worker_to_process_vars[w, p] for w in
                set(ww for ww in self.workers if self.skill_capable[(ww, p)] == 1)) == 1)

        # Fix worker to processes
        model.fix_worker_cons = pyo.ConstraintList()
        for p, process in self.processes.items():
            if process.fixed_worker_code:
                w = self.worker_code_to_id[process.fixed_worker_code]
                model.fix_worker_cons.add(expr=model.assign_worker_to_process_vars[w, p] == 1)
                for w_ in set(self.workers.keys()) - {w}:
                    model.fix_worker_cons.add(expr=model.assign_worker_to_process_vars[w_, p] == 0)

        model.workload_vars = pyo.Var(self.workers, domain=pyo.NonNegativeReals, initialize=0)

        # Define workload without splitting process
        model.def_workload_cons = pyo.Constraint(
            self.workers,
            rule=lambda m, w:
            m.workload_vars[w] == sum(
                self.processes[p].standard_oper_time * m.assign_worker_to_process_vars[w, p]
                / self._get_efficiency(w, p) for p in self._get_worker_capable_process(w)))

        # Mean workload constraint
        model.mean_workload_var = pyo.Var(domain=pyo.NonNegativeReals, initialize=0)
        model.mean_workload_cons = pyo.Constraint(
            expr=model.mean_workload_var == sum(model.workload_vars[w] for w in self.workers) / self.worker_num)

        # Max workload constraint
        model.max_workload_var = pyo.Var(domain=pyo.NonNegativeReals, initialize=0)
        model.max_workload_cons = pyo.Constraint(
            self.workers, rule=lambda m, w: m.max_workload_var >= m.workload_vars[w])

        # Workload volatility_rate constraint
        model.workload_volatility_a_cons = pyo.Constraint(
            self.workers,
            rule=lambda m, w:
            m.workload_vars[w] <= m.mean_workload_var * (1 - shrink[0] + self.volatility_rate))
        model.workload_volatility_b_cons = pyo.Constraint(
            self.workers,
            rule=lambda m, w:
            m.workload_vars[w] >= m.mean_workload_var * (1 + shrink[1] - self.volatility_rate))

        # Objective
        # HINT: learn a weighted proxy objective for the target objective, but upload later
        model.objective = pyo.Objective(
            expr=obj_weight[0] * model.max_workload_var + obj_weight[1] * model.mean_workload_var + obj_weight[2] * 0,
            sense=pyo.minimize)
        # HINT: accelerate instance-5.txt and instance-43.txt, with self.max_cycle_count = 2
        # model.objective = pyo.Objective(expr=0, sense=pyo.minimize)

        # Initialize an empty ConstraintList for Combinatorial Benders (CB) cuts
        model.cb_cuts = pyo.ConstraintList()
        # Initialize an empty ConstraintList for Local Branching cuts
        model.local_branching_cuts = pyo.ConstraintList()

        # HINT: heuristics for the master problem -> more likely to be feasible
        # HINT: add a cut based on cycle number estimation
        # TODO: add a subproblem relaxation to the master problem -> more likely to be feasible
        # model.process_to_station = pyo.Set(initialize=product(self.processes, self.ext_stations))
        # model.worker_to_station = pyo.Set(initialize=product(self.workers, self.stations))
        # model.worker_pro_sta = pyo.Set(initialize=product(self.workers, self.processes, self.ext_stations))
        # model.assign_process_to_station_vars = pyo.Var(model.process_to_station, domain=pyo.Binary, initialize=0)
        # model.assign_worker_to_station_vars = pyo.Var(model.worker_to_station, domain=pyo.Binary, initialize=0)
        # model.assign_worker_process_station_vars = pyo.Var(model.worker_pro_sta, domain=pyo.Binary, initialize=0)
        # # Link p-s with w-p-s
        # model.process_to_station_a_cons = pyo.Constraint(
        #     model.process_to_station,
        #     rule=lambda m, p, s:
        #     m.assign_process_to_station_vars[p, s]
        #     == sum(m.assign_worker_process_station_vars[w, p, s] for w in self.workers))
        # # Link p-s with w-p-s, process cannot be done by different workers at same station
        # model.process_to_station_c_cons = pyo.Constraint(
        #     model.process_to_station,
        #     rule=lambda m, p, s:
        #     sum(m.assign_worker_process_station_vars[w, p, s] for w in self.workers) <= 1)
        # # Link w-s with w-p-s
        # model.worker_to_station_cons = pyo.Constraint(
        #     model.worker_to_station,
        #     rule=lambda m, w, s:
        #     m.assign_worker_to_station_vars[w, s]
        #     >= sum(m.assign_worker_process_station_vars[w, p, ss] for p in self.processes
        #            for ss in self.__get_dummy_stations(s))
        #     / self.max_cycle_count / self.process_num)
        # model.worker_to_station_b_cons = pyo.Constraint(
        #     model.worker_to_station,
        #     rule=lambda m, w, s:
        #     m.assign_worker_to_station_vars[w, s]
        #     <= sum(m.assign_worker_process_station_vars[w, p, ss] for p in self.processes
        #            for ss in self.__get_dummy_stations(s)))
        # # Link w-p with w-p-s
        # model.worker_to_process_cons = pyo.Constraint(
        #     model.worker_to_process,
        #     rule=lambda m, w, p:
        #     m.assign_worker_to_process_vars[w, p]
        #     == sum(m.assign_worker_process_station_vars[w, p, ss] for ss in self.ext_stations))
        # # Link w-p with w-p-s, Worker cannot do same process at different station
        # model.worker_to_process_c_cons = pyo.Constraint(
        #     model.worker_to_process,
        #     rule=lambda m, w, p:
        #     sum(m.assign_worker_process_station_vars[w, p, ss] for ss in self.ext_stations) <= 1)
        # # Immediate precedence constraints
        # model.precedence_cons = pyo.ConstraintList()
        # for p1, p2 in self.immediate_precedence:
        #     model.precedence_cons.add(
        #         expr=sum(s * model.assign_process_to_station_vars[(p1, s)] for s in self.ext_stations) <=
        #              sum(s * model.assign_process_to_station_vars[(p2, s)] for s in self.ext_stations))
        # # Each station has no more than one worker
        # model.station_worker_cons = pyo.Constraint(
        #     self.stations,
        #     rule=lambda m, s:
        #     sum(m.assign_worker_to_station_vars[w, s] for w in self.workers) <= 1)
        # # Maximum number of stations for each worker
        # model.worker_max_stations_cons = pyo.Constraint(
        #     self.workers,
        #     rule=lambda m, w:
        #     sum(m.assign_worker_to_station_vars[w, s] for s in self.stations) <= self.max_station_per_worker)

        return model

    def make_cp_sub_problem(self, assign_worker_to_process_vals, split_task=False, cycle_count=None):
        # Extend stations by duplication according to max_cycle_count
        if cycle_count is None:
            cycle_count = self.max_cycle_count
        dummy_stations = dict()
        for s in range(self.station_num + 1, cycle_count * self.station_num + 1):
            original_s = (s - 1) % self.station_num + 1
            dummy_stations.update({s: self.stations[original_s]})
        ext_stations = {**self.stations, **dummy_stations}

        def __get_dummy_stations(s):
            return [a for a in ext_stations if (a - 1) % self.station_num + 1 == s]

        cp = cp_model.CpModel()

        # HINT: do not use ext_stations, use process order instead
        cp_process_to_station = {
            (p, s): cp.NewBoolVar('p_{}_s_{}'.format(p, s))
            for p, s in product(self.processes, ext_stations)}
        cp_aux_process_to_station = {
            (p, s): cp.NewBoolVar('p_{}_ss_{}'.format(p, s))
            for p, s in product(self.processes, self.stations)}
        cp_worker_to_station = {
            (w, s): cp.NewBoolVar('w_{}_s_{}'.format(w, s))
            for w, s in product(self.workers, self.stations)}
        cp_machine_to_station = {
            (m, s): cp.NewBoolVar('m_{}_s_{}'.format(m, s))
            for m, s in product(self.aux_machines, self.stations)}

        """
        Linking constraints
        """
        # Link p-s with p-s_
        for p, s in product(self.processes, self.stations):
            for s_ in __get_dummy_stations(s):
                cp.AddImplication(cp_process_to_station[(p, s_)], cp_aux_process_to_station[(p, s)])

        # Link w-p with w-s
        for w, p, s in product(self.workers, self.processes, self.stations):
            cp.AddImplication(
                assign_worker_to_process_vals[w, p] * cp_aux_process_to_station[(p, s)],
                cp_worker_to_station[(w, s)])

        # Each worker must be assigned to at least one station
        for w in self.workers:
            cp.Add(sum(cp_worker_to_station[(w, s)] for s in self.stations) >= 1)

        """
        工艺规则约束
        """
        # Each process must be assigned to a station
        for p in self.processes:
            cp.Add(sum(cp_process_to_station[(p, s)] for s in ext_stations) == 1)

        # Precedence constraints
        # DEBUG: complicating constraints
        # for p1, p2 in self.immediate_precedence:
        #     cp.Add(sum(s * cp_process_to_station[(p1, s)] for s in ext_stations) <=
        #            sum(s * cp_process_to_station[(p2, s)] for s in ext_stations))
        # Alternative (from https://github.com/google/or-tools/blob/stable/examples/python/line_balancing_sat.py)
        possible = {
            (p, s): cp.NewBoolVar('p_{}_s_{}'.format(p, s))
            for p, s in product(self.processes, ext_stations)}
        for p, s in product(self.processes, list(ext_stations.keys())[:-1]):
            cp.AddImplication(possible[(p, s)], possible[(p, s + 1)])
        for p, s in product(self.processes, ext_stations):
            cp.AddImplication(cp_process_to_station[(p, s)], possible[(p, s)])
        for p, s in product(self.processes, list(ext_stations.keys())[1:]):
            cp.AddImplication(cp_process_to_station[(p, s)], possible[(p, s - 1)].Not())
        for p1, p2 in self.immediate_precedence:
            for s in list(ext_stations.keys())[1:]:
                cp.AddImplication(cp_process_to_station[p1, s], possible[p2, s - 1].Not())

        """
        人员规则约束
        """
        # Maximum number of stations for each worker
        for w in self.workers:
            cp.Add(sum(cp_worker_to_station[(w, s)] for s in self.stations) <= self.max_station_per_worker)

        # Fix station to processes
        for p, process in self.processes.items():
            if process.fixed_station_code:
                s = self.station_code_to_id[process.fixed_station_code]
                cp.Add(sum(cp_process_to_station[(p, s_)] for s_ in __get_dummy_stations(s)) == 1)
                for s_ in set(self.stations.keys()) - {s}:
                    cp.Add(sum(cp_process_to_station[(p, s_)] for s_ in __get_dummy_stations(s_)) == 0)

        """
        设备规则约束
        """
        # Required machines should be prepared at the station
        for p, s in product(self.processes, ext_stations):
            cp.Add(cp_process_to_station[(p, s)] <=
                   cp_machine_to_station[(self.processes_required_machine[p], (s - 1) % self.station_num + 1)])

        # Maximum number of machines in each station
        # Fixed: add "if v.is_machine_needed" after Q&A 0819
        for s in self.stations:
            cp.Add(
                sum(cp_machine_to_station[(k, s)] for k, v in self.aux_machines.items() if v.is_machine_needed)
                <= self.max_machine_per_station)

        # Mono-machine constraint
        for s, k1 in product(self.stations, self.mono_aux_machines):
            for k in set(self.aux_machines.keys()) - {k1}:
                cp.Add(cp_machine_to_station[(k, s)] <= 1 - cp_machine_to_station[(k1, s)])

        # Fixed machine constraint (wrong)
        # for s, k in ((k, v) for k, v in self.stations_fixed_machine.items() if v != None):
        #     cp.Add(cp_machine_to_station[(k, s)] == 1)
        # Fixed machine constraint
        for s, fix_machine in self.stations_fixed_machine.items():
            if fix_machine is not None:
                cp.Add(cp_machine_to_station[(fix_machine, s)] == 1)
                for k in self.fixed_machines - {fix_machine}:
                    cp.Add(cp_machine_to_station[(k, s)] == 0)
            else:
                for k in self.fixed_machines:
                    cp.Add(cp_machine_to_station[(k, s)] == 0)

        """
        工位规则约束
        """
        # DEBUG: complicating constraints
        # Each station has no more than one worker
        for s in self.stations:
            cp.Add(sum(cp_worker_to_station[(w, s)] for w in self.workers) <= 1)

        """
        其他约束 (Revisit constraint)
        """
        # HINT: do not use auxiliary variables
        # Define variables for maximum revisit constraint
        cp_visit = {
            (s, c): cp.NewBoolVar(name=f"visit_{s}_{c}")
            for s, c in product(self.stations, range(cycle_count))}
        cp_revisit = {
            s: cp.NewIntVar(name=f"revisit_{s}", lb=0, ub=cycle_count)
            for s in self.stations}

        # Calculate visit_vars
        for s, c, p in product(self.stations, range(cycle_count), self.processes):
            cp.AddImplication(cp_process_to_station[(p, s + c * self.station_num)], cp_visit[(s, c)])

        # Calculate revisit_vars
        for s in self.stations:
            cp.AddMaxEquality(cp_revisit[s], [sum(cp_visit[(s, c)] for c in range(cycle_count)) - 1, 0])

        # Maximum revisit count (2) for each station without unmovable machine
        for s in self.station_with_no_unmovable_machine:
            cp.Add(cp_revisit[s] <= self._max_revisit_no_unmovable)

        # Maximum total revisit count constraint
        # DEBUG: complicating constraints -> check feasibility, and add cuts
        # HINT: how to add constraints to the master problem, such that this constraint is not likely validated
        cp.Add(sum(cp_revisit[s] for s in self.stations) <= self.max_revisited_station_count)

        """
        Symmetry breaking
        """
        # HINT: if a station do not have fixed machine, it should not be skipped
        # HINT: there are cases that station 100 is used, but stations corresponds to it but before it are not used
        # cp_station_can_be_used = {s: cp.NewBoolVar(name=f"station_can_be_used_{s}") for s in ext_stations}
        # cp_station_can_be_skipped = {s: cp.NewBoolVar(name=f"station_can_be_skipped_{s}") for s in ext_stations}
        # for c, s in product(range(cycle_count), self.stations):
        #     pass

        """
        Objective
        """
        cp.Minimize(0)

        return cp, cp_process_to_station

    def fast_detect_revisit(self, worker_to_process):
        min_required_station_num = 1
        process_to_worker = {p: w for w, p in worker_to_process if worker_to_process[w, p] == 1}
        wp_set = set()
        for p1, p2 in zip(self.task_tp_order[:-1], self.task_tp_order[1:]):
            p1_w = process_to_worker[p1]
            p2_w = process_to_worker[p2]
            if (p1, p2) in self.immediate_precedence and p1_w != p2_w:
                min_required_station_num += 1
                wp_set.add((p1_w, p1))
                wp_set.add((p2_w, p2))
                if min_required_station_num > self.station_num + self.max_revisited_station_count:
                    print("revisit_p_set", wp_set)
                    return False, wp_set, min_required_station_num
        return True, wp_set, min_required_station_num

    def fast_detect_cycle(self, worker_to_process):
        station_used = 1
        process_to_worker = {p: w for w, p in worker_to_process if worker_to_process[w, p] == 1}
        wp_set = set()
        for p1, p2 in zip(self.task_tp_order[:-1], self.task_tp_order[1:]):
            p1_w = process_to_worker[p1]
            p2_w = process_to_worker[p2]
            if p1_w != p2_w:
                station_used += 1
                wp_set.add((p1_w, p1))
                wp_set.add((p2_w, p2))
                if station_used // self.station_num + 1 > self.max_cycle_count:
                    print("cycle_p_set", wp_set)
                    return False, wp_set, station_used // self.station_num + 1
        return True, wp_set, station_used // self.station_num + 1

    def estimate_cycle(self, worker_to_process):
        station_used = 1
        process_to_worker = {p: w for w, p in worker_to_process if worker_to_process[w, p] == 1}
        processes_required_mono_machine = {
            k: v for k, v in self.processes_required_machine.items() if v in self.mono_aux_machines}
        # print("processes_required_mono_machine", processes_required_mono_machine)

        for p1, p2 in self.immediate_precedence:
            p1_w = process_to_worker[p1]
            p2_w = process_to_worker[p2]
            if p1_w != p2_w:
                station_used += 1
            else:
                if p1 in processes_required_mono_machine and p2 in processes_required_mono_machine:
                    if processes_required_mono_machine[p1] != processes_required_mono_machine[p2]:
                        station_used += 1
        cycle = station_used // self.station_num + 1
        print("station |" + f" used={station_used}, have={self.station_num}, cycle={cycle}")
        return cycle

    def __add_cb_cut(self, model, worker_to_process, wp_set):
        worker_to_process_one = {(w, p) for w, p in wp_set if worker_to_process[w, p] == 1}
        worker_to_process_zero = {(w, p) for w, p in wp_set if worker_to_process[w, p] == 0}

        # Add Combinatorial Benders (CB) cuts
        model.cb_cuts.add(
            expr=
            sum(model.assign_worker_to_process_vars[w, p] for w, p in worker_to_process_zero) +
            sum(1 - model.assign_worker_to_process_vars[w, p] for w, p in worker_to_process_one)
            >= 1)

        return model

    def __add_local_branching_cut(self, model, worker_to_process, left_or_right, k):
        worker_to_process_one = {(w, p) for w, p in worker_to_process if worker_to_process[w, p] == 1}
        worker_to_process_zero = {(w, p) for w, p in worker_to_process if worker_to_process[w, p] == 0}

        # Left branch
        if left_or_right == "left":
            model.local_branching_cuts.add(
                expr=
                sum(model.assign_worker_to_process_vars[w, p] for w, p in worker_to_process_zero) +
                sum(1 - model.assign_worker_to_process_vars[w, p] for w, p in worker_to_process_one)
                <= k)

        # Right branch
        if left_or_right == "right":
            model.local_branching_cuts.add(
                expr=
                sum(model.assign_worker_to_process_vars[w, p] for w, p in worker_to_process_zero) +
                sum(1 - model.assign_worker_to_process_vars[w, p] for w, p in worker_to_process_one)
                >= k + 1)

        return model

    def __local_branching(self, model, worker_to_process, k=5):
        # HINT: local branching based on self.max_cycle_count, critical for efficiency
        f_m = lambda w, p, vars: 1 if vars[w, p].value > 0.5 else 0
        f_s = lambda a, b, vars: 1 if solver.Value(vars[a, b]) > 0.5 else 0

        for kk, cycle_count in product([6], [self.max_cycle_count]):
            model = self.__add_local_branching_cut(model, worker_to_process, "right", kk)
            # optimizer = pyo.SolverFactory('gurobi')
            optimizer = pyo.SolverFactory('appsi_highs')
            try:
                results = optimizer.solve(model, tee=False)
            except:
                model.local_branching_cuts.clear()
                return 10, None, None

            if not results.solver.termination_condition == TerminationCondition.infeasible:
                worker_to_process_ = {
                    (w, p): f_m(w, p, model.assign_worker_to_process_vars) for w, p in model.worker_to_process}
                cp_sub_model, cp_process_to_station = self.make_cp_sub_problem(
                    worker_to_process_, cycle_count=cycle_count)
                solver = cp_model.CpSolver()
                solver.parameters.log_search_progress = False
                solver.parameters.max_time_in_seconds = 15
                status = solver.Solve(cp_sub_model)
                if status == cp_model.OPTIMAL:
                    process_to_station_ = {
                        (p, s): f_s(p, s, cp_process_to_station) for p, s in cp_process_to_station}
                    real_obj = self.get_real_objective(model)
                    print(f"🟩 subproblem FEASIBLE, by local branching >>> real_obj = {real_obj}")
                    model.local_branching_cuts.clear()
                    return real_obj, worker_to_process_, process_to_station_
            model.local_branching_cuts.clear()
        model.local_branching_cuts.clear()
        return 10, None, None

    def __add_cp_cut(self, cp_model, process_to_station, cp_process_to_station):
        process_to_station_one = {(p, s) for p, s in process_to_station if process_to_station[p, s] == 1}
        process_to_station_zero = {(p, s) for p, s in process_to_station if process_to_station[p, s] == 0}

        cp_model.Add(
            sum(cp_process_to_station[(p, s)] for p, s in process_to_station_zero) +
            sum(1 - cp_process_to_station[(p, s)] for p, s in process_to_station_one)
            >= 1)

        return cp_model

    def alchemy(self, cp_time_limit=10):
        # larger cp_time_limit
        f_m = lambda w, p, vars: 1 if vars[w, p].value > 0.5 else 0
        f_s = lambda a, b, vars: 1 if solver.Value(vars[a, b]) > 0.5 else 0

        obj_weights = [
            (1, 0, 0),  # max
            (0, 1, 0),  # mean
            # (0, 0, 1),  # 0
            # (0.5, 0.5, 0),  # 0.5 max + 0.5 mean
        ]
        shrinks = [(0, 0), (0.01, 0.01)]

        for obj_weight, shrink in product(obj_weights, shrinks):
            model = self.make_master_problem(obj_weight=obj_weight, shrink=shrink)
            optimizer = pyo.SolverFactory('appsi_highs')
            # optimizer = pyo.SolverFactory('gurobi')
            # optimizer.options["MIPFocus"] = 1
            try:
                results = optimizer.solve(model, tee=False)
            except:
                return 10, None, None, self.max_cycle_count

            if results.solver.termination_condition == TerminationCondition.infeasible:
                print(f"🟥 master problem INFEASIBLE obj_weight={obj_weight}, shrink={shrink}")

            else:
                real_obj = self.get_real_objective(model)
                worker_to_process = {
                    (w, p): f_m(w, p, model.assign_worker_to_process_vars) for w, p in model.worker_to_process}
                process_worker = {p: w for w, p in worker_to_process if worker_to_process[w, p] == 1}
                print("tasks   |" + " | ".join([str(p).rjust(3) for p in self.task_tp_order]))
                print("workers |" + " | ".join([str(process_worker[p]).rjust(3) for p in self.task_tp_order]))
                cp_sub_model, cp_process_to_station = self.make_cp_sub_problem(worker_to_process)
                solver = cp_model.CpSolver()
                solver.parameters.log_search_progress = False
                solver.parameters.max_time_in_seconds = cp_time_limit
                status = solver.Solve(cp_sub_model)
                if status == cp_model.OPTIMAL:
                    process_to_station = {
                        (p, s): f_s(p, s, cp_process_to_station) for p, s in cp_process_to_station}
                    process_station = {
                        p: s for p, s in process_to_station if process_to_station[p, s] == 1}
                    print(f"🟩 subproblem FEASIBLE >>> real_obj={real_obj}, obj_weight={obj_weight}, shrink={shrink}")
                    print("stations|" + " | ".join([str(process_station[p]).rjust(3) for p in self.task_tp_order]))
                    real_obj = self.get_real_objective(model)
                    return real_obj, worker_to_process, process_to_station, self.max_cycle_count

        return 10, None, None, self.max_cycle_count

    def solve(self, cp_time_limit=5, total_time_limit=120):
        start_time = time.time()
        status = cp_model.INFEASIBLE
        iter = 0

        f_m = lambda w, p, vars: 1 if vars[w, p].value > 0.5 else 0
        f_s = lambda a, b, vars: 1 if solver.Value(vars[a, b]) > 0.5 else 0

        # Solve master problem using MIP
        model = self.make_master_problem()
        optimizer = pyo.SolverFactory('appsi_highs')
        # optimizer = pyo.SolverFactory('gurobi')
        # optimizer.options["MIPFocus"] = 1
        try:
            results = optimizer.solve(model, tee=False)
        except:
            return 10, None, None, self.max_cycle_count

        # CASE: master problem is infeasible
        if results.solver.termination_condition == TerminationCondition.infeasible:
            print(f"🟥 master problem INFEASIBLE")
            return 10, None, None, self.max_cycle_count

        # CASE: master problem is feasible, continue to solve subproblem
        worker_to_process = {
            (w, p): f_m(w, p, model.assign_worker_to_process_vars) for w, p in model.worker_to_process}

        while time.time() - start_time <= total_time_limit and not status == cp_model.OPTIMAL:
            iter += 1
            process_worker = {p: w for w, p in worker_to_process if worker_to_process[w, p] == 1}
            print("tasks   |" + " | ".join([str(p).rjust(3) for p in self.task_tp_order]))
            print("workers |" + " | ".join([str(process_worker[p]).rjust(3) for p in self.task_tp_order]))

            # valid_revisit, revisit_wp_set, min_station_num = self.fast_detect_revisit(worker_to_process)
            # valid_cycle, cycle_wp_set, min_cycle_num = self.fast_detect_cycle(worker_to_process)
            valid_revisit, revisit_wp_set = True, set(product(self.workers, self.processes))
            valid_cycle, cycle_wp_set = True, set(product(self.workers, self.processes))

            if valid_revisit and valid_cycle:
                cp_sub_model, cp_process_to_station = self.make_cp_sub_problem(worker_to_process)
                solver = cp_model.CpSolver()
                solver.parameters.log_search_progress = False
                solver.parameters.max_time_in_seconds = cp_time_limit
                status = solver.Solve(cp_sub_model)

                # CASE: subproblem is feasible, global optimal solution found
                if status == cp_model.OPTIMAL:
                    process_to_station = {
                        (p, s): f_s(p, s, cp_process_to_station) for p, s in cp_process_to_station}
                    process_station = {
                        p: s for p, s in process_to_station if process_to_station[p, s] == 1}
                    print(f"{str(iter).rjust(3)}     |" + "---".join(['---' for _ in self.task_tp_order]))
                    print("stations|" + " | ".join([str(process_station[p]).rjust(3) for p in self.task_tp_order]))
                    real_obj = self.get_real_objective(model)
                    print("time    |" + f" {time.time() - start_time:.2f} seconds")
                    print(f"🟩 subproblem FEASIBLE >>> real_obj = {real_obj}")
                    return real_obj, worker_to_process, process_to_station, self.max_cycle_count

                # CASA: subproblem is infeasible, add a cut to the master problem
                else:
                    real_obj_, worker_to_process_, process_to_station_ = self.__local_branching(model,
                                                                                                worker_to_process)
                    if real_obj_ != 10:
                        return real_obj_, worker_to_process_, process_to_station_, self.max_cycle_count
                    model = self.__add_cb_cut(
                        model, worker_to_process, wp_set=list(product(self.workers, self.processes)))
                    results = optimizer.solve(model, tee=False)
                    worker_to_process = {
                        (w, p): f_m(w, p, model.assign_worker_to_process_vars) for w, p in model.worker_to_process}
                    if results.solver.termination_condition == TerminationCondition.infeasible:
                        print(f"🟥 master problem INFEASIBLE")
                        return 10, None, None, self.max_cycle_count

            # CASE: subproblem is infeasible (by fast detecting), add a cut to the master problem
            else:  # not valid_revisit or not valid_cycle
                status = cp_model.INFEASIBLE
                real_obj_, worker_to_process_, process_to_station_ = self.__local_branching(model, worker_to_process)
                if real_obj_ != 10:
                    return real_obj_, worker_to_process_, process_to_station_, self.max_cycle_count
                if not valid_revisit:
                    model = self.__add_cb_cut(model, worker_to_process, wp_set=revisit_wp_set)
                if not valid_cycle:
                    model = self.__add_cb_cut(model, worker_to_process, wp_set=cycle_wp_set)
                results = optimizer.solve(model, tee=False)
                worker_to_process = {
                    (w, p): f_m(w, p, model.assign_worker_to_process_vars) for w, p in model.worker_to_process}
                if results.solver.termination_condition == TerminationCondition.infeasible:
                    print(f"🟥 master problem INFEASIBLE")
                    return 10, None, None, self.max_cycle_count

            print("time    |" + f" {time.time() - start_time:.2f} seconds")
            print(f"{str(iter).rjust(3)}     |" + "---".join(['---' for _ in self.task_tp_order]))

        # CASE: time limit exceeded
        print(f"🟨 NO SOLUTION FOUND")
        return 10, None, None, self.max_cycle_count

    def get_real_objective(self, master_model):
        workloads = [pyo.value(master_model.workload_vars[w]) for w in self.workers]
        mean_workload = sum(workloads) / self.worker_num
        std_dev = (sum([(x - mean_workload) ** 2 for x in workloads]) / self.worker_num) ** 0.5
        score1 = mean_workload / max(workloads)
        score2 = std_dev / mean_workload
        real_objective = (1 - score1) * self.upph_weight + score2 * self.volatility_weight
        return real_objective


if __name__ == '__main__':
    # instance_li = INSTANCES
    instance_li = ["instance-53.txt", "instance-33.txt", "instance-21.txt"]

    start_time = time.time()
    real_objectives = {}
    instance_count = 0
    for instance in instance_li:
        instance_count += 1
        instance_start_time = time.time()

        print(f"Solving {instance} [{instance_count}/{len(instance_li)}] ...")
        S = Solver(load_json(f"instances/{instance}"))

        real_obj, worker_to_process, process_to_station, cycle_num = S.solve(cp_time_limit=15, total_time_limit=330)

        # if S.process_num >= 60:
        #     print("METHOD: alchemy()")
        #     real_obj, worker_to_process, process_to_station, cycle_num = S.alchemy(cp_time_limit=20)
        # else:
        #     print("METHOD: solve()")
        #     real_obj, worker_to_process, process_to_station, cycle_num = S.solve(cp_time_limit=15, total_time_limit=330)
        # if S.process_num >= 60 and real_obj == 10:
        #     print("METHOD: solve() [alchemy() failed]")
        #     real_obj, worker_to_process, process_to_station, cycle_num = S.solve(cp_time_limit=15, total_time_limit=250)

        if real_obj != 10:
            solution = Solution(S.instance_data, worker_to_process, process_to_station, cycle_num)

        real_objectives[instance] = real_obj
        print(f"Runtime         : {time.time() - instance_start_time} seconds")
        print()

    print(f"Real objectives : {list(real_objectives.values())}")
    print(f"Mean objective  : {sum(real_objectives.values()) / len(real_objectives)}")
    print(f"Total Runtime   : {time.time() - start_time} seconds")
