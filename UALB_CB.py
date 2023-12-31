# coding:utf-8
# By Penghui Guo (https://guo.ph) for "苏州园区“华为云杯”2023人工智能应用创新大赛（创客）" 2023. All rights reserved.

import time
from copy import deepcopy
from itertools import product

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
from ortools.sat.python import cp_model

from instance import Instance
from config import INSTANCES, PARAMETERS
from solution import Solution
from utility import load_json, save_json
from UALB_CB2 import Solver as Solver2

optimizer = pyo.SolverFactory('appsi_highs')
optimizer.config.load_solution = False


class Solver(Instance):
    def __init__(self, instance_data):
        super().__init__(instance_data)

        self.no_need_split = None
        self.solved = False

    def make_master_problem(self, split_task, obj_weight=(1, 0, 0), shrink=(0, 0)):
        model = pyo.ConcreteModel()

        model.worker_to_process = pyo.Set(initialize=product(self.workers, self.processes))
        model.assign_worker_to_process_vars = pyo.Var(
            model.worker_to_process, domain=pyo.Binary, initialize=0)

        if not split_task:
            # Each process must be assigned to exactly one worker
            model.assign_worker_to_process_cons = pyo.Constraint(
                self.processes,
                rule=lambda m, p:
                sum(m.assign_worker_to_process_vars[w, p] for w in self.workers) == 1)
        else:
            # Each process must be assigned to at least one worker
            model.assign_worker_to_process_cons = pyo.Constraint(
                self.processes,
                rule=lambda m, p:
                sum(m.assign_worker_to_process_vars[w, p] for w in self.workers) >= 1)

        if split_task:
            model.process_split_vars = pyo.Var(self.processes, domain=pyo.Binary, initialize=0)
            # if a task [t] has two worker, then model.process_split_vars[t] = 1
            model.process_split_cons = pyo.Constraint(
                self.processes,
                rule=lambda m, p:
                m.process_split_vars[p] >=
                (sum(m.assign_worker_to_process_vars[w, p] for w in self.workers) - 1)
                / (min(self.max_worker_per_oper, self.max_station_per_oper) - 1))
            # HINT: stronger than required
            # MAX_SPLIT_TASKS = self.max_split_num  # 3 is sufficient for at least feasible solutions
            MAX_SPLIT_TASKS = self.max_split_num if PARAMETERS["MAX_SPLIT_TASK_NUM"] is None else PARAMETERS[
                "MAX_SPLIT_TASK_NUM"]
            model.max_split_process_cons = pyo.Constraint(
                expr=sum(model.process_split_vars[p] for p in self.processes) <= MAX_SPLIT_TASKS)

        if not split_task:
            model.max_worker_per_operation_cons = pyo.Constraint(
                self.processes,
                rule=lambda m, p:
                sum(m.assign_worker_to_process_vars[w, p] for w in self.workers) <= self.max_worker_per_oper)
        # Maximum worker per operation
        if split_task:
            # HINT: stronger than required
            model.max_worker_per_operation_cons = pyo.Constraint(
                self.processes,
                rule=lambda m, p:
                sum(m.assign_worker_to_process_vars[w, p] for w in self.workers) <=
                min(self.max_worker_per_oper, self.max_station_per_oper))

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
        if not split_task:
            for p in self.pros_have_capable_skill_workers:
                model.worker_skill_capable_b_cons.add(
                    expr=sum(
                        model.assign_worker_to_process_vars[w, p] for w in self.pros_skill_capable_workers[p]) == 1)

                for w in self.workers:
                    if w not in self.pros_skill_capable_workers[p]:
                        model.worker_skill_capable_b_cons.add(expr=model.assign_worker_to_process_vars[w, p] == 0)
        else:
            for p in self.pros_have_capable_skill_workers:
                model.worker_skill_capable_b_cons.add(
                    expr=sum(
                        model.assign_worker_to_process_vars[w, p] for w in self.pros_skill_capable_workers[p]) >= 1)

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

        model.workload_vars = pyo.Var(self.workers, domain=pyo.NonNegativeReals, initialize=0)
        if not split_task:
            # Define workload without splitting process
            model.def_workload_cons = pyo.Constraint(
                self.workers,
                rule=lambda m, w:
                m.workload_vars[w] == sum(
                    self.processes[p].standard_oper_time / self._get_efficiency(w, p)
                    * m.assign_worker_to_process_vars[w, p] for p in self._get_worker_capable_process(w)))
        else:
            # Define workload with splitting process
            model.aux_workload_vars = pyo.Var(
                list(product(self.workers, self.processes)), domain=pyo.NonNegativeReals, initialize=0)
            model.aux_aux_workload_vars = pyo.Var(
                list(product(self.workers, self.processes, self.workers)), domain=pyo.NonNegativeReals, initialize=0)

            model.def_workload_a_cons = pyo.Constraint(
                self.workers,
                rule=lambda m, w:
                m.workload_vars[w] == sum(
                    self.processes[p].standard_oper_time / self._get_efficiency(w, p)
                    * m.aux_workload_vars[w, p] for p in self._get_worker_capable_process(w)))

            model.def_workload_b1_cons = pyo.Constraint(
                list((w, p, ww) for w in self.workers for p in self._get_worker_capable_process(w) for ww in
                     self.workers),
                rule=lambda m, w, p, ww:
                m.aux_aux_workload_vars[w, p, ww] - m.aux_workload_vars[w, p] >=
                - (1 - m.assign_worker_to_process_vars[ww, p]))
            model.def_workload_b2_cons = pyo.Constraint(
                list((w, p, ww) for w in self.workers for p in self._get_worker_capable_process(w) for ww in
                     self.workers),
                rule=lambda m, w, p, ww:
                m.aux_aux_workload_vars[w, p, ww] - m.aux_workload_vars[w, p] <=
                (1 - m.assign_worker_to_process_vars[ww, p]))

            model.def_workload_c1_cons = pyo.Constraint(
                list((w, p, ww) for w in self.workers for p in self._get_worker_capable_process(w) for ww in
                     self.workers),
                rule=lambda m, w, p, ww:
                m.aux_aux_workload_vars[w, p, ww] >= - m.assign_worker_to_process_vars[ww, p])
            model.def_workload_c2_cons = pyo.Constraint(
                list((w, p, ww) for w in self.workers for p in self._get_worker_capable_process(w) for ww in
                     self.workers),
                rule=lambda m, w, p, ww:
                m.aux_aux_workload_vars[w, p, ww] <= m.assign_worker_to_process_vars[ww, p])

            model.def_workload_d_cons = pyo.Constraint(
                list((w, p) for w in self.workers for p in self._get_worker_capable_process(w)),
                rule=lambda m, w, p:
                m.assign_worker_to_process_vars[w, p] ==
                sum(m.aux_aux_workload_vars[w, p, ww] for ww in self.workers))

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
            m.workload_vars[w] <= m.mean_workload_var * (1 + self.volatility_rate - shrink[0]))
        model.workload_volatility_b_cons = pyo.Constraint(
            self.workers,
            rule=lambda m, w:
            m.workload_vars[w] >= m.mean_workload_var * (1 - self.volatility_rate + shrink[1]))

        # Objective: weighted proxy objective
        model.objective = pyo.Objective(
            expr=obj_weight[0] * model.max_workload_var + obj_weight[1] * model.mean_workload_var + obj_weight[2] * 0,
            sense=pyo.minimize)

        # Objective: feasibility checking
        # model.objective = pyo.Objective(expr=0, sense=pyo.minimize)

        # Objective: mean workload
        # model.objective = pyo.Objective(expr=model.max_workload_var, sense=pyo.minimize)

        # Objective: Gini deviation -> slower
        # model.abs_obj_vars = pyo.Var(
        #     list(product(self.workers, self.workers)), domain=pyo.NonNegativeReals, initialize=0)
        # model.abs_obj_cons = pyo.ConstraintList()
        # for w1, w2 in product(self.workers, self.workers):
        #     model.abs_obj_cons.add(
        #         expr=model.abs_obj_vars[w1, w2] >= model.workload_vars[w1] - model.workload_vars[w2])
        #     model.abs_obj_cons.add(
        #         expr=model.abs_obj_vars[w1, w2] >= model.workload_vars[w2] - model.workload_vars[w1])
        #     model.abs_obj_cons.add(
        #         expr=model.abs_obj_vars[w1, w2] >= 0)
        # model.objective = pyo.Objective(
        #     expr=sum(model.abs_obj_vars[w1, w2] for w1, w2 in product(self.workers, self.workers)),
        #     sense=pyo.minimize)

        # Initialize an empty ConstraintList for Combinatorial Benders (CB) cuts
        model.cb_cuts = pyo.ConstraintList()
        # Initialize an empty ConstraintList for Local Branching cuts
        model.local_branching_cuts = pyo.ConstraintList()

        return model

    def make_cp_sub_problem(self, assign_worker_to_process_vals, split_task, cycle_count=None):
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

        if split_task:
            (processes, immediate_precedence, assign_worker_to_process, processes_required_machine,
             process_map) = self._make_dummy_process(assign_worker_to_process_vals)
            if process_map == {}:
                processes, immediate_precedence, assign_worker_to_process, processes_required_machine, process_map = (
                    self.processes, self.immediate_precedence, assign_worker_to_process_vals,
                    self.processes_required_machine, None)
        else:
            processes, immediate_precedence, assign_worker_to_process, processes_required_machine, process_map = (
                self.processes, self.immediate_precedence, assign_worker_to_process_vals,
                self.processes_required_machine, None)

        cp = cp_model.CpModel()

        cp_process_to_station = {
            (p, s): cp.NewBoolVar('p_{}_s_{}'.format(p, s))
            for p, s in product(processes, ext_stations)}
        cp_aux_process_to_station = {
            (p, s): cp.NewBoolVar('p_{}_ss_{}'.format(p, s))
            for p, s in product(processes, self.stations)}
        cp_worker_to_station = {
            (w, s): cp.NewBoolVar('w_{}_s_{}'.format(w, s))
            for w, s in product(self.workers, self.stations)}
        cp_machine_to_station = {
            (m, s): cp.NewBoolVar('m_{}_s_{}'.format(m, s))
            for m, s in product(self.aux_machines, self.stations)}

        # HINT: stronger than required
        if split_task:
            for p, s in product(process_map.keys(), ext_stations):
                for p_ in set(processes.keys()) - {p}:
                    for c in range(self.max_cycle_count):
                        cp.AddImplication(
                            cp_process_to_station[(p, s)],
                            cp_process_to_station[(p_, ((s - 1) % self.station_num + 1) + c * self.station_num)].Not())

        """
        Linking
        """
        # Link p-s with p-s_
        for p, s in product(processes, self.stations):
            for s_ in __get_dummy_stations(s):
                cp.AddImplication(cp_process_to_station[(p, s_)], cp_aux_process_to_station[(p, s)])

        # Link w-p with w-s
        for w, p, s in product(self.workers, processes, self.stations):
            cp.AddImplication(
                assign_worker_to_process[w, p] * cp_aux_process_to_station[(p, s)],
                cp_worker_to_station[(w, s)])

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

        # Fix station to processes
        for p, process in processes.items():
            if process.fixed_station_code:
                s = self.station_code_to_id[process.fixed_station_code]
                cp.Add(sum(cp_process_to_station[(p, s_)] for s_ in __get_dummy_stations(s)) == 1)
                for s_ in set(self.stations.keys()) - {s}:
                    for s__ in __get_dummy_stations(s_):
                        cp.Add(cp_process_to_station[(p, s__)] == 0)

        """
        Worker
        """
        # Each worker must be assigned to at least one station
        for w in self.workers:
            cp.Add(sum(cp_worker_to_station[(w, s)] for s in self.stations) >= 1)

        # Maximum number of stations for each worker
        for w in self.workers:
            cp.Add(sum(cp_worker_to_station[(w, s)] for s in self.stations) <= self.max_station_per_worker)

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
                # cp.AddImplication(cp_machine_to_station[(k1, s)], cp_machine_to_station[(k, s)].Not())

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
        # Each station has no more than one worker
        for s in self.stations:
            cp.Add(sum(cp_worker_to_station[(w, s)] for w in self.workers) <= 1)

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

        return cp, cp_process_to_station, cp_worker_to_station, process_map

    def calc_station_lb(self, worker_to_process, reverse=False):
        process_workers = {p: [] for p in self.processes}
        for w, p in worker_to_process:
            if worker_to_process[w, p] == 1:
                process_workers[p].append(w)

        required_station_num = 0
        worker_set_before = set()
        p_set = set()
        minimal_p_set = set(k for k in self.processes.keys())
        minimal_p_set_found = False
        li = self.task_tp_order_set if not reverse else reversed(self.task_tp_order_set)
        for i, pros in enumerate(li):
            p_set |= set(pros)
            worker_set = list(set(w for p in pros for w in process_workers[p]))
            for j, w in enumerate(worker_set):
                if w not in worker_set_before and w not in worker_set[:j]:
                    required_station_num += 1
                    feasible = required_station_num <= self.station_num + self.max_revisited_station_count
                    if not feasible and not minimal_p_set_found:
                        minimal_p_set = deepcopy(p_set)
                        minimal_p_set_found = True
            worker_set_before = worker_set

        feasible_possibility = required_station_num <= self.station_num + self.max_revisited_station_count
        # print(
        #     "required_station_num:", required_station_num,
        #     "// station_num:", self.station_num,
        #     "// allowed revisit:", self.max_revisited_station_count,
        #     "// minimal_p_set:", minimal_p_set,
        #     "// feasible possibility:", feasible_possibility
        # )
        return required_station_num, minimal_p_set, feasible_possibility

    def __add_cb_cut(self, model, worker_to_process, wp_set):
        worker_to_process_zero = {(w, p) for w, p in wp_set if worker_to_process[w, p] == 0}
        worker_to_process_one = {(w, p) for w, p in wp_set if worker_to_process[w, p] == 1}

        # Add Combinatorial Benders (CB) cuts
        model.cb_cuts.add(
            expr=
            sum(model.assign_worker_to_process_vars[w, p] for w, p in worker_to_process_zero)
            + sum(1 - model.assign_worker_to_process_vars[w, p] for w, p in worker_to_process_one)
            >= 1)

        return model

    def __add_local_branching_cut(self, model, worker_to_process, left_or_right, k):
        worker_to_process_one = {(w, p) for w, p in worker_to_process if worker_to_process[w, p] == 1}
        worker_to_process_zero = {(w, p) for w, p in worker_to_process if worker_to_process[w, p] == 0}

        if left_or_right == "left":
            model.local_branching_cuts.add(
                expr=
                sum(model.assign_worker_to_process_vars[w, p] for w, p in worker_to_process_zero) +
                sum(1 - model.assign_worker_to_process_vars[w, p] for w, p in worker_to_process_one)
                <= k)

        if left_or_right == "right":
            model.local_branching_cuts.add(
                expr=
                sum(model.assign_worker_to_process_vars[w, p] for w, p in worker_to_process_zero) +
                sum(1 - model.assign_worker_to_process_vars[w, p] for w, p in worker_to_process_one)
                >= k + 1)

        return model

    def __local_branching(self, model, worker_to_process, split_task):
        f_s = lambda a, b, vars: 1 if solver.Value(vars[a, b]) > 0.5 else 0

        perm_worker_to_process = deepcopy(worker_to_process)

        for k, side in product(PARAMETERS["LOCAL_BRANCHING_K_SET"], ["right"]):
            model = self.__add_local_branching_cut(model, perm_worker_to_process, side, k)
            model, local_worker_to_process = self.solve_master_problem(model)

            if local_worker_to_process is not None:
                _, minimal_p_set, can_be_feasible = self.calc_station_lb(local_worker_to_process)
                _, minimal_p_set_r, can_be_feasible_r = self.calc_station_lb(local_worker_to_process, reverse=True)
                can_be_feasible = can_be_feasible or can_be_feasible_r

                if can_be_feasible:
                    cp_sub_model, cp_process_to_station, cp_worker_to_station, process_map = self.make_cp_sub_problem(
                        local_worker_to_process, split_task=split_task)
                    solver = cp_model.CpSolver()
                    solver.parameters.log_search_progress = False
                    solver.parameters.max_time_in_seconds = PARAMETERS["CP_TIME_LIMIT"]
                    sub_status = solver.Solve(cp_sub_model)

                    if sub_status == cp_model.OPTIMAL:
                        process_to_station = {
                            (p, s): f_s(p, s, cp_process_to_station) for p, s in cp_process_to_station}
                        worker_to_station = {
                            (w, s): f_s(w, s, cp_worker_to_station) for w, s in cp_worker_to_station}
                        # self.print_opt(model, local_worker_to_process, process_to_station)
                        real_obj = self.get_real_objective(model)
                        model.local_branching_cuts.clear()
                        return real_obj, local_worker_to_process, process_to_station, worker_to_station, process_map
                    else:
                        model.local_branching_cuts.clear()
                        model = self.__add_cb_cut(model, local_worker_to_process, product(self.workers, self.processes))
                else:
                    model.local_branching_cuts.clear()
                    model = self.__add_cb_cut(model, local_worker_to_process, product(self.workers, minimal_p_set))
                    model = self.__add_cb_cut(model, local_worker_to_process, product(self.workers, minimal_p_set_r))
            else:
                model.local_branching_cuts.clear()
                break

        model.local_branching_cuts.clear()
        return 10, None, None, None, None

    def solve_master_problem(self, model):
        f_m = lambda w, p, vars: 1 if vars[w, p].value > 0.5 else 0

        worker_to_process = None
        results = optimizer.solve(model, tee=False, load_solutions=False)

        if results.solver.termination_condition == TerminationCondition.infeasible:
            print(f"🟥 master problem is INFEASIBLE")
            pass

        if results.solver.termination_condition == TerminationCondition.optimal:
            model.solutions.load_from(results)
            worker_to_process = {
                (w, p): f_m(w, p, model.assign_worker_to_process_vars) for w, p in model.worker_to_process}

        return model, worker_to_process

    def solve(self, split_task, cp_time_limit, total_time_limit, obj_weight):
        f_s = lambda a, b, vars: 1 if solver.Value(vars[a, b]) > 0.5 else 0

        start_time = time.time()
        sub_status = cp_model.INFEASIBLE
        iter = 0
        self.obj_changed = False

        """
        Solve the Master Problem (MP)
        """
        model = self.make_master_problem(split_task=split_task, obj_weight=obj_weight)
        _, worker_to_process = self.solve_master_problem(model)
        if worker_to_process is None:
            return 10, None, None, None, self.max_cycle_count, None
        else:
            if split_task == False:
                self.no_need_split = True

        while time.time() - start_time <= total_time_limit and not sub_status == cp_model.OPTIMAL:

            if (iter > PARAMETERS["CHANGE_OBJ_ITER"]
                or time.time() - start_time > PARAMETERS["CHANGE_OBJ_TIME"]) and not self.obj_changed:
                model.del_component(model.objective)
                model.objective = pyo.Objective(expr=model.max_workload_var, sense=pyo.minimize)
                self.obj_changed = True

            iter += 1
            """
            Print (worker, process) assignment
            """
            process_worker = {p: w for w, p in worker_to_process if worker_to_process[w, p] == 1}
            task_line = "tasks   |"
            worker_line = "workers |"
            for pros in self.task_tp_order_set:
                task_line += " ".join([str(p).rjust(3) for p in pros]) + " | "
                worker_line += " ".join([str(process_worker[p]).rjust(3) for p in pros]) + " | "
            print(task_line[:-3])
            print(worker_line[:-3])

            """
            Fast check MP feasibility
            """
            _, minimal_p_set, can_be_feasible = self.calc_station_lb(worker_to_process)
            _, minimal_p_set_r, can_be_feasible_r = self.calc_station_lb(worker_to_process, reverse=True)
            can_be_feasible = can_be_feasible or can_be_feasible_r

            if can_be_feasible:
                cp_sub_model, cp_process_to_station, cp_worker_to_station, process_map = self.make_cp_sub_problem(
                    worker_to_process, split_task=split_task)
                solver = cp_model.CpSolver()
                solver.parameters.log_search_progress = False
                solver.parameters.max_time_in_seconds = cp_time_limit
                sub_status = solver.Solve(cp_sub_model)

                """
                Global optimum reached
                """
                if sub_status == cp_model.OPTIMAL:
                    process_to_station = {(p, s): f_s(p, s, cp_process_to_station) for p, s in cp_process_to_station}
                    worker_to_station = {(w, s): f_s(w, s, cp_worker_to_station) for w, s in cp_worker_to_station}
                    self.solved = True
                    self.print_opt(model, worker_to_process, process_to_station)
                    real_obj = self.get_real_objective(model)
                    return real_obj, worker_to_process, process_to_station, worker_to_station, self.max_cycle_count, process_map

                else:
                    """
                    Sub Problem (SP) is infeasible -> Local Branching
                    """
                    (real_obj_, worker_to_process_, process_to_station_, worker_to_station_, process_map_
                     ) = self.__local_branching(model, worker_to_process, split_task=split_task)
                    if real_obj_ != 10:
                        print(f"🟦 LOCAL BRANCHING finds a FEASIBLE solution")
                        self.solved = True
                        self.print_opt(model, worker_to_process_, process_to_station_)
                        return real_obj_, worker_to_process_, process_to_station_, worker_to_station_, self.max_cycle_count, process_map_

                    """
                    Local Branching does not find feasible solution -> add CB cut
                    """
                    model = self.__add_cb_cut(model, worker_to_process, wp_set=product(self.workers, self.processes))
                    _, worker_to_process = self.solve_master_problem(model)
                    if worker_to_process is None:
                        return 10, None, None, None, self.max_cycle_count, None

            else:
                """
                Sub Problem (SP) is infeasible -> Local Branching
                """
                (real_obj_, worker_to_process_, process_to_station_, worker_to_station_, process_map_
                 ) = self.__local_branching(model, worker_to_process, split_task=split_task)
                if real_obj_ != 10:
                    print(f"🟦 LOCAL BRANCHING finds a FEASIBLE solution")
                    self.solved = True
                    self.print_opt(model, worker_to_process_, process_to_station_)
                    return real_obj_, worker_to_process_, process_to_station_, worker_to_station_, self.max_cycle_count, process_map_

                """
                Sub Problem (SP) is infeasible, add CB cut
                """
                sub_status = cp_model.INFEASIBLE
                model = self.__add_cb_cut(model, worker_to_process, wp_set=product(self.workers, minimal_p_set))
                model = self.__add_cb_cut(model, worker_to_process, wp_set=product(self.workers, minimal_p_set_r))
                _, worker_to_process = self.solve_master_problem(model)
                if worker_to_process is None:
                    return 10, None, None, None, self.max_cycle_count, None

            print(f"{str(iter).rjust(5)}   |" + f" {time.time() - start_time:.2f} seconds")

        # CASE: time limit exceeded
        print(f"🟨 NO SOLUTION FOUND within {total_time_limit} seconds")
        return 10, None, None, None, self.max_cycle_count, None

    def print_opt(self, model, worker_to_process, process_to_station):
        # DEBUG: task can have multiple stations and workers

        process_worker = {p: w for w, p in worker_to_process if worker_to_process[w, p] == 1}
        process_station = {p: s for p, s in process_to_station if process_to_station[p, s] == 1}
        print("OPTIMAL | " + '* * * ' * 20)
        task_line = "tasks   |"
        worker_line = "workers |"
        station_line = "stations|"
        for pros in self.task_tp_order_set:
            task_line += " ".join([str(p).rjust(3) for p in pros]) + " | "
            worker_line += " ".join([str(process_worker[p]).rjust(3) for p in pros]) + " | "
            station_line += " ".join([str(process_station[p]).rjust(3) for p in pros]) + " | "
        print(f"{task_line[:-3]}\n{worker_line[:-3]}\n{station_line[:-3]}")
        real_obj = self.get_real_objective(model)
        proxy_obj = pyo.value(model.objective)
        print(f"🟩 OPTIMAL >>> real_obj = {real_obj}, proxy_obj = {proxy_obj}")

    def get_real_objective(self, master_model):
        workloads = [pyo.value(master_model.workload_vars[w]) for w in self.workers]
        mean_workload = sum(workloads) / self.worker_num
        std_dev = (sum([(x - mean_workload) ** 2 for x in workloads]) / self.worker_num) ** 0.5
        score1 = mean_workload / max(workloads)
        score2 = std_dev / mean_workload
        real_objective = (1 - score1) * self.upph_weight + score2 * self.volatility_weight
        return real_objective

    def run(self):
        output_json = None
        real_obj = 10

        for split_task in [False, True]:
            if self.no_need_split and split_task:
                break

            real_obj, worker_to_process, process_to_station, worker_to_station, cycle_num, process_map = self.solve(
                split_task=split_task, obj_weight=PARAMETERS["OBJ_WEIGHT"],
                cp_time_limit=PARAMETERS["CP_TIME_LIMIT"], total_time_limit=PARAMETERS["UALB_CB_TIME_LIMIT"])

            if not self.solved and self.no_need_split:
                SS = Solver2(self.instance_data)
                real_obj, worker_to_process, process_to_station, worker_to_station, cycle_num, process_map = SS.solve(
                    cp_time_limit=PARAMETERS["CP_TIME_LIMIT"], total_time_limit=PARAMETERS["UALB_CB2_TIME_LIMIT"])
                self.solved = SS.solved

            if self.solved:
                solution = Solution(
                    self.instance_data,
                    worker_to_process, process_to_station, worker_to_station, cycle_num, split_task=split_task)
                output_json = solution.write_solution()
                break

        return output_json, real_obj


if __name__ == '__main__':
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
            # from rich import print as pprint
            # pprint(output_json)
        except Exception as e:
            print(e)
            # raise e

        real_objectives[instance] = real_obj
        print(f"Ins. Runtime    : {time.time() - instance_start_time} seconds")
        print()

    print(f"Real objectives : {list(real_objectives.values())}")
    print(f"Mean objective  : {sum(real_objectives.values()) / len(real_objectives)}")
    print(f"Total Runtime   : {time.time() - start_time} seconds")
