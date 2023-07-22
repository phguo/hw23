# coding:utf-8
# By Penghui Guo (https://guo.ph) for "苏州园区“华为云杯”2023人工智能应用创新大赛（创客）" 2023, all rights reserved.

from pyomo.opt import SolverStatus, TerminationCondition
import pyomo.environ as pyo
from itertools import product

from config import PARAMETERS, INSTANCES
from instance import Instance
from utility import load_json, save_json


class Solver(Instance):
    def __init__(self, instance_data):
        super().__init__(instance_data)

        self.model = self.init_model()

    def __get_dummy_stations(self, s):
        return [a for a in self.ext_stations if (a - 1) % self.station_num + 1 == s]

    def __get_dummy_stations_(self, s):
        return [a for a in self.ext_stations if (a - 1) % self.station_num + 1 == s and a != s]

    def init_model(self):
        model = pyo.ConcreteModel()

        # Extend stations by duplication according to max_cycle_count
        dummy_stations = dict()
        for s in range(self.station_num + 1, self.max_cycle_count * self.station_num + 1):
            original_s = (s - 1) % self.station_num + 1
            dummy_stations.update({s: self.stations[original_s]})
        self.ext_stations = {**self.stations, **dummy_stations}

        # Initialize sets
        model.process_to_station = pyo.Set(initialize=product(self.processes, self.ext_stations))
        model.worker_to_station = pyo.Set(initialize=product(self.workers, self.stations))
        model.worker_to_process = pyo.Set(initialize=product(self.workers, self.processes))
        model.machine_to_station = pyo.Set(initialize=product(self.aux_machines, self.stations))

        # Initialize variables
        model.assign_process_to_station_vars = pyo.Var(model.process_to_station, domain=pyo.Binary, initialize=0)
        model.assign_worker_to_station_vars = pyo.Var(model.worker_to_station, domain=pyo.Binary, initialize=0)
        model.assign_worker_to_process_vars = pyo.Var(model.worker_to_process, domain=pyo.Binary, initialize=0)
        model.assign_machine_to_station_vars = pyo.Var(model.machine_to_station, domain=pyo.Binary, initialize=0)

        """
        Linking constraints
        """
        model.worker_pro_sta = pyo.Set(initialize=product(self.workers, self.processes, self.ext_stations))
        model.assign_worker_process_station_vars = pyo.Var(model.worker_pro_sta, domain=pyo.Binary, initialize=0)

        # Link p-s with w-p-s
        model.process_to_station_a_cons = pyo.Constraint(
            model.process_to_station,
            rule=lambda m, p, s:
            m.assign_process_to_station_vars[p, s]
            == sum(m.assign_worker_process_station_vars[w, p, s] for w in self.workers))
        # Link p-s with w-p-s, process cannot be done by different workers at same station
        model.process_to_station_c_cons = pyo.Constraint(
            model.process_to_station,
            rule=lambda m, p, s:
            sum(m.assign_worker_process_station_vars[w, p, s] for w in self.workers) <= 1)

        # Link w-s with w-p-s
        model.worker_to_station_cons = pyo.Constraint(
            model.worker_to_station,
            rule=lambda m, w, s:
            m.assign_worker_to_station_vars[w, s]
            >= sum(m.assign_worker_process_station_vars[w, p, ss] for p in self.processes
                   for ss in self.__get_dummy_stations(s))
            / self.max_cycle_count / self.process_num)
        model.worker_to_station_b_cons = pyo.Constraint(
            model.worker_to_station,
            rule=lambda m, w, s:
            m.assign_worker_to_station_vars[w, s]
            <= sum(m.assign_worker_process_station_vars[w, p, ss] for p in self.processes
                   for ss in self.__get_dummy_stations(s)))

        # Link w-p with w-p-s
        model.worker_to_process_cons = pyo.Constraint(
            model.worker_to_process,
            rule=lambda m, w, p:
            m.assign_worker_to_process_vars[w, p]
            == sum(m.assign_worker_process_station_vars[w, p, ss] for ss in self.ext_stations))
        # Link w-p with w-p-s, Worker cannot do same process at different station
        model.worker_to_process_c_cons = pyo.Constraint(
            model.worker_to_process,
            rule=lambda m, w, p:
            sum(m.assign_worker_process_station_vars[w, p, ss] for ss in self.ext_stations) <= 1)

        """
        工艺规则约束
        """
        # Each process must be assigned to a station
        # DEBUG: instances 37, 56 are infeasible (will not be feasible until 1 <= * <= max_split_num)
        model.process_must_be_assigned_cons = pyo.Constraint(
            self.processes,
            rule=lambda m, p:
            sum(m.assign_process_to_station_vars[p, s] for s in self.ext_stations) == 1)

        # Precedence constraints
        model.precedence_cons = pyo.Constraint(
            self.immediate_precedence,
            rule=lambda m, p1, p2:
            sum(s * m.assign_process_to_station_vars[p1, s] for s in self.ext_stations)
            <= sum(s * m.assign_process_to_station_vars[p2, s] for s in self.ext_stations))

        # Maximum worker per operation
        model.max_worker_per_operation_cons = pyo.Constraint(
            self.processes,
            rule=lambda m, p:
            sum(m.assign_worker_to_process_vars[w, p] for w in self.workers) <= self.max_worker_per_oper)

        """
        人员规则约束
        """
        # Each worker must be assigned to at least one process
        model.worker_must_have_process_cons = pyo.Constraint(
            self.workers,
            rule=lambda m, w:
            sum(m.assign_worker_to_process_vars[w, p] for p in self.processes) >= 1)

        # Maximum number of stations for each worker
        model.worker_max_stations_cons = pyo.Constraint(
            self.workers,
            rule=lambda m, w:
            sum(m.assign_worker_to_station_vars[w, s] for s in self.stations) <= self.max_station_per_worker)

        # Skill & skill category constraints
        model.worker_skill_capable_cons = pyo.ConstraintList()
        for w, p in product(self.workers, self.processes):
            model.worker_skill_capable_cons.add(
                model.assign_worker_to_process_vars[w, p] <= self.skill_capable[(w, p)] + self.category_capable[(w, p)])
        # Must assign processes that have capable-skill workers to at least one of them
        model.worker_skill_capable_b_cons = pyo.Constraint(
            self.pros_have_capable_skill_workers,
            rule=lambda m, p:
            sum(m.assign_worker_to_process_vars[w, p] for w in
                set(w for w, p in self.skill_capable if self.skill_capable[(w, p)] == 1)) == 1)

        # Fix worker & station for processes
        model.fix_station_cons = pyo.ConstraintList()
        model.fix_worker_cons = pyo.ConstraintList()
        for p, process in self.processes.items():
            if process.fixed_worker_code:
                w = self.worker_code_to_id[process.fixed_worker_code]
                model.fix_worker_cons.add(expr=model.assign_worker_to_process_vars[w, p] == 1)
                # TODO: VI - set others to zero
            if process.fixed_station_code:
                s = self.station_code_to_id[process.fixed_station_code]
                model.fix_station_cons.add(
                    expr=sum(model.assign_process_to_station_vars[p, ss] for ss in self.__get_dummy_stations(s)) == 1)
                # TODO: VI - set others to zero

        """
        设备规则约束
        """
        # Required machines should be prepared at the station
        model.station_min_machines_cons = pyo.Constraint(
            model.process_to_station,
            rule=lambda m, p, s:
            m.assign_process_to_station_vars[p, s]
            <= m.assign_machine_to_station_vars[self.processes_required_machine[p], (s - 1) % self.station_num + 1])

        # Maximum number of machines in each station
        model.station_max_machines_cons = pyo.Constraint(
            self.stations,
            rule=lambda m, s:
            sum(m.assign_machine_to_station_vars[k, s] for k in self.aux_machines) <= self.max_machine_per_station)

        # Mono-machine constraint
        model.mono_machine_cons = pyo.ConstraintList()
        for s, k1 in product(self.stations, self.mono_aux_machines):
            for k in set(self.aux_machines.keys()) - {k1}:
                model.mono_machine_cons.add(
                    expr=model.assign_machine_to_station_vars[k, s]
                         <= 1 - model.assign_machine_to_station_vars[k1, s])

        # Fixed machine constraint
        model.fixed_machine_cons = pyo.Constraint(
            list((k, v) for k, v in self.stations_fixed_machine.items() if v != None),
            rule=lambda m, s, k:
            m.assign_machine_to_station_vars[k, s] == 1)

        """
        工位规则约束
        """
        # TODO: VI - Each station that has been assigned a process must also be assigned at least one worker

        # Each station has no more than one worker
        model.station_worker_cons = pyo.Constraint(
            self.stations,
            rule=lambda m, s:
            sum(m.assign_worker_to_station_vars[w, s] for w in self.workers) <= 1)

        """
        其他约束
        """
        # Define variables for maximum revisit constraint
        model.station_cycle = pyo.Set(initialize=product(self.stations, range(self.max_cycle_count)))
        model.revisit_vars = pyo.Var(
            model.station_cycle, domain=pyo.Binary, initialize=0)
        # Linearize for revisit_vars[s, c] = 1 if assign_process_to_station_vars[*, s] >= 1 else 0
        model.revisit_a_cons = pyo.Constraint(
            model.station_cycle,
            rule=lambda m, s, c:
            m.revisit_vars[s, c] >=
            sum(m.assign_process_to_station_vars[p, s] for p in self.processes) / len(self.processes))
        model.revisit_b_cons = pyo.Constraint(
            model.station_cycle,
            rule=lambda m, s, c:
            m.revisit_vars[s, c] <=
            sum(m.assign_process_to_station_vars[p, s] for p in self.processes))

        # Maximum revisit count (2) for each station without unmovable machine
        model.station_max_revisit_cons = pyo.Constraint(
            self.station_with_no_unmovable_machine,
            rule=lambda m, s:
            sum(m.revisit_vars[s, c] - 1 for c in range(1, self.max_cycle_count)) <= 2)

        # Maximum total revisit count constraint
        model.total_revisit_cons = pyo.Constraint(
            expr=sum(model.revisit_vars[s, c] - 1 for s, c in model.station_cycle) <= self.max_revisited_station_count)

        # Valid inequality (VI) - cannot use cycle c+1 if cycle c is not used
        model.station_cycle_ = pyo.Set(initialize=product(self.stations, range(self.max_cycle_count - 1)))
        model.revisit_vi_cons = pyo.Constraint(
            model.station_cycle_,
            rule=lambda m, s, c:
            m.revisit_vars[s, c] >= m.revisit_vars[s, c + 1])

        # Define variables for workload
        def __get_worker_capable_process(w):
            res = self.workers_capble_pro[w].union(self.workers_category_capable_pro[w])
            return res

        def __get_efficiency(w, p):
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

        model.workload_vars = pyo.Var(self.workers, domain=pyo.NonNegativeReals, initialize=0)
        model.def_workload_cons = pyo.Constraint(
            self.workers,
            rule=lambda m, w:
            m.workload_vars[w] == sum(
                self.processes[p].standard_oper_time / __get_efficiency(w, p) * m.assign_worker_to_process_vars[w, p]
                for p in __get_worker_capable_process(w)))

        # Workload volatility_rate constraint
        model.workload_volatility_a_cons = pyo.Constraint(
            self.workers,
            rule=lambda m, w:
            m.workload_vars[w] <=
            sum(m.workload_vars[ww] for ww in self.workers) / self.worker_num
            * (1 + self.volatility_rate))
        model.workload_volatility_b_cons = pyo.Constraint(
            self.workers,
            rule=lambda m, w:
            m.workload_vars[w] >=
            sum(m.workload_vars[ww] for ww in self.workers) / self.worker_num
            * (1 - self.volatility_rate))

        """
        Objective
        """
        # # Proxy objective 1: minimize the maximum workload
        # model.max_workload_var = pyo.Var(domain=pyo.NonNegativeReals, initialize=0)
        # model.max_workload_cons = pyo.Constraint(
        #     self.workers, rule=lambda m, w: m.max_workload_var >= m.workload_vars[w])
        # model.objective = pyo.Objective(expr=model.max_workload_var, sense=pyo.minimize)

        model.objective = pyo.Objective(expr=0, sense=pyo.minimize)

        return model

    def solve(self):
        # > conda install gcg papilo scip soplex zimpl
        # optimizer = pyo.SolverFactory('scip', executable="~/miniconda3/envs/py311/bin/scip")
        # optimizer = pyo.SolverFactory('appsi_highs')
        optimizer = pyo.SolverFactory('gurobi')
        results = optimizer.solve(
            self.model,
            tee=False,
            # validate=False
            # timelimit=5,
            # logfile="solver.log",
        )
        self.get_result()

        if results.solver.termination_condition == TerminationCondition.infeasible:
            print(">< Infeasible")
            # self.model.write("model.lp", io_options={"symbolic_solver_labels": True})
            # from gurobipy import read as g_read
            # g_model = g_read("model.lp")
            # g_model.computeIIS()
            # g_model.write("model.ilp")

    def get_result(self):
        result = {"dispatch_results": []}

        for s, station in self.stations.items():
            station_result = dict()
            station_result["station_code"] = station.station_code
            station_result["worker_code"] = ''

            for w, worker in self.workers.items():
                # Use > 0.5 instead of == 1 to avoid floating point error
                if pyo.value(self.model.assign_worker_to_station_vars[w, s]) > 0.5:
                    station_result["worker_code"] = worker.worker_code
                    break

            station_result["operation_list"] = []
            for ss in self.__get_dummy_stations(s):
                for p, process in self.processes.items():
                    # Use > 0.5 instead of == 1 to avoid floating point error
                    if pyo.value(self.model.assign_process_to_station_vars[p, ss]) > 0.5:
                        station_result["operation_list"].append({
                            "operation": process.operation,
                            "operation_number": process.operation_number})

            result["dispatch_results"].append(station_result)

        self.result = result

        # Stringify
        # self.result = json.dumps(result)

        return result


if __name__ == '__main__':
    # instance_li = ["instance-37.txt", "instance-56.txt"]
    instance_li = INSTANCES

    for i, instance in enumerate(instance_li):
        S = Solver(load_json(f"instances/{instance}"))
        print(f"Solving ({i + 1}/{len(instance_li)}): {instance}, {S}")
        S.solve()
        save_json(S.result, f"solutions/{instance}_result.txt")
