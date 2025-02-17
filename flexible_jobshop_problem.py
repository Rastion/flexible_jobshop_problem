from qubots.base_problem import BaseProblem
import random
import os

# Constant used to represent an incompatible machine.
INFINITE = 1000000

class FlexibleJobShopProblem(BaseProblem):
    """
    Flexible Job Shop Scheduling Problem for Qubots.
    
    A set of jobs, each consisting of a sequence of operations, must be scheduled on a set of machines.
    Each operation can be performed on one of several compatible machines (each with its own processing time).
    Operations within a job must be processed sequentially, and each machine can process only one operation at a time.
    The objective is to minimize the makespan, i.e., the time when all jobs are completed.
    
    **Solution Representation:**
      A dictionary mapping each job index (0-based) to a list of operations.
      Each operation is a dictionary with keys:
         - "machine": the chosen machine (0-based)
         - "start": the start time of the operation
         - "end": the finish time (which should equal start + processing time on the chosen machine)
    """
    
    def __init__(self, instance_file: str):
        (self.nb_jobs, self.nb_machines, self.nb_tasks, self.task_processing_time_data,
         self.job_operation_task, self.nb_operations, self.max_start) = self._read_instance(instance_file)
    
    def _read_instance(self, filename: str):
        """
        Reads an instance file with the following format:
        
          - First line: two (or three) integers:
              * number of jobs
              * number of machines
              * (an extra integer that can be ignored)
          - For each job (next nb_jobs lines):
              * The first integer is the number of operations in that job.
              * Then, for each operation:
                    - An integer indicating the number of compatible machines.
                    - For each compatible machine: a pair of integers (machine id, processing time).
                      (Machine ids are given as 1-indexed.)
        """

        # Resolve relative path with respect to this module’s directory.
        if not os.path.isabs(filename):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(base_dir, filename)

        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Clean up lines.
        lines = [line.strip() for line in lines if line.strip()]
        
        # First line: number of jobs, number of machines, and an extra value (ignored).
        parts = lines[0].split()
        nb_jobs = int(parts[0])
        nb_machines = int(parts[1])
        
        # Read the number of operations per job.
        nb_operations = []
        for j in range(nb_jobs):
            line = lines[j + 1].split()
            nb_operations.append(int(line[0]))
        
        nb_tasks = sum(nb_operations)
        
        # Initialize the processing time data for each task (operation).
        # For each task, create a list of length nb_machines filled with INFINITE.
        task_processing_time_data = [[INFINITE for _ in range(nb_machines)] for _ in range(nb_tasks)]
        
        # For each job, for each operation, record the corresponding task id and fill in the processing times.
        job_operation_task = []
        task_id = 0
        for j in range(nb_jobs):
            job_line = lines[j + 1].split()
            ops = []
            tmp = 1  # Start after the first integer (which is the number of operations).
            for o in range(nb_operations[j]):
                nb_machines_op = int(job_line[tmp])
                tmp += 1
                for i in range(nb_machines_op):
                    # Machine id is provided as 1-indexed, so subtract 1.
                    machine = int(job_line[tmp + 2 * i]) - 1
                    time = int(job_line[tmp + 2 * i + 1])
                    task_processing_time_data[task_id][machine] = time
                ops.append(task_id)
                task_id += 1
                tmp += 2 * nb_machines_op
            job_operation_task.append(ops)
        
        # Compute a trivial upper bound for start times: sum of the maximum processing times for each task.
        max_start = 0
        for i in range(nb_tasks):
            valid_times = [t for t in task_processing_time_data[i] if t != INFINITE]
            max_time = max(valid_times, default=0)
            max_start += max_time
        
        return nb_jobs, nb_machines, nb_tasks, task_processing_time_data, job_operation_task, nb_operations, max_start
    
    def evaluate_solution(self, solution) -> float:
        """
        Evaluates a candidate solution.
        
        Expects:
          solution: a dictionary mapping each job (0-indexed) to a list of operations.
          Each operation is a dictionary with keys 'machine', 'start', and 'end'.
        
        Returns:
          - The makespan (maximum completion time over all jobs) if the solution is feasible.
          - A penalty value (1e9) if any constraint is violated.
        """
        penalty = 1e9
        
        if not isinstance(solution, dict):
            return penalty
        
        # Check each job's operations.
        for j in range(self.nb_jobs):
            if j not in solution:
                return penalty
            ops = solution[j]
            if len(ops) != self.nb_operations[j]:
                return penalty
            for o, op in enumerate(ops):
                if not isinstance(op, dict):
                    return penalty
                for key in ['machine', 'start', 'end']:
                    if key not in op:
                        return penalty
                m = op['machine']
                start = op['start']
                end = op['end']
                if not (0 <= m < self.nb_machines):
                    return penalty
                # Map the operation to its corresponding task.
                task_id = self.job_operation_task[j][o]
                proc_time = self.task_processing_time_data[task_id][m]
                if proc_time == INFINITE:
                    return penalty
                if end != start + proc_time:
                    return penalty
                if o > 0:
                    # Ensure that operations in the same job do not overlap.
                    prev_end = ops[o - 1]['end']
                    if start < prev_end:
                        return penalty
        
        # Check disjunctive (machine) constraints: no overlapping operations on the same machine.
        machine_ops = {m: [] for m in range(self.nb_machines)}
        for j in range(self.nb_jobs):
            for op in solution[j]:
                m = op['machine']
                machine_ops[m].append((op['start'], op['end']))
        for m in range(self.nb_machines):
            intervals = sorted(machine_ops[m], key=lambda x: x[0])
            for i in range(len(intervals) - 1):
                if intervals[i][1] > intervals[i + 1][0]:
                    return penalty
        
        # If all constraints are met, the objective is the makespan.
        makespan = max(op['end'] for j in range(self.nb_jobs) for op in solution[j])
        return makespan
    
    def random_solution(self):
        """
        Generates a random candidate solution.
        
        For each job, operations are scheduled sequentially.
        For each operation, a compatible machine is randomly selected,
        and a start time is chosen such that the operation begins no earlier than
        the previous operation’s finish time.
        
        Note: This method does not resolve machine conflicts across different jobs.
        """
        solution = {}
        for j in range(self.nb_jobs):
            ops = []
            current_time = random.randint(0, self.max_start // 4)
            for o in range(self.nb_operations[j]):
                task_id = self.job_operation_task[j][o]
                # Determine all compatible machines for this operation.
                compatible = [m for m in range(self.nb_machines) 
                              if self.task_processing_time_data[task_id][m] != INFINITE]
                if not compatible:
                    m = 0
                else:
                    m = random.choice(compatible)
                proc_time = self.task_processing_time_data[task_id][m]
                start = current_time
                end = start + proc_time
                ops.append({
                    "machine": m,
                    "start": start,
                    "end": end
                })
                # Ensure the next operation starts after the current one finishes.
                current_time = end + random.randint(0, self.max_start // 10)
            solution[j] = ops
        return solution
