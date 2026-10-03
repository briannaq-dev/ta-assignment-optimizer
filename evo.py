import random as rnd
import copy
from functools import reduce
import numpy as np
import csv

from profiler import Profiler, profile

class Environment:

    def __init__(self):
        self.pop = {}  # evaluation --> solution
        self.fitness = {}  # objective function
        self.agents = {}  # name --> (operator function, num_solutions_input)

    def add_fitness_criteria(self, name, f, args):  # add_objective
        """ Register an objective with the environment """
        self.fitness[name] = (f, args)

    def add_agent(self, name, op, k=1):
        """ Register an agent with the environment
        The operator (op) defines how the agent tweaks a solution.
        k defines the number of solutions input to the agent. """
        self.agents[name] = (op, k)

    def add_solution(self, sol):
        """Add a solution to the population"""
        eval = tuple([(name, f(*(sol[arg] for arg in args))) for name, (f, args) in self.fitness.items()])
        self.pop[eval] = sol

    def get_random_solutions(self, k=1):
        """ Pick k random solutions from the population """
        if len(self.pop) == 0:  # no solutions in the population (This should never happen!)
            return []
        else:
            solutions = tuple(self.pop.values())
            # Doing a deep copy of a randomly chosen solution (k times)
            return [copy.deepcopy(rnd.choice(solutions)) for _ in range(k)]

    @profile
    def run_agent(self, name):
        """ Invoked named agent on the population """
        op, k = self.agents[name]
        picks = self.get_random_solutions(k)

        # Assuming the solutions in 'picks' are dictionaries with 'assignments', 'tas', and 'sections' keys
        if name in ['greedy_assignment', 'time_conflict_resolution', 'random_perturbation', 'unwillingness_reduction']:
            for pick in picks:
                new_solution = op(pick['assignments'], pick['tas'], pick['sections'])
                new_solution_dict = {
                    'assignments': new_solution,
                    'tas': pick['tas'],
                    'sections': pick['sections']
                }
                self.add_solution(new_solution_dict)
        else:
            new_solution = op(picks)
            self.add_solution(new_solution)

    @profile
    def dominates(self, p, q):
        """
        p = evaluation of one solution: ((obj1, score1), (obj2, score2), ... )
        q = evaluation of another solution: ((obj1, score1), (obj2, score2), ... )
        """
        pscores = np.array([score for name, score in p])
        qscores = np.array([score for name, score in q])
        score_diffs = qscores - pscores
        return min(score_diffs) >= 0 and max(score_diffs) > 0.0

    @profile
    def reduce_nds(self, S, p):
        return S - {q for q in S if self.dominates(p, q)}

    @profile
    def remove_dominated(self):
        nds = reduce(self.reduce_nds, self.pop.keys(), self.pop.keys())
        self.pop = {k: self.pop[k] for k in nds}

    @profile
    def evolve(self, n=75000, dom=100, status=1000, time_limit=300):
        agent_names = list(self.agents.keys())

        Profiler.set_time_limit(time_limit)

        for i in range(n):
            try:
                pick = rnd.choice(agent_names)
                self.run_agent(pick)

                if i % dom == 0:
                    self.remove_dominated()

                if i % status == 0:
                    self.remove_dominated()
            except TimeoutError as e:
                print(str(e))
                break

        self.remove_dominated()

    def __str__(self):
        """ Output the solutions in the population """
        rslt = ""
        for eval, sol in self.pop.items():
            rslt += str(eval) + ":\t" + str(sol) + "\n"
        return rslt

    @profile
    def get_non_dominated_solutions(self):
        """ Return the non-dominated solutions in the population """
        non_dominated_solutions = reduce(self.reduce_nds, self.pop.keys(), self.pop.keys())
        return [(eval, self.pop[eval]) for eval in non_dominated_solutions]

    @profile
    def output_pareto_front(self, filename="pareto_front.csv", groupname="BNTAs"):
        """ Output the non-dominated solutions in CSV format """
        non_dominated_solutions = self.get_non_dominated_solutions()

        # Prepare data for CSV
        data = []
        for eval, sol in non_dominated_solutions:
            # Extract the objective function values from the evaluation tuple
            obj_values = [score for name, score in eval]
            data.append([groupname] + obj_values)

        # Write the data to a CSV file
        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            # Write header row
            writer.writerow(['groupname', 'overallocation', 'conflicts', 'undersupport', 'unwilling', 'unpreferred'])
            # Write data rows
            writer.writerows(data)

        print(f"Pareto front output saved to {filename}")