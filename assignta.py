import pandas as pd
import random as rnd
import numpy as np

from profiler import Profiler, profile
from evo import Environment

# Load section data from CSV and turn into dictionary
def load_sections(filename='sections.csv'):
    sections_df = pd.read_csv(filename)

    sections = {}
    for _, row in sections_df.iterrows():
        section_id = row['section']
        instructor = row['instructor']
        daytime = row['daytime']
        location = row['location']
        students = row['students']
        topic = row['topic']
        min_ta = row['min_ta']
        max_ta = row['max_ta']

        sections[section_id] = {
            'instructor': instructor,
            'daytime': daytime,
            'location': location,
            'students': students,
            'topic': topic,
            'min_ta': min_ta,
            'max_ta': max_ta,
            'ta_ids': []
        }
    return sections

# Load TA data from CSV
def load_tas(filename='tas.csv'):
    tas_df = pd.read_csv(filename)

    tas = {}
    for _, row in tas_df.iterrows():
        ta_id = row['ta_id']
        name = row['name']
        max_assigned = row['max_assigned']

        # Create a dictionary of preferences for each TA
        preferences = {str(i): row[str(i)] for i in range(17)}

        tas[ta_id] = {
            'name': name,
            'max_assigned': max_assigned,
            'preferences': preferences,
            'availability': preferences
        }
    return tas

# OBJECTIVE FUNCTIONS
@profile
def overallocation(assignments, tas):
    """ Computes penalty for TAs who are assigned to more sections than they can handle (max_assigned) """
    total_penalty = 0
    for ta_id, ta_info in tas.items():
        max_assigned = ta_info['max_assigned']
        assigned_sections = np.sum(assignments[ta_id, :])
        if assigned_sections > max_assigned:
            penalty = assigned_sections - max_assigned
            total_penalty += penalty
    return total_penalty

@profile
def conflicts(assignments, sections):
    """ Counts the number of TAs that have scheduling conflicts """
    conflicting_tas = set()
    for ta_idx in range(assignments.shape[0]):
        assigned_sections = np.where(assignments[ta_idx, :] == 1)[0]
        assigned_daytimes = [sections[section_id]['daytime'] for section_id in assigned_sections]
        if len(set(assigned_daytimes)) < len(assigned_daytimes):
            conflicting_tas.add(ta_idx)
    return len(conflicting_tas)

@profile
def undersupport(assignments, sections, tas):
    """ Measures how many sections do not have enough TAs assigned (min_ta) """
    total_penalty = 0
    assigned_to_sections = np.sum(assignments, axis=0)
    for section_id, section_info in sections.items():
        required_ta = section_info['min_ta']
        assigned_ta = assigned_to_sections[section_id]
        if assigned_ta < required_ta:
            penalty = required_ta - assigned_ta
            total_penalty += penalty
    return total_penalty

@profile
def unwilling(assignments, tas):
    """ Calculates how many times a TA is assigned to a section they are unwilling to support """
    total_penalty = 0
    for ta_id, ta_info in tas.items():
        preferences = ta_info['preferences']
        assigned_sections = np.where(assignments[ta_id, :] == 1)[0]
        for sec_id in assigned_sections:
            if preferences[str(sec_id)] == 'U':
                total_penalty += 1
    return total_penalty

@profile
def unpreferred(assignments, tas):
    """ Measures how many times a TA is assigned to a section they are "willing" to support but not "preferred" """
    total_penalty = 0
    for ta_id, ta_info in tas.items():
        preferences = ta_info['preferences']
        for section_id, assigned in enumerate(assignments[ta_id, :]):
            if assigned == 1 and preferences[str(section_id)] == 'W':
                total_penalty += 1
    return total_penalty

# AGENTS
@profile
def greedy_assignment(assignments, tas, sections):
    """ Ensures that no TA is assigned more sections than they can handle """
    new_assignments = assignments.copy()
    for ta_id, ta_info in tas.items():
        max_assigned = ta_info['max_assigned']
        assigned_sections = np.where(new_assignments[ta_id, :] == 1)[0]
        current_assigned = len(assigned_sections)
        if current_assigned > max_assigned:
            excess_sections = assigned_sections[max_assigned:]
            for sec in excess_sections:
                new_assignments[ta_id, sec] = 0
    return new_assignments

@profile
def unwillingness_reduction(assignments, tas, sections):
    """ Reduces the number of assignments to sections that a TA is unwilling to support """
    new_assignments = assignments.copy()
    for ta_id, ta_info in tas.items():
        preferences = ta_info['preferences']
        unwilling_sections = [sec_id for sec_id, pref in preferences.items()
                              if pref == 'U' and new_assignments[ta_id, int(sec_id)] == 1]
        for sec_id in unwilling_sections:
            possible_sections = [sec_id for sec_id, pref in preferences.items()
                                 if pref != 'U' and new_assignments[ta_id, int(sec_id)] == 0]
            if possible_sections:
                new_section = rnd.choice(possible_sections)
                new_assignments[ta_id, int(sec_id)] = 0
                new_assignments[ta_id, int(new_section)] = 1
    return new_assignments

@profile
def time_conflict_resolution(assignments, tas, sections):
    """ Resolves time conflicts by shifting assignments for TAs scheduled to work in more than one section at a time """
    new_assignments = assignments.copy()
    daytime_groups = {}
    for sec_id, section_info in sections.items():
        daytime = section_info['daytime']
        if daytime not in daytime_groups:
            daytime_groups[daytime] = []
        daytime_groups[daytime].append(sec_id)
    for ta_id in range(assignments.shape[0]):
        assigned_sections = np.where(new_assignments[ta_id, :] == 1)[0]
        assigned_daytimes = [sections[sec_id]['daytime'] for sec_id in assigned_sections]
        conflicting_daytimes = [daytime for daytime in assigned_daytimes if assigned_daytimes.count(daytime) > 1]
        if conflicting_daytimes:
            for conflict_daytime in set(conflicting_daytimes):
                conflicting_sections = [sec_id for sec_id in assigned_sections
                                        if sections[sec_id]['daytime'] == conflict_daytime]
                for sec_id in conflicting_sections:
                    new_section = rnd.choice(daytime_groups[conflict_daytime])
                    if new_section != sec_id:
                        new_assignments[ta_id, sec_id] = 0
                        new_assignments[ta_id, new_section] = 1
                        break
    return new_assignments

@profile
def random_perturbation(assignments, tas, sections):
    """ Randomly changes assignments by unassigning and reassigning TAs to different sections """
    new_assignments = assignments.copy()
    num_tas_to_perturb = rnd.randint(1, 5)
    for _ in range(num_tas_to_perturb):
        ta_id = rnd.choice(range(assignments.shape[0]))
        assigned_sections = np.where(new_assignments[ta_id, :] == 1)[0]
        if assigned_sections.size > 0:
            sec_to_unassign = rnd.choice(assigned_sections)
            new_assignments[ta_id, sec_to_unassign] = 0
        unassigned_sections = np.where(new_assignments[ta_id, :] == 0)[0]
        if unassigned_sections.size > 0:
            new_section = rnd.choice(unassigned_sections)
            new_assignments[ta_id, new_section] = 1
    return new_assignments

def extract_best_solution(pareto="pareto_front.csv"):
    pareto_df = pd.read_csv(pareto)

    # Define weights for each objective function
    weights = {
        'overallocation': 1.5,
        'conflicts': 2.5,
        'undersupport': 2,
        'unwilling': 3,
        'unpreferred': 1
    }

    # Compute a combined score for each row
    pareto_df['combined_score'] = (
        weights['overallocation'] * pareto_df['overallocation'] +
        weights['conflicts'] * pareto_df['conflicts'] +
        weights['undersupport'] * pareto_df['undersupport'] +
        weights['unwilling'] * pareto_df['unwilling'] +
        weights['unpreferred'] * pareto_df['unpreferred']
    )

    # Find the best solution
    best_solution_row = pareto_df.loc[pareto_df['combined_score'].idxmin()]
    solution_index = best_solution_row.name
    return solution_index, best_solution_row

def print_best_solution(solution_index, tas, sections, assignments, solution_evaluation, output_filename="best_solution.txt"):
    # Open the file for writing
    with open(output_filename, 'w') as f:
        f.write(f"Best Solution (Pareto Index {solution_index}):\n\n")

        # 1. Scores for each objective function for the selected solution
        f.write("Objective Function Scores:\n")
        for objective, score in solution_evaluation:
            f.write(f"{objective}: {score}\n")

        # 2. List of sections for each TA
        f.write("\nAssigned Sections for Each TA:\n")
        for ta_id, ta_info in tas.items():
            assigned_sections = np.where(assignments[ta_id, :] == 1)[0]  # Get the sections assigned to this TA
            assigned_section_ids = [list(sections.keys())[sec_id] for sec_id in assigned_sections]  # Get the section IDs
            f.write(f"TA {ta_info['name']} (ID: {ta_id}) is assigned to sections: {assigned_section_ids}\n")

        # 3. List of teaching assistants for each section
        f.write("\nAssigned TAs for Each Section:\n")
        for sec_id, section_info in sections.items():
            assigned_tas = np.where(assignments[:, sec_id] == 1)[0]  # Get the TAs assigned to this section
            assigned_ta_names = [tas[ta_id]['name'] for ta_id in assigned_tas]  # Get the TA names
            f.write(f"Section {sec_id} ({section_info['topic']}) has TAs: {assigned_ta_names}\n")

    print(f"Best solution saved to {output_filename}")

def main():
    env = Environment()

    # Add fitness criteria to the environment
    env.add_fitness_criteria('overallocation', overallocation, ['assignments', 'tas'])
    env.add_fitness_criteria('conflicts', conflicts, ['assignments', 'sections'])
    env.add_fitness_criteria('undersupport', undersupport, ['assignments', 'sections', 'tas'])
    env.add_fitness_criteria('unwilling', unwilling, ['assignments', 'tas'])
    env.add_fitness_criteria('unpreferred', unpreferred, ['assignments', 'tas'])

    env.add_agent('greedy_assignment', greedy_assignment, k=1)
    env.add_agent('unwillingness_reduction', unwillingness_reduction, k=1)
    env.add_agent('time_conflict_resolution', time_conflict_resolution, k=1)
    env.add_agent('random_perturbation', random_perturbation, k=1)

    # Initialize the data and add initial solutions
    tas = load_tas('tas.csv')
    sections = load_sections('sections.csv')

    # Create a random initial assignment
    initial_assignments = np.zeros((len(tas), len(sections)))
    for ta_id in tas.keys():
        for section_id in sections.keys():
            if rnd.random() > 0.7:  # Randomly assign TAs to sections
                initial_assignments[int(ta_id), int(section_id)] = 1

    initial_solution = {
        'assignments': initial_assignments,
        'tas': tas,
        'sections': sections
    }

    # Add initial solution to the environment
    env.add_solution(initial_solution)

    # Run the evo process for 5 minutes max
    env.evolve(n=100000, dom=100, status=1000, time_limit=300)

    # Generate the Pareto after evolving solutions
    env.output_pareto_front(filename="pareto_front.csv", groupname="BNTAs")

    # Extract the best solution from the Pareto
    solution_index, solution_row = extract_best_solution()

    # Retrieve the best solution's objective function scores
    best_solution_eval = list(env.pop.keys())[solution_index]  # Get the solution's evaluation from the population
    best_solution = env.pop[best_solution_eval]  # Retrieve the corresponding solution

    # Extract the assignments from the best solution
    best_assignments = best_solution['assignments']

    print_best_solution(solution_index, tas, sections, best_assignments, best_solution_eval)

    # Generate the profiler report
    Profiler.report(filename='profiler_report.txt')

if __name__ == "__main__":
    main()