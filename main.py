from random import choices, randint, random
from typing import List, Dict, Tuple

Operation = int # Definition of an Operation.
Job = List[Operation] # Definition of a Job as a list of operations.
Jobs = List[Job] # Definition of Jobs as a list of jobs. 
Machine = int # Definition of a Machine.
Machines = List[Machine] # Definition of Machines as a list that contains all the machines on the problem.
Available = List[Machine] # Definition of Available as a list of selectable machines.
Selectable = List[List[Available]] # Definition of Selectable as a matrix where each position represents the Job and Operation respectively and the final lists are de machines Available for each operation.
Cost = Dict[Tuple[Job, Operation, Machine], int] # Definition of Cost as a dictionary where the key is the Operation of a Job executed in a Machine and the value is the time that takes the operation to run in the Machine.
Executions = List[List[int]] # Definition of Executions where the position represents a Machine and the value are the Operations executed in the Machine in order from left to right. Operations are presented like the real positions in the individual.

Individual = List[Machine] # Definition of the Individual as a list where the position represents an operation and the value is the machine assigned to the operation. 
Population = List[Individual] # Definition of a Population.


def generate_individual(jobs:Jobs, selectables:Selectable, amount_op:int) -> Individual:
    """
        Function that generates a new inidvidual with random values.
        Each position represents an operation and each value represents
        the machine asigned to each operation.
        Params:
            jobs (Jobs): The jobs to asign.
            selectables (Selectable): Machines selectables for each operation.
            amount_op (int): The total operations of the problem.
        Returns:
            Individual: The individual generated.
    """
    individual = [-1 for _ in range(amount_op)]
    pivot = 0 # To know in which job is in the individual.
    for p_job in range(len(jobs)):
        for op in jobs[p_job]:
            p_in = pivot + op # With the pivot, move to the operation in the individual.
            individual[p_in] = choices(selectables[p_job][op], k=1)[0]
        pivot += len(jobs[p_job])
    return individual

def generate_population(size:int, jobs:Jobs, selectables:Selectable, amount_op:int) -> Population:
    """
        Funciton that generates a new population with random individuals.
        Params:
            size (int): The amount of individuals to generate.
            jobs (Jobs): The jobs to asign.
            selectables (Selectable): Machines selectables for each operation.
            amount_op (int): The total operations of the problem.
        Returns:
            Population: The population generated.
    """
    return [generate_individual(jobs, selectables, amount_op) for _ in range(size)]

def fitness(individual:Individual, jobs:Jobs, costs:Cost, amount_m:int, K:int) -> float:
    times_machines = {k:0 for k in range(amount_m)} # The total times of process of each machine.
    pivot = 0 # To know in which job is in the individual.
    for p_job, job in enumerate(jobs):
        for op in job:
            p_in = pivot + op # With the pivot, move to the operation in the individual.
            machine = individual[p_in]
            if op > 0:
                machine1 = individual[p_in-1] # The machine asigned to the previous operation.
                # Verify if the previous operation has finished before the actual operation
                if times_machines[machine1] <= times_machines[machine]:
                    times_machines[machine] += costs[(p_job, op, machine)] # Continue iterating with the new final execution of the machine.
                else:
                    return 0 # Give the worst score due to problems in order of operations.
            else:
                times_machines[machine] += costs[(p_job, op, machine)] # Is the first operation. It doesn't have any problem.
        pivot += len(job)
    if max(times_machines.values()) > K:
        return 0
    else:
        return 1 / max(times_machines.values()) # Return the maximum execution time of all machines.

def selection_pair(population:Population, population_fitness:List[float]) -> Population:
    """
        Function that selects two pairs of individuals as canditates of crossover.
        Params:
            population (Population): The initial population.
            population_fitness (List[float]): The fitness of each individual.
    """
    return choices(population=population, weights=population_fitness, k=2)

def crossover(parent1:Individual, parent2:Individual) -> Tuple[Individual, Individual]:
    p = randint(1, len(parent1)-1) # Select a random position to make the cut.
    return parent1[0:p] + parent2[p:], parent2[0:p] + parent1[p:] # Cross each segment to make two new childs.

def mutate(individual:Individual, jobs:Jobs, selectables:Selectable, num:int=1, probability:float=0.5) -> Individual:
    new_individual = individual.copy() # Make a copy of the individual
    for _ in range(num):
        if random() > probability:
            p_job = randint(0, len(jobs)) # Select randomly a job.
            p_op = randint(0, len(jobs[p_job])) # Select randomly a operation of the job.
            machine = choices(selectables[p_job][p_op], k=len(selectables[p_job][p_op])) # Select a random machine for the operation.
            p_in = sum([len(jobs[i]) for i in range(p_job)]) + p_op # Get the position on the individual of the operation.
            new_individual[p_in] = machine # Asign the new machine.
    return new_individual

def genetic_algorithm(jobs:Jobs, selectables:Selectable, machines:Machines, costs:Cost, K:int, size_population:int, generations:int, mutation_probability:float) -> Tuple[Individual, float]:
    amount_op = sum([len(job) for job in jobs]) # Get the total amount of operations.
    population = generate_population(size_population, jobs, selectables, amount_op)

    best_individual = None
    best_result = -1

    for _ in range(generations):
        population_fitness = [fitness(individual, jobs, costs, len(machines), K) for individual in population]
        print(population_fitness)

        best_individual_generated = population[population_fitness.index(max(population_fitness))]
        best_fitness = max(population_fitness)

        # Delete the two worst individuals.
        worst_individual = population_fitness.index(min(population_fitness))
        del population_fitness[worst_individual]
        del population[worst_individual]
        worst_individual = population_fitness.index(min(population_fitness))
        del population_fitness[worst_individual]
        del population[worst_individual]

        if best_fitness > best_result:
            best_result = best_fitness
            best_individual = best_individual_generated

        parent1, parent2 = selection_pair(population, population_fitness)
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1, jobs, selectables, probability=mutation_probability)
        child2 = mutate(child2, jobs, selectables, probability=mutation_probability)
        population.append(child1)
        population.append(child2)

    return best_individual, best_result

jobs = [[1, 2, 3], [1, 2, 3, 4, 5], [1, 2, 3, 4], [1, 2, 3], [1, 2, 3, 4, 5], [1, 2, 3, 4], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 2, 3], [1, 2, 3, 4, 5], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3]]
selectables = [[[1, 2, 3, 4, 5], [3], [3, 5]], [[1, 2, 3, 4, 5], [3, 5], [3, 4], [1, 2, 4], [1]], [[1, 2, 3, 4, 5], [1, 3, 5], [1, 2, 3], [4]], [[1, 2, 3, 4, 5], [3, 4], [2, 3, 5]], [[1, 2, 3, 4, 5], [1, 3], [1, 3], [5], [2, 4, 5]], [[1, 2, 3, 4, 5], [1, 4], [3, 4], [1, 2, 3]], [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [5]], [[1, 2, 3, 4, 5], [3, 4], [1, 4]], [[1, 2, 3, 4, 5], [3, 5], [4]], [[1, 2, 3, 4, 5], [3], [1], [1, 3]], [[1, 2, 3, 4, 5], [1, 2, 3, 5], [2, 4], [1, 4], [1, 2, 5]], [[1, 2, 3, 4, 5], [3], [1, 3, 5]], [[1, 2, 3, 4, 5], [1, 3, 4], [3], [1, 2, 4], [2, 4, 5]], [[1, 2, 3, 4, 5], [1, 4], [3, 5]], [[1, 2, 3, 4, 5], [5], [2, 3, 4, 5], [2, 5]], [[1, 2, 3, 4, 5], [1, 2, 4, 5], [5], [5]], [[1, 2, 3, 4, 5], [4], [2, 4], [2], [4, 5]], [[1, 2, 3, 4, 5], [2, 5], [2, 4, 5], [2, 3, 4, 5], [1, 3, 4, 5]], [[1, 2, 3, 4, 5], [1, 3, 5], [1, 3], [2], [1, 2]], [[1, 2, 3, 4, 5], [1, 5], [1, 2, 4, 5]]]
machines = [1,2,3,4,5]
costs = {(1, 1, 1): 2, (1, 1, 2): 8, (1, 1, 3): 1, (1, 1, 4): 5, (1, 1, 5): 5, (1, 2, 3): 7, (1, 3, 3): 7, (1, 3, 5): 4, (2, 1, 1): 1, (2, 1, 2): 2, (2, 1, 3): 7, (2, 1, 4): 4, (2, 1, 5): 8, (2, 2, 3): 2, (2, 2, 5): 7, (2, 3, 3): 1, (2, 3, 4): 3, (2, 4, 1): 2, (2, 4, 2): 6, (2, 4, 4): 1, (2, 5, 1): 8, (3, 1, 1): 4, (3, 1, 2): 1, (3, 1, 3): 6, (3, 1, 4): 3, (3, 1, 5): 8, (3, 2, 1): 6, (3, 2, 3): 8, (3, 2, 5): 3, (3, 3, 1): 1, (3, 3, 2): 3, (3, 3, 3): 1, (3, 4, 4): 3, (4, 1, 1): 5, (4, 1, 2): 1, (4, 1, 3): 8, (4, 1, 4): 8, (4, 1, 5): 6, (4, 2, 3): 5, (4, 2, 4): 5, (4, 3, 2): 4, (4, 3, 3): 1, (4, 3, 5): 5, (5, 1, 1): 1, (5, 1, 2): 6, (5, 1, 3): 1, (5, 1, 4): 3, (5, 1, 5): 1, (5, 2, 1): 6, (5, 2, 3): 5, (5, 3, 1): 2, (5, 3, 3): 8, (5, 4, 5): 8, (5, 5, 2): 4, (5, 5, 4): 6, (5, 5, 5): 5, (6, 1, 1): 3, (6, 1, 2): 4, (6, 1, 3): 5, (6, 1, 4): 4, (6, 1, 5): 5, (6, 2, 1): 3, (6, 2, 4): 4, (6, 3, 3): 3, (6, 3, 4): 8, (6, 4, 1): 7, (6, 4, 2): 5, (6, 4, 3): 7, (7, 1, 1): 2, (7, 1, 2): 6, (7, 1, 3): 5, (7, 1, 4): 1, (7, 1, 5): 7, (7, 2, 1): 2, (7, 2, 2): 7, (7, 2, 3): 5, (7, 2, 4): 2, (7, 2, 5): 2, (7, 3, 5): 4, (8, 1, 1): 6, (8, 1, 2): 7, (8, 1, 3): 3, (8, 1, 4): 5, (8, 1, 5): 8, (8, 2, 3): 1, (8, 2, 4): 5, (8, 3, 1): 6, (8, 3, 4): 4, (9, 1, 1): 5, (9, 1, 2): 7, (9, 1, 3): 8, (9, 1, 4): 5, (9, 1, 5): 5, (9, 2, 3): 6, (9, 2, 5): 5, (9, 3, 4): 7, (10, 1, 1): 2, (10, 1, 2): 3, (10, 1, 3): 7, (10, 1, 4): 7, (10, 1, 5): 8, (10, 2, 3): 8, (10, 3, 1): 3, (10, 4, 1): 2, (10, 4, 3): 2, (11, 1, 1): 5, (11, 1, 2): 4, (11, 1, 3): 1, (11, 1, 4): 5, (11, 1, 5): 3, (11, 2, 1): 4, (11, 2, 2): 6, (11, 2, 3): 7, (11, 2, 5): 6, (11, 3, 2): 1, (11, 3, 4): 5, (11, 4, 1): 3, (11, 4, 4): 6, (11, 5, 1): 6, (11, 5, 2): 8, (11, 5, 5): 4, (12, 1, 1): 8, (12, 1, 2): 3, (12, 1, 3): 7, (12, 1, 4): 3, (12, 1, 5): 7, (12, 2, 3): 4, (12, 3, 1): 3, (12, 3, 3): 7, (12, 3, 5): 4, (13, 1, 1): 5, (13, 1, 2): 7, (13, 1, 3): 4, (13, 1, 4): 1, (13, 1, 5): 1, (13, 2, 1): 7, (13, 2, 3): 8, (13, 2, 4): 1, (13, 3, 3): 1, (13, 4, 1): 4, (13, 4, 2): 7, (13, 4, 4): 4, (13, 5, 2): 1, (13, 5, 4): 8, (13, 5, 5): 3, (14, 1, 1): 8, (14, 1, 2): 4, (14, 1, 3): 7, (14, 1, 4): 8, (14, 1, 5): 2, (14, 2, 1): 3, (14, 2, 4): 4, (14, 3, 3): 1, (14, 3, 5): 2, (15, 1, 1): 4, (15, 1, 2): 5, (15, 1, 3): 3, (15, 1, 4): 7, (15, 1, 5): 2, (15, 2, 5): 5, (15, 3, 2): 2, (15, 3, 3): 8, (15, 3, 4): 1, (15, 3, 5): 7, (15, 4, 2): 5, (15, 4, 5): 1, (16, 1, 1): 5, (16, 1, 2): 2, (16, 1, 3): 8, (16, 1, 4): 1, (16, 1, 5): 8, (16, 2, 1): 5, (16, 2, 2): 3, (16, 2, 4): 8, (16, 2, 5): 7, (16, 3, 5): 5, (16, 4, 5): 6, (17, 1, 1): 4, (17, 1, 2): 3, (17, 1, 3): 2, (17, 1, 4): 6, (17, 1, 5): 5, (17, 2, 4): 5, (17, 3, 2): 6, (17, 3, 4): 3, (17, 4, 2): 3, (17, 5, 4): 7, (17, 5, 5): 6, (18, 1, 1): 5, (18, 1, 2): 5, (18, 1, 3): 5, (18, 1, 4): 8, (18, 1, 5): 6, (18, 2, 2): 4, (18, 2, 5): 1, (18, 3, 2): 4, (18, 3, 4): 3, (18, 3, 5): 8, (18, 4, 2): 2, (18, 4, 3): 6, (18, 4, 4): 6, (18, 4, 5): 4, (18, 5, 1): 4, (18, 5, 3): 7, (18, 5, 4): 2, (18, 5, 5): 5, (19, 1, 1): 7, (19, 1, 2): 2, (19, 1, 3): 3, (19, 1, 4): 8, (19, 1, 5): 5, (19, 2, 1): 3, (19, 2, 3): 3, (19, 2, 5): 2, (19, 3, 1): 4, (19, 3, 3): 6, (19, 4, 2): 6, (19, 5, 1): 3, (19, 5, 2): 6, (20, 1, 1): 1, (20, 1, 2): 2, (20, 1, 3): 2, (20, 1, 4): 6, (20, 1, 5): 1, (20, 2, 1): 8, (20, 2, 5): 1, (20, 3, 1): 7, (20, 3, 2): 1, (20, 3, 4): 1, (20, 3, 5): 3}

jobs = [[x - 1 for x in sublist] for sublist in jobs]
selectables = [[[x - 1 for x in sublist] for sublist in group] for group in selectables]
machines = [x - 1 for x in machines]
costs = {tuple(x - 1 for x in key): value for key, value in costs.items()}
K = 199

print(genetic_algorithm(jobs, selectables, machines, costs, K, 100, 100, 0.1))