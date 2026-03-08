import random
import numpy as np
from typing import List, Tuple
import time


class SubsetSumProblem:
    def __init__(self, dataset_size: int = 100, target_sum: int = 100, seed: int = 42): # Default values not used
        self.dataset_size = dataset_size
        self.target_sum = target_sum
        self.seed = seed
        self.dataset = self._generate_dataset()

    def _generate_dataset(self) -> List[int]:
        random.seed(self.seed)
        np.random.seed(self.seed)
        # Generate random integers between 1 and 50
        return [random.randint(1, 50) for _ in range(self.dataset_size)]

    def get_problem_info(self):
        return {
            'seed': self.seed,
            'dataset': self.dataset,
            'target_sum': self.target_sum,
            'dataset_size': self.dataset_size
        }


class GeneticAlgorithm:
    def __init__(self, population_size: int = 100, crossover_rate: float = 0.8,
                 mutation_rate: float = 0.01, elitism_count: int = 2, max_generations: int = 1000): # Default values not used
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        self.max_generations = max_generations
        self.fitness_evaluations = 0

    def initialize_population(self, chromosome_length: int) -> List[List[int]]:
        return [[random.randint(0, 1) for _ in range(chromosome_length)]
                for _ in range(self.population_size)]

    def calculate_fitness(self, individual: List[int], dataset: List[int], target_sum: int) -> float:
        self.fitness_evaluations += 1
        subset_sum = sum(dataset[i] for i in range(len(individual)) if individual[i] == 1)
        return abs(target_sum - subset_sum)

    def roulette_wheel_selection(self, population: List[List[int]], fitnesses: List[float]) -> List[int]:
        # Convert to maximization problem
        max_fitness = max(fitnesses)
        inverted_fitnesses = [max_fitness - f + 1 for f in fitnesses]
        total_fitness = sum(inverted_fitnesses)

        if total_fitness == 0:
            return random.choice(population)

        pick = random.uniform(0, total_fitness)
        current = 0

        for i, individual in enumerate(population):
            current += inverted_fitnesses[i]
            if current > pick:
                return individual
        return population[-1]

    def single_point_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def bit_flip_mutation(self, individual: List[int]) -> List[int]:
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]  # Flip bit
        return mutated

    def run(self, dataset: List[int], target_sum: int) -> Tuple[List[int], int, int, float]:
        start_time = time.time()  # Start timing

        chromosome_length = len(dataset)
        population = self.initialize_population(chromosome_length)
        self.fitness_evaluations = 0

        best_individual = None
        best_fitness = float('inf')

        for generation in range(self.max_generations):
            # Evaluate fitness
            fitnesses = [self.calculate_fitness(ind, dataset, target_sum) for ind in population]

            # Check for perfect solution
            for i, fitness in enumerate(fitnesses):
                if fitness == 0:
                    runtime = time.time() - start_time  # Calculate runtime
                    return population[i], generation + 1, self.fitness_evaluations, runtime
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_individual = population[i]

            # Create new population with elitism
            new_population = []

            # Select elites
            elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:self.elitism_count]
            for idx in elite_indices:
                new_population.append(population[idx])

            # Generate offspring
            while len(new_population) < self.population_size:
                parent1 = self.roulette_wheel_selection(population, fitnesses)
                parent2 = self.roulette_wheel_selection(population, fitnesses)

                child1, child2 = self.single_point_crossover(parent1, parent2)

                child1 = self.bit_flip_mutation(child1)
                child2 = self.bit_flip_mutation(child2)

                new_population.extend([child1, child2])

            population = new_population[:self.population_size]

        runtime = time.time() - start_time  # Calculate runtime
        return best_individual, self.max_generations, self.fitness_evaluations, runtime


class BinaryDifferentialEvolution:
    def __init__(self, population_size: int = 100, F_prob: float = 0.5,
                 CR: float = 0.7, max_generations: int = 1000): # Default values not used
        self.population_size = population_size
        self.F_prob = F_prob  # Probability for bits in F vector to be 1
        self.CR = CR  # Crossover rate
        self.max_generations = max_generations
        self.fitness_evaluations = 0

    def initialize_population(self, chromosome_length: int) -> List[List[int]]:
        return [[random.randint(0, 1) for _ in range(chromosome_length)]
                for _ in range(self.population_size)]

    def calculate_fitness(self, individual: List[int], dataset: List[int], target_sum: int) -> float:
        self.fitness_evaluations += 1
        subset_sum = sum(dataset[i] for i in range(len(individual)) if individual[i] == 1)
        return abs(target_sum - subset_sum)

    def binary_mutation(self, population: List[List[int]]) -> List[List[int]]:
        mutants = []

        for i in range(self.population_size):
            # Select three distinct random individuals
            indices = random.sample(range(self.population_size), 3)
            while i in indices:
                indices = random.sample(range(self.population_size), 3)

            r1, r2, r3 = indices
            x_r1 = population[r1]
            x_r2 = population[r2]
            x_r3 = population[r3]

            # Create F vector (random binary vector)
            F = [1 if random.random() < self.F_prob else 0 for _ in range(len(x_r1))]

            # Calculate perturbation: F AND (x_r2 XOR x_r3)
            perturbation = [F[j] & (x_r2[j] ^ x_r3[j]) for j in range(len(x_r1))]

            # Create mutant: x_r1 XOR perturbation
            mutant = [x_r1[j] ^ perturbation[j] for j in range(len(x_r1))]

            mutants.append(mutant)

        return mutants

    def uniform_crossover(self, population: List[List[int]], mutants: List[List[int]]) -> List[List[int]]:
        trial_vectors = []

        for i in range(self.population_size):
            trial = population[i].copy()
            mutant = mutants[i]

            # Ensure at least one dimension from mutant
            random_idx = random.randint(0, len(trial) - 1)
            trial[random_idx] = mutant[random_idx]

            # Crossover for other dimensions
            for j in range(len(trial)):
                if j != random_idx and random.random() < self.CR:
                    trial[j] = mutant[j]

            trial_vectors.append(trial)

        return trial_vectors

    def run(self, dataset: List[int], target_sum: int) -> Tuple[List[int], int, int, float]:
        start_time = time.time()  # Start timing

        chromosome_length = len(dataset)
        population = self.initialize_population(chromosome_length)
        self.fitness_evaluations = 0

        best_individual = None
        best_fitness = float('inf')

        for generation in range(self.max_generations):
            # Evaluate current population
            fitnesses = [self.calculate_fitness(ind, dataset, target_sum) for ind in population]

            # Check for perfect solution
            for i, fitness in enumerate(fitnesses):
                if fitness == 0:
                    runtime = time.time() - start_time  # Calculate runtime
                    return population[i], generation + 1, self.fitness_evaluations, runtime
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_individual = population[i]

            # Create mutant population
            mutants = self.binary_mutation(population)

            # Create trial population through crossover
            trial_population = self.uniform_crossover(population, mutants)

            # Evaluate trial population
            trial_fitnesses = [self.calculate_fitness(ind, dataset, target_sum) for ind in trial_population]

            # Selection: replace if trial is better or equal
            new_population = []
            for i in range(self.population_size):
                if trial_fitnesses[i] <= fitnesses[i]:
                    new_population.append(trial_population[i])
                else:
                    new_population.append(population[i])

            population = new_population

        runtime = time.time() - start_time  # Calculate runtime
        return best_individual, self.max_generations, self.fitness_evaluations, runtime


def compare_algorithms():
    # Initialize the subset sum problem
    problem = SubsetSumProblem(dataset_size=100, target_sum=100, seed=random.randint(a=0, b=999999))
    problem_info = problem.get_problem_info()

    print("Subset Sum Problem Configuration:")
    print(f"Seed: {problem_info['seed']}")
    print(f"Dataset size: {problem_info['dataset_size']}")
    print(f"Target sum: {problem_info['target_sum']}")
    print("-" * 80)

    # Initialize algorithms
    ga = GeneticAlgorithm(population_size=100, crossover_rate=0.8,
                          mutation_rate=0.01, elitism_count=2, max_generations=10000)
    bde = BinaryDifferentialEvolution(population_size=100, F_prob=0.3,
                                      CR=0.3, max_generations=10000)

    # Storage for results
    ga_results = []
    bde_results = []

    print("Running Genetic Algorithm 10 times...")
    for run in range(10):
        ga_solution, ga_generations, ga_evaluations, ga_runtime = ga.run(problem_info['dataset'],
                                                                         problem_info['target_sum'])
        ga_subset_sum = sum(problem_info['dataset'][i] for i in range(len(ga_solution)) if ga_solution[i] == 1)
        ga_difference = abs(problem_info['target_sum'] - ga_subset_sum)

        ga_results.append({
            'run': run + 1,
            'solution': ga_solution,
            'generations': ga_generations,
            'evaluations': ga_evaluations,
            'runtime': ga_runtime,
            'subset_sum': ga_subset_sum,
            'difference': ga_difference
        })

        print(f"GA Run {run + 1}: Time = {ga_runtime:.4f}s, Generations = {ga_generations}, "
              f"Evaluations = {ga_evaluations}, Difference = {ga_difference}")

    print("-" * 80)

    print("Running Binary Differential Evolution 10 times...")
    for run in range(10):
        bde_solution, bde_generations, bde_evaluations, bde_runtime = bde.run(problem_info['dataset'],
                                                                              problem_info['target_sum'])
        bde_subset_sum = sum(problem_info['dataset'][i] for i in range(len(bde_solution)) if bde_solution[i] == 1)
        bde_difference = abs(problem_info['target_sum'] - bde_subset_sum)

        bde_results.append({
            'run': run + 1,
            'solution': bde_solution,
            'generations': bde_generations,
            'evaluations': bde_evaluations,
            'runtime': bde_runtime,
            'subset_sum': bde_subset_sum,
            'difference': bde_difference
        })

        print(f"BDE Run {run + 1}: Time = {bde_runtime:.4f}s, Generations = {bde_generations}, "
              f"Evaluations = {bde_evaluations}, Difference = {bde_difference}")

    print("-" * 80)

    # Calculate statistics
    ga_avg_runtime = sum(r['runtime'] for r in ga_results) / 10
    ga_avg_evaluations = sum(r['evaluations'] for r in ga_results) / 10
    ga_avg_generations = sum(r['generations'] for r in ga_results) / 10
    ga_avg_difference = sum(r['difference'] for r in ga_results) / 10
    ga_total_runtime = sum(r['runtime'] for r in ga_results)

    bde_avg_runtime = sum(r['runtime'] for r in bde_results) / 10
    bde_avg_evaluations = sum(r['evaluations'] for r in bde_results) / 10
    bde_avg_generations = sum(r['generations'] for r in bde_results) / 10
    bde_avg_difference = sum(r['difference'] for r in bde_results) / 10
    bde_total_runtime = sum(r['runtime'] for r in bde_results)

    # Print comparison summary
    print("COMPARISON SUMMARY (Average of 10 runs):")
    print(f"Algorithm        | Avg Time (s) | Total Time (s) | Avg Evaluations | Avg Generations | Avg Difference")
    print(
        f"GA               | {ga_avg_runtime:.4f}       | {ga_total_runtime:.4f}        | {ga_avg_evaluations:.0f}           | {ga_avg_generations:.0f}            | {ga_avg_difference:.2f}")
    print(
        f"BDE              | {bde_avg_runtime:.4f}       | {bde_total_runtime:.4f}        | {bde_avg_evaluations:.0f}           | {bde_avg_generations:.0f}            | {bde_avg_difference:.2f}")

    # Determine which algorithm was more efficient
    perfect_ga_runs = sum(1 for r in ga_results if r['difference'] == 0)
    perfect_bde_runs = sum(1 for r in bde_results if r['difference'] == 0)

    print(f"\nPerfect solutions found: GA = {perfect_ga_runs}/10, BDE = {perfect_bde_runs}/10")

if __name__ == "__main__":
    compare_algorithms()