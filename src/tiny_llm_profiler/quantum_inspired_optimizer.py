"""
Quantum-Inspired Optimization Engine
Advanced optimization using quantum computing principles for edge AI profiling.
"""

import asyncio
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

from .exceptions import ProfilerError


class QuantumOptimizationMethod(str, Enum):
    """Quantum-inspired optimization methods."""

    QUANTUM_ANNEALING = "quantum_annealing"
    VARIATIONAL_QUANTUM_EIGENSOLVER = "vqe"
    QUANTUM_APPROXIMATE_OPTIMIZATION = "qaoa"
    QUANTUM_GENETIC_ALGORITHM = "qga"
    QUANTUM_PARTICLE_SWARM = "qpso"
    QUANTUM_NEURAL_NETWORK = "qnn"


class OptimizationObjective(str, Enum):
    """Optimization objectives for edge AI profiling."""

    MINIMIZE_LATENCY = "minimize_latency"
    MINIMIZE_MEMORY = "minimize_memory"
    MINIMIZE_ENERGY = "minimize_energy"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    MAXIMIZE_ACCURACY = "maximize_accuracy"
    PARETO_OPTIMAL = "pareto_optimal"


@dataclass
class QuantumState:
    """Quantum state representation for optimization."""

    amplitudes: np.ndarray
    phases: np.ndarray
    entanglement_matrix: np.ndarray
    coherence_time: float = 1.0

    def __post_init__(self):
        self.num_qubits = int(np.log2(len(self.amplitudes)))

    def normalize(self) -> "QuantumState":
        """Normalize quantum state amplitudes."""
        norm = np.linalg.norm(self.amplitudes)
        if norm > 0:
            self.amplitudes = self.amplitudes / norm
        return self

    def measure(self) -> int:
        """Measure quantum state and collapse to classical state."""
        probabilities = np.abs(self.amplitudes) ** 2
        return np.random.choice(len(probabilities), p=probabilities)

    def entangle_with(self, other: "QuantumState") -> "QuantumState":
        """Create entangled state with another quantum state."""
        combined_amplitudes = np.kron(self.amplitudes, other.amplitudes)
        combined_phases = np.kron(self.phases, other.phases)

        # Create entanglement matrix
        total_qubits = self.num_qubits + other.num_qubits
        entanglement_matrix = np.random.random((2**total_qubits, 2**total_qubits))
        entanglement_matrix = (entanglement_matrix + entanglement_matrix.T) / 2

        return QuantumState(
            amplitudes=combined_amplitudes,
            phases=combined_phases,
            entanglement_matrix=entanglement_matrix,
            coherence_time=min(self.coherence_time, other.coherence_time),
        )


@dataclass
class OptimizationParameter:
    """Parameter definition for quantum optimization."""

    name: str
    value_range: Tuple[float, float]
    current_value: float
    quantum_encoding: str = "binary"  # binary, gray, angle
    precision_bits: int = 8

    def encode_quantum(self) -> np.ndarray:
        """Encode parameter value as quantum state."""
        min_val, max_val = self.value_range
        normalized_value = (self.current_value - min_val) / (max_val - min_val)

        if self.quantum_encoding == "binary":
            # Binary encoding
            binary_value = int(normalized_value * (2**self.precision_bits - 1))
            amplitudes = np.zeros(2**self.precision_bits)
            amplitudes[binary_value] = 1.0

        elif self.quantum_encoding == "angle":
            # Angle encoding (more efficient for continuous parameters)
            angle = normalized_value * 2 * np.pi
            amplitudes = np.array([np.cos(angle / 2), np.sin(angle / 2)])

        else:  # gray encoding
            # Gray code encoding for better quantum gate efficiency
            binary_value = int(normalized_value * (2**self.precision_bits - 1))
            gray_value = binary_value ^ (binary_value >> 1)
            amplitudes = np.zeros(2**self.precision_bits)
            amplitudes[gray_value] = 1.0

        phases = np.zeros_like(amplitudes)
        return QuantumState(
            amplitudes=amplitudes,
            phases=phases,
            entanglement_matrix=np.eye(len(amplitudes)),
        )

    def decode_quantum(self, quantum_state: QuantumState) -> float:
        """Decode quantum state back to parameter value."""
        measured_state = quantum_state.measure()

        if self.quantum_encoding == "angle":
            # For angle encoding, extract angle from amplitudes
            if len(quantum_state.amplitudes) >= 2:
                angle = 2 * np.arctan2(
                    quantum_state.amplitudes[1], quantum_state.amplitudes[0]
                )
                normalized_value = angle / (2 * np.pi)
            else:
                normalized_value = 0.5
        else:
            # For binary/gray encoding
            if self.quantum_encoding == "gray":
                # Convert gray back to binary
                binary_value = measured_state
                gray_value = binary_value
                binary_value = gray_value
                mask = gray_value >> 1
                while mask:
                    binary_value ^= mask
                    mask >>= 1
                measured_state = binary_value

            normalized_value = measured_state / (2**self.precision_bits - 1)

        min_val, max_val = self.value_range
        return min_val + normalized_value * (max_val - min_val)


class QuantumGeneticAlgorithm:
    """Quantum-inspired genetic algorithm for optimization."""

    def __init__(
        self,
        population_size: int = 50,
        num_generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
    ):
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.logger = logging.getLogger(__name__)

    async def optimize(
        self,
        parameters: List[OptimizationParameter],
        objective_function: Callable[[Dict[str, float]], float],
        objective_type: OptimizationObjective = OptimizationObjective.MINIMIZE_LATENCY,
    ) -> Dict[str, Any]:
        """
        Perform quantum-inspired genetic algorithm optimization.

        Args:
            parameters: List of parameters to optimize
            objective_function: Function to optimize
            objective_type: Type of optimization objective

        Returns:
            Optimization results including best parameters and convergence history
        """
        self.logger.info(
            f"Starting quantum genetic algorithm optimization with {len(parameters)} parameters"
        )

        # Initialize quantum population
        population = self._initialize_quantum_population(parameters)

        best_fitness_history = []
        best_individual = None
        best_fitness = (
            float("inf") if "minimize" in objective_type.value else float("-inf")
        )

        for generation in range(self.num_generations):
            # Evaluate fitness for each individual
            fitness_scores = await self._evaluate_population_fitness(
                population, parameters, objective_function
            )

            # Update best individual
            current_best_idx = self._get_best_individual_index(
                fitness_scores, objective_type
            )
            current_best_fitness = fitness_scores[current_best_idx]

            if self._is_better_fitness(
                current_best_fitness, best_fitness, objective_type
            ):
                best_fitness = current_best_fitness
                best_individual = population[current_best_idx].copy()

            best_fitness_history.append(best_fitness)

            # Log progress
            if generation % 10 == 0:
                self.logger.info(
                    f"Generation {generation}: Best fitness = {best_fitness:.6f}"
                )

            # Create next generation
            population = await self._create_next_generation(
                population, fitness_scores, parameters, objective_type
            )

        # Decode best individual to parameter values
        best_parameters = self._decode_quantum_individual(best_individual, parameters)

        return {
            "best_parameters": best_parameters,
            "best_fitness": best_fitness,
            "convergence_history": best_fitness_history,
            "generations_completed": self.num_generations,
            "optimization_method": "quantum_genetic_algorithm",
        }

    def _initialize_quantum_population(
        self, parameters: List[OptimizationParameter]
    ) -> List[List[QuantumState]]:
        """Initialize population of quantum individuals."""
        population = []

        for _ in range(self.population_size):
            individual = []
            for param in parameters:
                # Create random quantum state for each parameter
                if param.quantum_encoding == "angle":
                    # Random angle encoding
                    angle = np.random.uniform(0, 2 * np.pi)
                    amplitudes = np.array([np.cos(angle / 2), np.sin(angle / 2)])
                    phases = np.random.uniform(0, 2 * np.pi, 2)
                else:
                    # Random superposition state
                    amplitudes = np.random.random(2**param.precision_bits)
                    amplitudes = amplitudes / np.linalg.norm(amplitudes)
                    phases = np.random.uniform(0, 2 * np.pi, len(amplitudes))

                quantum_state = QuantumState(
                    amplitudes=amplitudes,
                    phases=phases,
                    entanglement_matrix=np.eye(len(amplitudes)),
                )
                individual.append(quantum_state)

            population.append(individual)

        return population

    async def _evaluate_population_fitness(
        self,
        population: List[List[QuantumState]],
        parameters: List[OptimizationParameter],
        objective_function: Callable[[Dict[str, float]], float],
    ) -> List[float]:
        """Evaluate fitness for entire population."""
        fitness_scores = []

        # Use thread pool for parallel evaluation
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []

            for individual in population:
                future = executor.submit(
                    self._evaluate_individual_fitness,
                    individual,
                    parameters,
                    objective_function,
                )
                futures.append(future)

            # Collect results
            for future in futures:
                fitness = future.result()
                fitness_scores.append(fitness)

        return fitness_scores

    def _evaluate_individual_fitness(
        self,
        individual: List[QuantumState],
        parameters: List[OptimizationParameter],
        objective_function: Callable[[Dict[str, float]], float],
    ) -> float:
        """Evaluate fitness for a single individual."""
        # Decode quantum individual to parameter values
        param_values = self._decode_quantum_individual(individual, parameters)

        try:
            # Evaluate objective function
            fitness = objective_function(param_values)
            return fitness
        except Exception as e:
            self.logger.warning(f"Error evaluating individual: {str(e)}")
            return float("inf")  # Penalize invalid solutions

    def _decode_quantum_individual(
        self, individual: List[QuantumState], parameters: List[OptimizationParameter]
    ) -> Dict[str, float]:
        """Decode quantum individual to parameter values."""
        param_values = {}

        for i, (quantum_state, param) in enumerate(zip(individual, parameters)):
            value = param.decode_quantum(quantum_state)
            param_values[param.name] = value

        return param_values

    def _get_best_individual_index(
        self, fitness_scores: List[float], objective_type: OptimizationObjective
    ) -> int:
        """Get index of best individual based on objective type."""
        if "minimize" in objective_type.value:
            return np.argmin(fitness_scores)
        else:
            return np.argmax(fitness_scores)

    def _is_better_fitness(
        self,
        new_fitness: float,
        current_best: float,
        objective_type: OptimizationObjective,
    ) -> bool:
        """Check if new fitness is better than current best."""
        if "minimize" in objective_type.value:
            return new_fitness < current_best
        else:
            return new_fitness > current_best

    async def _create_next_generation(
        self,
        population: List[List[QuantumState]],
        fitness_scores: List[float],
        parameters: List[OptimizationParameter],
        objective_type: OptimizationObjective,
    ) -> List[List[QuantumState]]:
        """Create next generation using quantum genetic operators."""
        new_population = []

        # Elitism: keep best individual
        best_idx = self._get_best_individual_index(fitness_scores, objective_type)
        new_population.append(population[best_idx].copy())

        # Selection and reproduction
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(
                population, fitness_scores, objective_type
            )
            parent2 = self._tournament_selection(
                population, fitness_scores, objective_type
            )

            # Quantum crossover
            if np.random.random() < self.crossover_rate:
                child1, child2 = self._quantum_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Quantum mutation
            if np.random.random() < self.mutation_rate:
                child1 = self._quantum_mutation(child1, parameters)
            if np.random.random() < self.mutation_rate:
                child2 = self._quantum_mutation(child2, parameters)

            new_population.extend([child1, child2])

        # Trim to exact population size
        return new_population[: self.population_size]

    def _tournament_selection(
        self,
        population: List[List[QuantumState]],
        fitness_scores: List[float],
        objective_type: OptimizationObjective,
        tournament_size: int = 3,
    ) -> List[QuantumState]:
        """Tournament selection for parent selection."""
        tournament_indices = np.random.choice(
            len(population), size=tournament_size, replace=False
        )
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]

        if "minimize" in objective_type.value:
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        else:
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]

        return population[winner_idx].copy()

    def _quantum_crossover(
        self, parent1: List[QuantumState], parent2: List[QuantumState]
    ) -> Tuple[List[QuantumState], List[QuantumState]]:
        """Quantum crossover operation."""
        child1, child2 = [], []

        for q1, q2 in zip(parent1, parent2):
            # Quantum interference-based crossover
            alpha = np.random.random()
            beta = np.sqrt(1 - alpha**2)

            # Create entangled children through quantum superposition
            child1_amplitudes = alpha * q1.amplitudes + beta * q2.amplitudes
            child2_amplitudes = beta * q1.amplitudes - alpha * q2.amplitudes

            # Normalize
            child1_amplitudes = child1_amplitudes / np.linalg.norm(child1_amplitudes)
            child2_amplitudes = child2_amplitudes / np.linalg.norm(child2_amplitudes)

            # Combine phases
            child1_phases = (q1.phases + q2.phases) / 2
            child2_phases = (q1.phases - q2.phases) / 2

            child1_state = QuantumState(
                amplitudes=child1_amplitudes,
                phases=child1_phases,
                entanglement_matrix=q1.entanglement_matrix,
            )

            child2_state = QuantumState(
                amplitudes=child2_amplitudes,
                phases=child2_phases,
                entanglement_matrix=q2.entanglement_matrix,
            )

            child1.append(child1_state)
            child2.append(child2_state)

        return child1, child2

    def _quantum_mutation(
        self, individual: List[QuantumState], parameters: List[OptimizationParameter]
    ) -> List[QuantumState]:
        """Quantum mutation operation."""
        mutated_individual = []

        for quantum_state, param in zip(individual, parameters):
            # Apply quantum rotation (mutation)
            mutation_strength = 0.1  # Adjustable mutation strength
            rotation_angle = np.random.normal(0, mutation_strength)

            # Create rotation matrix
            cos_theta = np.cos(rotation_angle)
            sin_theta = np.sin(rotation_angle)

            if param.quantum_encoding == "angle":
                # For angle encoding, apply rotation to amplitudes
                rotation_matrix = np.array(
                    [[cos_theta, -sin_theta], [sin_theta, cos_theta]]
                )
                new_amplitudes = rotation_matrix @ quantum_state.amplitudes
            else:
                # For binary/gray encoding, apply phase rotation
                new_amplitudes = quantum_state.amplitudes.copy()
                phase_shift = np.random.uniform(0, 2 * np.pi)
                new_phases = quantum_state.phases + phase_shift

            mutated_state = QuantumState(
                amplitudes=new_amplitudes,
                phases=(
                    quantum_state.phases
                    if param.quantum_encoding == "angle"
                    else new_phases
                ),
                entanglement_matrix=quantum_state.entanglement_matrix,
            ).normalize()

            mutated_individual.append(mutated_state)

        return mutated_individual


class QuantumAnnealingOptimizer:
    """Quantum annealing-inspired optimizer for combinatorial problems."""

    def __init__(
        self,
        num_sweeps: int = 1000,
        initial_temperature: float = 10.0,
        final_temperature: float = 0.01,
    ):
        self.num_sweeps = num_sweeps
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.logger = logging.getLogger(__name__)

    async def optimize(
        self,
        parameters: List[OptimizationParameter],
        objective_function: Callable[[Dict[str, float]], float],
        objective_type: OptimizationObjective = OptimizationObjective.MINIMIZE_LATENCY,
    ) -> Dict[str, Any]:
        """
        Perform quantum annealing optimization.

        Args:
            parameters: List of parameters to optimize
            objective_function: Function to optimize
            objective_type: Type of optimization objective

        Returns:
            Optimization results
        """
        self.logger.info(f"Starting quantum annealing optimization")

        # Initialize random solution
        current_solution = self._initialize_random_solution(parameters)
        current_energy = await self._evaluate_energy(
            current_solution, objective_function
        )

        best_solution = current_solution.copy()
        best_energy = current_energy

        energy_history = []
        temperature_schedule = self._create_temperature_schedule()

        for sweep in range(self.num_sweeps):
            temperature = temperature_schedule[sweep]

            # Perform one Monte Carlo sweep
            for param in parameters:
                # Propose local change
                new_solution = self._propose_local_change(current_solution, param)
                new_energy = await self._evaluate_energy(
                    new_solution, objective_function
                )

                # Accept or reject based on quantum annealing criteria
                if self._accept_transition(current_energy, new_energy, temperature):
                    current_solution = new_solution
                    current_energy = new_energy

                    # Update best solution
                    if self._is_better_energy(new_energy, best_energy, objective_type):
                        best_solution = new_solution.copy()
                        best_energy = new_energy

            energy_history.append(current_energy)

            # Log progress
            if sweep % 100 == 0:
                self.logger.info(
                    f"Sweep {sweep}: Energy = {current_energy:.6f}, T = {temperature:.6f}"
                )

        return {
            "best_parameters": best_solution,
            "best_energy": best_energy,
            "energy_history": energy_history,
            "sweeps_completed": self.num_sweeps,
            "optimization_method": "quantum_annealing",
        }

    def _initialize_random_solution(
        self, parameters: List[OptimizationParameter]
    ) -> Dict[str, float]:
        """Initialize random solution."""
        solution = {}
        for param in parameters:
            min_val, max_val = param.value_range
            solution[param.name] = np.random.uniform(min_val, max_val)
        return solution

    async def _evaluate_energy(
        self,
        solution: Dict[str, float],
        objective_function: Callable[[Dict[str, float]], float],
    ) -> float:
        """Evaluate energy (objective function) for solution."""
        try:
            return objective_function(solution)
        except Exception as e:
            self.logger.warning(f"Error evaluating energy: {str(e)}")
            return float("inf")

    def _create_temperature_schedule(self) -> np.ndarray:
        """Create annealing temperature schedule."""
        # Exponential cooling schedule
        schedule = np.exp(
            np.linspace(
                np.log(self.initial_temperature),
                np.log(self.final_temperature),
                self.num_sweeps,
            )
        )
        return schedule

    def _propose_local_change(
        self, current_solution: Dict[str, float], parameter: OptimizationParameter
    ) -> Dict[str, float]:
        """Propose local change to solution."""
        new_solution = current_solution.copy()

        # Random perturbation
        min_val, max_val = parameter.value_range
        perturbation_range = (max_val - min_val) * 0.1  # 10% of range

        perturbation = np.random.normal(0, perturbation_range)
        new_value = current_solution[parameter.name] + perturbation

        # Ensure within bounds
        new_value = max(min_val, min(max_val, new_value))
        new_solution[parameter.name] = new_value

        return new_solution

    def _accept_transition(
        self, current_energy: float, new_energy: float, temperature: float
    ) -> bool:
        """Decide whether to accept transition based on quantum annealing."""
        if new_energy < current_energy:
            return True  # Always accept improvements

        # Quantum tunneling probability
        energy_diff = new_energy - current_energy
        if temperature > 0:
            tunneling_probability = np.exp(-energy_diff / temperature)
            return np.random.random() < tunneling_probability

        return False

    def _is_better_energy(
        self,
        new_energy: float,
        current_best: float,
        objective_type: OptimizationObjective,
    ) -> bool:
        """Check if new energy is better than current best."""
        if "minimize" in objective_type.value:
            return new_energy < current_best
        else:
            return new_energy > current_best


class QuantumInspiredOptimizer:
    """Main quantum-inspired optimization engine."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.optimization_history = []

    async def optimize_edge_profiling(
        self,
        optimization_config: Dict[str, Any],
        objective_function: Callable[[Dict[str, float]], float],
        method: QuantumOptimizationMethod = QuantumOptimizationMethod.QUANTUM_GENETIC_ALGORITHM,
        objective_type: OptimizationObjective = OptimizationObjective.PARETO_OPTIMAL,
    ) -> Dict[str, Any]:
        """
        Optimize edge AI profiling parameters using quantum-inspired methods.

        Args:
            optimization_config: Configuration with parameters to optimize
            objective_function: Function to optimize
            method: Quantum optimization method to use
            objective_type: Type of optimization objective

        Returns:
            Optimization results
        """
        self.logger.info(f"Starting quantum-inspired optimization using {method.value}")

        # Parse optimization parameters
        parameters = self._parse_optimization_config(optimization_config)

        # Select optimization algorithm
        if method == QuantumOptimizationMethod.QUANTUM_GENETIC_ALGORITHM:
            optimizer = QuantumGeneticAlgorithm()
        elif method == QuantumOptimizationMethod.QUANTUM_ANNEALING:
            optimizer = QuantumAnnealingOptimizer()
        else:
            raise ValueError(f"Unsupported optimization method: {method}")

        # Run optimization
        start_time = time.time()
        results = await optimizer.optimize(
            parameters, objective_function, objective_type
        )
        optimization_time = time.time() - start_time

        # Enhance results with additional analysis
        enhanced_results = {
            **results,
            "optimization_time_seconds": optimization_time,
            "quantum_method": method.value,
            "objective_type": objective_type.value,
            "parameter_count": len(parameters),
            "quantum_advantage_analysis": self._analyze_quantum_advantage(results),
            "recommendations": self._generate_optimization_recommendations(results),
        }

        # Store in history
        self.optimization_history.append(enhanced_results)

        return enhanced_results

    def _parse_optimization_config(
        self, config: Dict[str, Any]
    ) -> List[OptimizationParameter]:
        """Parse optimization configuration into parameters."""
        parameters = []

        for param_name, param_config in config.get("parameters", {}).items():
            parameter = OptimizationParameter(
                name=param_name,
                value_range=tuple(param_config["range"]),
                current_value=param_config.get(
                    "initial_value", sum(param_config["range"]) / 2
                ),  # Default to midpoint
                quantum_encoding=param_config.get("encoding", "binary"),
                precision_bits=param_config.get("precision_bits", 8),
            )
            parameters.append(parameter)

        return parameters

    def _analyze_quantum_advantage(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze potential quantum advantage achieved."""
        convergence_history = results.get("convergence_history", [])

        if not convergence_history:
            return {"quantum_advantage": "unknown"}

        # Analyze convergence speed
        initial_fitness = convergence_history[0]
        final_fitness = convergence_history[-1]
        improvement_ratio = (
            abs(final_fitness - initial_fitness) / abs(initial_fitness)
            if initial_fitness != 0
            else 0
        )

        # Analyze convergence stability
        if len(convergence_history) > 10:
            late_stage_variance = np.var(convergence_history[-10:])
            early_stage_variance = np.var(convergence_history[:10])
            stability_improvement = (
                early_stage_variance / late_stage_variance
                if late_stage_variance > 0
                else float("inf")
            )
        else:
            stability_improvement = 1.0

        return {
            "improvement_ratio": improvement_ratio,
            "stability_improvement": stability_improvement,
            "convergence_efficiency": len(convergence_history)
            / max(1, improvement_ratio),
            "quantum_advantage_score": min(
                1.0, improvement_ratio * stability_improvement / 10.0
            ),
        }

    def _generate_optimization_recommendations(
        self, results: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on optimization results."""
        recommendations = []

        best_parameters = results.get("best_parameters", {})
        quantum_advantage = results.get("quantum_advantage_analysis", {})

        # Parameter-specific recommendations
        for param_name, param_value in best_parameters.items():
            if "timeout" in param_name.lower():
                if param_value > 5.0:
                    recommendations.append(
                        f"Consider reducing {param_name} timeout if system stability allows"
                    )
            elif "memory" in param_name.lower():
                if param_value > 400:
                    recommendations.append(
                        f"High {param_name} value may require memory optimization"
                    )

        # Quantum advantage recommendations
        qa_score = quantum_advantage.get("quantum_advantage_score", 0)
        if qa_score > 0.7:
            recommendations.append(
                "Strong quantum advantage observed - consider this configuration for production"
            )
        elif qa_score < 0.3:
            recommendations.append(
                "Limited quantum advantage - classical optimization may be sufficient"
            )

        # Convergence recommendations
        convergence_history = results.get("convergence_history", [])
        if len(convergence_history) > 50:
            final_improvement = abs(
                convergence_history[-1] - convergence_history[-10]
            ) / abs(convergence_history[-10])
            if final_improvement < 0.01:
                recommendations.append(
                    "Optimization converged - consider early stopping for efficiency"
                )

        return recommendations

    async def multi_objective_optimization(
        self,
        optimization_config: Dict[str, Any],
        objective_functions: Dict[str, Callable[[Dict[str, float]], float]],
        method: QuantumOptimizationMethod = QuantumOptimizationMethod.QUANTUM_GENETIC_ALGORITHM,
    ) -> Dict[str, Any]:
        """
        Perform multi-objective optimization for Pareto-optimal solutions.

        Args:
            optimization_config: Configuration with parameters to optimize
            objective_functions: Dictionary of objective functions
            method: Quantum optimization method to use

        Returns:
            Pareto front and optimization results
        """
        self.logger.info("Starting multi-objective quantum optimization")

        # Create combined objective function for Pareto optimization
        def combined_objective(params: Dict[str, float]) -> float:
            objective_values = []
            for obj_name, obj_func in objective_functions.items():
                try:
                    value = obj_func(params)
                    objective_values.append(value)
                except Exception as e:
                    self.logger.warning(f"Error in objective {obj_name}: {str(e)}")
                    objective_values.append(float("inf"))

            # Use weighted sum approach (could be enhanced with true Pareto ranking)
            weights = [1.0 / len(objective_values)] * len(objective_values)
            return sum(w * v for w, v in zip(weights, objective_values))

        # Run optimization
        results = await self.optimize_edge_profiling(
            optimization_config,
            combined_objective,
            method,
            OptimizationObjective.PARETO_OPTIMAL,
        )

        # Enhance with Pareto analysis
        results["pareto_analysis"] = await self._analyze_pareto_front(
            optimization_config, objective_functions, results
        )

        return results

    async def _analyze_pareto_front(
        self,
        optimization_config: Dict[str, Any],
        objective_functions: Dict[str, Callable[[Dict[str, float]], float]],
        optimization_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze Pareto front for multi-objective optimization."""
        # This would implement full Pareto front analysis
        # For now, return a simplified analysis

        best_params = optimization_results.get("best_parameters", {})

        # Evaluate all objectives at best point
        objective_values = {}
        for obj_name, obj_func in objective_functions.items():
            try:
                value = obj_func(best_params)
                objective_values[obj_name] = value
            except Exception as e:
                objective_values[obj_name] = float("inf")

        return {
            "pareto_point": {
                "parameters": best_params,
                "objective_values": objective_values,
            },
            "trade_off_analysis": self._analyze_trade_offs(objective_values),
            "pareto_efficiency": self._calculate_pareto_efficiency(objective_values),
        }

    def _analyze_trade_offs(self, objective_values: Dict[str, float]) -> Dict[str, str]:
        """Analyze trade-offs between objectives."""
        trade_offs = {}

        obj_names = list(objective_values.keys())
        for i, obj1 in enumerate(obj_names):
            for obj2 in obj_names[i + 1 :]:
                val1, val2 = objective_values[obj1], objective_values[obj2]

                # Simple trade-off analysis
                if val1 < 100 and val2 > 200:
                    trade_offs[f"{obj1}_vs_{obj2}"] = (
                        f"Low {obj1} achieved at cost of high {obj2}"
                    )
                elif val1 > 200 and val2 < 100:
                    trade_offs[f"{obj1}_vs_{obj2}"] = (
                        f"Low {obj2} achieved at cost of high {obj1}"
                    )
                else:
                    trade_offs[f"{obj1}_vs_{obj2}"] = "Balanced trade-off achieved"

        return trade_offs

    def _calculate_pareto_efficiency(self, objective_values: Dict[str, float]) -> float:
        """Calculate Pareto efficiency score."""
        # Simplified efficiency calculation
        # In practice, this would compare against true Pareto front
        values = list(objective_values.values())
        normalized_values = [
            (
                (v - min(values)) / (max(values) - min(values))
                if max(values) != min(values)
                else 0
            )
            for v in values
        ]
        return 1.0 - np.mean(
            normalized_values
        )  # Higher score for lower normalized values


# Global quantum optimizer instance
_global_quantum_optimizer: Optional[QuantumInspiredOptimizer] = None


def get_quantum_optimizer() -> QuantumInspiredOptimizer:
    """Get global quantum optimizer instance."""
    global _global_quantum_optimizer
    if _global_quantum_optimizer is None:
        _global_quantum_optimizer = QuantumInspiredOptimizer()
    return _global_quantum_optimizer


async def optimize_with_quantum(
    optimization_config: Dict[str, Any],
    objective_function: Callable[[Dict[str, float]], float],
    method: QuantumOptimizationMethod = QuantumOptimizationMethod.QUANTUM_GENETIC_ALGORITHM,
    objective_type: OptimizationObjective = OptimizationObjective.MINIMIZE_LATENCY,
) -> Dict[str, Any]:
    """Convenience function for quantum optimization."""
    optimizer = get_quantum_optimizer()
    return await optimizer.optimize_edge_profiling(
        optimization_config, objective_function, method, objective_type
    )
