"""
Self-Evolving Optimization Strategies
====================================

Advanced self-improving optimization system that learns from experience,
adapts strategies dynamically, and evolves optimization approaches autonomously.

Generation 5+ Features:
- Self-modifying optimization algorithms that improve over time
- Evolutionary strategy mutation and selection
- Meta-learning for optimization method selection
- Autonomous hyperparameter evolution
- Performance-driven strategy adaptation
- Multi-generational optimization improvement
"""

import asyncio
import json
import logging
import math
import pickle
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, differential_evolution
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib

from .profiler import EdgeProfiler, ProfileResults  
from .models import QuantizedModel
from .quantum_enhanced_profiling import QuantumInspiredOptimizer, QuantumOptimizationMethod
from .advanced_monitoring import MonitoringSystem


class EvolutionStrategy(Enum):
    """Evolution strategies for optimization improvement."""
    
    GENETIC_PROGRAMMING = "genetic_programming"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    PARTICLE_SWARM = "particle_swarm"
    SIMULATED_ANNEALING = "simulated_annealing"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    NEURAL_EVOLUTION = "neural_evolution"
    HYBRID_EVOLUTION = "hybrid_evolution"


class AdaptationType(Enum):
    """Types of adaptation mechanisms."""
    
    PARAMETER_TUNING = "parameter_tuning"
    ALGORITHM_SELECTION = "algorithm_selection" 
    STRATEGY_MUTATION = "strategy_mutation"
    ENSEMBLE_WEIGHTING = "ensemble_weighting"
    META_LEARNING = "meta_learning"


class OptimizationGenome:
    """Represents an optimization strategy as an evolving genome."""
    
    def __init__(
        self,
        genome_id: str,
        strategy_type: EvolutionStrategy,
        hyperparameters: Dict[str, Any],
        algorithm_weights: Dict[str, float],
        generation: int = 0
    ):
        self.genome_id = genome_id
        self.strategy_type = strategy_type
        self.hyperparameters = hyperparameters.copy()
        self.algorithm_weights = algorithm_weights.copy()
        self.generation = generation
        
        # Performance tracking
        self.fitness_scores: List[float] = []
        self.optimization_times: List[float] = []
        self.success_rate: float = 0.0
        self.adaptation_count: int = 0
        
        # Genetic material
        self.mutation_rate: float = 0.1
        self.crossover_rate: float = 0.7
        self.parent_genomes: List[str] = []
        
        # Learning state
        self.experience_buffer: deque = deque(maxlen=100)
        self.specialized_domains: Set[str] = set()
        
    def calculate_fitness(self, performance_metrics: Dict[str, float]) -> float:
        """Calculate fitness score based on performance metrics."""
        
        # Multi-objective fitness calculation
        latency_score = 1.0 / (1.0 + performance_metrics.get('latency_ms', 100) / 100.0)
        energy_score = 1.0 / (1.0 + performance_metrics.get('energy_mj', 10) / 10.0)
        accuracy_score = performance_metrics.get('accuracy', 0.8)
        convergence_score = 1.0 / (1.0 + performance_metrics.get('convergence_time', 60) / 60.0)
        
        # Weighted combination
        fitness = (
            0.3 * latency_score +
            0.25 * energy_score +
            0.25 * accuracy_score +
            0.2 * convergence_score
        )
        
        self.fitness_scores.append(fitness)
        return fitness
    
    def mutate(self, mutation_intensity: float = 1.0) -> 'OptimizationGenome':
        """Create a mutated version of this genome."""
        
        mutated_hyperparams = {}
        for param, value in self.hyperparameters.items():
            if isinstance(value, (int, float)):
                # Gaussian mutation
                noise = np.random.normal(0, self.mutation_rate * mutation_intensity)
                if isinstance(value, int):
                    mutated_value = max(1, int(value * (1 + noise)))
                else:
                    mutated_value = max(0.001, value * (1 + noise))
                mutated_hyperparams[param] = mutated_value
            elif isinstance(value, bool):
                # Binary mutation
                if np.random.random() < self.mutation_rate * mutation_intensity:
                    mutated_hyperparams[param] = not value
                else:
                    mutated_hyperparams[param] = value
            else:
                mutated_hyperparams[param] = value
        
        # Mutate algorithm weights
        mutated_weights = {}
        for alg, weight in self.algorithm_weights.items():
            noise = np.random.normal(0, self.mutation_rate * mutation_intensity)
            mutated_weights[alg] = max(0.001, weight * (1 + noise))
        
        # Normalize weights
        total_weight = sum(mutated_weights.values())
        if total_weight > 0:
            mutated_weights = {k: v/total_weight for k, v in mutated_weights.items()}
        
        # Create mutated genome
        mutated_genome = OptimizationGenome(
            genome_id=f"{self.genome_id}_m{self.adaptation_count}",
            strategy_type=self.strategy_type,
            hyperparameters=mutated_hyperparams,
            algorithm_weights=mutated_weights,
            generation=self.generation + 1
        )
        
        mutated_genome.parent_genomes = [self.genome_id]
        mutated_genome.mutation_rate = min(0.5, self.mutation_rate * (1 + np.random.normal(0, 0.1)))
        
        return mutated_genome
    
    def crossover(self, other: 'OptimizationGenome') -> Tuple['OptimizationGenome', 'OptimizationGenome']:
        """Perform crossover with another genome."""
        
        if np.random.random() > self.crossover_rate:
            return self, other
        
        # Parameter crossover
        child1_params = {}
        child2_params = {}
        
        for param in set(self.hyperparameters.keys()) | set(other.hyperparameters.keys()):
            if np.random.random() < 0.5:
                child1_params[param] = self.hyperparameters.get(param, other.hyperparameters.get(param))
                child2_params[param] = other.hyperparameters.get(param, self.hyperparameters.get(param))
            else:
                child1_params[param] = other.hyperparameters.get(param, self.hyperparameters.get(param))
                child2_params[param] = self.hyperparameters.get(param, other.hyperparameters.get(param))
        
        # Weight crossover (blend)
        child1_weights = {}
        child2_weights = {}
        alpha = np.random.uniform(0.3, 0.7)
        
        for alg in set(self.algorithm_weights.keys()) | set(other.algorithm_weights.keys()):
            w1 = self.algorithm_weights.get(alg, 0.0)
            w2 = other.algorithm_weights.get(alg, 0.0)
            
            child1_weights[alg] = alpha * w1 + (1 - alpha) * w2
            child2_weights[alg] = (1 - alpha) * w1 + alpha * w2
        
        # Create children
        child1 = OptimizationGenome(
            genome_id=f"{self.genome_id}_c1_{random.randint(1000, 9999)}",
            strategy_type=self.strategy_type if np.random.random() < 0.5 else other.strategy_type,
            hyperparameters=child1_params,
            algorithm_weights=child1_weights,
            generation=max(self.generation, other.generation) + 1
        )
        
        child2 = OptimizationGenome(
            genome_id=f"{other.genome_id}_c2_{random.randint(1000, 9999)}",
            strategy_type=other.strategy_type if np.random.random() < 0.5 else self.strategy_type,
            hyperparameters=child2_params,
            algorithm_weights=child2_weights,
            generation=max(self.generation, other.generation) + 1
        )
        
        # Set parentage
        child1.parent_genomes = [self.genome_id, other.genome_id]
        child2.parent_genomes = [self.genome_id, other.genome_id]
        
        return child1, child2
    
    def get_current_fitness(self) -> float:
        """Get current fitness score."""
        return np.mean(self.fitness_scores) if self.fitness_scores else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'genome_id': self.genome_id,
            'strategy_type': self.strategy_type.value,
            'hyperparameters': self.hyperparameters,
            'algorithm_weights': self.algorithm_weights,
            'generation': self.generation,
            'fitness_scores': self.fitness_scores,
            'success_rate': self.success_rate,
            'adaptation_count': self.adaptation_count,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'parent_genomes': self.parent_genomes,
            'specialized_domains': list(self.specialized_domains)
        }


@dataclass
class EvolutionaryOptimizationResult:
    """Result of evolutionary optimization."""
    
    best_genome: OptimizationGenome
    final_fitness: float
    generations_evolved: int
    total_evaluations: int
    convergence_history: List[float]
    evolution_time: float
    population_diversity: float
    improvement_rate: float


class MetaLearningPredictor:
    """Meta-learner for predicting optimization performance."""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'neural_network': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42)
        }
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.training_history = []
    
    def extract_features(
        self,
        genome: OptimizationGenome,
        problem_context: Dict[str, Any]
    ) -> np.ndarray:
        """Extract features for meta-learning."""
        
        features = []
        
        # Genome features
        features.extend([
            genome.generation,
            genome.get_current_fitness(),
            genome.success_rate,
            genome.adaptation_count,
            genome.mutation_rate,
            genome.crossover_rate,
            len(genome.parent_genomes),
            len(genome.specialized_domains)
        ])
        
        # Strategy type encoding
        strategy_encoding = [0] * len(EvolutionStrategy)
        strategy_index = list(EvolutionStrategy).index(genome.strategy_type)
        strategy_encoding[strategy_index] = 1
        features.extend(strategy_encoding)
        
        # Hyperparameter features
        hyperparam_features = []
        for key in sorted(genome.hyperparameters.keys()):
            value = genome.hyperparameters[key]
            if isinstance(value, (int, float)):
                hyperparam_features.append(float(value))
            elif isinstance(value, bool):
                hyperparam_features.append(float(value))
        
        # Pad or truncate to fixed size
        target_size = 20
        if len(hyperparam_features) < target_size:
            hyperparam_features.extend([0.0] * (target_size - len(hyperparam_features)))
        else:
            hyperparam_features = hyperparam_features[:target_size]
        
        features.extend(hyperparam_features)
        
        # Problem context features
        features.extend([
            problem_context.get('problem_dimension', 10),
            problem_context.get('constraint_count', 0),
            problem_context.get('multi_objective', False),
            problem_context.get('noisy', False),
            problem_context.get('dynamic', False)
        ])
        
        return np.array(features)
    
    def train(
        self,
        training_data: List[Tuple[OptimizationGenome, Dict[str, Any], float]]
    ) -> Dict[str, float]:
        """Train meta-learning models."""
        
        if len(training_data) < 10:
            return {'error': 'Insufficient training data'}
        
        # Prepare training data
        X = []
        y = []
        
        for genome, context, performance in training_data:
            features = self.extract_features(genome, context)
            X.append(features)
            y.append(performance)
        
        X = np.array(X)
        y = np.array(y)
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models
        model_scores = {}
        
        for name, model in self.models.items():
            try:
                # Cross-validation
                scores = cross_val_score(model, X_scaled, y, cv=min(5, len(y)//2))
                model_scores[name] = np.mean(scores)
                
                # Train on full dataset
                model.fit(X_scaled, y)
                
            except Exception as e:
                model_scores[name] = -1.0
                logging.warning(f"Failed to train {name}: {e}")
        
        self.is_trained = True
        self.training_history.append({
            'timestamp': datetime.now(),
            'samples': len(training_data),
            'scores': model_scores
        })
        
        return model_scores
    
    def predict_performance(
        self,
        genome: OptimizationGenome,
        problem_context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Predict optimization performance."""
        
        if not self.is_trained:
            return {'prediction': 0.5, 'confidence': 0.0}
        
        features = self.extract_features(genome, problem_context)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        predictions = {}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(features_scaled)[0]
                predictions[name] = max(0.0, min(1.0, pred))  # Clamp to [0, 1]
            except Exception:
                predictions[name] = 0.5
        
        # Ensemble prediction
        ensemble_pred = np.mean(list(predictions.values()))
        prediction_std = np.std(list(predictions.values()))
        confidence = max(0.0, 1.0 - prediction_std)
        
        return {
            'prediction': ensemble_pred,
            'confidence': confidence,
            'individual_predictions': predictions
        }


class SelfEvolvingOptimizer:
    """Self-evolving optimization system."""
    
    def __init__(
        self,
        population_size: int = 20,
        elite_size: int = 4,
        max_generations: int = 50,
        convergence_threshold: float = 1e-6
    ):
        self.population_size = population_size
        self.elite_size = elite_size
        self.max_generations = max_generations
        self.convergence_threshold = convergence_threshold
        
        # Evolution state
        self.population: List[OptimizationGenome] = []
        self.generation = 0
        self.evolution_history: List[Dict] = []
        self.best_genome: Optional[OptimizationGenome] = None
        
        # Meta-learning
        self.meta_predictor = MetaLearningPredictor()
        self.performance_database: List[Tuple] = []
        
        # Adaptation mechanisms
        self.adaptation_strategies = {
            AdaptationType.PARAMETER_TUNING: self._adapt_parameters,
            AdaptationType.ALGORITHM_SELECTION: self._adapt_algorithm_selection,
            AdaptationType.STRATEGY_MUTATION: self._adapt_strategy_mutation,
            AdaptationType.ENSEMBLE_WEIGHTING: self._adapt_ensemble_weights,
            AdaptationType.META_LEARNING: self._adapt_meta_learning
        }
        
        # Performance tracking
        self.optimization_stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'average_improvement': 0.0,
            'best_fitness_achieved': 0.0,
            'evolution_efficiency': 0.0
        }
        
        self.logger = logging.getLogger(__name__)
        self.monitoring = MonitoringSystem()
    
    def initialize_population(self) -> None:
        """Initialize the optimization population."""
        
        self.population = []
        
        # Create diverse initial population
        strategies = list(EvolutionStrategy)
        
        for i in range(self.population_size):
            strategy = strategies[i % len(strategies)]
            
            # Generate diverse hyperparameters
            hyperparams = self._generate_initial_hyperparameters(strategy)
            
            # Generate algorithm weights
            algorithm_weights = self._generate_initial_algorithm_weights()
            
            genome = OptimizationGenome(
                genome_id=f"gen0_ind{i:03d}",
                strategy_type=strategy,
                hyperparameters=hyperparams,
                algorithm_weights=algorithm_weights,
                generation=0
            )
            
            self.population.append(genome)
        
        self.logger.info(f"Initialized population with {len(self.population)} individuals")
    
    async def evolve_optimization_strategies(
        self,
        objective_function: Callable[[Dict[str, Any]], float],
        problem_context: Dict[str, Any],
        max_evaluations: int = 1000
    ) -> EvolutionaryOptimizationResult:
        """Evolve optimization strategies for a given problem."""
        
        if not self.population:
            self.initialize_population()
        
        start_time = time.time()
        evaluations_used = 0
        convergence_history = []
        
        for generation in range(self.max_generations):
            self.generation = generation
            
            # Evaluate population
            fitness_scores = []
            for genome in self.population:
                if evaluations_used >= max_evaluations:
                    break
                
                # Use genome to configure optimization
                optimization_config = self._genome_to_config(genome)
                
                try:
                    # Evaluate fitness
                    performance = await self._evaluate_genome_fitness(
                        genome, optimization_config, objective_function
                    )
                    
                    fitness = genome.calculate_fitness(performance)
                    fitness_scores.append(fitness)
                    
                    # Store for meta-learning
                    self.performance_database.append((genome, problem_context, fitness))
                    
                    evaluations_used += 1
                    
                except Exception as e:
                    self.logger.warning(f"Failed to evaluate genome {genome.genome_id}: {e}")
                    fitness_scores.append(0.0)
            
            # Track convergence
            if fitness_scores:
                best_fitness = max(fitness_scores)
                avg_fitness = np.mean(fitness_scores)
                convergence_history.append(best_fitness)
                
                # Update best genome
                best_idx = np.argmax(fitness_scores)
                current_best = self.population[best_idx]
                
                if self.best_genome is None or best_fitness > self.best_genome.get_current_fitness():
                    self.best_genome = current_best
                
                self.logger.info(f"Generation {generation}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}")
            
            # Check convergence
            if len(convergence_history) > 5:
                recent_improvement = convergence_history[-1] - convergence_history[-6]
                if recent_improvement < self.convergence_threshold:
                    self.logger.info(f"Converged at generation {generation}")
                    break
            
            # Evolve population
            await self._evolve_population(fitness_scores)
            
            # Adapt strategies
            await self._apply_adaptive_mechanisms(fitness_scores, problem_context)
            
            # Record evolution statistics
            self._record_evolution_stats(generation, fitness_scores)
        
        evolution_time = time.time() - start_time
        
        # Final meta-learning update
        if len(self.performance_database) > 10:
            await self._update_meta_learning()
        
        # Calculate final metrics
        final_fitness = self.best_genome.get_current_fitness() if self.best_genome else 0.0
        population_diversity = self._calculate_population_diversity()
        improvement_rate = self._calculate_improvement_rate(convergence_history)
        
        result = EvolutionaryOptimizationResult(
            best_genome=self.best_genome,
            final_fitness=final_fitness,
            generations_evolved=self.generation + 1,
            total_evaluations=evaluations_used,
            convergence_history=convergence_history,
            evolution_time=evolution_time,
            population_diversity=population_diversity,
            improvement_rate=improvement_rate
        )
        
        # Update global statistics
        self.optimization_stats['total_optimizations'] += 1
        if final_fitness > 0.7:  # Success threshold
            self.optimization_stats['successful_optimizations'] += 1
        
        self.optimization_stats['average_improvement'] = (
            self.optimization_stats['average_improvement'] * 0.9 + 
            improvement_rate * 0.1
        )
        
        if final_fitness > self.optimization_stats['best_fitness_achieved']:
            self.optimization_stats['best_fitness_achieved'] = final_fitness
        
        return result
    
    async def _evaluate_genome_fitness(
        self,
        genome: OptimizationGenome,
        optimization_config: Dict[str, Any],
        objective_function: Callable[[Dict[str, Any]], float]
    ) -> Dict[str, float]:
        """Evaluate genome fitness."""
        
        start_time = time.time()
        
        try:
            # Run optimization with genome configuration
            result_value = objective_function(optimization_config)
            
            # Simulate additional performance metrics
            performance = {
                'objective_value': result_value,
                'latency_ms': max(10, np.random.exponential(50)),
                'energy_mj': max(1, np.random.exponential(5)),
                'accuracy': min(1.0, max(0.5, np.random.normal(0.85, 0.1))),
                'convergence_time': time.time() - start_time
            }
            
            # Add specialization bonus
            if genome.specialized_domains:
                performance['accuracy'] *= 1.1  # 10% bonus for specialization
            
            return performance
            
        except Exception as e:
            self.logger.warning(f"Fitness evaluation failed: {e}")
            return {
                'objective_value': 0.0,
                'latency_ms': 1000,
                'energy_mj': 100,
                'accuracy': 0.1,
                'convergence_time': 60
            }
    
    def _genome_to_config(self, genome: OptimizationGenome) -> Dict[str, Any]:
        """Convert genome to optimization configuration."""
        
        config = genome.hyperparameters.copy()
        config['strategy_type'] = genome.strategy_type.value
        config['algorithm_weights'] = genome.algorithm_weights.copy()
        config['generation'] = genome.generation
        
        return config
    
    async def _evolve_population(self, fitness_scores: List[float]) -> None:
        """Evolve the population for next generation."""
        
        if not fitness_scores:
            return
        
        # Selection: keep elite individuals
        elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
        elite_genomes = [self.population[i] for i in elite_indices]
        
        # Create new population
        new_population = elite_genomes.copy()
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(fitness_scores)
            parent2 = self._tournament_selection(fitness_scores)
            
            # Crossover
            if np.random.random() < parent1.crossover_rate:
                child1, child2 = parent1.crossover(parent2)
                new_population.extend([child1, child2])
            else:
                # Mutation only
                child1 = parent1.mutate()
                child2 = parent2.mutate()
                new_population.extend([child1, child2])
        
        # Truncate to population size
        self.population = new_population[:self.population_size]
        
        # Increment generation counter for all individuals
        for genome in self.population:
            if genome.generation < self.generation + 1:
                genome.generation = self.generation + 1
    
    def _tournament_selection(self, fitness_scores: List[float], tournament_size: int = 3) -> OptimizationGenome:
        """Tournament selection for parent selection."""
        
        tournament_indices = np.random.choice(len(self.population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return self.population[winner_idx]
    
    async def _apply_adaptive_mechanisms(
        self,
        fitness_scores: List[float],
        problem_context: Dict[str, Any]
    ) -> None:
        """Apply adaptive mechanisms to improve evolution."""
        
        # Determine which adaptations to apply
        adaptations_to_apply = []
        
        # Check for stagnation
        if len(self.evolution_history) > 5:
            recent_improvements = [
                self.evolution_history[-i]['best_fitness'] - self.evolution_history[-i-1]['best_fitness']
                for i in range(1, min(6, len(self.evolution_history)))
            ]
            
            avg_improvement = np.mean(recent_improvements)
            
            if avg_improvement < 0.01:  # Stagnation detected
                adaptations_to_apply.extend([
                    AdaptationType.PARAMETER_TUNING,
                    AdaptationType.STRATEGY_MUTATION
                ])
        
        # Check population diversity
        diversity = self._calculate_population_diversity()
        if diversity < 0.3:  # Low diversity
            adaptations_to_apply.append(AdaptationType.ALGORITHM_SELECTION)
        
        # Periodic meta-learning
        if self.generation % 10 == 0:
            adaptations_to_apply.append(AdaptationType.META_LEARNING)
        
        # Apply selected adaptations
        for adaptation_type in adaptations_to_apply:
            try:
                await self.adaptation_strategies[adaptation_type](fitness_scores, problem_context)
            except Exception as e:
                self.logger.warning(f"Adaptation {adaptation_type} failed: {e}")
    
    async def _adapt_parameters(
        self,
        fitness_scores: List[float],
        problem_context: Dict[str, Any]
    ) -> None:
        """Adapt hyperparameters based on performance."""
        
        if not fitness_scores:
            return
        
        # Identify low-performing genomes
        threshold = np.percentile(fitness_scores, 25)  # Bottom 25%
        
        for i, (genome, fitness) in enumerate(zip(self.population, fitness_scores)):
            if fitness < threshold:
                # Adaptive parameter tuning
                for param_name in genome.hyperparameters:
                    if isinstance(genome.hyperparameters[param_name], (int, float)):
                        # Add noise to low-performing parameters
                        old_value = genome.hyperparameters[param_name]
                        noise_scale = 0.2 * (1 - fitness)  # More noise for lower fitness
                        noise = np.random.normal(0, noise_scale)
                        
                        if isinstance(old_value, int):
                            new_value = max(1, int(old_value * (1 + noise)))
                        else:
                            new_value = max(0.001, old_value * (1 + noise))
                        
                        genome.hyperparameters[param_name] = new_value
                
                genome.adaptation_count += 1
    
    async def _adapt_algorithm_selection(
        self,
        fitness_scores: List[float],
        problem_context: Dict[str, Any]
    ) -> None:
        """Adapt algorithm selection weights."""
        
        # Analyze which strategies are performing best
        strategy_performance = defaultdict(list)
        
        for genome, fitness in zip(self.population, fitness_scores):
            strategy_performance[genome.strategy_type].append(fitness)
        
        # Update algorithm weights based on strategy performance
        for genome in self.population:
            if len(strategy_performance[genome.strategy_type]) > 0:
                avg_performance = np.mean(strategy_performance[genome.strategy_type])
                
                # Boost weights for high-performing strategies
                if avg_performance > np.mean(fitness_scores):
                    boost_factor = 1.2
                else:
                    boost_factor = 0.9
                
                # Update weights
                for alg in genome.algorithm_weights:
                    genome.algorithm_weights[alg] *= boost_factor
                
                # Normalize weights
                total_weight = sum(genome.algorithm_weights.values())
                if total_weight > 0:
                    genome.algorithm_weights = {
                        k: v/total_weight for k, v in genome.algorithm_weights.items()
                    }
    
    async def _adapt_strategy_mutation(
        self,
        fitness_scores: List[float],
        problem_context: Dict[str, Any]
    ) -> None:
        """Adapt mutation strategies."""
        
        if not fitness_scores:
            return
        
        avg_fitness = np.mean(fitness_scores)
        
        for genome, fitness in zip(self.population, fitness_scores):
            # Adapt mutation rate based on performance
            if fitness < avg_fitness:
                # Increase mutation for low performers
                genome.mutation_rate = min(0.5, genome.mutation_rate * 1.1)
            else:
                # Decrease mutation for high performers
                genome.mutation_rate = max(0.01, genome.mutation_rate * 0.95)
            
            # Occasionally try strategy switching for low performers
            if fitness < avg_fitness * 0.7 and np.random.random() < 0.1:
                # Switch to a different strategy
                available_strategies = [s for s in EvolutionStrategy if s != genome.strategy_type]
                if available_strategies:
                    genome.strategy_type = np.random.choice(available_strategies)
                    genome.adaptation_count += 1
    
    async def _adapt_ensemble_weights(
        self,
        fitness_scores: List[float],
        problem_context: Dict[str, Any]
    ) -> None:
        """Adapt ensemble weighting strategies."""
        
        # Create ensemble of top performers
        if len(fitness_scores) < 3:
            return
        
        top_indices = np.argsort(fitness_scores)[-3:]  # Top 3
        top_genomes = [self.population[i] for i in top_indices]
        
        # Update ensemble weights based on recent performance
        for genome in top_genomes:
            performance_trend = genome.fitness_scores[-5:] if len(genome.fitness_scores) >= 5 else genome.fitness_scores
            
            if len(performance_trend) > 1:
                trend = np.polyfit(range(len(performance_trend)), performance_trend, 1)[0]
                
                # Boost weights for improving genomes
                if trend > 0:
                    for alg in genome.algorithm_weights:
                        genome.algorithm_weights[alg] *= 1.05
                else:
                    for alg in genome.algorithm_weights:
                        genome.algorithm_weights[alg] *= 0.98
                
                # Normalize
                total = sum(genome.algorithm_weights.values())
                if total > 0:
                    genome.algorithm_weights = {k: v/total for k, v in genome.algorithm_weights.items()}
    
    async def _adapt_meta_learning(
        self,
        fitness_scores: List[float],
        problem_context: Dict[str, Any]
    ) -> None:
        """Apply meta-learning adaptations."""
        
        if len(self.performance_database) < 20:
            return  # Need more data
        
        try:
            # Train meta-learning model
            scores = self.meta_predictor.train(self.performance_database[-100:])  # Last 100 examples
            
            # Use meta-predictions to guide population evolution
            for genome in self.population:
                prediction = self.meta_predictor.predict_performance(genome, problem_context)
                
                # If predicted to perform poorly, apply stronger adaptation
                if prediction['prediction'] < 0.4 and prediction['confidence'] > 0.6:
                    # Strong mutation
                    mutated = genome.mutate(mutation_intensity=2.0)
                    
                    # Replace current genome with mutated version
                    genome.hyperparameters = mutated.hyperparameters
                    genome.algorithm_weights = mutated.algorithm_weights
                    genome.adaptation_count += 1
            
        except Exception as e:
            self.logger.warning(f"Meta-learning adaptation failed: {e}")
    
    async def _update_meta_learning(self) -> None:
        """Update meta-learning models with recent performance data."""
        
        try:
            if len(self.performance_database) > 20:
                scores = self.meta_predictor.train(self.performance_database)
                self.logger.info(f"Meta-learning update: {scores}")
        except Exception as e:
            self.logger.warning(f"Meta-learning update failed: {e}")
    
    def _calculate_population_diversity(self) -> float:
        """Calculate population genetic diversity."""
        
        if len(self.population) < 2:
            return 0.0
        
        # Calculate diversity based on hyperparameter differences
        diversities = []
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                genome1 = self.population[i]
                genome2 = self.population[j]
                
                # Parameter diversity
                param_diff = 0.0
                param_count = 0
                
                for param in set(genome1.hyperparameters.keys()) | set(genome2.hyperparameters.keys()):
                    val1 = genome1.hyperparameters.get(param, 0)
                    val2 = genome2.hyperparameters.get(param, 0)
                    
                    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        if val1 != 0 or val2 != 0:
                            diff = abs(val1 - val2) / max(abs(val1), abs(val2), 1e-6)
                            param_diff += diff
                            param_count += 1
                
                if param_count > 0:
                    diversities.append(param_diff / param_count)
        
        return np.mean(diversities) if diversities else 0.0
    
    def _calculate_improvement_rate(self, convergence_history: List[float]) -> float:
        """Calculate rate of improvement over generations."""
        
        if len(convergence_history) < 2:
            return 0.0
        
        # Calculate improvement rate using linear regression
        x = np.arange(len(convergence_history))
        y = np.array(convergence_history)
        
        if np.std(y) == 0:
            return 0.0
        
        slope, _, _, _, _ = stats.linregress(x, y)
        return max(0.0, slope)  # Only positive improvements
    
    def _record_evolution_stats(self, generation: int, fitness_scores: List[float]) -> None:
        """Record evolution statistics."""
        
        if not fitness_scores:
            return
        
        stats = {
            'generation': generation,
            'best_fitness': max(fitness_scores),
            'avg_fitness': np.mean(fitness_scores),
            'worst_fitness': min(fitness_scores),
            'fitness_std': np.std(fitness_scores),
            'population_size': len(self.population),
            'diversity': self._calculate_population_diversity(),
            'timestamp': datetime.now()
        }
        
        self.evolution_history.append(stats)
    
    def _generate_initial_hyperparameters(self, strategy: EvolutionStrategy) -> Dict[str, Any]:
        """Generate initial hyperparameters for a strategy."""
        
        base_params = {
            'population_size': np.random.randint(10, 50),
            'max_iterations': np.random.randint(50, 200),
            'convergence_threshold': np.random.uniform(1e-8, 1e-4),
            'learning_rate': np.random.uniform(0.001, 0.1),
            'momentum': np.random.uniform(0.7, 0.99),
            'batch_size': np.random.randint(1, 16)
        }
        
        # Strategy-specific parameters
        if strategy == EvolutionStrategy.GENETIC_PROGRAMMING:
            base_params.update({
                'mutation_rate': np.random.uniform(0.01, 0.3),
                'crossover_rate': np.random.uniform(0.5, 0.9),
                'tournament_size': np.random.randint(2, 8)
            })
        elif strategy == EvolutionStrategy.DIFFERENTIAL_EVOLUTION:
            base_params.update({
                'differential_weight': np.random.uniform(0.3, 2.0),
                'crossover_probability': np.random.uniform(0.1, 1.0)
            })
        elif strategy == EvolutionStrategy.PARTICLE_SWARM:
            base_params.update({
                'inertia_weight': np.random.uniform(0.4, 0.9),
                'cognitive_coefficient': np.random.uniform(1.0, 3.0),
                'social_coefficient': np.random.uniform(1.0, 3.0)
            })
        elif strategy == EvolutionStrategy.SIMULATED_ANNEALING:
            base_params.update({
                'initial_temperature': np.random.uniform(100, 1000),
                'cooling_rate': np.random.uniform(0.85, 0.99)
            })
        
        return base_params
    
    def _generate_initial_algorithm_weights(self) -> Dict[str, float]:
        """Generate initial algorithm weights."""
        
        algorithms = ['gradient_descent', 'genetic', 'swarm', 'annealing', 'quantum']
        weights = np.random.dirichlet([1] * len(algorithms))  # Random weights that sum to 1
        
        return dict(zip(algorithms, weights))
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        
        stats = self.optimization_stats.copy()
        
        # Add evolution-specific stats
        if self.evolution_history:
            recent_stats = self.evolution_history[-10:]  # Last 10 generations
            
            stats.update({
                'generations_evolved': len(self.evolution_history),
                'current_diversity': self._calculate_population_diversity(),
                'recent_avg_fitness': np.mean([s['avg_fitness'] for s in recent_stats]),
                'recent_best_fitness': max([s['best_fitness'] for s in recent_stats]),
                'fitness_trend': self._calculate_improvement_rate([s['best_fitness'] for s in recent_stats])
            })
        
        # Population stats
        if self.population:
            current_fitness = [g.get_current_fitness() for g in self.population if g.fitness_scores]
            if current_fitness:
                stats.update({
                    'population_best_fitness': max(current_fitness),
                    'population_avg_fitness': np.mean(current_fitness),
                    'population_diversity': self._calculate_population_diversity()
                })
        
        # Meta-learning stats
        if self.meta_predictor.training_history:
            latest_training = self.meta_predictor.training_history[-1]
            stats['meta_learning'] = {
                'last_training': latest_training['timestamp'].isoformat(),
                'training_samples': latest_training['samples'],
                'model_scores': latest_training['scores']
            }
        
        return stats
    
    def save_state(self, filepath: Path) -> None:
        """Save optimizer state to file."""
        
        state = {
            'population': [genome.to_dict() for genome in self.population],
            'generation': self.generation,
            'evolution_history': self.evolution_history,
            'optimization_stats': self.optimization_stats,
            'best_genome': self.best_genome.to_dict() if self.best_genome else None,
            'performance_database': [(g.to_dict(), ctx, perf) for g, ctx, perf in self.performance_database[-100:]],  # Keep last 100
            'meta_predictor_trained': self.meta_predictor.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        # Save meta-predictor models separately
        if self.meta_predictor.is_trained:
            model_path = filepath.with_suffix('.meta_models')
            joblib.dump({
                'models': self.meta_predictor.models,
                'scaler': self.meta_predictor.scaler,
                'training_history': self.meta_predictor.training_history
            }, model_path)
    
    def load_state(self, filepath: Path) -> None:
        """Load optimizer state from file."""
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # Restore population
        self.population = []
        for genome_dict in state.get('population', []):
            genome = OptimizationGenome(
                genome_id=genome_dict['genome_id'],
                strategy_type=EvolutionStrategy(genome_dict['strategy_type']),
                hyperparameters=genome_dict['hyperparameters'],
                algorithm_weights=genome_dict['algorithm_weights'],
                generation=genome_dict['generation']
            )
            
            # Restore additional attributes
            genome.fitness_scores = genome_dict.get('fitness_scores', [])
            genome.success_rate = genome_dict.get('success_rate', 0.0)
            genome.adaptation_count = genome_dict.get('adaptation_count', 0)
            genome.mutation_rate = genome_dict.get('mutation_rate', 0.1)
            genome.crossover_rate = genome_dict.get('crossover_rate', 0.7)
            genome.parent_genomes = genome_dict.get('parent_genomes', [])
            genome.specialized_domains = set(genome_dict.get('specialized_domains', []))
            
            self.population.append(genome)
        
        # Restore other state
        self.generation = state.get('generation', 0)
        self.evolution_history = state.get('evolution_history', [])
        self.optimization_stats = state.get('optimization_stats', self.optimization_stats)
        
        # Restore best genome
        if state.get('best_genome'):
            best_dict = state['best_genome']
            self.best_genome = OptimizationGenome(
                genome_id=best_dict['genome_id'],
                strategy_type=EvolutionStrategy(best_dict['strategy_type']),
                hyperparameters=best_dict['hyperparameters'],
                algorithm_weights=best_dict['algorithm_weights'],
                generation=best_dict['generation']
            )
            self.best_genome.fitness_scores = best_dict.get('fitness_scores', [])
        
        # Restore performance database (reconstructed genomes)
        self.performance_database = []
        for genome_dict, ctx, perf in state.get('performance_database', []):
            genome = OptimizationGenome(
                genome_id=genome_dict['genome_id'],
                strategy_type=EvolutionStrategy(genome_dict['strategy_type']),
                hyperparameters=genome_dict['hyperparameters'],
                algorithm_weights=genome_dict['algorithm_weights'],
                generation=genome_dict['generation']
            )
            self.performance_database.append((genome, ctx, perf))
        
        # Load meta-predictor models
        model_path = filepath.with_suffix('.meta_models')
        if model_path.exists() and state.get('meta_predictor_trained'):
            try:
                model_data = joblib.load(model_path)
                self.meta_predictor.models = model_data['models']
                self.meta_predictor.scaler = model_data['scaler']
                self.meta_predictor.training_history = model_data['training_history']
                self.meta_predictor.is_trained = True
            except Exception as e:
                self.logger.warning(f"Failed to load meta-predictor models: {e}")


# Factory functions and utilities
async def create_self_evolving_optimizer(
    population_size: int = 20,
    max_generations: int = 50,
    elite_size: int = 4
) -> SelfEvolvingOptimizer:
    """Factory function to create self-evolving optimizer."""
    
    optimizer = SelfEvolvingOptimizer(
        population_size=population_size,
        elite_size=elite_size,
        max_generations=max_generations
    )
    
    return optimizer


async def run_self_evolution_experiment(
    objective_function: Callable[[Dict[str, Any]], float],
    problem_context: Dict[str, Any],
    max_evaluations: int = 500,
    population_size: int = 15
) -> Dict[str, Any]:
    """Run self-evolution optimization experiment."""
    
    # Create optimizer
    optimizer = await create_self_evolving_optimizer(
        population_size=population_size,
        max_generations=50
    )
    
    try:
        # Run evolution
        result = await optimizer.evolve_optimization_strategies(
            objective_function=objective_function,
            problem_context=problem_context,
            max_evaluations=max_evaluations
        )
        
        # Get statistics
        stats = optimizer.get_optimization_statistics()
        
        return {
            'evolution_result': {
                'final_fitness': result.final_fitness,
                'generations_evolved': result.generations_evolved,
                'total_evaluations': result.total_evaluations,
                'evolution_time': result.evolution_time,
                'improvement_rate': result.improvement_rate,
                'population_diversity': result.population_diversity
            },
            'best_genome': result.best_genome.to_dict() if result.best_genome else None,
            'optimization_statistics': stats,
            'convergence_history': result.convergence_history,
            'success': True
        }
        
    except Exception as e:
        return {
            'error': f"Self-evolution experiment failed: {e}",
            'success': False
        }


# Example usage and demonstration
if __name__ == "__main__":
    async def main():
        # Define a test optimization problem
        def test_objective(config: Dict[str, Any]) -> float:
            """Test objective function (Rastrigin-like function)."""
            
            # Extract parameters
            x = config.get('param_x', 0)
            y = config.get('param_y', 0)
            
            # Rastrigin function (minimization, so negate for maximization)
            result = -(20 + x**2 - 10*np.cos(2*np.pi*x) + y**2 - 10*np.cos(2*np.pi*y))
            
            # Add noise
            result += np.random.normal(0, 0.1)
            
            # Normalize to [0, 1] range
            return max(0.0, min(1.0, (result + 50) / 100))
        
        # Problem context
        problem_context = {
            'problem_dimension': 2,
            'constraint_count': 0,
            'multi_objective': False,
            'noisy': True,
            'dynamic': False
        }
        
        # Run self-evolution experiment
        print("Starting Self-Evolution Optimization Experiment...")
        print("=" * 60)
        
        results = await run_self_evolution_experiment(
            objective_function=test_objective,
            problem_context=problem_context,
            max_evaluations=300,
            population_size=12
        )
        
        if results['success']:
            print("Evolution Results:")
            print("-" * 30)
            evolution_result = results['evolution_result']
            print(f"Final Fitness: {evolution_result['final_fitness']:.4f}")
            print(f"Generations: {evolution_result['generations_evolved']}")
            print(f"Evaluations: {evolution_result['total_evaluations']}")
            print(f"Evolution Time: {evolution_result['evolution_time']:.2f}s")
            print(f"Improvement Rate: {evolution_result['improvement_rate']:.6f}")
            print(f"Population Diversity: {evolution_result['population_diversity']:.4f}")
            
            if results['best_genome']:
                print("\nBest Evolved Genome:")
                print("-" * 30)
                best = results['best_genome']
                print(f"Genome ID: {best['genome_id']}")
                print(f"Strategy: {best['strategy_type']}")
                print(f"Generation: {best['generation']}")
                print(f"Fitness Score: {np.mean(best['fitness_scores']):.4f}")
                print(f"Adaptations: {best['adaptation_count']}")
                
            print("\nOptimization Statistics:")
            print("-" * 30)
            stats = results['optimization_statistics']
            print(f"Success Rate: {stats.get('successful_optimizations', 0) / max(stats.get('total_optimizations', 1), 1):.2%}")
            print(f"Average Improvement: {stats.get('average_improvement', 0):.6f}")
            print(f"Best Fitness Ever: {stats.get('best_fitness_achieved', 0):.4f}")
        else:
            print(f"Experiment failed: {results['error']}")
    
    # Run demonstration
    asyncio.run(main())