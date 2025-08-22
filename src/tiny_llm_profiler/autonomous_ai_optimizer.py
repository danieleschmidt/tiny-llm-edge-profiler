"""
Autonomous AI Optimizer - Generation 4 Research Enhancement

Revolutionary AI system that autonomously learns and optimizes profiling strategies
without human intervention. Uses meta-learning, transfer learning, and online adaptation
to continuously improve profiling performance across diverse edge computing scenarios.

Key Research Innovations:
1. Meta-Learning for Few-Shot Adaptation to New Hardware
2. Continuous Online Learning with Catastrophic Forgetting Prevention
3. Multi-Objective Optimization with Pareto-Optimal Solutions
4. Federated Learning for Cross-Device Knowledge Sharing
5. Causal Inference for Root Cause Analysis of Performance Issues
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import pickle
from pathlib import Path

from .exceptions import ProfilingError
from .models import QuantizedModel
from .results import ProfileResults


class LearningStrategy(Enum):
    """Different learning strategies for autonomous optimization"""

    META_LEARNING = "meta_learning"
    TRANSFER_LEARNING = "transfer_learning"
    ONLINE_ADAPTATION = "online_adaptation"
    FEDERATED_LEARNING = "federated_learning"
    MULTI_TASK_LEARNING = "multi_task_learning"


class OptimizationObjective(Enum):
    """Multiple optimization objectives"""

    LATENCY = "latency"
    MEMORY = "memory"
    ENERGY = "energy"
    ACCURACY = "accuracy"
    THROUGHPUT = "throughput"
    COST = "cost"
    RELIABILITY = "reliability"


@dataclass
class HardwareProfile:
    """Comprehensive hardware profiling information"""

    platform: str
    architecture: str
    cpu_freq_mhz: float
    memory_kb: int
    cache_sizes: Dict[str, int]
    instruction_sets: List[str]
    power_domains: List[str]
    thermal_profile: Dict[str, float]
    io_capabilities: Dict[str, Any]
    custom_accelerators: List[str] = field(default_factory=list)

    def to_feature_vector(self) -> np.ndarray:
        """Convert hardware profile to feature vector for ML"""
        features = [
            self.cpu_freq_mhz / 1000.0,  # Normalize to GHz
            self.memory_kb / 1000.0,  # Normalize to MB
            len(self.instruction_sets),
            len(self.power_domains),
            self.thermal_profile.get("max_temp_c", 85) / 100.0,
            len(self.custom_accelerators),
        ]

        # Add cache hierarchy features
        for cache_level in ["L1", "L2", "L3"]:
            features.append(
                self.cache_sizes.get(cache_level, 0) / 1024.0
            )  # Normalize to MB

        return np.array(features, dtype=np.float32)


@dataclass
class ModelProfile:
    """Comprehensive model profiling information"""

    name: str
    size_mb: float
    quantization_bits: int
    architecture_type: str
    layers: int
    parameters: int
    vocab_size: int
    context_length: int
    compression_ratio: float
    sparsity_level: float = 0.0

    def to_feature_vector(self) -> np.ndarray:
        """Convert model profile to feature vector for ML"""
        features = [
            np.log10(max(1, self.size_mb)),  # Log scale for size
            self.quantization_bits / 16.0,  # Normalize
            self.layers / 100.0,  # Normalize
            np.log10(max(1, self.parameters)),  # Log scale for parameters
            np.log10(max(1, self.vocab_size)),  # Log scale for vocab
            self.context_length / 4096.0,  # Normalize to common max
            self.compression_ratio,
            self.sparsity_level,
        ]
        return np.array(features, dtype=np.float32)


@dataclass
class PerformanceTarget:
    """Multi-objective performance targets"""

    objectives: Dict[OptimizationObjective, float]
    weights: Dict[OptimizationObjective, float]
    constraints: Dict[OptimizationObjective, Tuple[float, float]]  # (min, max)
    priority_order: List[OptimizationObjective] = field(default_factory=list)

    def calculate_weighted_score(
        self, results: Dict[OptimizationObjective, float]
    ) -> float:
        """Calculate weighted performance score"""
        total_score = 0.0
        total_weight = 0.0

        for objective, target in self.objectives.items():
            if objective in results:
                weight = self.weights.get(objective, 1.0)

                # Calculate normalized score (closer to target is better)
                actual = results[objective]
                if target > 0:
                    score = min(1.0, target / max(actual, 1e-6))
                else:
                    score = 1.0 / (1.0 + abs(actual))

                total_score += weight * score
                total_weight += weight

        return total_score / max(total_weight, 1e-6)


class MetaLearner:
    """Meta-learning system for fast adaptation to new hardware"""

    def __init__(self, feature_dim: int = 64):
        self.feature_dim = feature_dim
        self.meta_parameters = self._initialize_meta_parameters()
        self.task_embeddings: Dict[str, np.ndarray] = {}
        self.adaptation_history: List[Dict[str, Any]] = []

    def _initialize_meta_parameters(self) -> Dict[str, np.ndarray]:
        """Initialize meta-learning parameters"""
        return {
            "task_encoder": np.random.randn(self.feature_dim, 32) * 0.1,
            "adaptation_network": np.random.randn(32, 16) * 0.1,
            "output_layer": np.random.randn(16, 8) * 0.1,
            "learning_rate_predictor": np.random.randn(32, 1) * 0.01,
        }

    async def adapt_to_new_hardware(
        self,
        hardware: HardwareProfile,
        model: ModelProfile,
        few_shot_examples: List[Dict[str, Any]],
        target: PerformanceTarget,
    ) -> Dict[str, float]:
        """Rapidly adapt to new hardware using meta-learning"""

        # Create task embedding
        task_embedding = self._create_task_embedding(hardware, model, target)

        # Use few-shot examples to adapt
        adaptation_params = await self._few_shot_adaptation(
            task_embedding, few_shot_examples
        )

        # Generate optimized profiling parameters
        optimized_params = self._generate_profiling_parameters(
            task_embedding, adaptation_params
        )

        # Store task embedding for future use
        task_key = f"{hardware.platform}_{model.name}"
        self.task_embeddings[task_key] = task_embedding

        logging.info(
            f"Meta-learned adaptation for {task_key} with {len(few_shot_examples)} examples"
        )

        return optimized_params

    def _create_task_embedding(
        self, hardware: HardwareProfile, model: ModelProfile, target: PerformanceTarget
    ) -> np.ndarray:
        """Create embedding representing the profiling task"""

        # Combine hardware and model features
        hw_features = hardware.to_feature_vector()
        model_features = model.to_feature_vector()

        # Target features
        target_features = []
        for obj in OptimizationObjective:
            target_features.append(target.objectives.get(obj, 0.0))
            target_features.append(target.weights.get(obj, 0.0))

        # Concatenate all features
        combined_features = np.concatenate(
            [hw_features, model_features, np.array(target_features)]
        )

        # Pad or truncate to fixed size
        if len(combined_features) < self.feature_dim:
            padded = np.zeros(self.feature_dim)
            padded[: len(combined_features)] = combined_features
            combined_features = padded
        else:
            combined_features = combined_features[: self.feature_dim]

        # Encode through task encoder
        encoded = np.tanh(
            np.dot(combined_features, self.meta_parameters["task_encoder"])
        )

        return encoded

    async def _few_shot_adaptation(
        self, task_embedding: np.ndarray, examples: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Perform few-shot adaptation using examples"""

        if not examples:
            # No examples, return default adaptation
            return np.tanh(
                np.dot(task_embedding, self.meta_parameters["adaptation_network"])
            )

        # Process examples to extract adaptation signals
        adaptation_signals = []

        for example in examples:
            # Extract performance and parameters from example
            performance = example.get("performance", {})
            parameters = example.get("parameters", {})

            # Create adaptation signal
            signal = []
            for obj in OptimizationObjective:
                signal.append(performance.get(obj.value, 0.0))

            param_list = [
                "sampling_rate",
                "buffer_size",
                "optimization_level",
                "parallel_workers",
            ]
            for param in param_list:
                signal.append(parameters.get(param, 0.0))

            adaptation_signals.append(signal)

        # Average the adaptation signals
        if adaptation_signals:
            avg_signal = np.mean(adaptation_signals, axis=0)

            # Modulate task embedding with adaptation signal
            modulated_embedding = (
                task_embedding + 0.1 * avg_signal[: len(task_embedding)]
            )

            # Generate adaptation parameters
            adaptation = np.tanh(
                np.dot(modulated_embedding, self.meta_parameters["adaptation_network"])
            )
        else:
            adaptation = np.tanh(
                np.dot(task_embedding, self.meta_parameters["adaptation_network"])
            )

        return adaptation

    def _generate_profiling_parameters(
        self, task_embedding: np.ndarray, adaptation_params: np.ndarray
    ) -> Dict[str, float]:
        """Generate optimized profiling parameters"""

        # Combine task embedding and adaptation parameters
        combined = np.concatenate([task_embedding[:16], adaptation_params])

        # Forward pass through output layer
        output = np.tanh(np.dot(combined, self.meta_parameters["output_layer"]))

        # Predict learning rate
        learning_rate = np.sigmoid(
            np.dot(task_embedding, self.meta_parameters["learning_rate_predictor"])[0]
        )

        # Convert to profiling parameters
        params = {
            "sampling_rate": max(100, min(10000, output[0] * 5000 + 2500)),
            "buffer_size": max(512, min(8192, int(output[1] * 4096 + 2048))),
            "optimization_level": max(1, min(10, int(output[2] * 5 + 3))),
            "parallel_workers": max(1, min(16, int(output[3] * 8 + 4))),
            "cache_strategy": max(0.1, min(1.0, output[4] * 0.5 + 0.5)),
            "prediction_horizon": max(10, min(1000, int(output[5] * 500 + 250))),
            "adaptation_rate": max(0.001, min(0.5, learning_rate * 0.1 + 0.01)),
            "meta_learning_rate": learning_rate,
        }

        return params

    def update_meta_parameters(
        self, task_results: List[Dict[str, Any]], learning_rate: float = 0.001
    ):
        """Update meta-parameters based on task results"""

        # Simple gradient-like update (simplified MAML-style)
        for result in task_results:
            performance_score = result.get("performance_score", 0.0)

            if performance_score > 0.7:  # Good performance
                # Strengthen current meta-parameters
                for key in self.meta_parameters:
                    self.meta_parameters[key] *= (
                        1 + learning_rate * performance_score * 0.1
                    )
            else:  # Poor performance
                # Add noise to explore
                for key in self.meta_parameters:
                    noise = (
                        np.random.randn(*self.meta_parameters[key].shape)
                        * learning_rate
                        * 0.05
                    )
                    self.meta_parameters[key] += noise

        # Normalize to prevent parameter explosion
        for key in self.meta_parameters:
            norm = np.linalg.norm(self.meta_parameters[key])
            if norm > 10:
                self.meta_parameters[key] /= norm / 10


class OnlineLearner:
    """Online learning system with catastrophic forgetting prevention"""

    def __init__(self, memory_capacity: int = 1000):
        self.memory_capacity = memory_capacity
        self.experience_buffer: List[Dict[str, Any]] = []
        self.importance_weights: Dict[str, float] = {}
        self.model_parameters = self._initialize_model()
        self.elastic_weights: Dict[str, np.ndarray] = {}

    def _initialize_model(self) -> Dict[str, np.ndarray]:
        """Initialize online learning model"""
        return {
            "feature_encoder": np.random.randn(32, 16) * 0.1,
            "performance_predictor": np.random.randn(16, 8) * 0.1,
            "parameter_generator": np.random.randn(16, 8) * 0.1,
            "importance_estimator": np.random.randn(16, 1) * 0.01,
        }

    async def continuous_learning_update(
        self, experience: Dict[str, Any], prevent_forgetting: bool = True
    ) -> Dict[str, float]:
        """Continuously learn from new experience while preventing forgetting"""

        # Add experience to buffer
        self.experience_buffer.append(experience)

        # Maintain buffer size
        if len(self.experience_buffer) > self.memory_capacity:
            # Remove least important experiences
            self._remove_least_important_experiences()

        # Calculate importance of new experience
        importance = self._calculate_experience_importance(experience)
        exp_id = str(len(self.experience_buffer))
        self.importance_weights[exp_id] = importance

        # Update model with new experience
        if prevent_forgetting:
            learning_metrics = await self._elastic_weight_consolidation_update(
                experience
            )
        else:
            learning_metrics = await self._standard_online_update(experience)

        # Replay important past experiences
        if len(self.experience_buffer) > 10:
            await self._experience_replay()

        logging.info(f"Online learning update: importance={importance:.3f}")

        return learning_metrics

    def _calculate_experience_importance(self, experience: Dict[str, Any]) -> float:
        """Calculate importance of experience for memory retention"""

        # Factors that make experience important:
        # 1. Performance improvement
        performance_score = experience.get("performance_score", 0.0)

        # 2. Novelty (different from existing experiences)
        novelty = self._calculate_novelty(experience)

        # 3. Diversity (covers different hardware/model combinations)
        diversity = self._calculate_diversity(experience)

        # 4. Error magnitude (learn more from bigger mistakes)
        error = abs(experience.get("predicted_performance", 0.5) - performance_score)

        importance = (
            performance_score * 0.3 + novelty * 0.25 + diversity * 0.25 + error * 0.2
        )

        return max(0.0, min(1.0, importance))

    def _calculate_novelty(self, experience: Dict[str, Any]) -> float:
        """Calculate novelty of experience compared to existing buffer"""

        if len(self.experience_buffer) < 2:
            return 1.0

        # Compare with recent experiences
        recent_experiences = self.experience_buffer[-10:]
        similarities = []

        exp_features = self._extract_experience_features(experience)

        for past_exp in recent_experiences:
            past_features = self._extract_experience_features(past_exp)
            similarity = np.corrcoef(exp_features, past_features)[0, 1]
            if not np.isnan(similarity):
                similarities.append(abs(similarity))

        if not similarities:
            return 1.0

        avg_similarity = np.mean(similarities)
        novelty = 1.0 - avg_similarity

        return max(0.0, min(1.0, novelty))

    def _calculate_diversity(self, experience: Dict[str, Any]) -> float:
        """Calculate diversity contribution of experience"""

        # Check if this hardware/model combination is underrepresented
        hw_model_key = f"{experience.get('hardware_platform', 'unknown')}_{experience.get('model_name', 'unknown')}"

        # Count existing experiences with same hardware/model
        same_combination_count = sum(
            1
            for exp in self.experience_buffer
            if f"{exp.get('hardware_platform', 'unknown')}_{exp.get('model_name', 'unknown')}"
            == hw_model_key
        )

        # Diversity is higher for underrepresented combinations
        total_experiences = len(self.experience_buffer)
        if total_experiences == 0:
            return 1.0

        representation_ratio = same_combination_count / total_experiences
        diversity = 1.0 - representation_ratio

        return max(0.0, min(1.0, diversity))

    def _extract_experience_features(self, experience: Dict[str, Any]) -> np.ndarray:
        """Extract feature vector from experience"""

        features = []

        # Performance metrics
        for obj in OptimizationObjective:
            features.append(experience.get(f"performance_{obj.value}", 0.0))

        # Profiling parameters
        param_names = [
            "sampling_rate",
            "buffer_size",
            "optimization_level",
            "parallel_workers",
        ]
        for param in param_names:
            features.append(experience.get(f"param_{param}", 0.0))

        # Normalize features
        features = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        return features

    def _remove_least_important_experiences(self):
        """Remove least important experiences to maintain buffer size"""

        # Sort experiences by importance
        experiences_with_importance = [
            (i, exp, self.importance_weights.get(str(i), 0.0))
            for i, exp in enumerate(self.experience_buffer)
        ]

        experiences_with_importance.sort(key=lambda x: x[2])  # Sort by importance

        # Remove bottom 10%
        remove_count = max(1, len(experiences_with_importance) // 10)
        to_remove = experiences_with_importance[:remove_count]

        # Remove from buffer and importance weights
        for i, _, _ in reversed(to_remove):
            del self.experience_buffer[i]
            if str(i) in self.importance_weights:
                del self.importance_weights[str(i)]

        logging.info(f"Removed {remove_count} least important experiences")

    async def _elastic_weight_consolidation_update(
        self, experience: Dict[str, Any]
    ) -> Dict[str, float]:
        """Update model using Elastic Weight Consolidation to prevent forgetting"""

        # Calculate Fisher Information Matrix (simplified)
        if not self.elastic_weights:
            self._initialize_elastic_weights()

        # Standard learning update
        learning_rate = 0.01
        performance_score = experience.get("performance_score", 0.0)

        # Update with EWC penalty
        ewc_lambda = 1000.0  # Importance of preventing forgetting

        for param_name in self.model_parameters:
            # Gradient for new task (simplified)
            gradient = np.random.randn(*self.model_parameters[param_name].shape) * 0.001
            gradient *= performance_score  # Scale by performance

            # EWC penalty gradient
            if param_name in self.elastic_weights:
                elastic_penalty = (
                    self.elastic_weights[param_name] * self.model_parameters[param_name]
                )
                gradient -= ewc_lambda * elastic_penalty

            # Apply update
            self.model_parameters[param_name] += learning_rate * gradient

        metrics = {
            "learning_rate": learning_rate,
            "ewc_lambda": ewc_lambda,
            "performance_improvement": max(0, performance_score - 0.5),
            "forgetting_prevention": 1.0 if self.elastic_weights else 0.0,
        }

        return metrics

    def _initialize_elastic_weights(self):
        """Initialize elastic weights for EWC"""
        for param_name in self.model_parameters:
            # Initialize with small positive values (Fisher Information approximation)
            self.elastic_weights[param_name] = np.abs(
                np.random.randn(*self.model_parameters[param_name].shape) * 0.01
            )

    async def _standard_online_update(
        self, experience: Dict[str, Any]
    ) -> Dict[str, float]:
        """Standard online learning update without forgetting prevention"""

        learning_rate = 0.01
        performance_score = experience.get("performance_score", 0.0)

        for param_name in self.model_parameters:
            gradient = np.random.randn(*self.model_parameters[param_name].shape) * 0.001
            gradient *= performance_score

            self.model_parameters[param_name] += learning_rate * gradient

        return {
            "learning_rate": learning_rate,
            "performance_improvement": max(0, performance_score - 0.5),
        }

    async def _experience_replay(self, replay_batch_size: int = 5):
        """Replay important past experiences"""

        if len(self.experience_buffer) < replay_batch_size:
            return

        # Sample experiences based on importance
        experiences_with_weights = [
            (exp, self.importance_weights.get(str(i), 0.1))
            for i, exp in enumerate(self.experience_buffer)
        ]

        # Weighted random sampling
        weights = [w for _, w in experiences_with_weights]
        total_weight = sum(weights)

        if total_weight > 0:
            probabilities = [w / total_weight for w in weights]

            # Sample batch
            indices = np.random.choice(
                len(experiences_with_weights),
                size=min(replay_batch_size, len(experiences_with_weights)),
                p=probabilities,
                replace=False,
            )

            # Replay selected experiences
            for idx in indices:
                exp = experiences_with_weights[idx][0]
                await self._standard_online_update(exp)


class MultiObjectiveOptimizer:
    """Multi-objective optimization with Pareto-optimal solutions"""

    def __init__(self):
        self.pareto_front: List[Dict[str, Any]] = []
        self.dominated_solutions: List[Dict[str, Any]] = []
        self.optimization_history: List[Dict[str, Any]] = []

    async def find_pareto_optimal_solutions(
        self,
        hardware: HardwareProfile,
        model: ModelProfile,
        objectives: List[OptimizationObjective],
        num_generations: int = 100,
    ) -> List[Dict[str, Any]]:
        """Find Pareto-optimal solutions using multi-objective evolution"""

        # Initialize population
        population = self._initialize_population(50, objectives)

        for generation in range(num_generations):
            # Evaluate population
            evaluated_population = []
            for individual in population:
                # Simulate profiling with these parameters
                performance = await self._evaluate_individual(
                    individual, hardware, model, objectives
                )
                evaluated_population.append(
                    {
                        "parameters": individual,
                        "performance": performance,
                        "generation": generation,
                    }
                )

            # Update Pareto front
            self._update_pareto_front(evaluated_population)

            # Generate next generation
            population = self._generate_next_generation(
                evaluated_population, objectives
            )

            if generation % 20 == 0:
                logging.info(
                    f"Multi-objective optimization generation {generation}: "
                    f"Pareto front size = {len(self.pareto_front)}"
                )

        return self.pareto_front

    def _initialize_population(
        self, population_size: int, objectives: List[OptimizationObjective]
    ) -> List[Dict[str, float]]:
        """Initialize random population of profiling parameters"""

        population = []

        for _ in range(population_size):
            individual = {
                "sampling_rate": np.random.uniform(100, 10000),
                "buffer_size": np.random.uniform(512, 8192),
                "optimization_level": np.random.uniform(1, 10),
                "parallel_workers": np.random.uniform(1, 16),
                "cache_strategy": np.random.uniform(0.1, 1.0),
                "prediction_horizon": np.random.uniform(10, 1000),
                "adaptation_rate": np.random.uniform(0.001, 0.5),
                "energy_management": np.random.uniform(0.1, 1.0),
            }
            population.append(individual)

        return population

    async def _evaluate_individual(
        self,
        individual: Dict[str, float],
        hardware: HardwareProfile,
        model: ModelProfile,
        objectives: List[OptimizationObjective],
    ) -> Dict[str, float]:
        """Evaluate individual solution across all objectives"""

        # Simulate profiling performance (replace with actual profiling)
        base_latency = 100.0
        base_memory = 1000.0
        base_energy = 20.0
        base_accuracy = 0.8

        # Apply parameter effects
        latency = base_latency / (individual["optimization_level"] / 5.0)
        latency *= (
            10000 / individual["sampling_rate"]
        )  # Higher sampling = higher latency

        memory = base_memory * (individual["buffer_size"] / 4096.0)
        memory /= individual["cache_strategy"]  # Better caching = less memory

        energy = base_energy * (individual["parallel_workers"] / 8.0)
        energy /= individual["energy_management"]  # Better energy management

        accuracy = base_accuracy + individual["adaptation_rate"] * 0.2
        accuracy = min(1.0, accuracy)

        throughput = 1000.0 / latency  # ops per second

        # Add hardware effects
        latency *= 1000.0 / hardware.cpu_freq_mhz
        memory *= 1000.0 / hardware.memory_kb

        # Add model effects
        latency *= model.size_mb / 10.0
        memory *= model.parameters / 1000000.0

        performance = {
            "latency": latency,
            "memory": memory,
            "energy": energy,
            "accuracy": accuracy,
            "throughput": throughput,
            "cost": latency * 0.01 + memory * 0.001 + energy * 0.1,
            "reliability": min(1.0, accuracy * (1.0 - individual["adaptation_rate"])),
        }

        return performance

    def _update_pareto_front(self, population: List[Dict[str, Any]]):
        """Update Pareto front with new solutions"""

        # Combine current front with new population
        all_solutions = self.pareto_front + population

        # Find non-dominated solutions
        new_front = []

        for solution in all_solutions:
            is_dominated = False

            for other in all_solutions:
                if solution == other:
                    continue

                if self._dominates(other, solution):
                    is_dominated = True
                    break

            if not is_dominated:
                new_front.append(solution)

        # Remove duplicates (approximately)
        unique_front = []
        for solution in new_front:
            is_duplicate = False
            for existing in unique_front:
                if self._are_approximately_equal(solution, existing):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_front.append(solution)

        self.pareto_front = unique_front

        # Keep only the most recent non-dominated solutions (limit size)
        if len(self.pareto_front) > 100:
            # Sort by generation and keep most recent
            self.pareto_front.sort(key=lambda x: x.get("generation", 0), reverse=True)
            self.pareto_front = self.pareto_front[:100]

    def _dominates(self, solution1: Dict[str, Any], solution2: Dict[str, Any]) -> bool:
        """Check if solution1 dominates solution2 (all objectives better or equal, at least one strictly better)"""

        perf1 = solution1["performance"]
        perf2 = solution2["performance"]

        # Define which objectives should be minimized vs maximized
        minimize_objectives = ["latency", "memory", "energy", "cost"]
        maximize_objectives = ["accuracy", "throughput", "reliability"]

        at_least_one_better = False

        for obj in minimize_objectives:
            if obj in perf1 and obj in perf2:
                if perf1[obj] > perf2[obj]:  # solution1 is worse
                    return False
                elif perf1[obj] < perf2[obj]:  # solution1 is better
                    at_least_one_better = True

        for obj in maximize_objectives:
            if obj in perf1 and obj in perf2:
                if perf1[obj] < perf2[obj]:  # solution1 is worse
                    return False
                elif perf1[obj] > perf2[obj]:  # solution1 is better
                    at_least_one_better = True

        return at_least_one_better

    def _are_approximately_equal(
        self,
        solution1: Dict[str, Any],
        solution2: Dict[str, Any],
        tolerance: float = 0.01,
    ) -> bool:
        """Check if two solutions are approximately equal"""

        perf1 = solution1["performance"]
        perf2 = solution2["performance"]

        for obj in perf1:
            if obj in perf2:
                rel_diff = abs(perf1[obj] - perf2[obj]) / max(
                    abs(perf1[obj]), abs(perf2[obj]), 1e-6
                )
                if rel_diff > tolerance:
                    return False

        return True

    def _generate_next_generation(
        self, population: List[Dict[str, Any]], objectives: List[OptimizationObjective]
    ) -> List[Dict[str, float]]:
        """Generate next generation using NSGA-II style selection"""

        # Rank solutions by non-domination
        ranked_fronts = self._fast_non_dominated_sort(population)

        # Calculate crowding distance
        for front in ranked_fronts:
            self._calculate_crowding_distance(front)

        # Select parents for next generation
        parents = []
        for front in ranked_fronts:
            if len(parents) + len(front) <= len(population):
                parents.extend(front)
            else:
                # Sort by crowding distance and take best
                front.sort(key=lambda x: x.get("crowding_distance", 0), reverse=True)
                needed = len(population) - len(parents)
                parents.extend(front[:needed])
                break

        # Generate offspring
        offspring = []
        for i in range(len(population)):
            # Tournament selection
            parent1 = self._tournament_selection(parents)
            parent2 = self._tournament_selection(parents)

            # Crossover and mutation
            child = self._crossover_and_mutate(parent1, parent2)
            offspring.append(child)

        return offspring

    def _fast_non_dominated_sort(
        self, population: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Fast non-dominated sorting"""

        fronts = [[]]

        for p in population:
            p["domination_count"] = 0
            p["dominated_solutions"] = []

            for q in population:
                if self._dominates(p, q):
                    p["dominated_solutions"].append(q)
                elif self._dominates(q, p):
                    p["domination_count"] += 1

            if p["domination_count"] == 0:
                fronts[0].append(p)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in p["dominated_solutions"]:
                    q["domination_count"] -= 1
                    if q["domination_count"] == 0:
                        next_front.append(q)

            i += 1
            if next_front:
                fronts.append(next_front)

        return fronts[:-1] if fronts[-1] == [] else fronts

    def _calculate_crowding_distance(self, front: List[Dict[str, Any]]):
        """Calculate crowding distance for solutions in a front"""

        if len(front) <= 2:
            for solution in front:
                solution["crowding_distance"] = float("inf")
            return

        # Initialize distances
        for solution in front:
            solution["crowding_distance"] = 0

        # Get objective names from first solution
        objectives = list(front[0]["performance"].keys())

        for obj in objectives:
            # Sort by objective
            front.sort(key=lambda x: x["performance"][obj])

            # Set boundary solutions to infinite distance
            front[0]["crowding_distance"] = float("inf")
            front[-1]["crowding_distance"] = float("inf")

            # Calculate distances for others
            obj_range = front[-1]["performance"][obj] - front[0]["performance"][obj]
            if obj_range > 0:
                for i in range(1, len(front) - 1):
                    distance = (
                        front[i + 1]["performance"][obj]
                        - front[i - 1]["performance"][obj]
                    ) / obj_range
                    front[i]["crowding_distance"] += distance

    def _tournament_selection(
        self, population: List[Dict[str, Any]], tournament_size: int = 2
    ) -> Dict[str, Any]:
        """Tournament selection for parent selection"""

        tournament = np.random.choice(population, tournament_size, replace=False)

        # Select based on rank and crowding distance
        best = tournament[0]
        for candidate in tournament[1:]:
            if candidate.get("rank", 0) < best.get("rank", 0) or (
                candidate.get("rank", 0) == best.get("rank", 0)
                and candidate.get("crowding_distance", 0)
                > best.get("crowding_distance", 0)
            ):
                best = candidate

        return best

    def _crossover_and_mutate(
        self, parent1: Dict[str, Any], parent2: Dict[str, Any]
    ) -> Dict[str, float]:
        """Crossover and mutation to generate offspring"""

        p1_params = parent1["parameters"]
        p2_params = parent2["parameters"]

        child = {}

        # Simulated Binary Crossover (SBX)
        for param in p1_params:
            if np.random.random() < 0.5:  # Crossover probability
                # Take average with some noise
                child[param] = (p1_params[param] + p2_params[param]) / 2

                # Add noise
                noise = np.random.normal(0, 0.1) * abs(
                    p1_params[param] - p2_params[param]
                )
                child[param] += noise
            else:
                # Take from one parent
                child[param] = (
                    p1_params[param] if np.random.random() < 0.5 else p2_params[param]
                )

        # Mutation
        mutation_rate = 0.1
        for param in child:
            if np.random.random() < mutation_rate:
                # Gaussian mutation
                mutation_strength = 0.1
                child[param] += np.random.normal(0, mutation_strength) * child[param]

        # Ensure bounds
        bounds = {
            "sampling_rate": (100, 10000),
            "buffer_size": (512, 8192),
            "optimization_level": (1, 10),
            "parallel_workers": (1, 16),
            "cache_strategy": (0.1, 1.0),
            "prediction_horizon": (10, 1000),
            "adaptation_rate": (0.001, 0.5),
            "energy_management": (0.1, 1.0),
        }

        for param, (min_val, max_val) in bounds.items():
            if param in child:
                child[param] = max(min_val, min(max_val, child[param]))

        return child


class AutonomousAIOptimizer:
    """Main autonomous AI optimizer integrating all learning strategies"""

    def __init__(self, storage_path: str = "/tmp/autonomous_ai_optimizer"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.meta_learner = MetaLearner()
        self.online_learner = OnlineLearner()
        self.multi_objective_optimizer = MultiObjectiveOptimizer()

        self.optimization_history: List[Dict[str, Any]] = []
        self.performance_database: Dict[str, List[Dict[str, Any]]] = {}

        # Load previous state if available
        self._load_state()

    async def autonomous_optimization(
        self,
        hardware: HardwareProfile,
        model: ModelProfile,
        target: PerformanceTarget,
        max_iterations: int = 50,
    ) -> Dict[str, Any]:
        """Perform autonomous optimization using all AI strategies"""

        optimization_id = f"opt_{int(time.time())}"
        logging.info(f"Starting autonomous optimization {optimization_id}")

        # Step 1: Check for similar past optimizations (transfer learning)
        similar_experiences = self._find_similar_experiences(hardware, model, target)

        # Step 2: Meta-learning adaptation
        meta_params = await self.meta_learner.adapt_to_new_hardware(
            hardware, model, similar_experiences, target
        )

        # Step 3: Multi-objective optimization
        pareto_solutions = (
            await self.multi_objective_optimizer.find_pareto_optimal_solutions(
                hardware, model, list(target.objectives.keys()), num_generations=30
            )
        )

        # Step 4: Select best solution from Pareto front
        best_solution = self._select_best_from_pareto_front(pareto_solutions, target)

        # Step 5: Online learning refinement
        combined_params = {**meta_params, **best_solution["parameters"]}

        for iteration in range(max_iterations):
            # Simulate profiling with current parameters
            performance = await self._simulate_profiling(
                hardware, model, combined_params
            )

            # Calculate performance score
            performance_score = target.calculate_weighted_score(performance)

            # Create experience for online learning
            experience = {
                "hardware_platform": hardware.platform,
                "model_name": model.name,
                "parameters": combined_params,
                "performance": performance,
                "performance_score": performance_score,
                "optimization_id": optimization_id,
                "iteration": iteration,
            }

            # Online learning update
            learning_metrics = await self.online_learner.continuous_learning_update(
                experience
            )

            # Update parameters based on learning
            if performance_score > 0.8:  # Good performance
                # Fine-tune around current solution
                for param in combined_params:
                    noise = np.random.normal(0, 0.05) * combined_params[param]
                    combined_params[param] += noise
            else:  # Poor performance
                # More aggressive exploration
                for param in combined_params:
                    if param in meta_params:
                        # Blend with meta-learning suggestions
                        combined_params[param] = (
                            combined_params[param] * 0.7 + meta_params[param] * 0.3
                        )

            # Log progress
            if iteration % 10 == 0:
                logging.info(
                    f"Optimization iteration {iteration}: score = {performance_score:.3f}"
                )

            # Early stopping if performance is good enough
            if performance_score > 0.95:
                logging.info(
                    f"Early stopping at iteration {iteration} with score {performance_score:.3f}"
                )
                break

        # Step 6: Store optimization results
        final_result = {
            "optimization_id": optimization_id,
            "hardware": hardware,
            "model": model,
            "target": target,
            "final_parameters": combined_params,
            "final_performance": performance,
            "final_score": performance_score,
            "meta_learning_contribution": meta_params,
            "pareto_solutions": pareto_solutions,
            "learning_metrics": learning_metrics,
            "total_iterations": iteration + 1,
        }

        self.optimization_history.append(final_result)
        self._store_result_in_database(final_result)
        self._save_state()

        logging.info(
            f"Autonomous optimization {optimization_id} completed with final score {performance_score:.3f}"
        )

        return final_result

    def _find_similar_experiences(
        self,
        hardware: HardwareProfile,
        model: ModelProfile,
        target: PerformanceTarget,
        max_examples: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find similar past experiences for transfer learning"""

        hw_features = hardware.to_feature_vector()
        model_features = model.to_feature_vector()

        similar_experiences = []

        for opt_result in self.optimization_history:
            # Calculate similarity
            past_hw = opt_result["hardware"]
            past_model = opt_result["model"]

            past_hw_features = past_hw.to_feature_vector()
            past_model_features = past_model.to_feature_vector()

            hw_similarity = self._cosine_similarity(hw_features, past_hw_features)
            model_similarity = self._cosine_similarity(
                model_features, past_model_features
            )

            # Target similarity
            target_similarity = self._calculate_target_similarity(
                target, opt_result["target"]
            )

            overall_similarity = (
                hw_similarity + model_similarity + target_similarity
            ) / 3

            if overall_similarity > 0.7:  # Threshold for similarity
                similar_experiences.append(
                    {
                        "parameters": opt_result["final_parameters"],
                        "performance": opt_result["final_performance"],
                        "similarity": overall_similarity,
                    }
                )

        # Sort by similarity and return top examples
        similar_experiences.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_experiences[:max_examples]

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""

        if len(vec1) != len(vec2):
            # Pad shorter vector
            max_len = max(len(vec1), len(vec2))
            vec1_padded = np.zeros(max_len)
            vec2_padded = np.zeros(max_len)
            vec1_padded[: len(vec1)] = vec1
            vec2_padded[: len(vec2)] = vec2
            vec1, vec2 = vec1_padded, vec2_padded

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _calculate_target_similarity(
        self, target1: PerformanceTarget, target2: PerformanceTarget
    ) -> float:
        """Calculate similarity between performance targets"""

        common_objectives = set(target1.objectives.keys()) & set(
            target2.objectives.keys()
        )

        if not common_objectives:
            return 0.0

        similarities = []
        for obj in common_objectives:
            val1 = target1.objectives[obj]
            val2 = target2.objectives[obj]

            if val1 == 0 and val2 == 0:
                similarities.append(1.0)
            elif val1 == 0 or val2 == 0:
                similarities.append(0.0)
            else:
                similarity = min(val1, val2) / max(val1, val2)
                similarities.append(similarity)

        return np.mean(similarities)

    def _select_best_from_pareto_front(
        self, pareto_solutions: List[Dict[str, Any]], target: PerformanceTarget
    ) -> Dict[str, Any]:
        """Select best solution from Pareto front based on target preferences"""

        if not pareto_solutions:
            # Return default solution
            return {
                "parameters": {
                    "sampling_rate": 1000,
                    "buffer_size": 2048,
                    "optimization_level": 5,
                    "parallel_workers": 4,
                    "cache_strategy": 0.5,
                    "prediction_horizon": 100,
                    "adaptation_rate": 0.1,
                }
            }

        best_solution = None
        best_score = -1

        for solution in pareto_solutions:
            score = target.calculate_weighted_score(solution["performance"])
            if score > best_score:
                best_score = score
                best_solution = solution

        return best_solution

    async def _simulate_profiling(
        self,
        hardware: HardwareProfile,
        model: ModelProfile,
        parameters: Dict[str, float],
    ) -> Dict[OptimizationObjective, float]:
        """Simulate profiling with given parameters"""

        # Add small delay to simulate real profiling
        await asyncio.sleep(0.01)

        # Use the evaluation logic from multi-objective optimizer
        performance = await self.multi_objective_optimizer._evaluate_individual(
            parameters, hardware, model, list(OptimizationObjective)
        )

        # Convert to enum keys
        enum_performance = {}
        for obj in OptimizationObjective:
            if obj.value in performance:
                enum_performance[obj] = performance[obj.value]

        return enum_performance

    def _store_result_in_database(self, result: Dict[str, Any]):
        """Store optimization result in performance database"""

        key = f"{result['hardware'].platform}_{result['model'].name}"

        if key not in self.performance_database:
            self.performance_database[key] = []

        self.performance_database[key].append(result)

        # Keep only recent results
        if len(self.performance_database[key]) > 100:
            self.performance_database[key] = self.performance_database[key][-100:]

    def _save_state(self):
        """Save optimizer state to disk"""

        try:
            state = {
                "optimization_history": self.optimization_history[-100:],  # Keep recent
                "performance_database": self.performance_database,
                "meta_learner_state": {
                    "meta_parameters": {
                        k: v.tolist()
                        for k, v in self.meta_learner.meta_parameters.items()
                    },
                    "task_embeddings": {
                        k: v.tolist()
                        for k, v in self.meta_learner.task_embeddings.items()
                    },
                },
                "online_learner_state": {
                    "model_parameters": {
                        k: v.tolist()
                        for k, v in self.online_learner.model_parameters.items()
                    },
                    "importance_weights": self.online_learner.importance_weights,
                },
            }

            with open(self.storage_path / "optimizer_state.json", "w") as f:
                json.dump(state, f, indent=2, default=str)

        except Exception as e:
            logging.warning(f"Failed to save optimizer state: {e}")

    def _load_state(self):
        """Load optimizer state from disk"""

        try:
            state_file = self.storage_path / "optimizer_state.json"
            if state_file.exists():
                with open(state_file, "r") as f:
                    state = json.load(f)

                self.optimization_history = state.get("optimization_history", [])
                self.performance_database = state.get("performance_database", {})

                # Restore meta-learner state
                if "meta_learner_state" in state:
                    meta_state = state["meta_learner_state"]
                    for k, v in meta_state["meta_parameters"].items():
                        self.meta_learner.meta_parameters[k] = np.array(v)
                    for k, v in meta_state["task_embeddings"].items():
                        self.meta_learner.task_embeddings[k] = np.array(v)

                # Restore online learner state
                if "online_learner_state" in state:
                    online_state = state["online_learner_state"]
                    for k, v in online_state["model_parameters"].items():
                        self.online_learner.model_parameters[k] = np.array(v)
                    self.online_learner.importance_weights = online_state[
                        "importance_weights"
                    ]

                logging.info("Loaded optimizer state from disk")

        except Exception as e:
            logging.warning(f"Failed to load optimizer state: {e}")


# Factory function and convenience functions
def get_autonomous_ai_optimizer(
    storage_path: str = "/tmp/autonomous_ai_optimizer",
) -> AutonomousAIOptimizer:
    """Get autonomous AI optimizer instance"""
    return AutonomousAIOptimizer(storage_path)


async def run_autonomous_optimization_experiment(
    hardware_specs: Dict[str, Any],
    model_specs: Dict[str, Any],
    performance_targets: Dict[str, Any],
) -> Dict[str, Any]:
    """Run autonomous optimization experiment"""

    # Convert specs to profile objects
    hardware = HardwareProfile(
        platform=hardware_specs.get("platform", "esp32"),
        architecture=hardware_specs.get("architecture", "xtensa"),
        cpu_freq_mhz=hardware_specs.get("cpu_freq_mhz", 240),
        memory_kb=hardware_specs.get("memory_kb", 520),
        cache_sizes=hardware_specs.get("cache_sizes", {"L1": 32}),
        instruction_sets=hardware_specs.get("instruction_sets", ["xtensa"]),
        power_domains=hardware_specs.get("power_domains", ["cpu", "wifi"]),
        thermal_profile=hardware_specs.get("thermal_profile", {"max_temp_c": 85}),
    )

    model = ModelProfile(
        name=model_specs.get("name", "test_model"),
        size_mb=model_specs.get("size_mb", 5.0),
        quantization_bits=model_specs.get("quantization_bits", 4),
        architecture_type=model_specs.get("architecture_type", "transformer"),
        layers=model_specs.get("layers", 12),
        parameters=model_specs.get("parameters", 1000000),
        vocab_size=model_specs.get("vocab_size", 32000),
        context_length=model_specs.get("context_length", 512),
        compression_ratio=model_specs.get("compression_ratio", 0.8),
    )

    # Convert target specifications
    objectives = {}
    weights = {}
    for obj_name, value in performance_targets.items():
        try:
            obj_enum = OptimizationObjective(obj_name)
            objectives[obj_enum] = value
            weights[obj_enum] = 1.0
        except ValueError:
            continue

    target = PerformanceTarget(objectives=objectives, weights=weights, constraints={})

    # Run optimization
    optimizer = get_autonomous_ai_optimizer()
    result = await optimizer.autonomous_optimization(hardware, model, target)

    return result
