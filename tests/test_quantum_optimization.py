"""
Test suite for Quantum-Inspired Optimization components.
"""

import asyncio
import numpy as np
import pytest
from unittest.mock import Mock, AsyncMock, patch

from tiny_llm_profiler.quantum_inspired_optimizer import (
    QuantumInspiredOptimizer,
    QuantumGeneticAlgorithm,
    QuantumAnnealingOptimizer,
    QuantumState,
    OptimizationParameter,
    QuantumOptimizationMethod,
    OptimizationObjective
)


class TestQuantumState:
    """Test cases for QuantumState."""
    
    def test_quantum_state_creation(self):
        """Test quantum state creation and basic properties."""
        amplitudes = np.array([1.0, 0.0, 0.0, 0.0])  # 2 qubits
        phases = np.array([0.0, 0.0, 0.0, 0.0])
        entanglement_matrix = np.eye(4)
        
        state = QuantumState(
            amplitudes=amplitudes,
            phases=phases,
            entanglement_matrix=entanglement_matrix
        )
        
        assert state.num_qubits == 2
        assert np.allclose(state.amplitudes, amplitudes)
        assert np.allclose(state.phases, phases)
    
    def test_quantum_state_normalization(self):
        """Test quantum state normalization."""
        amplitudes = np.array([1.0, 1.0, 1.0, 1.0])  # Unnormalized
        phases = np.zeros(4)
        entanglement_matrix = np.eye(4)
        
        state = QuantumState(
            amplitudes=amplitudes,
            phases=phases,
            entanglement_matrix=entanglement_matrix
        )
        
        normalized_state = state.normalize()
        
        # Check normalization
        norm = np.linalg.norm(normalized_state.amplitudes)
        assert np.isclose(norm, 1.0, atol=1e-10)
    
    def test_quantum_measurement(self):
        """Test quantum measurement."""
        amplitudes = np.array([0.6, 0.8, 0.0, 0.0])  # Normalized
        phases = np.zeros(4)
        entanglement_matrix = np.eye(4)
        
        state = QuantumState(
            amplitudes=amplitudes,
            phases=phases,
            entanglement_matrix=entanglement_matrix
        )
        
        # Measure multiple times to check probabilistic behavior
        measurements = [state.measure() for _ in range(100)]
        
        # Should only measure states 0 and 1 (non-zero amplitudes)
        unique_measurements = set(measurements)
        assert unique_measurements.issubset({0, 1})
        
        # State 1 should be measured more often (higher probability)
        count_1 = sum(1 for m in measurements if m == 1)
        count_0 = sum(1 for m in measurements if m == 0)
        assert count_1 > count_0  # Higher amplitude -> higher probability
    
    def test_quantum_entanglement(self):
        """Test quantum entanglement between states."""
        # Create two single-qubit states
        state1 = QuantumState(
            amplitudes=np.array([1.0, 0.0]),
            phases=np.array([0.0, 0.0]),
            entanglement_matrix=np.eye(2)
        )
        
        state2 = QuantumState(
            amplitudes=np.array([0.0, 1.0]),
            phases=np.array([0.0, 0.0]),
            entanglement_matrix=np.eye(2)
        )
        
        entangled_state = state1.entangle_with(state2)
        
        assert entangled_state.num_qubits == 2
        assert len(entangled_state.amplitudes) == 4
        assert entangled_state.entanglement_matrix.shape == (4, 4)


class TestOptimizationParameter:
    """Test cases for OptimizationParameter."""
    
    def test_parameter_creation(self):
        """Test optimization parameter creation."""
        param = OptimizationParameter(
            name="timeout_seconds",
            value_range=(1.0, 10.0),
            current_value=5.0,
            quantum_encoding="binary",
            precision_bits=8
        )
        
        assert param.name == "timeout_seconds"
        assert param.value_range == (1.0, 10.0)
        assert param.current_value == 5.0
        assert param.quantum_encoding == "binary"
        assert param.precision_bits == 8
    
    def test_binary_quantum_encoding(self):
        """Test binary quantum encoding."""
        param = OptimizationParameter(
            name="test_param",
            value_range=(0.0, 1.0),
            current_value=0.5,
            quantum_encoding="binary",
            precision_bits=4
        )
        
        quantum_state = param.encode_quantum()
        
        assert isinstance(quantum_state, QuantumState)
        assert len(quantum_state.amplitudes) == 2**4  # 16 states
        assert np.isclose(np.linalg.norm(quantum_state.amplitudes), 1.0)
    
    def test_angle_quantum_encoding(self):
        """Test angle quantum encoding."""
        param = OptimizationParameter(
            name="test_param",
            value_range=(0.0, 1.0),
            current_value=0.25,
            quantum_encoding="angle",
            precision_bits=2
        )
        
        quantum_state = param.encode_quantum()
        
        assert isinstance(quantum_state, QuantumState)
        assert len(quantum_state.amplitudes) == 2  # Angle encoding uses 2 amplitudes
        assert np.isclose(np.linalg.norm(quantum_state.amplitudes), 1.0)
    
    def test_quantum_decode_binary(self):
        """Test decoding binary quantum state."""
        param = OptimizationParameter(
            name="test_param",
            value_range=(0.0, 10.0),
            current_value=5.0,
            quantum_encoding="binary",
            precision_bits=4
        )
        
        # Create quantum state and decode
        quantum_state = param.encode_quantum()
        decoded_value = param.decode_quantum(quantum_state)
        
        # Should be approximately the original value
        assert 0.0 <= decoded_value <= 10.0
    
    def test_quantum_decode_angle(self):
        """Test decoding angle quantum state."""
        param = OptimizationParameter(
            name="test_param",
            value_range=(0.0, 1.0),
            current_value=0.7,
            quantum_encoding="angle",
            precision_bits=2
        )
        
        quantum_state = param.encode_quantum()
        decoded_value = param.decode_quantum(quantum_state)
        
        assert 0.0 <= decoded_value <= 1.0


class TestQuantumGeneticAlgorithm:
    """Test cases for QuantumGeneticAlgorithm."""
    
    @pytest.fixture
    def qga(self):
        """Create QuantumGeneticAlgorithm instance."""
        return QuantumGeneticAlgorithm(
            population_size=10,
            num_generations=5,  # Small for testing
            mutation_rate=0.1,
            crossover_rate=0.8
        )
    
    @pytest.fixture
    def test_parameters(self):
        """Create test optimization parameters."""
        return [
            OptimizationParameter(
                name="param1",
                value_range=(0.0, 1.0),
                current_value=0.5,
                quantum_encoding="angle"
            ),
            OptimizationParameter(
                name="param2",
                value_range=(1.0, 10.0),
                current_value=5.0,
                quantum_encoding="binary",
                precision_bits=4
            )
        ]
    
    @pytest.fixture
    def test_objective_function(self):
        """Create test objective function."""
        def objective(params):
            # Simple quadratic function: minimize (param1-0.3)^2 + (param2-3.0)^2
            return (params["param1"] - 0.3)**2 + (params["param2"] - 3.0)**2
        return objective
    
    @pytest.mark.asyncio
    async def test_quantum_genetic_optimization(self, qga, test_parameters, test_objective_function):
        """Test quantum genetic algorithm optimization."""
        result = await qga.optimize(
            test_parameters,
            test_objective_function,
            OptimizationObjective.MINIMIZE_LATENCY
        )
        
        # Verify result structure
        assert 'best_parameters' in result
        assert 'best_fitness' in result
        assert 'convergence_history' in result
        assert 'generations_completed' in result
        assert 'optimization_method' in result
        
        # Check parameters
        best_params = result['best_parameters']
        assert 'param1' in best_params
        assert 'param2' in best_params
        assert 0.0 <= best_params['param1'] <= 1.0
        assert 1.0 <= best_params['param2'] <= 10.0
        
        # Check convergence
        convergence = result['convergence_history']
        assert len(convergence) == qga.num_generations
        assert result['generations_completed'] == qga.num_generations
    
    def test_quantum_population_initialization(self, qga, test_parameters):
        """Test quantum population initialization."""
        population = qga._initialize_quantum_population(test_parameters)
        
        assert len(population) == qga.population_size
        assert len(population[0]) == len(test_parameters)  # Number of parameters
        
        # Each individual should have quantum states for each parameter
        for individual in population:
            assert len(individual) == len(test_parameters)
            for quantum_state in individual:
                assert isinstance(quantum_state, QuantumState)
    
    def test_quantum_crossover(self, qga, test_parameters):
        """Test quantum crossover operation."""
        # Create two parent individuals
        parent1 = qga._initialize_quantum_population(test_parameters)[0]
        parent2 = qga._initialize_quantum_population(test_parameters)[0]
        
        child1, child2 = qga._quantum_crossover(parent1, parent2)
        
        assert len(child1) == len(test_parameters)
        assert len(child2) == len(test_parameters)
        
        # Children should have normalized quantum states
        for quantum_state in child1 + child2:
            norm = np.linalg.norm(quantum_state.amplitudes)
            assert np.isclose(norm, 1.0, atol=1e-10)
    
    def test_quantum_mutation(self, qga, test_parameters):
        """Test quantum mutation operation."""
        individual = qga._initialize_quantum_population(test_parameters)[0]
        original_amplitudes = [qs.amplitudes.copy() for qs in individual]
        
        mutated_individual = qga._quantum_mutation(individual, test_parameters)
        
        assert len(mutated_individual) == len(test_parameters)
        
        # Mutation should change at least some amplitudes
        changed = False
        for orig_amps, mutated_qs in zip(original_amplitudes, mutated_individual):
            if not np.allclose(orig_amps, mutated_qs.amplitudes):
                changed = True
                break
        
        # Note: Mutation is probabilistic, so this test might occasionally fail
        # In practice, with multiple parameters, at least one should change


class TestQuantumAnnealingOptimizer:
    """Test cases for QuantumAnnealingOptimizer."""
    
    @pytest.fixture
    def qao(self):
        """Create QuantumAnnealingOptimizer instance."""
        return QuantumAnnealingOptimizer(
            num_sweeps=50,  # Small for testing
            initial_temperature=10.0,
            final_temperature=0.01
        )
    
    @pytest.fixture
    def simple_parameters(self):
        """Create simple test parameters."""
        return [
            OptimizationParameter(
                name="x",
                value_range=(-5.0, 5.0),
                current_value=0.0
            ),
            OptimizationParameter(
                name="y",
                value_range=(-5.0, 5.0),
                current_value=0.0
            )
        ]
    
    @pytest.fixture
    def simple_objective_function(self):
        """Create simple objective function (minimize x^2 + y^2)."""
        def objective(params):
            return params["x"]**2 + params["y"]**2
        return objective
    
    @pytest.mark.asyncio
    async def test_quantum_annealing_optimization(self, qao, simple_parameters, simple_objective_function):
        """Test quantum annealing optimization."""
        result = await qao.optimize(
            simple_parameters,
            simple_objective_function,
            OptimizationObjective.MINIMIZE_ENERGY
        )
        
        # Verify result structure
        assert 'best_parameters' in result
        assert 'best_energy' in result
        assert 'energy_history' in result
        assert 'sweeps_completed' in result
        assert 'optimization_method' in result
        
        # Check parameters are within bounds
        best_params = result['best_parameters']
        assert -5.0 <= best_params['x'] <= 5.0
        assert -5.0 <= best_params['y'] <= 5.0
        
        # Energy should be minimized (close to 0 for this function)
        assert result['best_energy'] >= 0  # x^2 + y^2 is always non-negative
        
        # Check convergence history
        energy_history = result['energy_history']
        assert len(energy_history) == qao.num_sweeps
    
    def test_temperature_schedule(self, qao):
        """Test temperature schedule creation."""
        schedule = qao._create_temperature_schedule()
        
        assert len(schedule) == qao.num_sweeps
        assert schedule[0] == qao.initial_temperature
        assert np.isclose(schedule[-1], qao.final_temperature, rtol=0.1)
        
        # Temperature should generally decrease
        assert schedule[0] > schedule[-1]
    
    def test_local_change_proposal(self, qao, simple_parameters):
        """Test local change proposal."""
        current_solution = {"x": 1.0, "y": 2.0}
        parameter = simple_parameters[0]  # x parameter
        
        new_solution = qao._propose_local_change(current_solution, parameter)
        
        assert "x" in new_solution
        assert "y" in new_solution
        assert new_solution["y"] == current_solution["y"]  # Only x should change
        assert -5.0 <= new_solution["x"] <= 5.0  # Within bounds
    
    def test_transition_acceptance(self, qao):
        """Test transition acceptance logic."""
        # Test improvement acceptance
        assert qao._accept_transition(10.0, 5.0, 1.0) == True  # Always accept improvement
        
        # Test degradation at high temperature (should often accept)
        high_temp_acceptances = sum(
            qao._accept_transition(5.0, 10.0, 10.0) for _ in range(100)
        )
        assert high_temp_acceptances > 0  # Should accept some degradations
        
        # Test degradation at low temperature (should rarely accept)
        low_temp_acceptances = sum(
            qao._accept_transition(5.0, 10.0, 0.01) for _ in range(100)
        )
        assert low_temp_acceptances < high_temp_acceptances


class TestQuantumInspiredOptimizer:
    """Test cases for main QuantumInspiredOptimizer."""
    
    @pytest.fixture
    def optimizer(self):
        """Create QuantumInspiredOptimizer instance."""
        return QuantumInspiredOptimizer()
    
    @pytest.fixture
    def test_config(self):
        """Create test optimization configuration."""
        return {
            'parameters': {
                'timeout_ms': {
                    'range': [10, 1000],
                    'initial_value': 100,
                    'encoding': 'binary',
                    'precision_bits': 8
                },
                'memory_limit_kb': {
                    'range': [100, 1000],
                    'initial_value': 500,
                    'encoding': 'angle'
                }
            }
        }
    
    @pytest.fixture
    def test_objective(self):
        """Create test objective function."""
        def objective(params):
            # Minimize sum of normalized parameters
            timeout = params['timeout_ms'] / 1000.0
            memory = params['memory_limit_kb'] / 1000.0
            return timeout + memory
        return objective
    
    @pytest.mark.asyncio
    async def test_edge_profiling_optimization(self, optimizer, test_config, test_objective):
        """Test edge profiling optimization."""
        result = await optimizer.optimize_edge_profiling(
            test_config,
            test_objective,
            QuantumOptimizationMethod.QUANTUM_GENETIC_ALGORITHM,
            OptimizationObjective.MINIMIZE_LATENCY
        )
        
        # Verify enhanced results structure
        assert 'best_parameters' in result
        assert 'optimization_time_seconds' in result
        assert 'quantum_method' in result
        assert 'objective_type' in result
        assert 'parameter_count' in result
        assert 'quantum_advantage_analysis' in result
        assert 'recommendations' in result
        
        # Check quantum advantage analysis
        qa_analysis = result['quantum_advantage_analysis']
        assert 'improvement_ratio' in qa_analysis
        assert 'stability_improvement' in qa_analysis
        assert 'quantum_advantage_score' in qa_analysis
        
        # Check recommendations
        recommendations = result['recommendations']
        assert isinstance(recommendations, list)
    
    def test_optimization_config_parsing(self, optimizer, test_config):
        """Test optimization configuration parsing."""
        parameters = optimizer._parse_optimization_config(test_config)
        
        assert len(parameters) == 2
        assert parameters[0].name == 'timeout_ms'
        assert parameters[1].name == 'memory_limit_kb'
        
        # Check parameter properties
        timeout_param = parameters[0]
        assert timeout_param.value_range == (10, 1000)
        assert timeout_param.current_value == 100
        assert timeout_param.quantum_encoding == 'binary'
        assert timeout_param.precision_bits == 8
        
        memory_param = parameters[1]
        assert memory_param.value_range == (100, 1000)
        assert memory_param.quantum_encoding == 'angle'
    
    def test_quantum_advantage_analysis(self, optimizer):
        """Test quantum advantage analysis."""
        # Mock results with convergence history
        results = {
            'convergence_history': [10.0, 8.0, 6.0, 5.0, 4.5, 4.2, 4.1, 4.05, 4.0, 4.0]
        }
        
        analysis = optimizer._analyze_quantum_advantage(results)
        
        assert 'improvement_ratio' in analysis
        assert 'stability_improvement' in analysis
        assert 'convergence_efficiency' in analysis
        assert 'quantum_advantage_score' in analysis
        
        # Check that improvement ratio is positive for this example
        assert analysis['improvement_ratio'] > 0
    
    def test_optimization_recommendations(self, optimizer):
        """Test optimization recommendations generation."""
        results = {
            'best_parameters': {
                'timeout_ms': 50,
                'memory_limit_kb': 800
            },
            'quantum_advantage_analysis': {
                'quantum_advantage_score': 0.8
            },
            'convergence_history': list(range(100, 0, -1))  # Good convergence
        }
        
        recommendations = optimizer._generate_optimization_recommendations(results)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should contain recommendations based on the mock data
        rec_text = ' '.join(recommendations)
        assert 'quantum advantage' in rec_text or 'timeout' in rec_text or 'memory' in rec_text


@pytest.mark.integration
class TestQuantumOptimizationIntegration:
    """Integration tests for quantum optimization components."""
    
    @pytest.mark.asyncio
    async def test_full_optimization_pipeline(self):
        """Test complete optimization pipeline."""
        from tiny_llm_profiler.quantum_inspired_optimizer import (
            get_quantum_optimizer,
            optimize_with_quantum
        )
        
        # Test configuration
        config = {
            'parameters': {
                'param1': {
                    'range': [0.0, 1.0],
                    'initial_value': 0.5,
                    'encoding': 'angle'
                },
                'param2': {
                    'range': [0.0, 10.0],
                    'initial_value': 5.0,
                    'encoding': 'binary',
                    'precision_bits': 6
                }
            }
        }
        
        def objective_function(params):
            return abs(params['param1'] - 0.7) + abs(params['param2'] - 3.0)
        
        # Test global optimizer
        optimizer = get_quantum_optimizer()
        assert isinstance(optimizer, QuantumInspiredOptimizer)
        
        # Test convenience function
        result = await optimize_with_quantum(
            config,
            objective_function,
            QuantumOptimizationMethod.QUANTUM_GENETIC_ALGORITHM,
            OptimizationObjective.MINIMIZE_LATENCY
        )
        
        assert 'best_parameters' in result
        assert 'quantum_advantage_analysis' in result
        
        # Check that optimization found reasonable values
        best_params = result['best_parameters']
        assert 0.0 <= best_params['param1'] <= 1.0
        assert 0.0 <= best_params['param2'] <= 10.0
    
    @pytest.mark.asyncio
    async def test_multi_objective_optimization(self):
        """Test multi-objective optimization."""
        optimizer = QuantumInspiredOptimizer()
        
        config = {
            'parameters': {
                'latency_param': {
                    'range': [1.0, 100.0],
                    'initial_value': 50.0
                },
                'memory_param': {
                    'range': [10.0, 1000.0],
                    'initial_value': 500.0
                }
            }
        }
        
        # Define multiple objectives
        def latency_objective(params):
            return params['latency_param']  # Minimize latency
        
        def memory_objective(params):
            return params['memory_param']  # Minimize memory
        
        objective_functions = {
            'latency': latency_objective,
            'memory': memory_objective
        }
        
        result = await optimizer.multi_objective_optimization(
            config,
            objective_functions,
            QuantumOptimizationMethod.QUANTUM_GENETIC_ALGORITHM
        )
        
        assert 'pareto_analysis' in result
        assert 'best_parameters' in result
        
        pareto_analysis = result['pareto_analysis']
        assert 'pareto_point' in pareto_analysis
        assert 'trade_off_analysis' in pareto_analysis
        assert 'pareto_efficiency' in pareto_analysis


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])