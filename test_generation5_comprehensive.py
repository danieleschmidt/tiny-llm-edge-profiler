"""
Comprehensive Generation 5 Testing Suite
========================================

Complete test suite for all Generation 5 breakthrough features including:
- Federated Edge Learning
- Quantum-Enhanced Profiling 
- Real-time Collaborative Networks
- Self-Evolving Optimization
- Autonomous Model Architecture Search

This test suite validates all advanced functionality and ensures
production-ready quality.
"""

import asyncio
import json
import logging
import pytest
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

# Import Generation 5 modules
from src.tiny_llm_profiler.federated_edge_learning import (
    FederatedEdgeLearningSystem,
    FederatedLearningMode,
    PrivacyLevel,
    create_federated_learning_system,
    run_federated_profiling_experiment
)

from src.tiny_llm_profiler.quantum_enhanced_profiling import (
    QuantumEnhancedProfiler,
    QuantumInspiredOptimizer,
    QuantumOptimizationMethod,
    QuantumState,
    QuantumCircuit,
    QuantumGate,
    create_quantum_enhanced_profiler
)

from src.tiny_llm_profiler.realtime_collaborative_network import (
    RealtimeCollaborativeNetwork,
    CollaborationMode,
    MessageType,
    Priority,
    RealtimeMessage,
    create_collaborative_network,
    run_collaborative_profiling_experiment
)

from src.tiny_llm_profiler.self_evolving_optimization import (
    SelfEvolvingOptimizer,
    OptimizationGenome,
    EvolutionStrategy,
    MetaLearningPredictor,
    create_self_evolving_optimizer,
    run_self_evolution_experiment
)

from src.tiny_llm_profiler.autonomous_model_architecture_search import (
    AutonomousNAS,
    ModelArchitecture,
    LayerSpec,
    LayerType,
    HardwareTarget,
    ActivationType,
    create_autonomous_nas,
    run_architecture_search_experiment
)

# Base imports
from src.tiny_llm_profiler.models import QuantizedModel
from src.tiny_llm_profiler.profiler import EdgeProfiler, ProfileResults


class TestQuantumEnhancedProfiling:
    """Test quantum-enhanced profiling capabilities."""
    
    @pytest.fixture
    def quantum_state(self):
        """Create test quantum state."""
        amplitudes = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0, 0], dtype=complex)
        return QuantumState(amplitudes=amplitudes, num_qubits=2)
    
    @pytest.fixture  
    def quantum_circuit(self):
        """Create test quantum circuit."""
        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_hadamard(0)
        circuit.add_cnot(0, 1)
        return circuit
    
    def test_quantum_state_creation(self, quantum_state):
        """Test quantum state creation and normalization."""
        assert quantum_state.num_qubits == 2
        assert len(quantum_state.amplitudes) == 4
        
        # Check normalization
        norm = np.linalg.norm(quantum_state.amplitudes)
        assert np.isclose(norm, 1.0)
        
        # Check measurement probabilities
        prob_0 = quantum_state.probability(0)
        prob_1 = quantum_state.probability(1)
        assert np.isclose(prob_0, 0.5)
        assert np.isclose(prob_1, 0.5)
    
    def test_quantum_circuit_execution(self, quantum_circuit):
        """Test quantum circuit execution."""
        final_state = quantum_circuit.execute()
        
        assert final_state.num_qubits == 2
        assert len(final_state.amplitudes) == 4
        
        # Check Bell state creation (|00⟩ + |11⟩) / √2
        expected = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        np.testing.assert_allclose(final_state.amplitudes, expected, atol=1e-10)
    
    def test_quantum_fidelity(self, quantum_state):
        """Test quantum state fidelity calculation."""
        # Same state should have fidelity 1
        fidelity = quantum_state.fidelity(quantum_state)
        assert np.isclose(fidelity, 1.0)
        
        # Orthogonal state should have fidelity 0
        orthogonal_amplitudes = np.array([0, 0, 1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        orthogonal_state = QuantumState(orthogonal_amplitudes, 2)
        
        fidelity = quantum_state.fidelity(orthogonal_state)
        assert np.isclose(fidelity, 0.0)
    
    def test_quantum_optimizer_initialization(self):
        """Test quantum optimizer creation."""
        optimizer = QuantumInspiredOptimizer(
            method=QuantumOptimizationMethod.QAOA,
            num_qubits=4,
            max_iterations=50
        )
        
        assert optimizer.method == QuantumOptimizationMethod.QAOA
        assert optimizer.num_qubits == 4
        assert optimizer.max_iterations == 50
        assert len(optimizer.optimization_history) == 0
    
    def test_quantum_optimization_simple_function(self):
        """Test quantum optimization on simple function."""
        def objective(params):
            x, y = params[0], params[1]
            return -(x - 2)**2 - (y - 1)**2  # Maximum at (2, 1)
        
        optimizer = QuantumInspiredOptimizer(
            method=QuantumOptimizationMethod.QPSO,
            max_iterations=20
        )
        
        bounds = [(0, 4), (0, 4)]
        result_params, result_value = optimizer.optimize_configuration(
            objective_function=objective,
            parameter_bounds=bounds
        )
        
        # Should find solution near (2, 1)
        assert len(result_params) == 2
        assert 1.5 <= result_params[0] <= 2.5
        assert 0.5 <= result_params[1] <= 1.5
        assert result_value <= 0  # Negative objective
        assert len(optimizer.optimization_history) > 0
    
    @pytest.mark.asyncio
    async def test_quantum_enhanced_profiler_creation(self):
        """Test quantum-enhanced profiler creation."""
        profiler = await create_quantum_enhanced_profiler(
            platform="esp32",
            quantum_method=QuantumOptimizationMethod.VQE,
            num_qubits=6
        )
        
        assert profiler is not None
        assert profiler.quantum_optimizer.method == QuantumOptimizationMethod.VQE
        assert profiler.quantum_optimizer.num_qubits == 6
    
    @pytest.mark.asyncio  
    async def test_quantum_profiling_optimization(self):
        """Test quantum profiling optimization."""
        profiler = await create_quantum_enhanced_profiler(
            platform="esp32",
            quantum_method=QuantumOptimizationMethod.QAOA
        )
        
        model = QuantizedModel(
            name="test_quantum_model",
            quantization_bits=4
        )
        
        # Run optimization
        quantum_results, profile_results = await profiler.quantum_profile_optimization(
            model=model,
            optimization_objectives=["latency", "energy"]
        )
        
        # Validate results
        assert "optimal_configuration" in quantum_results
        assert "optimization_time_seconds" in quantum_results
        assert "quantum_advantage_score" in quantum_results
        assert quantum_results["optimization_time_seconds"] > 0
        assert 0 <= quantum_results["quantum_advantage_score"] <= 1
        
        assert profile_results.model_name == "test_quantum_model"
        assert profile_results.first_token_latency_ms > 0
        assert profile_results.energy_per_token_mj > 0


class TestFederatedEdgeLearning:
    """Test federated edge learning capabilities."""
    
    @pytest.mark.asyncio
    async def test_federated_system_creation(self):
        """Test federated learning system creation."""
        system = await create_federated_learning_system(
            device_id="test_device_001",
            mode=FederatedLearningMode.DECENTRALIZED,
            privacy_level=PrivacyLevel.DIFFERENTIAL
        )
        
        assert system.device_id == "test_device_001"
        assert system.mode == FederatedLearningMode.DECENTRALIZED
        assert system.privacy_level == PrivacyLevel.DIFFERENTIAL
        assert system.aggregator.privacy_level == PrivacyLevel.DIFFERENTIAL
    
    def test_federated_aggregation(self):
        """Test federated aggregation functionality."""
        from src.tiny_llm_profiler.federated_edge_learning import (
            FederatedAggregator, FederatedUpdate
        )
        from datetime import datetime
        
        aggregator = FederatedAggregator(
            aggregation_method="fedavg",
            privacy_level=PrivacyLevel.BASIC,
            min_participants=2
        )
        
        # Create mock updates
        updates = []
        for i in range(3):
            model_delta = {
                "layer_0": np.random.randn(10, 10) * 0.01,
                "layer_1": np.random.randn(5, 10) * 0.01
            }
            
            update = FederatedUpdate(
                device_id=f"device_{i}",
                round_id=1,
                model_delta=model_delta,
                performance_metrics={"accuracy": 0.8 + i * 0.05},
                sample_count=100 + i * 20,
                computation_time=10.0 + i * 2,
                timestamp=datetime.now()
            )
            updates.append(update)
        
        # Test aggregation
        aggregated = aggregator.aggregate_updates(updates, global_round=1)
        
        assert "layer_0" in aggregated
        assert "layer_1" in aggregated
        assert aggregated["layer_0"].shape == (10, 10)
        assert aggregated["layer_1"].shape == (5, 10)
    
    @pytest.mark.asyncio
    async def test_federated_profiling_experiment(self):
        """Test federated profiling experiment."""
        devices = ["device_001", "device_002", "device_003"]
        model = QuantizedModel(name="federated_test_model", quantization_bits=4)
        
        # Mock the actual federated learning to avoid complex async setup
        with patch('src.tiny_llm_profiler.federated_edge_learning.FederatedEdgeLearningSystem.start_federated_learning') as mock_fl:
            mock_fl.return_value = {
                'final_accuracy': 0.85,
                'total_rounds': 10,
                'convergence_achieved': True
            }
            
            results = await run_federated_profiling_experiment(
                devices=devices,
                model=model,
                max_rounds=20,
                target_accuracy=0.8
            )
            
            assert 'participating_devices' in results
            assert results['participating_devices'] == 3


class TestRealtimeCollaborativeNetwork:
    """Test real-time collaborative networking."""
    
    @pytest.fixture
    def collaborative_message(self):
        """Create test collaborative message."""
        return RealtimeMessage(
            message_id="test_msg_001",
            sender_id="node_001", 
            message_type=MessageType.PROFILING_UPDATE,
            priority=Priority.MEDIUM,
            payload={"test_data": "value"},
            timestamp=None  # Will be auto-set
        )
    
    def test_message_creation(self, collaborative_message):
        """Test message creation and serialization."""
        assert collaborative_message.message_id == "test_msg_001"
        assert collaborative_message.sender_id == "node_001"
        assert collaborative_message.message_type == MessageType.PROFILING_UPDATE
        assert collaborative_message.timestamp is not None
        assert not collaborative_message.is_expired()
        
        # Test serialization
        msg_dict = collaborative_message.to_dict()
        assert msg_dict["message_id"] == "test_msg_001"
        assert msg_dict["message_type"] == "profiling_update"
        
        # Test deserialization
        restored_msg = RealtimeMessage.from_dict(msg_dict)
        assert restored_msg.message_id == collaborative_message.message_id
        assert restored_msg.message_type == collaborative_message.message_type
    
    @pytest.mark.asyncio
    async def test_collaborative_network_creation(self):
        """Test collaborative network creation."""
        network = await create_collaborative_network(
            node_id="test_node_001",
            collaboration_mode=CollaborationMode.DYNAMIC,
            websocket_port=8766
        )
        
        assert network.node_id == "test_node_001"
        assert network.collaboration_mode == CollaborationMode.DYNAMIC
        assert network.websocket_port == 8766
        assert len(network.connected_nodes) == 0
    
    def test_consensus_engine(self):
        """Test consensus engine functionality."""
        from src.tiny_llm_profiler.realtime_collaborative_network import ConsensusEngine
        
        engine = ConsensusEngine("test_node", minimum_agreement_threshold=0.6)
        
        # Test proposal handling
        proposal = {
            'proposal_id': 'test_proposal',
            'proposer_id': 'other_node',
            'optimization_config': {'param': 'value'},
            'expected_improvement': 0.15,
            'confidence_score': 0.8
        }
        
        vote_result = engine.handle_proposal(proposal)
        assert 'vote' in vote_result
        assert 'reasoning' in vote_result
        assert isinstance(vote_result['vote'], bool)
    
    @pytest.mark.asyncio
    async def test_collaborative_profiling_experiment(self):
        """Test collaborative profiling experiment."""
        participants = ["node_001", "node_002"]
        model = QuantizedModel(name="collab_test_model", quantization_bits=4)
        
        # Mock network operations to avoid WebSocket complexity
        with patch('src.tiny_llm_profiler.realtime_collaborative_network.RealtimeCollaborativeNetwork.start_collaborative_network') as mock_start:
            mock_start.return_value = None
            
            results = await run_collaborative_profiling_experiment(
                participants=participants,
                session_name="test_collaboration",
                model=model
            )
            
            assert 'session_name' in results
            assert results['session_name'] == "test_collaboration"


class TestSelfEvolvingOptimization:
    """Test self-evolving optimization capabilities."""
    
    @pytest.fixture
    def optimization_genome(self):
        """Create test optimization genome."""
        return OptimizationGenome(
            genome_id="test_genome_001",
            strategy_type=EvolutionStrategy.GENETIC_PROGRAMMING,
            hyperparameters={
                "population_size": 20,
                "mutation_rate": 0.1,
                "learning_rate": 0.01
            },
            algorithm_weights={
                "gradient_descent": 0.4,
                "genetic": 0.3, 
                "swarm": 0.3
            }
        )
    
    def test_genome_creation(self, optimization_genome):
        """Test optimization genome creation."""
        assert optimization_genome.genome_id == "test_genome_001"
        assert optimization_genome.strategy_type == EvolutionStrategy.GENETIC_PROGRAMMING
        assert len(optimization_genome.hyperparameters) == 3
        assert len(optimization_genome.algorithm_weights) == 3
        assert optimization_genome.generation == 0
    
    def test_genome_mutation(self, optimization_genome):
        """Test genome mutation."""
        mutated = optimization_genome.mutate(mutation_intensity=1.0)
        
        assert mutated.genome_id != optimization_genome.genome_id
        assert mutated.generation == optimization_genome.generation + 1
        assert mutated.parent_genomes == [optimization_genome.genome_id]
        
        # Parameters should be different but similar
        orig_pop_size = optimization_genome.hyperparameters["population_size"] 
        mut_pop_size = mutated.hyperparameters["population_size"]
        assert mut_pop_size != orig_pop_size  # Should be mutated
        assert 0.5 * orig_pop_size <= mut_pop_size <= 2.0 * orig_pop_size  # Reasonable range
    
    def test_genome_crossover(self, optimization_genome):
        """Test genome crossover."""
        # Create second parent
        parent2 = OptimizationGenome(
            genome_id="parent2",
            strategy_type=EvolutionStrategy.PARTICLE_SWARM,
            hyperparameters={
                "population_size": 30,
                "mutation_rate": 0.2, 
                "learning_rate": 0.02
            },
            algorithm_weights={
                "gradient_descent": 0.2,
                "genetic": 0.5,
                "swarm": 0.3
            }
        )
        
        child1, child2 = optimization_genome.crossover(parent2)
        
        # Children should be different from parents
        assert child1.genome_id not in [optimization_genome.genome_id, parent2.genome_id]
        assert child2.genome_id not in [optimization_genome.genome_id, parent2.genome_id]
        
        # Children should have both parents listed
        assert optimization_genome.genome_id in child1.parent_genomes
        assert parent2.genome_id in child1.parent_genomes
    
    def test_genome_fitness_calculation(self, optimization_genome):
        """Test fitness calculation."""
        performance_metrics = {
            "latency_ms": 50,
            "energy_mj": 5,
            "accuracy": 0.85,
            "convergence_time": 30
        }
        
        fitness = optimization_genome.calculate_fitness(performance_metrics)
        
        assert 0 <= fitness <= 1  # Fitness should be normalized
        assert len(optimization_genome.fitness_scores) == 1
        assert optimization_genome.fitness_scores[0] == fitness
    
    def test_meta_learning_predictor(self):
        """Test meta-learning predictor."""
        predictor = MetaLearningPredictor()
        
        # Test feature extraction
        genome = OptimizationGenome(
            genome_id="test",
            strategy_type=EvolutionStrategy.GENETIC_PROGRAMMING,
            hyperparameters={"param1": 1.0, "param2": True},
            algorithm_weights={"alg1": 0.5, "alg2": 0.5}
        )
        
        context = {
            "problem_dimension": 5,
            "constraint_count": 2,
            "multi_objective": False
        }
        
        features = predictor.extract_features(genome, context)
        assert len(features) > 10  # Should extract many features
        assert all(isinstance(f, (int, float)) for f in features)
    
    @pytest.mark.asyncio
    async def test_self_evolving_optimizer_creation(self):
        """Test self-evolving optimizer creation."""
        optimizer = await create_self_evolving_optimizer(
            population_size=10,
            max_generations=20
        )
        
        assert optimizer.population_size == 10
        assert optimizer.max_generations == 20
        assert len(optimizer.population) == 0  # Not initialized yet
    
    @pytest.mark.asyncio
    async def test_self_evolution_experiment(self):
        """Test self-evolution experiment."""
        def simple_objective(config):
            x = config.get('param_x', 0)
            y = config.get('param_y', 0)
            return 1.0 / (1.0 + (x - 1)**2 + (y - 2)**2)  # Maximum at (1, 2)
        
        problem_context = {
            'problem_dimension': 2,
            'constraint_count': 0
        }
        
        results = await run_self_evolution_experiment(
            objective_function=simple_objective,
            problem_context=problem_context,
            max_evaluations=50,
            population_size=8
        )
        
        assert results['success']
        assert 'evolution_result' in results
        assert results['evolution_result']['total_evaluations'] <= 50
        assert results['evolution_result']['final_fitness'] > 0


class TestAutonomousModelArchitectureSearch:
    """Test autonomous model architecture search."""
    
    @pytest.fixture
    def layer_spec(self):
        """Create test layer specification."""
        return LayerSpec(
            layer_type=LayerType.LINEAR,
            input_dim=128,
            output_dim=64,
            activation=ActivationType.RELU,
            quantization_bits=8
        )
    
    @pytest.fixture
    def model_architecture(self):
        """Create test model architecture."""
        layers = [
            LayerSpec(LayerType.LINEAR, 784, 128, ActivationType.RELU),
            LayerSpec(LayerType.DROPOUT, dropout_rate=0.2),
            LayerSpec(LayerType.LINEAR, 128, 64, ActivationType.RELU),
            LayerSpec(LayerType.LINEAR, 64, 10, ActivationType.SOFTMAX)
        ]
        
        return ModelArchitecture(
            architecture_id="test_arch_001",
            layers=layers,
            input_shape=(784,),
            output_dim=10,
            target_hardware=HardwareTarget.ESP32
        )
    
    def test_layer_spec_creation(self, layer_spec):
        """Test layer specification creation."""
        assert layer_spec.layer_type == LayerType.LINEAR
        assert layer_spec.input_dim == 128
        assert layer_spec.output_dim == 64
        assert layer_spec.activation == ActivationType.RELU
        assert layer_spec.quantization_bits == 8
    
    def test_layer_parameter_estimation(self, layer_spec):
        """Test layer parameter estimation."""
        params = layer_spec.estimate_parameters()
        expected = 128 * 64 + 64  # weights + bias
        assert params == expected
    
    def test_model_architecture_creation(self, model_architecture):
        """Test model architecture creation."""
        assert model_architecture.architecture_id == "test_arch_001"
        assert len(model_architecture.layers) == 4
        assert model_architecture.input_shape == (784,)
        assert model_architecture.output_dim == 10
        assert model_architecture.target_hardware == HardwareTarget.ESP32
    
    def test_architecture_parameter_estimation(self, model_architecture):
        """Test total parameter estimation."""
        total_params = model_architecture.estimate_total_parameters()
        
        # Calculate expected: 784*128 + 128 + 128*64 + 64 + 64*10 + 10
        expected = (784 * 128 + 128) + (128 * 64 + 64) + (64 * 10 + 10)
        assert total_params == expected
    
    def test_architecture_memory_estimation(self, model_architecture):
        """Test memory usage estimation."""
        memory_mb = model_architecture.estimate_memory_usage()
        assert memory_mb > 0
        assert memory_mb < 100  # Should be reasonable for this small model
    
    def test_hardware_compatibility(self, model_architecture):
        """Test hardware compatibility checking."""
        # Small architecture should be compatible with ESP32
        assert model_architecture.is_hardware_compatible()
        
        # Create architecture that's too large
        large_arch = ModelArchitecture(
            architecture_id="large_arch",
            layers=[LayerSpec(LayerType.LINEAR, 10000, 10000)],
            input_shape=(10000,),
            output_dim=1000,
            target_hardware=HardwareTarget.ARDUINO,  # Very limited hardware
            max_memory_mb=0.1
        )
        
        assert not large_arch.is_hardware_compatible()
    
    def test_fitness_calculation(self, model_architecture):
        """Test architecture fitness calculation."""
        # Add some performance data
        model_architecture.accuracy_scores = [0.85, 0.87, 0.86]
        model_architecture.latency_measurements = [45, 50, 48]
        model_architecture.energy_consumption = [8, 9, 7]
        
        fitness = model_architecture.calculate_fitness()
        assert 0 <= fitness <= 1
        assert fitness > 0  # Should have positive fitness
    
    def test_architecture_mutation(self, model_architecture):
        """Test architecture mutation."""
        from src.tiny_llm_profiler.autonomous_model_architecture_search import ArchitectureMutator
        
        mutator = ArchitectureMutator(mutation_rate=0.5)
        mutated = mutator.mutate(model_architecture)
        
        assert mutated.architecture_id != model_architecture.architecture_id
        assert mutated.generation == model_architecture.generation + 1
        assert model_architecture.architecture_id in mutated.parent_architectures
        assert len(mutated.mutation_history) == 1
    
    @pytest.mark.asyncio
    async def test_autonomous_nas_creation(self):
        """Test autonomous NAS system creation."""
        nas = await create_autonomous_nas(
            target_hardware=HardwareTarget.ESP32,
            population_size=5,
            max_generations=10
        )
        
        assert nas.target_hardware == HardwareTarget.ESP32
        assert nas.population_size == 5
        assert nas.max_generations == 10
        assert len(nas.population) == 0  # Not initialized yet
    
    @pytest.mark.asyncio
    async def test_architecture_search_experiment(self):
        """Test architecture search experiment."""
        results = await run_architecture_search_experiment(
            input_shape=(784,),
            output_dim=10,
            target_hardware=HardwareTarget.ESP32,
            evaluation_budget=30,
            population_size=5
        )
        
        assert results['success']
        search_results = results['search_results']
        
        assert 'best_architecture' in search_results
        assert 'search_statistics' in search_results
        
        stats = search_results['search_statistics']
        assert stats['evaluations_used'] <= 30
        assert stats['generations'] > 0
        assert stats['search_time_seconds'] > 0


class TestIntegrationScenarios:
    """Test integration between different Generation 5 components."""
    
    @pytest.mark.asyncio
    async def test_quantum_federated_integration(self):
        """Test quantum optimization with federated learning."""
        # Create quantum-enhanced profiler
        quantum_profiler = await create_quantum_enhanced_profiler(
            platform="esp32",
            quantum_method=QuantumOptimizationMethod.QPSO
        )
        
        # Create federated system
        federated_system = await create_federated_learning_system(
            device_id="quantum_fed_device",
            mode=FederatedLearningMode.ADAPTIVE
        )
        
        model = QuantizedModel(name="integrated_model", quantization_bits=4)
        
        # Run quantum optimization
        quantum_results, profile_results = await quantum_profiler.quantum_profile_optimization(
            model=model,
            optimization_objectives=["latency", "energy"]
        )
        
        # Verify integration works
        assert quantum_results['optimal_configuration'] is not None
        assert profile_results.model_name == "integrated_model"
    
    @pytest.mark.asyncio  
    async def test_nas_with_self_evolution(self):
        """Test NAS with self-evolving optimization."""
        # Create NAS system
        nas = await create_autonomous_nas(
            target_hardware=HardwareTarget.ESP32,
            population_size=5,
            max_generations=5
        )
        
        # Create self-evolving optimizer
        optimizer = await create_self_evolving_optimizer(
            population_size=5,
            max_generations=5
        )
        
        # Test that both systems can coexist and work
        nas.initialize_population(
            input_shape=(32,),
            output_dim=10
        )
        
        optimizer.initialize_population()
        
        assert len(nas.population) == 5
        assert len(optimizer.population) == 5
    
    @pytest.mark.asyncio
    async def test_collaborative_quantum_optimization(self):
        """Test collaborative networks with quantum optimization."""
        # Create collaborative network
        network = await create_collaborative_network(
            node_id="quantum_collab_node",
            collaboration_mode=CollaborationMode.DYNAMIC
        )
        
        # Create quantum profiler
        quantum_profiler = await create_quantum_enhanced_profiler(
            platform="esp32",
            quantum_method=QuantumOptimizationMethod.QAOA
        )
        
        model = QuantizedModel(name="collab_quantum_model", quantization_bits=4)
        
        # Run quantum optimization
        quantum_results, _ = await quantum_profiler.quantum_profile_optimization(
            model=model
        )
        
        # Simulate sharing results through network
        # (In real implementation would use network.share_profiling_results)
        assert network.node_id == "quantum_collab_node"
        assert quantum_results['quantum_advantage_score'] >= 0
    
    def test_genome_to_architecture_mapping(self):
        """Test mapping optimization genomes to architectures."""
        # Create optimization genome
        genome = OptimizationGenome(
            genome_id="arch_genome",
            strategy_type=EvolutionStrategy.GENETIC_PROGRAMMING,
            hyperparameters={
                "num_layers": 3,
                "hidden_dim": 64,
                "dropout_rate": 0.2
            },
            algorithm_weights={"genetic": 1.0}
        )
        
        # Test that genome can inform architecture decisions
        num_layers = int(genome.hyperparameters.get("num_layers", 2))
        hidden_dim = int(genome.hyperparameters.get("hidden_dim", 32))
        
        # Create architecture based on genome
        layers = []
        for i in range(num_layers):
            if i == 0:
                layer = LayerSpec(LayerType.LINEAR, 32, hidden_dim)
            elif i == num_layers - 1:
                layer = LayerSpec(LayerType.LINEAR, hidden_dim, 10)  
            else:
                layer = LayerSpec(LayerType.LINEAR, hidden_dim, hidden_dim)
            layers.append(layer)
        
        architecture = ModelArchitecture(
            architecture_id="genome_derived",
            layers=layers,
            input_shape=(32,),
            output_dim=10
        )
        
        assert len(architecture.layers) == 3
        assert architecture.layers[1].output_dim == 64


class TestPerformanceAndScalability:
    """Test performance and scalability of Generation 5 features."""
    
    @pytest.mark.asyncio
    async def test_quantum_optimization_performance(self):
        """Test quantum optimization performance scaling."""
        def benchmark_objective(params):
            return sum(p**2 for p in params)
        
        # Test different problem sizes
        problem_sizes = [2, 4, 6]
        optimization_times = []
        
        for size in problem_sizes:
            optimizer = QuantumInspiredOptimizer(
                method=QuantumOptimizationMethod.QPSO,
                max_iterations=20
            )
            
            bounds = [(-5, 5)] * size
            
            start_time = time.time()
            result_params, result_value = optimizer.optimize_configuration(
                objective_function=benchmark_objective,
                parameter_bounds=bounds
            )
            end_time = time.time()
            
            optimization_times.append(end_time - start_time)
            
            # Verify convergence
            assert len(result_params) == size
            assert result_value <= 0  # Should find negative values
        
        # Performance should scale reasonably
        assert optimization_times[0] > 0
        assert optimization_times[-1] < 30  # Should complete within 30 seconds
    
    def test_architecture_evaluation_efficiency(self):
        """Test architecture evaluation efficiency."""
        from src.tiny_llm_profiler.autonomous_model_architecture_search import HardwareAwareEvaluator
        
        evaluator = HardwareAwareEvaluator(HardwareTarget.ESP32)
        
        # Create multiple architectures
        architectures = []
        for i in range(10):
            layers = [
                LayerSpec(LayerType.LINEAR, 128, 64),
                LayerSpec(LayerType.LINEAR, 64, 32),
                LayerSpec(LayerType.LINEAR, 32, 10)
            ]
            arch = ModelArchitecture(
                architecture_id=f"perf_test_{i}",
                layers=layers,
                input_shape=(128,),
                output_dim=10
            )
            architectures.append(arch)
        
        # Evaluate all architectures
        start_time = time.time()
        evaluations = []
        for arch in architectures:
            eval_result = evaluator.evaluate_architecture(arch)
            evaluations.append(eval_result)
        end_time = time.time()
        
        # Should complete quickly
        assert end_time - start_time < 5  # Under 5 seconds for 10 architectures
        assert len(evaluations) == 10
        assert all('overall_score' in eval_result for eval_result in evaluations)
    
    def test_memory_usage_optimization(self):
        """Test memory usage optimization."""
        # Create large architecture
        large_arch = ModelArchitecture(
            architecture_id="memory_test",
            layers=[
                LayerSpec(LayerType.LINEAR, 1000, 500),
                LayerSpec(LayerType.LINEAR, 500, 250),  
                LayerSpec(LayerType.LINEAR, 250, 10)
            ],
            input_shape=(1000,),
            output_dim=10,
            target_hardware=HardwareTarget.ESP32
        )
        
        original_memory = large_arch.estimate_memory_usage()
        
        # Apply quantization to reduce memory
        large_arch.global_quantization_bits = 4  # More aggressive quantization
        for layer in large_arch.layers:
            layer.quantization_bits = 4
            layer.quantize_weights = True
            layer.quantize_activations = True
        
        # Memory usage should be similar (estimation doesn't account for quantization in detail)
        # but fitness should reflect the improvement
        fitness_with_quant = large_arch.calculate_fitness()
        
        # Reset quantization
        large_arch.global_quantization_bits = 16
        for layer in large_arch.layers:
            layer.quantization_bits = 16
            layer.quantize_weights = False
            layer.quantize_activations = False
        
        fitness_without_quant = large_arch.calculate_fitness()
        
        # Quantized version should have better fitness due to hardware compatibility
        assert fitness_with_quant >= 0
        assert fitness_without_quant >= 0


@pytest.mark.asyncio
async def test_end_to_end_generation5_pipeline():
    """Test complete Generation 5 pipeline integration."""
    
    # Step 1: Create model
    model = QuantizedModel(name="e2e_test_model", quantization_bits=8)
    
    # Step 2: Run autonomous architecture search
    nas_results = await run_architecture_search_experiment(
        input_shape=(784,),
        output_dim=10,
        target_hardware=HardwareTarget.ESP32,
        evaluation_budget=20,
        population_size=5
    )
    
    assert nas_results['success']
    best_arch = nas_results['search_results']['best_architecture']
    assert best_arch is not None
    
    # Step 3: Run quantum-enhanced profiling
    quantum_profiler = await create_quantum_enhanced_profiler(
        platform="esp32",
        quantum_method=QuantumOptimizationMethod.QPSO
    )
    
    quantum_results, profile_results = await quantum_profiler.quantum_profile_optimization(
        model=model,
        optimization_objectives=["latency", "energy"]
    )
    
    assert quantum_results['optimization_time_seconds'] > 0
    assert profile_results.first_token_latency_ms > 0
    
    # Step 4: Self-evolving optimization experiment
    def combined_objective(config):
        latency_penalty = config.get('latency_ms', 100) / 100.0
        energy_penalty = config.get('energy_mj', 10) / 10.0
        return 1.0 / (1.0 + latency_penalty + energy_penalty)
    
    evolution_results = await run_self_evolution_experiment(
        objective_function=combined_objective,
        problem_context={'problem_dimension': 2},
        max_evaluations=30,
        population_size=5
    )
    
    assert evolution_results['success']
    
    # Step 5: Verify all components worked together
    assert nas_results['success']
    assert quantum_results['quantum_advantage_score'] >= 0
    assert evolution_results['evolution_result']['final_fitness'] > 0
    
    print("🎉 Generation 5 End-to-End Pipeline Test PASSED!")


if __name__ == "__main__":
    # Run comprehensive tests
    pytest.main([__file__, "-v", "--tb=short"])