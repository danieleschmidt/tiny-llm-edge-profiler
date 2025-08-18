"""
Neuromorphic Computing Framework
Bio-inspired computing patterns for adaptive edge AI profiling.
"""

import asyncio
import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

from .exceptions import ProfilerError


class NeuronType(str, Enum):
    """Types of artificial neurons in neuromorphic system."""
    LEAKY_INTEGRATE_FIRE = "lif"
    ADAPTIVE_EXPONENTIAL = "aef"
    IZHIKEVICH = "izhikevich"
    HODGKIN_HUXLEY = "hodgkin_huxley"
    SPIKE_RESPONSE = "spike_response"


class SynapticPlasticityType(str, Enum):
    """Types of synaptic plasticity mechanisms."""
    SPIKE_TIMING_DEPENDENT = "stdp"
    HOMEOSTATIC = "homeostatic"
    METAPLASTICITY = "metaplasticity"
    REWARD_MODULATED = "reward_modulated"
    STRUCTURAL = "structural"


class NetworkTopology(str, Enum):
    """Network topology patterns."""
    FEEDFORWARD = "feedforward"
    RECURRENT = "recurrent"
    RESERVOIR = "reservoir"
    SMALL_WORLD = "small_world"
    SCALE_FREE = "scale_free"
    HIERARCHICAL = "hierarchical"


@dataclass
class SpikingNeuron:
    """Spiking neuron model with adaptive properties."""
    neuron_id: str
    neuron_type: NeuronType
    membrane_potential: float = -70.0  # mV
    threshold: float = -55.0  # mV
    resting_potential: float = -70.0  # mV
    membrane_capacitance: float = 1.0  # nF
    membrane_resistance: float = 10.0  # MΩ
    refractory_period: float = 2.0  # ms
    last_spike_time: Optional[float] = None
    adaptation_current: float = 0.0
    spike_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Adaptive parameters
    threshold_adaptation: bool = True
    adaptation_time_constant: float = 100.0  # ms
    threshold_increase: float = 5.0  # mV
    
    def update(self, dt: float, input_current: float, current_time: float) -> bool:
        """
        Update neuron state and return True if spike occurred.
        
        Args:
            dt: Time step in milliseconds
            input_current: Input current in nA
            current_time: Current simulation time in ms
            
        Returns:
            True if neuron spiked, False otherwise
        """
        # Check refractory period
        if (self.last_spike_time is not None and 
            current_time - self.last_spike_time < self.refractory_period):
            return False
        
        # Update based on neuron type
        if self.neuron_type == NeuronType.LEAKY_INTEGRATE_FIRE:
            return self._update_lif(dt, input_current, current_time)
        elif self.neuron_type == NeuronType.ADAPTIVE_EXPONENTIAL:
            return self._update_aef(dt, input_current, current_time)
        elif self.neuron_type == NeuronType.IZHIKEVICH:
            return self._update_izhikevich(dt, input_current, current_time)
        else:
            return self._update_lif(dt, input_current, current_time)  # Default to LIF
    
    def _update_lif(self, dt: float, input_current: float, current_time: float) -> bool:
        """Update Leaky Integrate-and-Fire neuron."""
        # Membrane dynamics
        tau_m = self.membrane_resistance * self.membrane_capacitance
        
        # Update membrane potential
        dv_dt = (-(self.membrane_potential - self.resting_potential) + 
                self.membrane_resistance * (input_current - self.adaptation_current)) / tau_m
        
        self.membrane_potential += dv_dt * dt
        
        # Check for spike
        if self.membrane_potential >= self.threshold:
            self._spike(current_time)
            return True
        
        # Update adaptation current
        if self.adaptation_current > 0:
            self.adaptation_current *= np.exp(-dt / self.adaptation_time_constant)
        
        return False
    
    def _update_aef(self, dt: float, input_current: float, current_time: float) -> bool:
        """Update Adaptive Exponential Integrate-and-Fire neuron."""
        # Additional parameters for AEF
        delta_t = 2.0  # mV, sharpness of exponential
        v_t = -50.0   # mV, threshold potential
        
        # Exponential term
        exp_term = delta_t * np.exp((self.membrane_potential - v_t) / delta_t)
        
        # Membrane dynamics with exponential nonlinearity
        tau_m = self.membrane_resistance * self.membrane_capacitance
        dv_dt = (-(self.membrane_potential - self.resting_potential) + exp_term +
                self.membrane_resistance * (input_current - self.adaptation_current)) / tau_m
        
        self.membrane_potential += dv_dt * dt
        
        # Check for spike (using a higher threshold for AEF)
        if self.membrane_potential >= 20.0:  # mV
            self._spike(current_time)
            return True
        
        # Update adaptation current
        if self.adaptation_current > 0:
            self.adaptation_current *= np.exp(-dt / self.adaptation_time_constant)
        
        return False
    
    def _update_izhikevich(self, dt: float, input_current: float, current_time: float) -> bool:
        """Update Izhikevich neuron model."""
        # Izhikevich parameters (regular spiking)
        a, b, c, d = 0.02, 0.2, -65.0, 8.0
        
        # Recovery variable (not implemented in basic version)
        # This is a simplified implementation
        v = self.membrane_potential
        
        # Izhikevich dynamics (simplified)
        dv_dt = 0.04 * v**2 + 5 * v + 140 + input_current
        self.membrane_potential += dv_dt * dt
        
        # Check for spike
        if self.membrane_potential >= 30.0:  # mV
            self._spike(current_time)
            self.membrane_potential = c  # Reset
            return True
        
        return False
    
    def _spike(self, current_time: float) -> None:
        """Handle spike generation."""
        self.last_spike_time = current_time
        self.spike_history.append(current_time)
        
        # Reset membrane potential
        self.membrane_potential = self.resting_potential
        
        # Increase adaptation current
        self.adaptation_current += 5.0  # nA
        
        # Adaptive threshold
        if self.threshold_adaptation:
            self.threshold += self.threshold_increase
    
    def get_firing_rate(self, time_window: float = 1000.0) -> float:
        """Calculate firing rate over recent time window."""
        if not self.spike_history:
            return 0.0
        
        current_time = self.spike_history[-1] if self.spike_history else 0
        recent_spikes = [t for t in self.spike_history if current_time - t <= time_window]
        
        return len(recent_spikes) / (time_window / 1000.0)  # Hz
    
    def reset_adaptation(self) -> None:
        """Reset adaptive properties."""
        self.threshold = -55.0  # Reset to default
        self.adaptation_current = 0.0


@dataclass
class Synapse:
    """Synaptic connection between neurons with plasticity."""
    synapse_id: str
    pre_neuron_id: str
    post_neuron_id: str
    weight: float
    delay: float = 1.0  # ms
    plasticity_type: SynapticPlasticityType = SynapticPlasticityType.SPIKE_TIMING_DEPENDENT
    
    # STDP parameters
    tau_plus: float = 20.0  # ms
    tau_minus: float = 20.0  # ms
    a_plus: float = 0.01
    a_minus: float = 0.012
    
    # Homeostatic parameters
    target_rate: float = 10.0  # Hz
    homeostatic_scaling: float = 0.001
    
    # Weight bounds
    w_min: float = 0.0
    w_max: float = 1.0
    
    # Spike timing tracking
    pre_spike_times: deque = field(default_factory=lambda: deque(maxlen=100))
    post_spike_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def transmit_spike(self, pre_spike_time: float, current_time: float) -> float:
        """
        Transmit spike and return synaptic current.
        
        Args:
            pre_spike_time: Time of presynaptic spike
            current_time: Current simulation time
            
        Returns:
            Synaptic current to inject into postsynaptic neuron
        """
        # Check if spike should arrive now
        if current_time >= pre_spike_time + self.delay:
            # Record presynaptic spike
            self.pre_spike_times.append(pre_spike_time)
            
            # Return synaptic current
            return self.weight * self._synaptic_kernel(current_time - pre_spike_time - self.delay)
        
        return 0.0
    
    def _synaptic_kernel(self, dt: float) -> float:
        """Synaptic current kernel (exponential decay)."""
        tau_syn = 5.0  # ms
        if dt >= 0:
            return np.exp(-dt / tau_syn)
        return 0.0
    
    def update_plasticity(
        self,
        pre_spike_time: Optional[float],
        post_spike_time: Optional[float],
        post_firing_rate: float
    ) -> None:
        """Update synaptic weight based on plasticity rules."""
        if self.plasticity_type == SynapticPlasticityType.SPIKE_TIMING_DEPENDENT:
            self._update_stdp(pre_spike_time, post_spike_time)
        elif self.plasticity_type == SynapticPlasticityType.HOMEOSTATIC:
            self._update_homeostatic(post_firing_rate)
        elif self.plasticity_type == SynapticPlasticityType.REWARD_MODULATED:
            self._update_reward_modulated(pre_spike_time, post_spike_time)
    
    def _update_stdp(
        self,
        pre_spike_time: Optional[float],
        post_spike_time: Optional[float]
    ) -> None:
        """Update weight using Spike-Timing Dependent Plasticity."""
        if pre_spike_time is None or post_spike_time is None:
            return
        
        dt = post_spike_time - pre_spike_time
        
        if dt > 0:  # Post before pre (LTD)
            dw = -self.a_minus * np.exp(-dt / self.tau_minus)
        else:  # Pre before post (LTP)
            dw = self.a_plus * np.exp(dt / self.tau_plus)
        
        # Update weight
        self.weight = np.clip(self.weight + dw, self.w_min, self.w_max)
    
    def _update_homeostatic(self, post_firing_rate: float) -> None:
        """Update weight using homeostatic scaling."""
        rate_error = post_firing_rate - self.target_rate
        dw = -self.homeostatic_scaling * rate_error * self.weight
        
        self.weight = np.clip(self.weight + dw, self.w_min, self.w_max)
    
    def _update_reward_modulated(
        self,
        pre_spike_time: Optional[float],
        post_spike_time: Optional[float],
        reward_signal: float = 0.0
    ) -> None:
        """Update weight using reward-modulated STDP."""
        # Basic STDP modified by reward signal
        if pre_spike_time is None or post_spike_time is None:
            return
        
        dt = post_spike_time - pre_spike_time
        
        if dt > 0:
            stdp_term = -self.a_minus * np.exp(-dt / self.tau_minus)
        else:
            stdp_term = self.a_plus * np.exp(dt / self.tau_plus)
        
        # Modulate by reward
        dw = reward_signal * stdp_term
        self.weight = np.clip(self.weight + dw, self.w_min, self.w_max)


class SpikingNeuralNetwork:
    """Spiking neural network with neuromorphic computing patterns."""
    
    def __init__(
        self,
        network_id: str,
        topology: NetworkTopology = NetworkTopology.FEEDFORWARD
    ):
        self.network_id = network_id
        self.topology = topology
        self.logger = logging.getLogger(__name__)
        
        self.neurons: Dict[str, SpikingNeuron] = {}
        self.synapses: Dict[str, Synapse] = {}
        self.input_neurons: Set[str] = set()
        self.output_neurons: Set[str] = set()
        
        # Network dynamics
        self.current_time = 0.0
        self.dt = 0.1  # ms
        self.spike_buffer: Dict[str, List[float]] = defaultdict(list)
        
        # Learning parameters
        self.learning_enabled = True
        self.reward_signal = 0.0
        
        # Network activity tracking
        self.activity_history: deque = deque(maxlen=10000)
        self.performance_metrics = {}
    
    def add_neuron(
        self,
        neuron_id: str,
        neuron_type: NeuronType = NeuronType.LEAKY_INTEGRATE_FIRE,
        is_input: bool = False,
        is_output: bool = False
    ) -> None:
        """Add neuron to the network."""
        neuron = SpikingNeuron(neuron_id=neuron_id, neuron_type=neuron_type)
        self.neurons[neuron_id] = neuron
        
        if is_input:
            self.input_neurons.add(neuron_id)
        if is_output:
            self.output_neurons.add(neuron_id)
        
        self.logger.debug(f"Added neuron {neuron_id} of type {neuron_type.value}")
    
    def add_synapse(
        self,
        synapse_id: str,
        pre_neuron_id: str,
        post_neuron_id: str,
        weight: float,
        delay: float = 1.0,
        plasticity_type: SynapticPlasticityType = SynapticPlasticityType.SPIKE_TIMING_DEPENDENT
    ) -> None:
        """Add synaptic connection between neurons."""
        if pre_neuron_id not in self.neurons or post_neuron_id not in self.neurons:
            raise ValueError(f"Neurons {pre_neuron_id} or {post_neuron_id} not found")
        
        synapse = Synapse(
            synapse_id=synapse_id,
            pre_neuron_id=pre_neuron_id,
            post_neuron_id=post_neuron_id,
            weight=weight,
            delay=delay,
            plasticity_type=plasticity_type
        )
        
        self.synapses[synapse_id] = synapse
        self.logger.debug(f"Added synapse {synapse_id}: {pre_neuron_id} -> {post_neuron_id}")
    
    def create_network_topology(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int
    ) -> None:
        """Create network with specified topology."""
        # Create input layer
        for i in range(input_size):
            self.add_neuron(f"input_{i}", is_input=True)
        
        # Create hidden layers
        prev_layer_neurons = [f"input_{i}" for i in range(input_size)]
        
        for layer_idx, layer_size in enumerate(hidden_sizes):
            current_layer_neurons = []
            
            for i in range(layer_size):
                neuron_id = f"hidden_{layer_idx}_{i}"
                self.add_neuron(neuron_id, NeuronType.ADAPTIVE_EXPONENTIAL)
                current_layer_neurons.append(neuron_id)
                
                # Connect to previous layer
                for prev_neuron in prev_layer_neurons:
                    weight = np.random.normal(0.5, 0.1)
                    weight = np.clip(weight, 0.1, 1.0)
                    self.add_synapse(
                        f"syn_{prev_neuron}_{neuron_id}",
                        prev_neuron,
                        neuron_id,
                        weight
                    )
            
            prev_layer_neurons = current_layer_neurons
        
        # Create output layer
        for i in range(output_size):
            neuron_id = f"output_{i}"
            self.add_neuron(neuron_id, is_output=True)
            
            # Connect to last hidden layer
            for prev_neuron in prev_layer_neurons:
                weight = np.random.normal(0.5, 0.1)
                weight = np.clip(weight, 0.1, 1.0)
                self.add_synapse(
                    f"syn_{prev_neuron}_{neuron_id}",
                    prev_neuron,
                    neuron_id,
                    weight
                )
        
        self.logger.info(f"Created {self.topology.value} network: {input_size}-{hidden_sizes}-{output_size}")
    
    async def simulate_step(self, input_currents: Dict[str, float]) -> Dict[str, bool]:
        """
        Simulate one time step of the network.
        
        Args:
            input_currents: Dictionary of input currents for neurons
            
        Returns:
            Dictionary of neurons that spiked this step
        """
        spikes_this_step = {}
        
        # Calculate synaptic currents for each neuron
        synaptic_currents = defaultdict(float)
        
        for synapse in self.synapses.values():
            # Check for spikes to transmit
            pre_neuron = self.neurons[synapse.pre_neuron_id]
            if pre_neuron.last_spike_time is not None:
                current = synapse.transmit_spike(
                    pre_neuron.last_spike_time, 
                    self.current_time
                )
                synaptic_currents[synapse.post_neuron_id] += current
        
        # Update each neuron
        for neuron_id, neuron in self.neurons.items():
            # Total input current
            total_current = input_currents.get(neuron_id, 0.0) + synaptic_currents[neuron_id]
            
            # Update neuron
            spiked = neuron.update(self.dt, total_current, self.current_time)
            spikes_this_step[neuron_id] = spiked
            
            if spiked:
                self.spike_buffer[neuron_id].append(self.current_time)
        
        # Update synaptic plasticity
        if self.learning_enabled:
            await self._update_plasticity(spikes_this_step)
        
        # Track network activity
        total_spikes = sum(spikes_this_step.values())
        self.activity_history.append({
            'time': self.current_time,
            'total_spikes': total_spikes,
            'active_neurons': sum(1 for s in spikes_this_step.values() if s)
        })
        
        self.current_time += self.dt
        return spikes_this_step
    
    async def _update_plasticity(self, current_spikes: Dict[str, bool]) -> None:
        """Update synaptic plasticity based on recent activity."""
        for synapse in self.synapses.values():
            pre_neuron = self.neurons[synapse.pre_neuron_id]
            post_neuron = self.neurons[synapse.post_neuron_id]
            
            # Get recent spike times
            pre_spike_time = pre_neuron.last_spike_time if current_spikes.get(synapse.pre_neuron_id, False) else None
            post_spike_time = post_neuron.last_spike_time if current_spikes.get(synapse.post_neuron_id, False) else None
            
            # Get postsynaptic firing rate
            post_firing_rate = post_neuron.get_firing_rate(1000.0)  # Last 1 second
            
            # Update plasticity
            synapse.update_plasticity(pre_spike_time, post_spike_time, post_firing_rate)
    
    async def process_input_pattern(
        self,
        input_pattern: List[float],
        simulation_time: float = 100.0
    ) -> Dict[str, Any]:
        """
        Process input pattern through the network.
        
        Args:
            input_pattern: Input values to encode as spike trains
            simulation_time: Duration of simulation in ms
            
        Returns:
            Network response and activity metrics
        """
        if len(input_pattern) != len(self.input_neurons):
            raise ValueError(f"Input pattern size {len(input_pattern)} doesn't match input neurons {len(self.input_neurons)}")
        
        # Convert input values to spike trains (rate coding)
        input_spike_trains = self._encode_input_pattern(input_pattern, simulation_time)
        
        # Reset network state
        self._reset_network_state()
        
        # Simulate network
        output_spikes = defaultdict(list)
        steps = int(simulation_time / self.dt)
        
        for step in range(steps):
            # Generate input currents from spike trains
            input_currents = self._generate_input_currents(input_spike_trains, self.current_time)
            
            # Simulate step
            spikes = await self.simulate_step(input_currents)
            
            # Record output spikes
            for neuron_id in self.output_neurons:
                if spikes.get(neuron_id, False):
                    output_spikes[neuron_id].append(self.current_time)
        
        # Analyze response
        response_analysis = self._analyze_network_response(output_spikes, simulation_time)
        
        return {
            'output_spikes': dict(output_spikes),
            'response_analysis': response_analysis,
            'network_activity': list(self.activity_history)[-steps:],
            'simulation_time': simulation_time
        }
    
    def _encode_input_pattern(
        self,
        input_pattern: List[float],
        simulation_time: float
    ) -> Dict[str, List[float]]:
        """Encode input pattern as spike trains using rate coding."""
        spike_trains = {}
        input_neuron_ids = list(self.input_neurons)
        
        for i, value in enumerate(input_pattern):
            neuron_id = input_neuron_ids[i]
            
            # Convert value to firing rate (0-100 Hz)
            firing_rate = max(0, min(100, value * 100))  # Assume input values in [0, 1]
            
            # Generate Poisson spike train
            spike_times = []
            dt_ms = 1.0  # 1 ms resolution
            t = 0
            
            while t < simulation_time:
                if np.random.random() < (firing_rate / 1000.0) * dt_ms:
                    spike_times.append(t)
                t += dt_ms
            
            spike_trains[neuron_id] = spike_times
        
        return spike_trains
    
    def _generate_input_currents(
        self,
        spike_trains: Dict[str, List[float]],
        current_time: float
    ) -> Dict[str, float]:
        """Generate input currents based on spike trains."""
        input_currents = {}
        
        for neuron_id, spike_times in spike_trains.items():
            current = 0.0
            
            # Check for recent spikes
            for spike_time in spike_times:
                dt = current_time - spike_time
                if 0 <= dt <= 5.0:  # 5 ms window
                    # Exponential current injection
                    current += 10.0 * np.exp(-dt / 2.0)  # nA
            
            input_currents[neuron_id] = current
        
        return input_currents
    
    def _reset_network_state(self) -> None:
        """Reset network to initial state."""
        self.current_time = 0.0
        self.spike_buffer.clear()
        self.activity_history.clear()
        
        for neuron in self.neurons.values():
            neuron.membrane_potential = neuron.resting_potential
            neuron.last_spike_time = None
            neuron.spike_history.clear()
            neuron.reset_adaptation()
    
    def _analyze_network_response(
        self,
        output_spikes: Dict[str, List[float]],
        simulation_time: float
    ) -> Dict[str, Any]:
        """Analyze network response patterns."""
        analysis = {}
        
        # Calculate output firing rates
        firing_rates = {}
        for neuron_id, spike_times in output_spikes.items():
            firing_rates[neuron_id] = len(spike_times) / (simulation_time / 1000.0)  # Hz
        
        analysis['output_firing_rates'] = firing_rates
        analysis['total_output_spikes'] = sum(len(spikes) for spikes in output_spikes.values())
        analysis['response_latency'] = self._calculate_response_latency(output_spikes)
        analysis['spike_synchrony'] = self._calculate_spike_synchrony(output_spikes)
        
        return analysis
    
    def _calculate_response_latency(self, output_spikes: Dict[str, List[float]]) -> float:
        """Calculate response latency (time to first output spike)."""
        all_first_spikes = []
        
        for spike_times in output_spikes.values():
            if spike_times:
                all_first_spikes.append(min(spike_times))
        
        return min(all_first_spikes) if all_first_spikes else float('inf')
    
    def _calculate_spike_synchrony(self, output_spikes: Dict[str, List[float]]) -> float:
        """Calculate synchrony of output spikes."""
        all_spike_times = []
        for spike_times in output_spikes.values():
            all_spike_times.extend(spike_times)
        
        if len(all_spike_times) < 2:
            return 0.0
        
        all_spike_times.sort()
        
        # Calculate coefficient of variation of interspike intervals
        intervals = np.diff(all_spike_times)
        if len(intervals) == 0:
            return 0.0
        
        cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
        
        # Higher synchrony corresponds to lower CV
        return max(0, 1 - cv)
    
    def set_reward_signal(self, reward: float) -> None:
        """Set reward signal for reward-modulated plasticity."""
        self.reward_signal = reward
        
        # Update all reward-modulated synapses
        for synapse in self.synapses.values():
            if synapse.plasticity_type == SynapticPlasticityType.REWARD_MODULATED:
                # Apply reward modulation to recent plastic changes
                pass  # Implementation would track recent plasticity changes
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network statistics."""
        stats = {
            'topology': self.topology.value,
            'neuron_count': len(self.neurons),
            'synapse_count': len(self.synapses),
            'input_neurons': len(self.input_neurons),
            'output_neurons': len(self.output_neurons),
            'current_time': self.current_time
        }
        
        # Weight statistics
        weights = [synapse.weight for synapse in self.synapses.values()]
        if weights:
            stats['weight_statistics'] = {
                'mean': np.mean(weights),
                'std': np.std(weights),
                'min': np.min(weights),
                'max': np.max(weights)
            }
        
        # Activity statistics
        if self.activity_history:
            recent_activity = list(self.activity_history)[-100:]  # Last 100 steps
            stats['activity_statistics'] = {
                'mean_spikes_per_step': np.mean([a['total_spikes'] for a in recent_activity]),
                'mean_active_neurons': np.mean([a['active_neurons'] for a in recent_activity])
            }
        
        return stats


class NeuromorphicProfiler:
    """Neuromorphic computing framework for adaptive edge AI profiling."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.networks: Dict[str, SpikingNeuralNetwork] = {}
        self.profiling_history = []
    
    async def create_adaptive_profiling_network(
        self,
        network_id: str,
        input_metrics: List[str],
        output_decisions: List[str],
        hidden_layers: List[int] = [20, 10]
    ) -> str:
        """
        Create neuromorphic network for adaptive profiling decisions.
        
        Args:
            network_id: Unique identifier for the network
            input_metrics: List of input metric names
            output_decisions: List of output decision types
            hidden_layers: Sizes of hidden layers
            
        Returns:
            Network ID
        """
        network = SpikingNeuralNetwork(network_id, NetworkTopology.FEEDFORWARD)
        
        # Create network topology
        network.create_network_topology(
            input_size=len(input_metrics),
            hidden_sizes=hidden_layers,
            output_size=len(output_decisions)
        )
        
        self.networks[network_id] = network
        self.logger.info(f"Created neuromorphic profiling network: {network_id}")
        
        return network_id
    
    async def process_profiling_metrics(
        self,
        network_id: str,
        metrics: Dict[str, float],
        simulation_time: float = 100.0
    ) -> Dict[str, Any]:
        """
        Process profiling metrics through neuromorphic network.
        
        Args:
            network_id: ID of the network to use
            metrics: Dictionary of metric values
            simulation_time: Simulation duration in ms
            
        Returns:
            Neuromorphic processing results
        """
        if network_id not in self.networks:
            raise ValueError(f"Network {network_id} not found")
        
        network = self.networks[network_id]
        
        # Normalize metrics to [0, 1] range
        normalized_metrics = self._normalize_metrics(metrics)
        
        # Convert to input pattern
        input_pattern = list(normalized_metrics.values())
        
        # Process through network
        response = await network.process_input_pattern(input_pattern, simulation_time)
        
        # Interpret output
        interpretation = self._interpret_network_output(response, list(metrics.keys()))
        
        # Store in history
        profiling_result = {
            'network_id': network_id,
            'timestamp': time.time(),
            'input_metrics': metrics,
            'normalized_input': normalized_metrics,
            'network_response': response,
            'interpretation': interpretation
        }
        
        self.profiling_history.append(profiling_result)
        
        return profiling_result
    
    def _normalize_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Normalize metrics to [0, 1] range."""
        normalized = {}
        
        # Define typical ranges for common metrics
        metric_ranges = {
            'latency_ms': (0, 1000),
            'memory_kb': (0, 1000),
            'cpu_percent': (0, 100),
            'energy_mj': (0, 100),
            'temperature_c': (20, 80),
            'throughput': (0, 100)
        }
        
        for metric_name, value in metrics.items():
            # Find appropriate range
            range_key = None
            for key in metric_ranges:
                if key in metric_name.lower():
                    range_key = key
                    break
            
            if range_key:
                min_val, max_val = metric_ranges[range_key]
                normalized_value = (value - min_val) / (max_val - min_val)
                normalized_value = max(0, min(1, normalized_value))
            else:
                # Default normalization
                normalized_value = max(0, min(1, value / 100))
            
            normalized[metric_name] = normalized_value
        
        return normalized
    
    def _interpret_network_output(
        self,
        response: Dict[str, Any],
        input_metric_names: List[str]
    ) -> Dict[str, Any]:
        """Interpret neuromorphic network output."""
        output_spikes = response['output_spikes']
        response_analysis = response['response_analysis']
        
        # Determine dominant output based on firing rates
        firing_rates = response_analysis['output_firing_rates']
        
        if not firing_rates:
            decision = 'no_response'
            confidence = 0.0
        else:
            # Find output with highest firing rate
            max_rate_neuron = max(firing_rates.items(), key=lambda x: x[1])
            decision = max_rate_neuron[0]
            
            # Calculate confidence based on relative firing rates
            total_rate = sum(firing_rates.values())
            confidence = max_rate_neuron[1] / total_rate if total_rate > 0 else 0
        
        # Generate adaptive recommendations
        recommendations = self._generate_neuromorphic_recommendations(
            response_analysis, input_metric_names
        )
        
        return {
            'primary_decision': decision,
            'confidence': confidence,
            'firing_rates': firing_rates,
            'response_latency_ms': response_analysis['response_latency'],
            'spike_synchrony': response_analysis['spike_synchrony'],
            'recommendations': recommendations
        }
    
    def _generate_neuromorphic_recommendations(
        self,
        response_analysis: Dict[str, Any],
        input_metrics: List[str]
    ) -> List[str]:
        """Generate recommendations based on neuromorphic processing."""
        recommendations = []
        
        # Analyze response characteristics
        latency = response_analysis['response_latency']
        synchrony = response_analysis['spike_synchrony']
        total_spikes = response_analysis['total_output_spikes']
        
        if latency > 50:  # ms
            recommendations.append("Network response latency high - consider optimization")
        
        if synchrony < 0.3:
            recommendations.append("Low output synchrony - decision uncertainty detected")
        
        if total_spikes < 5:
            recommendations.append("Low network activity - input may be outside trained range")
        
        # Input-specific recommendations
        for metric in input_metrics:
            if 'memory' in metric.lower():
                recommendations.append("Consider memory optimization strategies")
            elif 'latency' in metric.lower():
                recommendations.append("Monitor latency trends for performance optimization")
        
        return recommendations
    
    async def train_network_with_feedback(
        self,
        network_id: str,
        training_examples: List[Tuple[Dict[str, float], str]],
        reward_feedback: List[float]
    ) -> Dict[str, Any]:
        """
        Train neuromorphic network with supervised feedback.
        
        Args:
            network_id: ID of network to train
            training_examples: List of (metrics, expected_decision) pairs
            reward_feedback: Reward signals for each example
            
        Returns:
            Training results
        """
        if network_id not in self.networks:
            raise ValueError(f"Network {network_id} not found")
        
        network = self.networks[network_id]
        training_results = []
        
        for i, ((metrics, expected_decision), reward) in enumerate(zip(training_examples, reward_feedback)):
            # Set reward signal
            network.set_reward_signal(reward)
            
            # Process example
            result = await self.process_profiling_metrics(network_id, metrics)
            
            # Evaluate performance
            actual_decision = result['interpretation']['primary_decision']
            correct = actual_decision == expected_decision
            
            training_results.append({
                'example_index': i,
                'metrics': metrics,
                'expected_decision': expected_decision,
                'actual_decision': actual_decision,
                'correct': correct,
                'reward': reward,
                'confidence': result['interpretation']['confidence']
            })
        
        # Calculate training statistics
        accuracy = sum(1 for r in training_results if r['correct']) / len(training_results)
        avg_confidence = np.mean([r['confidence'] for r in training_results])
        
        training_summary = {
            'network_id': network_id,
            'training_examples': len(training_examples),
            'accuracy': accuracy,
            'average_confidence': avg_confidence,
            'detailed_results': training_results,
            'network_statistics': network.get_network_statistics()
        }
        
        self.logger.info(f"Training completed for {network_id}: {accuracy:.2%} accuracy")
        
        return training_summary
    
    def get_network_status(self, network_id: str) -> Dict[str, Any]:
        """Get comprehensive status of neuromorphic network."""
        if network_id not in self.networks:
            return {'error': f'Network {network_id} not found'}
        
        network = self.networks[network_id]
        
        return {
            'network_id': network_id,
            'statistics': network.get_network_statistics(),
            'recent_activity': list(network.activity_history)[-10:],  # Last 10 activity records
            'profiling_history_count': len([h for h in self.profiling_history if h['network_id'] == network_id])
        }


# Global neuromorphic profiler instance
_global_neuromorphic_profiler: Optional[NeuromorphicProfiler] = None

def get_neuromorphic_profiler() -> NeuromorphicProfiler:
    """Get global neuromorphic profiler instance."""
    global _global_neuromorphic_profiler
    if _global_neuromorphic_profiler is None:
        _global_neuromorphic_profiler = NeuromorphicProfiler()
    return _global_neuromorphic_profiler

async def create_neuromorphic_network(
    network_id: str,
    input_metrics: List[str],
    output_decisions: List[str],
    hidden_layers: List[int] = [20, 10]
) -> str:
    """Create neuromorphic profiling network."""
    profiler = get_neuromorphic_profiler()
    return await profiler.create_adaptive_profiling_network(
        network_id, input_metrics, output_decisions, hidden_layers
    )

async def process_with_neuromorphic(
    network_id: str,
    metrics: Dict[str, float],
    simulation_time: float = 100.0
) -> Dict[str, Any]:
    """Process metrics through neuromorphic network."""
    profiler = get_neuromorphic_profiler()
    return await profiler.process_profiling_metrics(network_id, metrics, simulation_time)


# Simple NeuromorphicProcessor for integration with other systems
class NeuromorphicProcessor:
    """Simplified neuromorphic processor for integration with other systems."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.processing_history: List[Dict[str, Any]] = []
        self.adaptation_level = 0.5
        
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input through neuromorphic-inspired algorithms."""
        start_time = time.time()
        
        # Simulate neuromorphic processing
        confidence = self._calculate_confidence(input_data)
        decision = self._make_decision(input_data, confidence)
        
        processing_time = time.time() - start_time
        
        result = {
            "input": input_data,
            "decision": decision,
            "confidence": confidence,
            "processing_time_s": processing_time,
            "adaptation_level": self.adaptation_level
        }
        
        self.processing_history.append(result)
        self._update_adaptation(result)
        
        return result
    
    def _calculate_confidence(self, input_data: Dict[str, Any]) -> float:
        """Calculate confidence in processing."""
        # Simple heuristic based on input consistency
        values = [v for v in input_data.values() if isinstance(v, (int, float))]
        if not values:
            return 0.5
        
        # Higher values generally indicate higher confidence
        normalized_sum = sum(max(0, min(1, v)) for v in values) / len(values)
        return normalized_sum
    
    def _make_decision(self, input_data: Dict[str, Any], confidence: float) -> Dict[str, Any]:
        """Make decision based on input and confidence."""
        if confidence > 0.8:
            decision = "high_confidence_action"
        elif confidence > 0.5:
            decision = "moderate_confidence_action"
        else:
            decision = "low_confidence_action"
        
        return {
            "decision": decision,
            "confidence": confidence,
            "reasoning": "Neuromorphic pattern matching"
        }
    
    def _update_adaptation(self, result: Dict[str, Any]):
        """Update adaptation level based on results."""
        confidence = result["confidence"]
        
        # Adapt based on confidence trends
        if confidence > 0.7:
            self.adaptation_level = min(1.0, self.adaptation_level * 1.01)
        else:
            self.adaptation_level = max(0.1, self.adaptation_level * 0.99)
    
    def get_processor_state(self) -> Dict[str, Any]:
        """Get current processor state."""
        return {
            "adaptation_level": self.adaptation_level,
            "processing_count": len(self.processing_history),
            "avg_confidence": sum(r["confidence"] for r in self.processing_history[-10:]) / min(10, len(self.processing_history)) if self.processing_history else 0.0
        }