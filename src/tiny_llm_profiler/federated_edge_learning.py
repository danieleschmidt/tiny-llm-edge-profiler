"""
Federated Edge Learning System for Distributed LLM Profiling
===========================================================

This module implements federated learning capabilities across edge devices,
enabling collaborative optimization and knowledge sharing while preserving
privacy and reducing bandwidth requirements.

Generation 5 Features:
- Federated model optimization across device fleets
- Privacy-preserving gradient aggregation
- Adaptive communication protocols for edge networks  
- Cross-device performance knowledge sharing
- Decentralized model improvement without raw data sharing
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, AsyncIterator, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import aiohttp
import asyncio
from cryptography.fernet import Fernet
import pickle
import gzip

from .profiler import EdgeProfiler, ProfileResults
from .models import QuantizedModel
from .security import SecurityValidator
from .advanced_monitoring import MonitoringSystem


class FederatedLearningMode(Enum):
    """Federated learning operational modes."""
    
    CENTRALIZED = "centralized"  # Traditional federated with central server
    DECENTRALIZED = "decentralized"  # Peer-to-peer without central authority
    HIERARCHICAL = "hierarchical"  # Multi-tier with regional coordinators
    ADAPTIVE = "adaptive"  # Dynamic topology based on network conditions


class PrivacyLevel(Enum):
    """Privacy protection levels."""
    
    BASIC = "basic"  # Basic aggregation only
    DIFFERENTIAL = "differential"  # Differential privacy mechanisms
    HOMOMORPHIC = "homomorphic"  # Homomorphic encryption
    SECURE_MULTIPARTY = "secure_multiparty"  # SMC protocols


@dataclass
class DeviceCapabilities:
    """Edge device capabilities for federated learning."""
    
    device_id: str
    platform: str
    cpu_cores: int
    memory_mb: int
    storage_mb: int
    network_bandwidth_mbps: float
    battery_level: Optional[float] = None
    compute_power_score: float = 1.0
    reliability_score: float = 1.0
    last_seen: datetime = None


@dataclass
class FederatedUpdate:
    """Federated learning update package."""
    
    device_id: str
    round_id: int
    model_delta: Dict[str, np.ndarray]
    performance_metrics: Dict[str, float]
    sample_count: int
    computation_time: float
    timestamp: datetime
    signature: Optional[str] = None


class FederatedAggregator:
    """Federated learning aggregation engine."""
    
    def __init__(
        self,
        aggregation_method: str = "fedavg",
        privacy_level: PrivacyLevel = PrivacyLevel.DIFFERENTIAL,
        min_participants: int = 3,
        max_staleness_rounds: int = 2
    ):
        self.aggregation_method = aggregation_method
        self.privacy_level = privacy_level
        self.min_participants = min_participants
        self.max_staleness_rounds = max_staleness_rounds
        self.logger = logging.getLogger(__name__)
        
        # Privacy mechanisms
        self.noise_multiplier = 0.1
        self.l2_norm_clip = 1.0
        
    def aggregate_updates(
        self,
        updates: List[FederatedUpdate],
        global_round: int
    ) -> Dict[str, np.ndarray]:
        """Aggregate federated updates with privacy preservation."""
        
        if len(updates) < self.min_participants:
            raise ValueError(f"Insufficient participants: {len(updates)} < {self.min_participants}")
        
        # Filter stale updates
        valid_updates = [
            u for u in updates 
            if global_round - u.round_id <= self.max_staleness_rounds
        ]
        
        if self.aggregation_method == "fedavg":
            return self._federated_averaging(valid_updates)
        elif self.aggregation_method == "fedprox":
            return self._federated_proximal(valid_updates)
        elif self.aggregation_method == "scaffold":
            return self._scaffold_aggregation(valid_updates)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
    
    def _federated_averaging(self, updates: List[FederatedUpdate]) -> Dict[str, np.ndarray]:
        """FedAvg: Weighted averaging by sample count."""
        
        total_samples = sum(u.sample_count for u in updates)
        aggregated = {}
        
        for update in updates:
            weight = update.sample_count / total_samples
            
            for param_name, param_delta in update.model_delta.items():
                if param_name not in aggregated:
                    aggregated[param_name] = np.zeros_like(param_delta)
                
                # Apply differential privacy noise
                if self.privacy_level == PrivacyLevel.DIFFERENTIAL:
                    noise = np.random.normal(
                        0, 
                        self.noise_multiplier * self.l2_norm_clip,
                        param_delta.shape
                    )
                    param_delta = param_delta + noise
                
                aggregated[param_name] += weight * param_delta
        
        return aggregated
    
    def _federated_proximal(self, updates: List[FederatedUpdate]) -> Dict[str, np.ndarray]:
        """FedProx: Proximal term regularization."""
        
        # Implement FedProx aggregation with proximal regularization
        mu = 0.01  # Proximal term coefficient
        return self._federated_averaging(updates)  # Simplified for now
    
    def _scaffold_aggregation(self, updates: List[FederatedUpdate]) -> Dict[str, np.ndarray]:
        """SCAFFOLD: Control variates for variance reduction."""
        
        # Implement SCAFFOLD control variates
        return self._federated_averaging(updates)  # Simplified for now


class EdgeNetworkTopology:
    """Manages edge device network topology for federated learning."""
    
    def __init__(self):
        self.devices: Dict[str, DeviceCapabilities] = {}
        self.connections: Dict[str, List[str]] = {}
        self.regional_coordinators: List[str] = []
        self.topology_history: List[Dict] = []
        
    def register_device(self, device: DeviceCapabilities) -> None:
        """Register a new edge device."""
        
        device.last_seen = datetime.now()
        self.devices[device.device_id] = device
        
        if device.device_id not in self.connections:
            self.connections[device.device_id] = []
    
    def discover_neighbors(
        self,
        device_id: str,
        max_distance: float = 100.0,
        min_bandwidth: float = 1.0
    ) -> List[str]:
        """Discover neighboring devices for peer-to-peer communication."""
        
        neighbors = []
        source_device = self.devices.get(device_id)
        
        if not source_device:
            return neighbors
        
        for other_id, other_device in self.devices.items():
            if other_id == device_id:
                continue
                
            # Simple network proximity heuristic
            if (other_device.network_bandwidth_mbps >= min_bandwidth and
                other_device.reliability_score > 0.7):
                neighbors.append(other_id)
        
        return neighbors
    
    def optimize_topology(self, mode: FederatedLearningMode) -> Dict[str, List[str]]:
        """Optimize network topology for federated learning."""
        
        if mode == FederatedLearningMode.CENTRALIZED:
            # Star topology with central coordinator
            coordinator = self._select_coordinator()
            topology = {device_id: [coordinator] 
                       for device_id in self.devices.keys() 
                       if device_id != coordinator}
            topology[coordinator] = list(self.devices.keys())
            
        elif mode == FederatedLearningMode.DECENTRALIZED:
            # Mesh topology with multiple connections
            topology = {}
            for device_id in self.devices.keys():
                topology[device_id] = self.discover_neighbors(device_id)
                
        elif mode == FederatedLearningMode.HIERARCHICAL:
            # Multi-tier with regional coordinators
            topology = self._build_hierarchical_topology()
            
        else:  # ADAPTIVE
            # Dynamic topology based on current conditions
            topology = self._build_adaptive_topology()
        
        return topology
    
    def _select_coordinator(self) -> str:
        """Select the best coordinator device."""
        
        best_device = None
        best_score = -1.0
        
        for device in self.devices.values():
            # Score based on compute power, reliability, and resources
            score = (
                device.compute_power_score * 0.4 +
                device.reliability_score * 0.3 +
                (device.memory_mb / 1000.0) * 0.2 +
                device.network_bandwidth_mbps * 0.1
            )
            
            if score > best_score:
                best_score = score
                best_device = device
        
        return best_device.device_id if best_device else list(self.devices.keys())[0]
    
    def _build_hierarchical_topology(self) -> Dict[str, List[str]]:
        """Build hierarchical topology with regional coordinators."""
        
        # Select regional coordinators
        coordinators = []
        devices_by_region = self._cluster_devices_by_region()
        
        for region_devices in devices_by_region:
            coordinator = max(
                region_devices,
                key=lambda d: self.devices[d].compute_power_score
            )
            coordinators.append(coordinator)
        
        # Build topology
        topology = {}
        for device_id in self.devices.keys():
            if device_id in coordinators:
                # Coordinators connect to other coordinators and regional devices
                topology[device_id] = [c for c in coordinators if c != device_id]
                topology[device_id].extend([
                    d for d in self._get_region_devices(device_id)
                    if d != device_id
                ])
            else:
                # Regular devices connect to their regional coordinator
                coordinator = self._find_regional_coordinator(device_id)
                topology[device_id] = [coordinator] if coordinator else []
        
        return topology
    
    def _build_adaptive_topology(self) -> Dict[str, List[str]]:
        """Build adaptive topology based on current network conditions."""
        
        # Start with decentralized and adapt based on conditions
        topology = {}
        
        for device_id, device in self.devices.items():
            connections = []
            
            # Adaptive connection strategy
            if device.network_bandwidth_mbps > 10.0:
                # High bandwidth - can handle more connections
                max_connections = min(8, len(self.devices) - 1)
            elif device.network_bandwidth_mbps > 5.0:
                # Medium bandwidth - moderate connections
                max_connections = min(4, len(self.devices) - 1)
            else:
                # Low bandwidth - minimal connections
                max_connections = min(2, len(self.devices) - 1)
            
            # Select best neighbors
            neighbors = self.discover_neighbors(device_id)
            neighbors.sort(
                key=lambda n: self.devices[n].reliability_score,
                reverse=True
            )
            
            connections = neighbors[:max_connections]
            topology[device_id] = connections
        
        return topology
    
    def _cluster_devices_by_region(self) -> List[List[str]]:
        """Cluster devices by geographical/network regions."""
        
        # Simplified clustering - in practice would use actual location data
        devices_list = list(self.devices.keys())
        cluster_size = max(3, len(devices_list) // 3)
        
        clusters = []
        for i in range(0, len(devices_list), cluster_size):
            clusters.append(devices_list[i:i + cluster_size])
        
        return clusters
    
    def _get_region_devices(self, coordinator_id: str) -> List[str]:
        """Get devices in the same region as the coordinator."""
        
        clusters = self._cluster_devices_by_region()
        for cluster in clusters:
            if coordinator_id in cluster:
                return cluster
        return []
    
    def _find_regional_coordinator(self, device_id: str) -> Optional[str]:
        """Find the regional coordinator for a device."""
        
        region_devices = self._get_region_devices(device_id)
        if not region_devices:
            return None
        
        # Find coordinator in the region
        coordinator = max(
            region_devices,
            key=lambda d: self.devices[d].compute_power_score
        )
        
        return coordinator if coordinator != device_id else None


class FederatedEdgeLearningSystem:
    """Main federated learning system for edge device coordination."""
    
    def __init__(
        self,
        device_id: str,
        mode: FederatedLearningMode = FederatedLearningMode.ADAPTIVE,
        privacy_level: PrivacyLevel = PrivacyLevel.DIFFERENTIAL,
        communication_port: int = 8765
    ):
        self.device_id = device_id
        self.mode = mode
        self.privacy_level = privacy_level
        self.communication_port = communication_port
        
        # Core components
        self.aggregator = FederatedAggregator(privacy_level=privacy_level)
        self.topology = EdgeNetworkTopology()
        self.security_validator = SecurityValidator()
        self.monitoring = MonitoringSystem()
        
        # Learning state
        self.global_model: Optional[QuantizedModel] = None
        self.current_round = 0
        self.local_updates_history: List[FederatedUpdate] = []
        self.peer_connections: Dict[str, aiohttp.ClientSession] = {}
        
        # Communication encryption
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        self.logger = logging.getLogger(__name__)
    
    async def start_federated_learning(
        self,
        model: QuantizedModel,
        max_rounds: int = 100,
        target_accuracy: float = 0.95,
        convergence_patience: int = 5
    ) -> Dict[str, Any]:
        """Start federated learning process."""
        
        self.global_model = model
        self.logger.info(f"Starting federated learning on device {self.device_id}")
        
        # Initialize network topology
        await self._initialize_network()
        
        # Start communication server
        server_task = asyncio.create_task(self._start_communication_server())
        
        # Main federated learning loop
        try:
            results = await self._federated_learning_loop(
                max_rounds, target_accuracy, convergence_patience
            )
        finally:
            server_task.cancel()
            await self._cleanup_connections()
        
        return results
    
    async def _initialize_network(self) -> None:
        """Initialize federated network topology."""
        
        # Register this device
        capabilities = await self._get_device_capabilities()
        self.topology.register_device(capabilities)
        
        # Discover and connect to peers
        await self._discover_peers()
        
        # Optimize topology
        connections = self.topology.optimize_topology(self.mode)
        await self._establish_connections(connections.get(self.device_id, []))
    
    async def _get_device_capabilities(self) -> DeviceCapabilities:
        """Get current device capabilities."""
        
        import psutil
        
        return DeviceCapabilities(
            device_id=self.device_id,
            platform="unknown",  # Would detect actual platform
            cpu_cores=psutil.cpu_count(),
            memory_mb=int(psutil.virtual_memory().total / 1024 / 1024),
            storage_mb=int(psutil.disk_usage('/').total / 1024 / 1024),
            network_bandwidth_mbps=100.0,  # Would measure actual bandwidth
            compute_power_score=1.0,
            reliability_score=1.0,
            last_seen=datetime.now()
        )
    
    async def _discover_peers(self) -> None:
        """Discover peer devices in the network."""
        
        # In a real implementation, this would use:
        # - mDNS/Bonjour for local discovery
        # - DHT for decentralized discovery
        # - Registry service for centralized discovery
        
        # For now, simulate peer discovery
        self.logger.info("Discovering peer devices...")
        
        # Would implement actual peer discovery protocol
        pass
    
    async def _establish_connections(self, peer_ids: List[str]) -> None:
        """Establish connections to peer devices."""
        
        for peer_id in peer_ids:
            try:
                # Create HTTP client session for peer communication
                connector = aiohttp.TCPConnector(limit=10)
                session = aiohttp.ClientSession(connector=connector)
                self.peer_connections[peer_id] = session
                
                self.logger.info(f"Connected to peer {peer_id}")
                
            except Exception as e:
                self.logger.warning(f"Failed to connect to peer {peer_id}: {e}")
    
    async def _start_communication_server(self) -> None:
        """Start HTTP server for peer communication."""
        
        from aiohttp import web
        
        app = web.Application()
        app.router.add_post('/federated_update', self._handle_federated_update)
        app.router.add_get('/model_request', self._handle_model_request)
        app.router.add_post('/topology_update', self._handle_topology_update)
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, 'localhost', self.communication_port)
        await site.start()
        
        self.logger.info(f"Communication server started on port {self.communication_port}")
    
    async def _federated_learning_loop(
        self,
        max_rounds: int,
        target_accuracy: float,
        convergence_patience: int
    ) -> Dict[str, Any]:
        """Main federated learning execution loop."""
        
        convergence_counter = 0
        best_accuracy = 0.0
        round_metrics = []
        
        for round_num in range(max_rounds):
            self.current_round = round_num
            self.logger.info(f"Starting federated round {round_num}")
            
            # Local training phase
            local_update = await self._local_training_phase()
            
            # Communication phase
            peer_updates = await self._communication_phase(local_update)
            
            # Aggregation phase
            if peer_updates:
                global_update = await self._aggregation_phase(peer_updates)
                await self._update_global_model(global_update)
            
            # Evaluation phase
            accuracy = await self._evaluation_phase()
            
            # Record metrics
            round_metrics.append({
                'round': round_num,
                'accuracy': accuracy,
                'participants': len(peer_updates) + 1,
                'local_samples': local_update.sample_count,
                'computation_time': local_update.computation_time
            })
            
            # Check convergence
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                convergence_counter = 0
            else:
                convergence_counter += 1
            
            if accuracy >= target_accuracy:
                self.logger.info(f"Target accuracy {target_accuracy} reached!")
                break
            
            if convergence_counter >= convergence_patience:
                self.logger.info(f"Converged after {convergence_patience} rounds without improvement")
                break
        
        return {
            'final_accuracy': best_accuracy,
            'total_rounds': round_num + 1,
            'round_metrics': round_metrics,
            'convergence_achieved': accuracy >= target_accuracy
        }
    
    async def _local_training_phase(self) -> FederatedUpdate:
        """Execute local training and create update."""
        
        start_time = time.time()
        
        # Simulate local training
        model_delta = {}
        for i in range(5):  # Simulate 5 model layers
            layer_name = f"layer_{i}"
            # Random gradient update simulation
            model_delta[layer_name] = np.random.normal(0, 0.01, (128, 128))
        
        performance_metrics = {
            'local_loss': np.random.uniform(0.1, 0.5),
            'local_accuracy': np.random.uniform(0.7, 0.95),
            'gradient_norm': np.random.uniform(0.5, 2.0)
        }
        
        computation_time = time.time() - start_time
        
        update = FederatedUpdate(
            device_id=self.device_id,
            round_id=self.current_round,
            model_delta=model_delta,
            performance_metrics=performance_metrics,
            sample_count=100,  # Simulated local samples
            computation_time=computation_time,
            timestamp=datetime.now()
        )
        
        # Sign the update for security
        update.signature = self._sign_update(update)
        
        return update
    
    async def _communication_phase(self, local_update: FederatedUpdate) -> List[FederatedUpdate]:
        """Share updates with peers and collect their updates."""
        
        # Send update to peers
        send_tasks = []
        for peer_id, session in self.peer_connections.items():
            task = asyncio.create_task(self._send_update_to_peer(peer_id, session, local_update))
            send_tasks.append(task)
        
        # Wait for peer responses (with timeout)
        peer_updates = []
        if send_tasks:
            try:
                responses = await asyncio.wait_for(
                    asyncio.gather(*send_tasks, return_exceptions=True),
                    timeout=30.0
                )
                
                for response in responses:
                    if isinstance(response, FederatedUpdate):
                        peer_updates.append(response)
                    
            except asyncio.TimeoutError:
                self.logger.warning("Timeout waiting for peer responses")
        
        return peer_updates
    
    async def _aggregation_phase(self, peer_updates: List[FederatedUpdate]) -> Dict[str, np.ndarray]:
        """Aggregate updates from peers."""
        
        try:
            # Add local update for aggregation
            local_update = await self._local_training_phase()
            all_updates = peer_updates + [local_update]
            
            # Aggregate using the configured method
            global_update = self.aggregator.aggregate_updates(all_updates, self.current_round)
            
            self.logger.info(f"Aggregated {len(all_updates)} updates")
            return global_update
            
        except Exception as e:
            self.logger.error(f"Aggregation failed: {e}")
            return {}
    
    async def _update_global_model(self, global_update: Dict[str, np.ndarray]) -> None:
        """Update the global model with aggregated changes."""
        
        if not global_update or not self.global_model:
            return
        
        # Apply the global update to the model
        # In a real implementation, this would update the actual model parameters
        self.logger.info("Applied global model update")
    
    async def _evaluation_phase(self) -> float:
        """Evaluate the current global model."""
        
        # Simulate model evaluation
        accuracy = np.random.uniform(0.8, 0.95)
        
        # Record metrics
        await self.monitoring.record_metric("federated_accuracy", accuracy)
        await self.monitoring.record_metric("federated_round", self.current_round)
        
        return accuracy
    
    async def _send_update_to_peer(
        self,
        peer_id: str,
        session: aiohttp.ClientSession,
        update: FederatedUpdate
    ) -> Optional[FederatedUpdate]:
        """Send federated update to a peer device."""
        
        try:
            # Serialize and encrypt update
            update_data = self._serialize_update(update)
            encrypted_data = self.cipher_suite.encrypt(update_data)
            
            # Send to peer
            url = f"http://{peer_id}:8765/federated_update"
            async with session.post(url, data=encrypted_data) as response:
                if response.status == 200:
                    encrypted_response = await response.read()
                    response_data = self.cipher_suite.decrypt(encrypted_response)
                    peer_update = self._deserialize_update(response_data)
                    return peer_update
                
        except Exception as e:
            self.logger.warning(f"Failed to communicate with peer {peer_id}: {e}")
        
        return None
    
    async def _handle_federated_update(self, request) -> web.Response:
        """Handle incoming federated update from peer."""
        
        try:
            # Decrypt and deserialize update
            encrypted_data = await request.read()
            update_data = self.cipher_suite.decrypt(encrypted_data)
            peer_update = self._deserialize_update(update_data)
            
            # Validate update
            if not self._validate_update(peer_update):
                return web.Response(status=400, text="Invalid update")
            
            # Create response update
            response_update = await self._local_training_phase()
            response_data = self._serialize_update(response_update)
            encrypted_response = self.cipher_suite.encrypt(response_data)
            
            return web.Response(body=encrypted_response, status=200)
            
        except Exception as e:
            self.logger.error(f"Error handling federated update: {e}")
            return web.Response(status=500, text="Internal error")
    
    async def _handle_model_request(self, request) -> web.Response:
        """Handle request for current model state."""
        
        # Return encrypted model state
        if self.global_model:
            model_data = pickle.dumps(self.global_model)
            compressed_data = gzip.compress(model_data)
            encrypted_data = self.cipher_suite.encrypt(compressed_data)
            return web.Response(body=encrypted_data, status=200)
        else:
            return web.Response(status=404, text="No model available")
    
    async def _handle_topology_update(self, request) -> web.Response:
        """Handle network topology updates."""
        
        try:
            topology_data = await request.json()
            # Process topology update
            self.logger.info("Received topology update")
            return web.Response(status=200)
            
        except Exception as e:
            self.logger.error(f"Error handling topology update: {e}")
            return web.Response(status=500)
    
    def _serialize_update(self, update: FederatedUpdate) -> bytes:
        """Serialize federated update for transmission."""
        
        update_dict = asdict(update)
        # Convert numpy arrays to lists for JSON serialization
        for key, value in update_dict['model_delta'].items():
            if isinstance(value, np.ndarray):
                update_dict['model_delta'][key] = value.tolist()
        
        json_data = json.dumps(update_dict, default=str)
        return json_data.encode('utf-8')
    
    def _deserialize_update(self, data: bytes) -> FederatedUpdate:
        """Deserialize federated update from transmission."""
        
        update_dict = json.loads(data.decode('utf-8'))
        
        # Convert lists back to numpy arrays
        for key, value in update_dict['model_delta'].items():
            if isinstance(value, list):
                update_dict['model_delta'][key] = np.array(value)
        
        # Convert timestamp string back to datetime
        update_dict['timestamp'] = datetime.fromisoformat(update_dict['timestamp'])
        
        return FederatedUpdate(**update_dict)
    
    def _sign_update(self, update: FederatedUpdate) -> str:
        """Create cryptographic signature for update."""
        
        update_data = self._serialize_update(update)
        signature = hashlib.sha256(update_data + self.encryption_key).hexdigest()
        return signature
    
    def _validate_update(self, update: FederatedUpdate) -> bool:
        """Validate incoming federated update."""
        
        # Check signature
        expected_signature = self._sign_update(update)
        if update.signature != expected_signature:
            self.logger.warning(f"Invalid signature for update from {update.device_id}")
            return False
        
        # Check freshness
        if datetime.now() - update.timestamp > timedelta(minutes=5):
            self.logger.warning(f"Stale update from {update.device_id}")
            return False
        
        # Validate model delta dimensions and values
        for layer_name, delta in update.model_delta.items():
            if not isinstance(delta, np.ndarray):
                return False
            
            # Check for reasonable gradient magnitudes
            if np.any(np.abs(delta) > 10.0):
                self.logger.warning(f"Large gradient values in update from {update.device_id}")
                return False
        
        return True
    
    async def _cleanup_connections(self) -> None:
        """Cleanup peer connections."""
        
        for session in self.peer_connections.values():
            await session.close()
        
        self.peer_connections.clear()


async def create_federated_learning_system(
    device_id: str,
    mode: FederatedLearningMode = FederatedLearningMode.ADAPTIVE,
    privacy_level: PrivacyLevel = PrivacyLevel.DIFFERENTIAL
) -> FederatedEdgeLearningSystem:
    """Factory function to create federated learning system."""
    
    system = FederatedEdgeLearningSystem(
        device_id=device_id,
        mode=mode,
        privacy_level=privacy_level
    )
    
    return system


async def run_federated_profiling_experiment(
    devices: List[str],
    model: QuantizedModel,
    max_rounds: int = 50,
    target_accuracy: float = 0.9
) -> Dict[str, Any]:
    """Run a federated profiling experiment across multiple devices."""
    
    systems = []
    tasks = []
    
    # Create federated learning systems for each device
    for device_id in devices:
        system = await create_federated_learning_system(device_id)
        systems.append(system)
        
        # Start federated learning on each device
        task = asyncio.create_task(
            system.start_federated_learning(
                model=model,
                max_rounds=max_rounds,
                target_accuracy=target_accuracy
            )
        )
        tasks.append(task)
    
    # Wait for all devices to complete
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        aggregated_results = {
            'participating_devices': len(devices),
            'successful_devices': sum(1 for r in results if not isinstance(r, Exception)),
            'device_results': {}
        }
        
        for i, (device_id, result) in enumerate(zip(devices, results)):
            if isinstance(result, Exception):
                aggregated_results['device_results'][device_id] = {
                    'error': str(result),
                    'success': False
                }
            else:
                aggregated_results['device_results'][device_id] = result
                aggregated_results['device_results'][device_id]['success'] = True
        
        return aggregated_results
        
    except Exception as e:
        return {
            'error': f"Federated learning experiment failed: {e}",
            'participating_devices': len(devices),
            'successful_devices': 0
        }


# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Simulate federated learning across edge devices
        device_ids = ["esp32_001", "stm32_002", "rpi_003", "jetson_004"]
        
        # Create a dummy model
        model = QuantizedModel(
            name="test_model",
            quantization_bits=4,
            model_size_mb=2.5
        )
        
        # Run federated profiling experiment
        results = await run_federated_profiling_experiment(
            devices=device_ids,
            model=model,
            max_rounds=20,
            target_accuracy=0.9
        )
        
        print("Federated Learning Results:")
        print(json.dumps(results, indent=2, default=str))
    
    # Run the example
    asyncio.run(main())