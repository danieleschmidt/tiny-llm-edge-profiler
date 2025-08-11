"""
Distributed Profiling System for Generation 3
Provides comprehensive distributed profiling capabilities including:
- Multi-device concurrent profiling coordination
- Distributed profiling across multiple systems
- Load balancing for profiling workloads
- Resource pool management for large-scale operations
- Network communication and synchronization
- Fault tolerance and recovery mechanisms
"""

import time
import asyncio
import threading
import socket
import json
import uuid
import pickle
import hashlib
from typing import Dict, List, Optional, Any, Callable, Set, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
import multiprocessing as mp
from contextlib import asynccontextmanager
import heapq

from .exceptions import TinyLLMProfilerError, ResourceError, ProfilingError
from .logging_config import get_logger, PerformanceLogger
from .models import QuantizedModel
from .results import ProfileResults
from .concurrent_utils import ProfilingTask, TaskResult, TaskStatus
from .resource_pool import ResourcePoolManager
from .auto_scaling import AutoScaler, LoadBalancer

logger = get_logger("distributed_profiler")
perf_logger = PerformanceLogger()


class NodeRole(str, Enum):
    """Node role in distributed system."""
    COORDINATOR = "coordinator"   # Central orchestrator
    WORKER = "worker"            # Profiling worker node
    HYBRID = "hybrid"            # Can act as both coordinator and worker


class NodeStatus(str, Enum):
    """Status of a distributed node."""
    STARTING = "starting"
    ACTIVE = "active"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    FAILING = "failing"
    OFFLINE = "offline"


class DistributedTaskStatus(str, Enum):
    """Status of distributed profiling tasks."""
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


@dataclass
class NodeInfo:
    """Information about a distributed node."""
    node_id: str
    role: NodeRole
    status: NodeStatus
    host: str
    port: int
    last_heartbeat: datetime = field(default_factory=datetime.now)
    
    # Capabilities
    supported_platforms: Set[str] = field(default_factory=set)
    max_concurrent_tasks: int = 4
    current_task_count: int = 0
    
    # Performance metrics
    total_tasks_completed: int = 0
    avg_task_duration: float = 0.0
    error_rate: float = 0.0
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Load metrics
    load_score: float = 0.0
    priority_bonus: float = 0.0
    
    def is_healthy(self, timeout_seconds: int = 60) -> bool:
        """Check if node is healthy based on heartbeat."""
        if self.status == NodeStatus.OFFLINE:
            return False
        
        time_since_heartbeat = (datetime.now() - self.last_heartbeat).total_seconds()
        return time_since_heartbeat < timeout_seconds
    
    def can_accept_task(self) -> bool:
        """Check if node can accept more tasks."""
        return (
            self.status in [NodeStatus.ACTIVE, NodeStatus.BUSY] and
            self.current_task_count < self.max_concurrent_tasks and
            self.is_healthy()
        )
    
    def calculate_load_score(self) -> float:
        """Calculate current load score for load balancing."""
        # Base load from task count
        task_load = (self.current_task_count / max(self.max_concurrent_tasks, 1)) * 100
        
        # CPU and memory load
        system_load = (self.cpu_usage + self.memory_usage_mb / 1024) / 2
        
        # Error rate penalty
        error_penalty = self.error_rate * 50
        
        # Calculate final score (lower is better)
        self.load_score = task_load + system_load + error_penalty - self.priority_bonus
        return self.load_score


@dataclass
class DistributedTask:
    """Distributed profiling task with routing information."""
    task_id: str
    profiling_task: ProfilingTask
    status: DistributedTaskStatus = DistributedTaskStatus.QUEUED
    
    # Assignment info
    assigned_node_id: Optional[str] = None
    assignment_time: Optional[datetime] = None
    
    # Execution info
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Retry info
    retry_count: int = 0
    max_retries: int = 3
    
    # Priority and constraints
    priority: int = 0  # Higher is more important
    required_platforms: Set[str] = field(default_factory=set)
    preferred_nodes: Set[str] = field(default_factory=set)
    excluded_nodes: Set[str] = field(default_factory=set)
    
    # Results
    result: Optional[TaskResult] = None
    error_message: Optional[str] = None
    
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return (
            self.status == DistributedTaskStatus.FAILED and
            self.retry_count < self.max_retries
        )
    
    def get_execution_duration(self) -> Optional[float]:
        """Get task execution duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class NetworkProtocol:
    """Network protocol for distributed communication."""
    
    # Message types
    HEARTBEAT = "heartbeat"
    TASK_ASSIGNMENT = "task_assignment"
    TASK_RESULT = "task_result"
    TASK_STATUS = "task_status"
    NODE_REGISTRATION = "node_registration"
    NODE_SHUTDOWN = "node_shutdown"
    COORDINATION_MESSAGE = "coordination"
    
    @staticmethod
    def create_message(msg_type: str, data: Any, sender_id: str) -> Dict[str, Any]:
        """Create a standardized network message."""
        return {
            "type": msg_type,
            "data": data,
            "sender_id": sender_id,
            "timestamp": datetime.now().isoformat(),
            "message_id": str(uuid.uuid4())
        }
    
    @staticmethod
    def serialize_message(message: Dict[str, Any]) -> bytes:
        """Serialize message for network transmission."""
        try:
            json_str = json.dumps(message, default=str)
            return json_str.encode('utf-8')
        except Exception as e:
            logger.error(f"Failed to serialize message: {e}")
            raise
    
    @staticmethod
    def deserialize_message(data: bytes) -> Dict[str, Any]:
        """Deserialize network message."""
        try:
            json_str = data.decode('utf-8')
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Failed to deserialize message: {e}")
            raise


class NetworkCommunicator:
    """Handles network communication between distributed nodes."""
    
    def __init__(self, node_id: str, host: str = "localhost", port: int = 0):
        self.node_id = node_id
        self.host = host
        self.port = port
        
        # Network setup
        self.socket: Optional[socket.socket] = None
        self.server_socket: Optional[socket.socket] = None
        self.running = False
        
        # Message handling
        self.message_handlers: Dict[str, Callable] = {}
        self.pending_responses: Dict[str, asyncio.Future] = {}
        
        # Connection management
        self.connections: Dict[str, socket.socket] = {}
        self.connection_lock = threading.Lock()
        
        # Communication stats
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "connection_errors": 0,
            "active_connections": 0
        }
    
    def start_server(self) -> int:
        """Start network server and return the port."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            
            # Get actual port if 0 was specified
            actual_port = self.server_socket.getsockname()[1]
            self.port = actual_port
            
            self.server_socket.listen(10)
            self.running = True
            
            # Start server thread
            server_thread = threading.Thread(target=self._server_loop, daemon=True)
            server_thread.start()
            
            logger.info(f"Network server started on {self.host}:{self.port}")
            return actual_port
            
        except Exception as e:
            logger.error(f"Failed to start network server: {e}")
            raise
    
    def stop_server(self):
        """Stop network server."""
        self.running = False
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception as e:
                logger.error(f"Error closing server socket: {e}")
        
        # Close all connections
        with self.connection_lock:
            for conn in self.connections.values():
                try:
                    conn.close()
                except:
                    pass
            self.connections.clear()
        
        logger.info("Network server stopped")
    
    def _server_loop(self):
        """Main server loop for handling incoming connections."""
        while self.running and self.server_socket:
            try:
                client_socket, address = self.server_socket.accept()
                
                # Handle client in separate thread
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address),
                    daemon=True
                )
                client_thread.start()
                
                self.stats["active_connections"] += 1
                
            except Exception as e:
                if self.running:  # Only log if not shutting down
                    logger.error(f"Server loop error: {e}")
    
    def _handle_client(self, client_socket: socket.socket, address: Tuple[str, int]):
        """Handle individual client connection."""
        try:
            while self.running:
                # Receive message
                data = self._receive_message(client_socket)
                if not data:
                    break
                
                # Process message
                try:
                    message = NetworkProtocol.deserialize_message(data)
                    self._process_message(message, client_socket)
                    self.stats["messages_received"] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process message: {e}")
        
        except Exception as e:
            logger.error(f"Client handler error: {e}")
        
        finally:
            client_socket.close()
            self.stats["active_connections"] -= 1
    
    def _receive_message(self, sock: socket.socket) -> Optional[bytes]:
        """Receive a complete message from socket."""
        try:
            # First, receive message length (4 bytes)
            length_data = sock.recv(4)
            if len(length_data) < 4:
                return None
            
            message_length = int.from_bytes(length_data, byteorder='big')
            
            # Receive the message
            message_data = b''
            while len(message_data) < message_length:
                chunk = sock.recv(message_length - len(message_data))
                if not chunk:
                    return None
                message_data += chunk
            
            return message_data
            
        except Exception as e:
            logger.error(f"Failed to receive message: {e}")
            return None
    
    def _send_message(self, sock: socket.socket, message: bytes):
        """Send a complete message through socket."""
        try:
            # Send message length first
            length_bytes = len(message).to_bytes(4, byteorder='big')
            sock.sendall(length_bytes + message)
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise
    
    def _process_message(self, message: Dict[str, Any], client_socket: socket.socket):
        """Process received message."""
        msg_type = message.get("type")
        
        if msg_type in self.message_handlers:
            try:
                handler = self.message_handlers[msg_type]
                handler(message, client_socket)
            except Exception as e:
                logger.error(f"Message handler error for {msg_type}: {e}")
        else:
            logger.warning(f"No handler for message type: {msg_type}")
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register a message handler."""
        self.message_handlers[message_type] = handler
    
    def send_to_node(self, target_host: str, target_port: int, message: Dict[str, Any]) -> bool:
        """Send message to specific node."""
        try:
            # Serialize message
            message_data = NetworkProtocol.serialize_message(message)
            
            # Create connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10.0)  # 10 second timeout
            
            try:
                sock.connect((target_host, target_port))
                self._send_message(sock, message_data)
                self.stats["messages_sent"] += 1
                return True
                
            finally:
                sock.close()
        
        except Exception as e:
            logger.error(f"Failed to send message to {target_host}:{target_port}: {e}")
            self.stats["connection_errors"] += 1
            return False
    
    def broadcast_to_nodes(self, nodes: List[Tuple[str, int]], message: Dict[str, Any]) -> int:
        """Broadcast message to multiple nodes."""
        success_count = 0
        
        for host, port in nodes:
            if self.send_to_node(host, port, message):
                success_count += 1
        
        return success_count
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        return self.stats.copy()


class DistributedCoordinator:
    """Coordinates distributed profiling across multiple nodes."""
    
    def __init__(
        self,
        coordinator_id: Optional[str] = None,
        host: str = "localhost",
        port: int = 8000
    ):
        self.coordinator_id = coordinator_id or f"coordinator_{uuid.uuid4().hex[:8]}"
        self.host = host
        self.port = port
        
        # Node management
        self.nodes: Dict[str, NodeInfo] = {}
        self.node_lock = threading.RLock()
        
        # Task management
        self.task_queue: List[DistributedTask] = []  # Priority queue
        self.active_tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks: Dict[str, DistributedTask] = {}
        self.task_lock = threading.RLock()
        
        # Network communication
        self.communicator = NetworkCommunicator(
            self.coordinator_id, host, port
        )
        self._setup_message_handlers()
        
        # Load balancing
        self.load_balancer = LoadBalancer()
        
        # Auto-scaling
        self.auto_scaler = AutoScaler()
        
        # Monitoring
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "nodes_registered": 0,
            "nodes_active": 0,
            "total_execution_time": 0.0
        }
        
        # Background tasks
        self.running = False
        self.coordinator_threads: List[threading.Thread] = []
    
    def start(self):
        """Start the distributed coordinator."""
        logger.info(f"Starting distributed coordinator: {self.coordinator_id}")
        
        # Start network server
        actual_port = self.communicator.start_server()
        self.port = actual_port
        
        self.running = True
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info(f"Distributed coordinator started on {self.host}:{self.port}")
    
    def stop(self):
        """Stop the distributed coordinator."""
        logger.info("Stopping distributed coordinator...")
        
        self.running = False
        
        # Send shutdown message to all nodes
        self._notify_nodes_shutdown()
        
        # Stop network communication
        self.communicator.stop_server()
        
        # Wait for background threads
        for thread in self.coordinator_threads:
            thread.join(timeout=5.0)
        
        logger.info("Distributed coordinator stopped")
    
    def _setup_message_handlers(self):
        """Setup message handlers for network communication."""
        handlers = {
            NetworkProtocol.HEARTBEAT: self._handle_heartbeat,
            NetworkProtocol.NODE_REGISTRATION: self._handle_node_registration,
            NetworkProtocol.NODE_SHUTDOWN: self._handle_node_shutdown,
            NetworkProtocol.TASK_RESULT: self._handle_task_result,
            NetworkProtocol.TASK_STATUS: self._handle_task_status
        }
        
        for msg_type, handler in handlers.items():
            self.communicator.register_handler(msg_type, handler)
    
    def _start_background_tasks(self):
        """Start background coordinator tasks."""
        tasks = [
            ("task_scheduler", self._task_scheduling_loop),
            ("health_monitor", self._health_monitoring_loop),
            ("load_balancer", self._load_balancing_loop)
        ]
        
        for name, target in tasks:
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self.coordinator_threads.append(thread)
    
    def register_node(
        self,
        node_id: str,
        role: NodeRole,
        host: str,
        port: int,
        supported_platforms: Set[str],
        max_concurrent_tasks: int = 4
    ) -> bool:
        """Register a new distributed node."""
        with self.node_lock:
            node_info = NodeInfo(
                node_id=node_id,
                role=role,
                status=NodeStatus.ACTIVE,
                host=host,
                port=port,
                supported_platforms=supported_platforms,
                max_concurrent_tasks=max_concurrent_tasks
            )
            
            self.nodes[node_id] = node_info
            self.stats["nodes_registered"] += 1
            
            # Register with load balancer
            self.load_balancer.register_worker(
                node_id,
                capabilities={"specialized_platforms": list(supported_platforms)}
            )
            
            logger.info(f"Registered node: {node_id} ({role}) at {host}:{port}")
            return True
    
    def submit_distributed_task(
        self,
        profiling_task: ProfilingTask,
        priority: int = 0,
        required_platforms: Optional[Set[str]] = None,
        preferred_nodes: Optional[Set[str]] = None,
        excluded_nodes: Optional[Set[str]] = None
    ) -> str:
        """
        Submit a task for distributed execution.
        
        Args:
            profiling_task: The profiling task to execute
            priority: Task priority (higher = more important)
            required_platforms: Platforms that must support this task
            preferred_nodes: Preferred nodes for execution
            excluded_nodes: Nodes to exclude from execution
            
        Returns:
            Distributed task ID
        """
        task_id = f"dist_task_{uuid.uuid4().hex[:12]}"
        
        distributed_task = DistributedTask(
            task_id=task_id,
            profiling_task=profiling_task,
            priority=priority,
            required_platforms=required_platforms or {profiling_task.platform},
            preferred_nodes=preferred_nodes or set(),
            excluded_nodes=excluded_nodes or set()
        )
        
        with self.task_lock:
            # Insert into priority queue (heap)
            heapq.heappush(self.task_queue, (-priority, task_id, distributed_task))
            self.stats["tasks_submitted"] += 1
        
        logger.info(f"Submitted distributed task: {task_id} (priority: {priority})")
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[DistributedTaskStatus]:
        """Get status of a distributed task."""
        with self.task_lock:
            # Check active tasks
            if task_id in self.active_tasks:
                return self.active_tasks[task_id].status
            
            # Check completed tasks
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id].status
            
            # Check queue
            for _, tid, task in self.task_queue:
                if tid == task_id:
                    return task.status
        
        return None
    
    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Get result of a completed distributed task."""
        with self.task_lock:
            if task_id in self.completed_tasks:
                task = self.completed_tasks[task_id]
                return task.result
        
        return None
    
    def _task_scheduling_loop(self):
        """Main task scheduling loop."""
        while self.running:
            try:
                # Get next task from priority queue
                task_to_schedule = None
                
                with self.task_lock:
                    if self.task_queue:
                        _, task_id, task = heapq.heappop(self.task_queue)
                        task_to_schedule = task
                
                if task_to_schedule:
                    # Try to assign task to a node
                    assigned = self._assign_task_to_node(task_to_schedule)
                    
                    if not assigned:
                        # Put task back in queue with lower priority
                        with self.task_lock:
                            heapq.heappush(
                                self.task_queue,
                                (-task_to_schedule.priority + 1, task_to_schedule.task_id, task_to_schedule)
                            )
                
                # Sleep between scheduling attempts
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Task scheduling error: {e}")
                time.sleep(5.0)
    
    def _assign_task_to_node(self, task: DistributedTask) -> bool:
        """Assign a task to an appropriate node."""
        with self.node_lock:
            # Find candidate nodes
            candidate_nodes = []
            
            for node_id, node_info in self.nodes.items():
                if not node_info.can_accept_task():
                    continue
                
                # Check platform requirements
                if task.required_platforms and not (
                    task.required_platforms & node_info.supported_platforms
                ):
                    continue
                
                # Check exclusions
                if node_id in task.excluded_nodes:
                    continue
                
                candidate_nodes.append((node_id, node_info))
            
            if not candidate_nodes:
                return False
            
            # Select best node using load balancer
            selected_node_id = None
            
            if task.preferred_nodes:
                # Check preferred nodes first
                preferred_candidates = [
                    (nid, ninfo) for nid, ninfo in candidate_nodes
                    if nid in task.preferred_nodes
                ]
                if preferred_candidates:
                    candidate_nodes = preferred_candidates
            
            # Use load balancer to select node
            if len(candidate_nodes) == 1:
                selected_node_id = candidate_nodes[0][0]
            else:
                # Calculate load scores and select best node
                best_node_id = None
                best_score = float('inf')
                
                for node_id, node_info in candidate_nodes:
                    score = node_info.calculate_load_score()
                    if score < best_score:
                        best_score = score
                        best_node_id = node_id
                
                selected_node_id = best_node_id
            
            if selected_node_id:
                # Assign task
                task.assigned_node_id = selected_node_id
                task.assignment_time = datetime.now()
                task.status = DistributedTaskStatus.ASSIGNED
                
                # Update node info
                selected_node = self.nodes[selected_node_id]
                selected_node.current_task_count += 1
                
                # Move to active tasks
                with self.task_lock:
                    self.active_tasks[task.task_id] = task
                
                # Send task to node
                success = self._send_task_to_node(task, selected_node)
                
                if success:
                    logger.info(f"Assigned task {task.task_id} to node {selected_node_id}")
                    return True
                else:
                    # Assignment failed, revert changes
                    selected_node.current_task_count -= 1
                    task.assigned_node_id = None
                    task.status = DistributedTaskStatus.QUEUED
                    
                    with self.task_lock:
                        del self.active_tasks[task.task_id]
        
        return False
    
    def _send_task_to_node(self, task: DistributedTask, node_info: NodeInfo) -> bool:
        """Send task assignment to a node."""
        try:
            # Prepare task data
            task_data = {
                "task_id": task.task_id,
                "profiling_task": asdict(task.profiling_task),
                "priority": task.priority
            }
            
            message = NetworkProtocol.create_message(
                NetworkProtocol.TASK_ASSIGNMENT,
                task_data,
                self.coordinator_id
            )
            
            # Send to node
            success = self.communicator.send_to_node(
                node_info.host,
                node_info.port,
                message
            )
            
            if success:
                task.status = DistributedTaskStatus.RUNNING
                task.start_time = datetime.now()
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to send task to node: {e}")
            return False
    
    def _handle_heartbeat(self, message: Dict[str, Any], client_socket: socket.socket):
        """Handle heartbeat message from a node."""
        try:
            sender_id = message.get("sender_id")
            data = message.get("data", {})
            
            with self.node_lock:
                if sender_id in self.nodes:
                    node = self.nodes[sender_id]
                    node.last_heartbeat = datetime.now()
                    
                    # Update node metrics from heartbeat
                    node.cpu_usage = data.get("cpu_usage", 0.0)
                    node.memory_usage_mb = data.get("memory_usage_mb", 0.0)
                    node.current_task_count = data.get("current_task_count", 0)
                    node.error_rate = data.get("error_rate", 0.0)
                    
                    # Update status based on load
                    if node.current_task_count >= node.max_concurrent_tasks:
                        node.status = NodeStatus.OVERLOADED
                    elif node.current_task_count > 0:
                        node.status = NodeStatus.BUSY
                    else:
                        node.status = NodeStatus.ACTIVE
        
        except Exception as e:
            logger.error(f"Failed to handle heartbeat: {e}")
    
    def _handle_node_registration(self, message: Dict[str, Any], client_socket: socket.socket):
        """Handle node registration message."""
        try:
            data = message.get("data", {})
            
            success = self.register_node(
                node_id=data["node_id"],
                role=NodeRole(data["role"]),
                host=data["host"],
                port=data["port"],
                supported_platforms=set(data["supported_platforms"]),
                max_concurrent_tasks=data.get("max_concurrent_tasks", 4)
            )
            
            # Send response
            response = NetworkProtocol.create_message(
                "registration_response",
                {"success": success},
                self.coordinator_id
            )
            
            response_data = NetworkProtocol.serialize_message(response)
            self.communicator._send_message(client_socket, response_data)
            
        except Exception as e:
            logger.error(f"Failed to handle node registration: {e}")
    
    def _handle_node_shutdown(self, message: Dict[str, Any], client_socket: socket.socket):
        """Handle node shutdown message."""
        try:
            sender_id = message.get("sender_id")
            
            with self.node_lock:
                if sender_id in self.nodes:
                    self.nodes[sender_id].status = NodeStatus.OFFLINE
                    logger.info(f"Node {sender_id} shutdown")
        
        except Exception as e:
            logger.error(f"Failed to handle node shutdown: {e}")
    
    def _handle_task_result(self, message: Dict[str, Any], client_socket: socket.socket):
        """Handle task result message from a node."""
        try:
            data = message.get("data", {})
            task_id = data.get("task_id")
            sender_id = message.get("sender_id")
            
            with self.task_lock:
                if task_id in self.active_tasks:
                    task = self.active_tasks[task_id]
                    
                    # Update task with result
                    task.end_time = datetime.now()
                    
                    if data.get("success", False):
                        task.status = DistributedTaskStatus.COMPLETED
                        task.result = TaskResult(
                            task_id=task_id,
                            success=True,
                            result=data.get("result"),
                            duration_seconds=task.get_execution_duration()
                        )
                        self.stats["tasks_completed"] += 1
                    else:
                        task.status = DistributedTaskStatus.FAILED
                        task.error_message = data.get("error")
                        task.result = TaskResult(
                            task_id=task_id,
                            success=False,
                            error=Exception(task.error_message)
                        )
                        self.stats["tasks_failed"] += 1
                        
                        # Consider retry
                        if task.can_retry():
                            task.retry_count += 1
                            task.status = DistributedTaskStatus.RETRYING
                            task.assigned_node_id = None
                            # Put back in queue
                            heapq.heappush(
                                self.task_queue,
                                (-task.priority, task.task_id, task)
                            )
                    
                    # Move to completed if not retrying
                    if task.status != DistributedTaskStatus.RETRYING:
                        self.completed_tasks[task_id] = task
                        del self.active_tasks[task_id]
                        
                        # Update execution time stats
                        if task.get_execution_duration():
                            self.stats["total_execution_time"] += task.get_execution_duration()
                    
                    # Update node info
                    with self.node_lock:
                        if sender_id in self.nodes:
                            node = self.nodes[sender_id]
                            node.current_task_count = max(0, node.current_task_count - 1)
                            node.total_tasks_completed += 1
                            
                            # Update average task duration
                            if task.get_execution_duration():
                                if node.avg_task_duration == 0:
                                    node.avg_task_duration = task.get_execution_duration()
                                else:
                                    node.avg_task_duration = (
                                        0.9 * node.avg_task_duration + 
                                        0.1 * task.get_execution_duration()
                                    )
        
        except Exception as e:
            logger.error(f"Failed to handle task result: {e}")
    
    def _handle_task_status(self, message: Dict[str, Any], client_socket: socket.socket):
        """Handle task status update from a node."""
        try:
            data = message.get("data", {})
            task_id = data.get("task_id")
            status = DistributedTaskStatus(data.get("status"))
            
            with self.task_lock:
                if task_id in self.active_tasks:
                    task = self.active_tasks[task_id]
                    task.status = status
        
        except Exception as e:
            logger.error(f"Failed to handle task status: {e}")
    
    def _health_monitoring_loop(self):
        """Monitor health of distributed nodes."""
        while self.running:
            try:
                with self.node_lock:
                    active_count = 0
                    unhealthy_nodes = []
                    
                    for node_id, node_info in self.nodes.items():
                        if node_info.is_healthy():
                            active_count += 1
                        else:
                            unhealthy_nodes.append(node_id)
                            node_info.status = NodeStatus.FAILING
                    
                    self.stats["nodes_active"] = active_count
                    
                    # Handle unhealthy nodes
                    for node_id in unhealthy_nodes:
                        self._handle_unhealthy_node(node_id)
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(60)
    
    def _handle_unhealthy_node(self, node_id: str):
        """Handle an unhealthy or failed node."""
        with self.node_lock:
            node = self.nodes[node_id]
            logger.warning(f"Node {node_id} is unhealthy: {node.status}")
            
            # Reassign active tasks from this node
            tasks_to_reassign = []
            
            with self.task_lock:
                for task_id, task in self.active_tasks.items():
                    if task.assigned_node_id == node_id:
                        tasks_to_reassign.append(task)
                
                # Reassign tasks
                for task in tasks_to_reassign:
                    task.assigned_node_id = None
                    task.status = DistributedTaskStatus.QUEUED
                    task.retry_count += 1
                    
                    # Put back in queue
                    heapq.heappush(
                        self.task_queue,
                        (-task.priority, task.task_id, task)
                    )
                    
                    # Remove from active tasks
                    del self.active_tasks[task.task_id]
                    
                    logger.info(f"Reassigned task {task.task_id} from failed node {node_id}")
    
    def _load_balancing_loop(self):
        """Periodic load balancing and optimization."""
        while self.running:
            try:
                # Update load balancer metrics
                with self.node_lock:
                    for node_id, node_info in self.nodes.items():
                        # Update load balancer with node stats
                        if node_info.is_healthy():
                            node_score = node_info.calculate_load_score()
                            
                            # Simple load balancing: mark overloaded nodes
                            if node_score > 80:
                                node_info.status = NodeStatus.OVERLOADED
                            elif node_score > 50:
                                node_info.status = NodeStatus.BUSY
                            else:
                                node_info.status = NodeStatus.ACTIVE
                
                time.sleep(60)  # Balance every minute
                
            except Exception as e:
                logger.error(f"Load balancing error: {e}")
                time.sleep(120)
    
    def _notify_nodes_shutdown(self):
        """Notify all nodes of coordinator shutdown."""
        message = NetworkProtocol.create_message(
            "coordinator_shutdown",
            {},
            self.coordinator_id
        )
        
        with self.node_lock:
            node_addresses = [
                (node.host, node.port) 
                for node in self.nodes.values() 
                if node.status != NodeStatus.OFFLINE
            ]
        
        if node_addresses:
            self.communicator.broadcast_to_nodes(node_addresses, message)
    
    def get_distributed_stats(self) -> Dict[str, Any]:
        """Get comprehensive distributed profiling statistics."""
        with self.node_lock, self.task_lock:
            node_stats = {}
            for node_id, node_info in self.nodes.items():
                node_stats[node_id] = {
                    "status": node_info.status.value,
                    "role": node_info.role.value,
                    "supported_platforms": list(node_info.supported_platforms),
                    "current_task_count": node_info.current_task_count,
                    "max_concurrent_tasks": node_info.max_concurrent_tasks,
                    "total_tasks_completed": node_info.total_tasks_completed,
                    "avg_task_duration": node_info.avg_task_duration,
                    "error_rate": node_info.error_rate,
                    "load_score": node_info.load_score,
                    "is_healthy": node_info.is_healthy()
                }
            
            return {
                "coordinator_id": self.coordinator_id,
                "coordinator_address": f"{self.host}:{self.port}",
                "stats": self.stats,
                "nodes": node_stats,
                "task_queue_size": len(self.task_queue),
                "active_tasks": len(self.active_tasks),
                "completed_tasks": len(self.completed_tasks),
                "communication_stats": self.communicator.get_communication_stats()
            }


# Global distributed coordinator instance
_global_coordinator: Optional[DistributedCoordinator] = None


def get_distributed_coordinator(
    coordinator_id: Optional[str] = None,
    host: str = "localhost", 
    port: int = 8000
) -> DistributedCoordinator:
    """Get or create the global distributed coordinator."""
    global _global_coordinator
    
    if _global_coordinator is None:
        _global_coordinator = DistributedCoordinator(coordinator_id, host, port)
    
    return _global_coordinator


def start_distributed_profiling(
    coordinator_host: str = "localhost",
    coordinator_port: int = 8000
) -> DistributedCoordinator:
    """Start distributed profiling system."""
    coordinator = get_distributed_coordinator(
        host=coordinator_host, 
        port=coordinator_port
    )
    coordinator.start()
    return coordinator


def submit_distributed_profiling_task(
    model: QuantizedModel,
    platform: str,
    test_prompts: List[str],
    device_path: Optional[str] = None,
    priority: int = 0,
    preferred_nodes: Optional[Set[str]] = None
) -> str:
    """Submit a profiling task for distributed execution."""
    coordinator = get_distributed_coordinator()
    
    # Create profiling task
    task_id = f"profile_{platform}_{model.name}_{int(time.time() * 1000)}"
    profiling_task = ProfilingTask(
        task_id=task_id,
        platform=platform,
        model=model,
        device_path=device_path,
        test_prompts=test_prompts or ["Hello world"],
        metrics=["latency", "memory"],
        priority=priority
    )
    
    return coordinator.submit_distributed_task(
        profiling_task,
        priority=priority,
        preferred_nodes=preferred_nodes
    )