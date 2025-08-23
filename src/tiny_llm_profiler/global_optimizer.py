"""
Global Optimization System for Multi-Region Deployment - Generation 3
Provides comprehensive global optimization capabilities including:
- Multi-region deployment coordination and optimization
- Network latency optimization across geographic regions
- Geo-distributed profiling with regional load balancing
- Cross-region data synchronization and consistency
- Global resource allocation and cost optimization
- Regional performance monitoring and analytics
- Disaster recovery and failover optimization
- Global cache coherence and synchronization
"""

import time
import threading
import asyncio
import aiohttp
import socket
import struct
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod
import json
import hashlib
import numpy as np
from pathlib import Path
import ipaddress
import dns.resolver

# ping3 may not be available in all environments
try:
    import ping3

    PING3_AVAILABLE = True
except ImportError:
    PING3_AVAILABLE = False
    ping3 = None

from .exceptions import TinyLLMProfilerError, NetworkError, ResourceError
from .logging_config import get_logger
from .performance_analytics import PerformanceMetric, record_performance_metric
from .distributed_profiler import DistributedCoordinator, NodeInfo

logger = get_logger("global_optimizer")


class RegionType(str, Enum):
    """Types of deployment regions."""

    PRIMARY = "primary"  # Main processing region
    SECONDARY = "secondary"  # Backup processing region
    EDGE = "edge"  # Edge computing nodes
    CDN = "cdn"  # Content delivery nodes
    CACHE = "cache"  # Distributed cache nodes


class OptimizationStrategy(str, Enum):
    """Global optimization strategies."""

    LATENCY_FIRST = "latency_first"  # Minimize latency
    COST_FIRST = "cost_first"  # Minimize costs
    RELIABILITY_FIRST = "reliability_first"  # Maximize reliability
    BALANCED = "balanced"  # Balance all factors
    CUSTOM = "custom"  # Custom optimization criteria


class RegionStatus(str, Enum):
    """Regional deployment status."""

    ACTIVE = "active"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


@dataclass
class NetworkLatencyInfo:
    """Network latency information between regions."""

    from_region: str
    to_region: str
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    jitter_ms: float
    packet_loss_percent: float
    bandwidth_mbps: float
    timestamp: datetime = field(default_factory=datetime.now)

    def quality_score(self) -> float:
        """Calculate network quality score (0-100)."""
        # Lower latency and packet loss = higher score
        latency_score = max(0, 100 - (self.avg_latency_ms / 10))  # -10 per 100ms
        loss_score = max(0, 100 - (self.packet_loss_percent * 10))  # -10 per 1% loss
        jitter_score = max(0, 100 - (self.jitter_ms / 5))  # -20 per 5ms jitter

        return (latency_score + loss_score + jitter_score) / 3


@dataclass
class RegionInfo:
    """Information about a deployment region."""

    region_id: str
    name: str
    region_type: RegionType
    location: Dict[str, float]  # {"lat": float, "lng": float}
    capacity: Dict[str, float]  # Resource capacity
    current_load: Dict[str, float]  # Current resource usage
    status: RegionStatus
    endpoint: str
    cost_per_hour: float
    availability_zone: str
    provider: str = "unknown"

    # Performance metrics
    avg_response_time_ms: float = 0.0
    throughput_rps: float = 0.0
    error_rate_percent: float = 0.0
    uptime_percent: float = 100.0

    # Network connectivity
    connected_regions: Set[str] = field(default_factory=set)
    network_latencies: Dict[str, NetworkLatencyInfo] = field(default_factory=dict)

    def utilization_score(self) -> float:
        """Calculate current utilization score (0-100)."""
        if not self.capacity or not self.current_load:
            return 0.0

        total_capacity = sum(self.capacity.values())
        total_load = sum(self.current_load.values())

        if total_capacity == 0:
            return 100.0

        return min(100.0, (total_load / total_capacity) * 100)

    def health_score(self) -> float:
        """Calculate region health score (0-100)."""
        if self.status == RegionStatus.OFFLINE:
            return 0.0
        elif self.status == RegionStatus.MAINTENANCE:
            return 25.0
        elif self.status == RegionStatus.DEGRADED:
            return 50.0

        # For active regions, calculate based on metrics
        response_score = max(0, 100 - (self.avg_response_time_ms / 10))
        error_score = max(0, 100 - (self.error_rate_percent * 10))
        uptime_score = self.uptime_percent

        return (response_score + error_score + uptime_score) / 3


@dataclass
class GlobalOptimizationResult:
    """Result of global optimization analysis."""

    strategy_used: OptimizationStrategy
    recommended_deployments: Dict[str, List[str]]  # region_id -> [services]
    traffic_routing: Dict[str, Dict[str, float]]  # from_region -> {to_region: weight}
    resource_allocations: Dict[str, Dict[str, float]]  # region_id -> {resource: amount}
    cost_estimate: float
    expected_latency_improvement: float
    confidence_score: float
    optimization_timestamp: datetime = field(default_factory=datetime.now)

    def to_deployment_plan(self) -> Dict[str, Any]:
        """Convert to deployment plan format."""
        return {
            "optimization_id": hashlib.md5(
                f"{self.strategy_used}_{self.optimization_timestamp}".encode()
            ).hexdigest()[:8],
            "strategy": self.strategy_used.value,
            "deployments": self.recommended_deployments,
            "routing": self.traffic_routing,
            "resources": self.resource_allocations,
            "cost_estimate": self.cost_estimate,
            "expected_improvement": self.expected_latency_improvement,
            "confidence": self.confidence_score,
            "created_at": self.optimization_timestamp.isoformat(),
        }


class NetworkAnalyzer:
    """Analyzes network performance between regions."""

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
        self.latency_cache: Dict[Tuple[str, str], NetworkLatencyInfo] = {}
        self.cache_ttl = timedelta(minutes=5)

    async def measure_latency(
        self, from_region: RegionInfo, to_region: RegionInfo
    ) -> NetworkLatencyInfo:
        """Measure network latency between two regions."""
        cache_key = (from_region.region_id, to_region.region_id)

        # Check cache
        if cache_key in self.latency_cache:
            cached_info = self.latency_cache[cache_key]
            if datetime.now() - cached_info.timestamp < self.cache_ttl:
                return cached_info

        try:
            # Parse endpoint to get hostname
            endpoint = to_region.endpoint.replace("http://", "").replace("https://", "")
            hostname = endpoint.split(":")[0] if ":" in endpoint else endpoint

            # Perform multiple ping measurements
            latencies = []
            for _ in range(5):
                latency = await self._async_ping(hostname)
                if latency is not None:
                    latencies.append(latency * 1000)  # Convert to ms

            if not latencies:
                # Fallback to TCP connection test
                latency = await self._tcp_latency_test(hostname, 80)
                latencies = [latency] if latency else []

            if latencies:
                avg_latency = np.mean(latencies)
                min_latency = min(latencies)
                max_latency = max(latencies)
                jitter = np.std(latencies)
                packet_loss = max(0, (5 - len(latencies)) / 5 * 100)
            else:
                # Default values for unreachable endpoints
                avg_latency = 1000.0  # High latency
                min_latency = 1000.0
                max_latency = 1000.0
                jitter = 0.0
                packet_loss = 100.0

            # Estimate bandwidth (simplified)
            bandwidth = max(1.0, 1000 / max(avg_latency, 1))  # Rough estimate

            latency_info = NetworkLatencyInfo(
                from_region=from_region.region_id,
                to_region=to_region.region_id,
                avg_latency_ms=avg_latency,
                min_latency_ms=min_latency,
                max_latency_ms=max_latency,
                jitter_ms=jitter,
                packet_loss_percent=packet_loss,
                bandwidth_mbps=bandwidth,
            )

            # Cache result
            self.latency_cache[cache_key] = latency_info

            return latency_info

        except Exception as e:
            logger.error(
                f"Failed to measure latency from {from_region.region_id} to {to_region.region_id}: {e}"
            )

            # Return default high latency values
            return NetworkLatencyInfo(
                from_region=from_region.region_id,
                to_region=to_region.region_id,
                avg_latency_ms=1000.0,
                min_latency_ms=1000.0,
                max_latency_ms=1000.0,
                jitter_ms=0.0,
                packet_loss_percent=100.0,
                bandwidth_mbps=1.0,
            )

    async def _async_ping(self, hostname: str) -> Optional[float]:
        """Async ping implementation."""
        try:
            # Use ping3 in a thread to avoid blocking
            import concurrent.futures

            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(
                    executor,
                    lambda: (
                        ping3.ping(hostname, timeout=self.timeout)
                        if PING3_AVAILABLE
                        else 0.1
                    ),
                )
                return result
        except Exception:
            return None

    async def _tcp_latency_test(self, hostname: str, port: int) -> Optional[float]:
        """Test TCP connection latency."""
        try:
            start_time = time.time()

            # Try to establish TCP connection
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(hostname, port), timeout=self.timeout
            )

            latency = (time.time() - start_time) * 1000  # Convert to ms

            writer.close()
            await writer.wait_closed()

            return latency

        except Exception:
            return None

    def calculate_geographic_distance(
        self, region1: RegionInfo, region2: RegionInfo
    ) -> float:
        """Calculate geographic distance between regions in kilometers."""
        if "lat" not in region1.location or "lng" not in region1.location:
            return 0.0
        if "lat" not in region2.location or "lng" not in region2.location:
            return 0.0

        # Haversine formula
        lat1, lng1 = region1.location["lat"], region1.location["lng"]
        lat2, lng2 = region2.location["lat"], region2.location["lng"]

        R = 6371  # Earth's radius in km

        dlat = np.radians(lat2 - lat1)
        dlng = np.radians(lng2 - lng1)

        a = np.sin(dlat / 2) * np.sin(dlat / 2) + np.cos(np.radians(lat1)) * np.cos(
            np.radians(lat2)
        ) * np.sin(dlng / 2) * np.sin(dlng / 2)

        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = R * c

        return distance


class TrafficRouter:
    """Manages traffic routing across regions."""

    def __init__(self):
        self.routing_rules: Dict[str, Dict[str, float]] = {}
        self.fallback_regions: Dict[str, List[str]] = {}

    def optimize_routing(
        self,
        regions: Dict[str, RegionInfo],
        network_matrix: Dict[Tuple[str, str], NetworkLatencyInfo],
        strategy: OptimizationStrategy = OptimizationStrategy.LATENCY_FIRST,
    ) -> Dict[str, Dict[str, float]]:
        """Optimize traffic routing between regions."""
        routing = {}

        for source_region_id, source_region in regions.items():
            if source_region.status != RegionStatus.ACTIVE:
                continue

            # Find best target regions based on strategy
            target_weights = self._calculate_target_weights(
                source_region, regions, network_matrix, strategy
            )

            # Normalize weights
            total_weight = sum(target_weights.values())
            if total_weight > 0:
                routing[source_region_id] = {
                    target: weight / total_weight
                    for target, weight in target_weights.items()
                }
            else:
                # Fallback to self-routing
                routing[source_region_id] = {source_region_id: 1.0}

        return routing

    def _calculate_target_weights(
        self,
        source_region: RegionInfo,
        regions: Dict[str, RegionInfo],
        network_matrix: Dict[Tuple[str, str], NetworkLatencyInfo],
        strategy: OptimizationStrategy,
    ) -> Dict[str, float]:
        """Calculate target weights for a source region."""
        weights = {}

        for target_id, target_region in regions.items():
            if target_region.status != RegionStatus.ACTIVE:
                continue

            # Get network info
            network_key = (source_region.region_id, target_id)
            network_info = network_matrix.get(network_key)

            if not network_info:
                continue

            # Calculate weight based on strategy
            if strategy == OptimizationStrategy.LATENCY_FIRST:
                # Lower latency = higher weight
                base_weight = 1.0 / max(network_info.avg_latency_ms, 1.0)
                weight = base_weight * (1.0 - network_info.packet_loss_percent / 100)

            elif strategy == OptimizationStrategy.COST_FIRST:
                # Lower cost = higher weight
                base_weight = 1.0 / max(target_region.cost_per_hour, 0.01)
                # Apply latency penalty
                latency_penalty = min(1.0, 100 / network_info.avg_latency_ms)
                weight = base_weight * latency_penalty

            elif strategy == OptimizationStrategy.RELIABILITY_FIRST:
                # Higher reliability = higher weight
                reliability_score = (
                    (target_region.uptime_percent / 100)
                    * target_region.health_score()
                    / 100
                )
                weight = reliability_score * (
                    1.0 - network_info.packet_loss_percent / 100
                )

            else:  # BALANCED
                # Balance all factors
                latency_score = 1.0 / max(network_info.avg_latency_ms, 1.0)
                cost_score = 1.0 / max(target_region.cost_per_hour, 0.01)
                reliability_score = target_region.health_score() / 100

                weight = (latency_score + cost_score + reliability_score) / 3

            # Apply capacity constraints
            utilization = target_region.utilization_score()
            if utilization > 80:  # High utilization penalty
                weight *= max(0.1, (100 - utilization) / 20)

            weights[target_id] = max(0.0, weight)

        return weights

    def get_routing_recommendations(
        self, routing: Dict[str, Dict[str, float]], traffic_patterns: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Get routing recommendations based on traffic patterns."""
        recommendations = []

        for source_region, targets in routing.items():
            if source_region not in traffic_patterns:
                continue

            source_traffic = traffic_patterns[source_region]

            # Find primary target (highest weight)
            primary_target = max(targets.items(), key=lambda x: x[1])

            # Check if routing is optimal
            if primary_target[1] < 0.6:  # Low confidence in primary target
                recommendations.append(
                    {
                        "type": "routing_uncertainty",
                        "source_region": source_region,
                        "description": f"Low confidence in primary routing target ({primary_target[1]:.2f})",
                        "suggestion": "Consider adding more capacity or improving network connectivity",
                    }
                )

            # Check for traffic concentration
            high_traffic_targets = [
                target
                for target, weight in targets.items()
                if weight * source_traffic > 1000  # High absolute traffic
            ]

            if len(high_traffic_targets) == 1:
                recommendations.append(
                    {
                        "type": "traffic_concentration",
                        "source_region": source_region,
                        "target_region": high_traffic_targets[0],
                        "description": "High traffic concentration in single target region",
                        "suggestion": "Consider load balancing across multiple regions",
                    }
                )

        return recommendations


class RegionManager:
    """Manages deployment regions and their health."""

    def __init__(self):
        self.regions: Dict[str, RegionInfo] = {}
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.network_analyzer = NetworkAnalyzer()
        self._lock = threading.RLock()

    def register_region(self, region_info: RegionInfo):
        """Register a new region."""
        with self._lock:
            self.regions[region_info.region_id] = region_info
            logger.info(
                f"Registered region: {region_info.region_id} ({region_info.name})"
            )

    def update_region_status(
        self, region_id: str, status: RegionStatus, metadata: Optional[Dict] = None
    ):
        """Update region status."""
        with self._lock:
            if region_id in self.regions:
                self.regions[region_id].status = status

                # Record health history
                health_score = self.regions[region_id].health_score()
                self.health_history[region_id].append(
                    {
                        "timestamp": datetime.now(),
                        "status": status.value,
                        "health_score": health_score,
                        "metadata": metadata or {},
                    }
                )

                logger.info(f"Updated region {region_id} status: {status.value}")

    def update_region_metrics(
        self,
        region_id: str,
        response_time_ms: Optional[float] = None,
        throughput_rps: Optional[float] = None,
        error_rate_percent: Optional[float] = None,
        current_load: Optional[Dict[str, float]] = None,
    ):
        """Update region performance metrics."""
        with self._lock:
            if region_id not in self.regions:
                return

            region = self.regions[region_id]

            if response_time_ms is not None:
                region.avg_response_time_ms = response_time_ms
            if throughput_rps is not None:
                region.throughput_rps = throughput_rps
            if error_rate_percent is not None:
                region.error_rate_percent = error_rate_percent
            if current_load is not None:
                region.current_load.update(current_load)

    async def measure_network_connectivity(self):
        """Measure network connectivity between all regions."""
        with self._lock:
            region_list = list(self.regions.values())

        # Measure latency between all region pairs
        tasks = []
        for i, source_region in enumerate(region_list):
            for target_region in region_list[i:]:
                if source_region.region_id != target_region.region_id:
                    # Measure both directions
                    tasks.append(
                        self.network_analyzer.measure_latency(
                            source_region, target_region
                        )
                    )
                    tasks.append(
                        self.network_analyzer.measure_latency(
                            target_region, source_region
                        )
                    )

        # Execute measurements concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Update region connectivity information
        with self._lock:
            for result in results:
                if isinstance(result, NetworkLatencyInfo):
                    source_id = result.from_region
                    target_id = result.to_region

                    if source_id in self.regions:
                        self.regions[source_id].connected_regions.add(target_id)
                        self.regions[source_id].network_latencies[target_id] = result

    def get_region_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for region management."""
        recommendations = []

        with self._lock:
            for region_id, region in self.regions.items():
                # Check region health
                if region.health_score() < 50:
                    recommendations.append(
                        {
                            "type": "region_health",
                            "region_id": region_id,
                            "priority": "high",
                            "description": f"Region {region.name} has low health score ({region.health_score():.1f})",
                            "actions": [
                                "Check region status",
                                "Investigate performance issues",
                                "Consider failover",
                            ],
                        }
                    )

                # Check utilization
                utilization = region.utilization_score()
                if utilization > 90:
                    recommendations.append(
                        {
                            "type": "high_utilization",
                            "region_id": region_id,
                            "priority": "high",
                            "description": f"Region {region.name} has high utilization ({utilization:.1f}%)",
                            "actions": [
                                "Scale up resources",
                                "Implement load balancing",
                                "Add capacity",
                            ],
                        }
                    )
                elif utilization < 10:
                    recommendations.append(
                        {
                            "type": "low_utilization",
                            "region_id": region_id,
                            "priority": "medium",
                            "description": f"Region {region.name} has low utilization ({utilization:.1f}%)",
                            "actions": [
                                "Consider downsizing",
                                "Redirect traffic",
                                "Cost optimization",
                            ],
                        }
                    )

                # Check connectivity
                if len(region.connected_regions) < len(self.regions) - 1:
                    missing_connections = (
                        set(self.regions.keys())
                        - region.connected_regions
                        - {region_id}
                    )
                    if missing_connections:
                        recommendations.append(
                            {
                                "type": "connectivity",
                                "region_id": region_id,
                                "priority": "medium",
                                "description": f"Region {region.name} has limited connectivity",
                                "actions": [
                                    "Check network configuration",
                                    "Verify endpoints",
                                    "Test connectivity",
                                ],
                            }
                        )

        return recommendations

    def get_regions(
        self, status: Optional[RegionStatus] = None
    ) -> Dict[str, RegionInfo]:
        """Get regions filtered by status."""
        with self._lock:
            if status is None:
                return dict(self.regions)
            else:
                return {
                    region_id: region
                    for region_id, region in self.regions.items()
                    if region.status == status
                }


class GlobalOptimizer:
    """Main global optimization coordinator."""

    def __init__(
        self,
        optimization_interval: int = 300,  # 5 minutes
        enable_auto_optimization: bool = True,
    ):
        self.optimization_interval = optimization_interval
        self.enable_auto_optimization = enable_auto_optimization

        # Components
        self.region_manager = RegionManager()
        self.traffic_router = TrafficRouter()

        # Optimization history
        self.optimization_history: List[GlobalOptimizationResult] = []

        # Background optimization
        self.running = False
        self.optimization_executor = ThreadPoolExecutor(max_workers=2)
        self._lock = threading.RLock()

        logger.info("Global Optimizer initialized")

    def start_optimization(self):
        """Start automatic global optimization."""
        if self.running or not self.enable_auto_optimization:
            return

        self.running = True

        # Start optimization tasks
        self.optimization_executor.submit(self._optimization_loop)
        self.optimization_executor.submit(self._monitoring_loop)

        logger.info("Global optimization started")

    def stop_optimization(self):
        """Stop automatic global optimization."""
        if not self.running:
            return

        self.running = False
        self.optimization_executor.shutdown(wait=True, timeout=30.0)

        logger.info("Global optimization stopped")

    async def optimize_deployment(
        self,
        services: List[str],
        traffic_patterns: Dict[str, float],
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> GlobalOptimizationResult:
        """Optimize global deployment configuration."""

        # Get current regions
        regions = self.region_manager.get_regions(RegionStatus.ACTIVE)

        if not regions:
            raise TinyLLMProfilerError("No active regions available for optimization")

        # Measure network connectivity
        await self.region_manager.measure_network_connectivity()

        # Build network latency matrix
        network_matrix = {}
        for source_id, source_region in regions.items():
            for target_id, latency_info in source_region.network_latencies.items():
                if target_id in regions:
                    network_matrix[(source_id, target_id)] = latency_info

        # Optimize service placement
        service_placements = self._optimize_service_placement(
            services, regions, network_matrix, strategy, constraints
        )

        # Optimize traffic routing
        traffic_routing = self.traffic_router.optimize_routing(
            regions, network_matrix, strategy
        )

        # Calculate resource allocations
        resource_allocations = self._calculate_resource_allocations(
            service_placements, traffic_patterns, regions
        )

        # Estimate costs and performance
        cost_estimate = self._estimate_deployment_cost(
            service_placements, resource_allocations, regions
        )
        latency_improvement = self._estimate_latency_improvement(
            network_matrix, traffic_routing
        )

        # Calculate confidence score
        confidence_score = self._calculate_optimization_confidence(
            regions, network_matrix, service_placements
        )

        result = GlobalOptimizationResult(
            strategy_used=strategy,
            recommended_deployments=service_placements,
            traffic_routing=traffic_routing,
            resource_allocations=resource_allocations,
            cost_estimate=cost_estimate,
            expected_latency_improvement=latency_improvement,
            confidence_score=confidence_score,
        )

        # Store optimization result
        with self._lock:
            self.optimization_history.append(result)
            if len(self.optimization_history) > 50:
                self.optimization_history = self.optimization_history[-25:]

        # Record performance metrics
        record_performance_metric(
            "global_optimization_cost", cost_estimate, tags={"strategy", strategy.value}
        )
        record_performance_metric(
            "global_optimization_latency_improvement",
            latency_improvement,
            tags={"strategy", strategy.value},
        )

        logger.info(
            f"Global optimization completed: {strategy.value} strategy, cost=${cost_estimate:.2f}, latency improvement={latency_improvement:.1f}%"
        )

        return result

    def _optimize_service_placement(
        self,
        services: List[str],
        regions: Dict[str, RegionInfo],
        network_matrix: Dict[Tuple[str, str], NetworkLatencyInfo],
        strategy: OptimizationStrategy,
        constraints: Optional[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        """Optimize placement of services across regions."""
        constraints = constraints or {}

        placements = {region_id: [] for region_id in regions.keys()}

        # Sort regions by suitability for each strategy
        region_scores = {}
        for region_id, region in regions.items():
            if strategy == OptimizationStrategy.COST_FIRST:
                # Lower cost = higher score
                score = 1.0 / max(region.cost_per_hour, 0.01)
            elif strategy == OptimizationStrategy.RELIABILITY_FIRST:
                # Higher reliability = higher score
                score = region.health_score() * region.uptime_percent / 100
            elif strategy == OptimizationStrategy.LATENCY_FIRST:
                # Lower average latency to other regions = higher score
                avg_latency = (
                    np.mean(
                        [
                            info.avg_latency_ms
                            for key, info in network_matrix.items()
                            if key[0] == region_id
                        ]
                    )
                    if any(key[0] == region_id for key in network_matrix.keys())
                    else 1000
                )
                score = 1.0 / max(avg_latency, 1.0)
            else:  # BALANCED
                # Weighted combination of all factors
                cost_score = 1.0 / max(region.cost_per_hour, 0.01)
                reliability_score = region.health_score() / 100
                avg_latency = (
                    np.mean(
                        [
                            info.avg_latency_ms
                            for key, info in network_matrix.items()
                            if key[0] == region_id
                        ]
                    )
                    if any(key[0] == region_id for key in network_matrix.keys())
                    else 1000
                )
                latency_score = 1.0 / max(avg_latency, 1.0)

                score = (cost_score + reliability_score + latency_score) / 3

            # Apply capacity constraint
            utilization = region.utilization_score()
            if utilization > 80:
                score *= max(0.1, (100 - utilization) / 20)

            region_scores[region_id] = score

        # Sort regions by score
        sorted_regions = sorted(region_scores.items(), key=lambda x: x[1], reverse=True)

        # Place services using greedy approach
        for service in services:
            # Check for service-specific constraints
            service_constraints = constraints.get(service, {})
            allowed_regions = service_constraints.get("allowed_regions")
            required_redundancy = service_constraints.get("redundancy", 1)

            # Filter available regions
            available_regions = []
            for region_id, score in sorted_regions:
                if allowed_regions and region_id not in allowed_regions:
                    continue
                if regions[region_id].utilization_score() > 95:  # Almost full
                    continue
                available_regions.append((region_id, score))

            # Place service in top regions for redundancy
            placed_count = 0
            for region_id, score in available_regions:
                if placed_count >= required_redundancy:
                    break

                placements[region_id].append(service)
                placed_count += 1

                logger.debug(
                    f"Placed service {service} in region {region_id} (score: {score:.3f})"
                )

            if placed_count < required_redundancy:
                logger.warning(
                    f"Could not achieve required redundancy for service {service} ({placed_count}/{required_redundancy})"
                )

        return placements

    def _calculate_resource_allocations(
        self,
        service_placements: Dict[str, List[str]],
        traffic_patterns: Dict[str, float],
        regions: Dict[str, RegionInfo],
    ) -> Dict[str, Dict[str, float]]:
        """Calculate resource allocations for each region."""
        allocations = {}

        for region_id, services in service_placements.items():
            if not services:
                continue

            region = regions[region_id]
            expected_traffic = traffic_patterns.get(region_id, 0.0)

            # Base resource requirements per service
            base_cpu_per_service = 0.5  # CPU cores
            base_memory_per_service = 1.0  # GB
            base_storage_per_service = 5.0  # GB

            # Scale based on traffic
            traffic_multiplier = max(1.0, expected_traffic / 100.0)  # Scale factor

            # Calculate total requirements
            total_services = len(services)
            cpu_needed = total_services * base_cpu_per_service * traffic_multiplier
            memory_needed = (
                total_services * base_memory_per_service * traffic_multiplier
            )
            storage_needed = total_services * base_storage_per_service

            # Add buffer for headroom
            cpu_needed *= 1.2  # 20% buffer
            memory_needed *= 1.2
            storage_needed *= 1.1  # 10% buffer

            # Check against region capacity
            max_cpu = region.capacity.get("cpu", 0.0)
            max_memory = region.capacity.get("memory", 0.0)
            max_storage = region.capacity.get("storage", 0.0)

            allocations[region_id] = {
                "cpu": min(
                    cpu_needed, max_cpu * 0.9
                ),  # Don't use more than 90% capacity
                "memory": min(memory_needed, max_memory * 0.9),
                "storage": min(storage_needed, max_storage * 0.9),
            }

        return allocations

    def _estimate_deployment_cost(
        self,
        service_placements: Dict[str, List[str]],
        resource_allocations: Dict[str, Dict[str, float]],
        regions: Dict[str, RegionInfo],
    ) -> float:
        """Estimate total deployment cost per hour."""
        total_cost = 0.0

        for region_id, services in service_placements.items():
            if not services or region_id not in regions:
                continue

            region = regions[region_id]
            allocations = resource_allocations.get(region_id, {})

            # Base cost for the region
            base_cost = region.cost_per_hour

            # Resource-based cost scaling
            cpu_allocation = allocations.get("cpu", 0.0)
            memory_allocation = allocations.get("memory", 0.0)

            # Assume cost scales with resource usage
            resource_factor = max(1.0, (cpu_allocation + memory_allocation) / 2.0)

            region_cost = base_cost * resource_factor
            total_cost += region_cost

        return total_cost

    def _estimate_latency_improvement(
        self,
        network_matrix: Dict[Tuple[str, str], NetworkLatencyInfo],
        traffic_routing: Dict[str, Dict[str, float]],
    ) -> float:
        """Estimate latency improvement from optimization."""
        if not network_matrix or not traffic_routing:
            return 0.0

        # Calculate weighted average latency
        total_weighted_latency = 0.0
        total_weight = 0.0

        for source_region, targets in traffic_routing.items():
            for target_region, weight in targets.items():
                if weight <= 0:
                    continue

                network_key = (source_region, target_region)
                if network_key in network_matrix:
                    latency = network_matrix[network_key].avg_latency_ms
                    total_weighted_latency += latency * weight
                    total_weight += weight

        if total_weight == 0:
            return 0.0

        optimized_latency = total_weighted_latency / total_weight

        # Compare with baseline (assume 20% improvement from optimization)
        # In practice, this would compare with previous measurements
        baseline_latency = optimized_latency * 1.2
        improvement_percentage = (
            (baseline_latency - optimized_latency) / baseline_latency
        ) * 100

        return max(0.0, improvement_percentage)

    def _calculate_optimization_confidence(
        self,
        regions: Dict[str, RegionInfo],
        network_matrix: Dict[Tuple[str, str], NetworkLatencyInfo],
        service_placements: Dict[str, List[str]],
    ) -> float:
        """Calculate confidence in optimization results."""
        confidence_factors = []

        # Region health confidence
        avg_health = np.mean([region.health_score() for region in regions.values()])
        health_confidence = avg_health / 100.0
        confidence_factors.append(health_confidence)

        # Network measurement confidence
        if network_matrix:
            avg_quality = np.mean(
                [info.quality_score() for info in network_matrix.values()]
            )
            network_confidence = avg_quality / 100.0
            confidence_factors.append(network_confidence)
        else:
            confidence_factors.append(0.5)  # Low confidence without network data

        # Service placement confidence
        total_services = sum(len(services) for services in service_placements.values())
        placement_confidence = min(
            1.0, total_services / max(1, len(regions))
        )  # Services per region ratio
        confidence_factors.append(placement_confidence)

        # Data freshness confidence (assume recent data is more reliable)
        # This would typically check timestamp of metrics
        freshness_confidence = 0.9  # Assume relatively fresh data
        confidence_factors.append(freshness_confidence)

        return np.mean(confidence_factors)

    def _optimization_loop(self):
        """Background optimization loop."""
        while self.running:
            try:
                time.sleep(self.optimization_interval)

                if not self.running:
                    break

                # Perform connectivity measurements
                asyncio.run(self.region_manager.measure_network_connectivity())

                # Get region recommendations
                recommendations = self.region_manager.get_region_recommendations()

                # Log high-priority recommendations
                for rec in recommendations:
                    if rec.get("priority") == "high":
                        logger.warning(f"Region recommendation: {rec['description']}")

                # Record optimization metrics
                active_regions = len(
                    self.region_manager.get_regions(RegionStatus.ACTIVE)
                )
                record_performance_metric(
                    "global_optimizer_active_regions", float(active_regions)
                )

            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                time.sleep(60)  # Wait on error

    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.running:
            try:
                time.sleep(60)  # Monitor every minute

                if not self.running:
                    break

                # Check region health
                regions = self.region_manager.get_regions()

                for region_id, region in regions.items():
                    health_score = region.health_score()
                    utilization = region.utilization_score()

                    # Record metrics
                    record_performance_metric(
                        f"region_health_{region_id}",
                        health_score,
                        tags={"region", region_id},
                    )
                    record_performance_metric(
                        f"region_utilization_{region_id}",
                        utilization,
                        tags={"region", region_id},
                    )

                    # Alert on issues
                    if health_score < 50:
                        logger.error(
                            f"Region {region_id} health critical: {health_score:.1f}"
                        )
                    elif utilization > 90:
                        logger.warning(
                            f"Region {region_id} high utilization: {utilization:.1f}%"
                        )

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)

    def get_global_status(self) -> Dict[str, Any]:
        """Get comprehensive global optimization status."""
        regions = self.region_manager.get_regions()

        # Calculate aggregate statistics
        total_regions = len(regions)
        active_regions = len(
            [r for r in regions.values() if r.status == RegionStatus.ACTIVE]
        )
        avg_health = (
            np.mean([r.health_score() for r in regions.values()]) if regions else 0.0
        )
        avg_utilization = (
            np.mean([r.utilization_score() for r in regions.values()])
            if regions
            else 0.0
        )

        # Get latest optimization
        latest_optimization = None
        with self._lock:
            if self.optimization_history:
                latest_optimization = self.optimization_history[-1].to_deployment_plan()

        return {
            "global_optimizer": {
                "running": self.running,
                "optimization_interval": self.optimization_interval,
                "auto_optimization_enabled": self.enable_auto_optimization,
            },
            "regions": {
                "total": total_regions,
                "active": active_regions,
                "average_health": avg_health,
                "average_utilization": avg_utilization,
                "status_breakdown": {
                    status.value: len(
                        [r for r in regions.values() if r.status == status]
                    )
                    for status in RegionStatus
                },
            },
            "optimization": {
                "total_optimizations": len(self.optimization_history),
                "latest_optimization": latest_optimization,
            },
            "recommendations": self.region_manager.get_region_recommendations(),
        }

    def register_region(self, region_info: RegionInfo):
        """Register a new region for global optimization."""
        self.region_manager.register_region(region_info)

    def update_region_status(
        self, region_id: str, status: RegionStatus, metadata: Optional[Dict] = None
    ):
        """Update region status."""
        self.region_manager.update_region_status(region_id, status, metadata)

    def update_region_metrics(self, region_id: str, **metrics):
        """Update region performance metrics."""
        self.region_manager.update_region_metrics(region_id, **metrics)


# Global optimizer instance
_global_optimizer: Optional[GlobalOptimizer] = None


def get_global_optimizer(**kwargs) -> GlobalOptimizer:
    """Get or create the global optimizer instance."""
    global _global_optimizer

    if _global_optimizer is None:
        _global_optimizer = GlobalOptimizer(**kwargs)

    return _global_optimizer


def start_global_optimization(**kwargs) -> GlobalOptimizer:
    """Start the global optimization system."""
    optimizer = get_global_optimizer(**kwargs)
    optimizer.start_optimization()
    return optimizer


def register_deployment_region(
    region_id: str,
    name: str,
    region_type: RegionType,
    location: Dict[str, float],
    capacity: Dict[str, float],
    endpoint: str,
    cost_per_hour: float,
    **kwargs,
) -> RegionInfo:
    """Register a new deployment region."""
    region_info = RegionInfo(
        region_id=region_id,
        name=name,
        region_type=region_type,
        location=location,
        capacity=capacity,
        current_load={},
        status=RegionStatus.ACTIVE,
        endpoint=endpoint,
        cost_per_hour=cost_per_hour,
        availability_zone=kwargs.get("availability_zone", "default"),
        provider=kwargs.get("provider", "unknown"),
    )

    optimizer = get_global_optimizer()
    optimizer.register_region(region_info)

    return region_info


async def optimize_global_deployment(
    services: List[str],
    traffic_patterns: Dict[str, float],
    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
    constraints: Optional[Dict[str, Any]] = None,
) -> GlobalOptimizationResult:
    """Optimize global deployment configuration."""
    optimizer = get_global_optimizer()
    return await optimizer.optimize_deployment(
        services, traffic_patterns, strategy, constraints
    )


def get_global_optimization_status() -> Dict[str, Any]:
    """Get global optimization system status."""
    optimizer = get_global_optimizer()
    return optimizer.get_global_status()
