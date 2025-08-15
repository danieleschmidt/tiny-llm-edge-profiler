"""
Multi-Region Orchestrator for Self-Healing Pipeline Guard
Global deployment coordination, edge computing, and regional optimization
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
import json
import hashlib
import random

logger = logging.getLogger(__name__)


class GlobalRegion(Enum):
    NORTH_AMERICA_EAST = "na-east"
    NORTH_AMERICA_WEST = "na-west"
    EUROPE_WEST = "eu-west"
    EUROPE_CENTRAL = "eu-central"
    ASIA_PACIFIC_SOUTHEAST = "ap-southeast"
    ASIA_PACIFIC_NORTHEAST = "ap-northeast"
    MIDDLE_EAST = "me-south"
    AFRICA = "af-south"
    SOUTH_AMERICA = "sa-east"
    OCEANIA = "oc-southeast"


class DeploymentStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPLOYING = "deploying"
    FAILED = "failed"
    MAINTENANCE = "maintenance"
    SCALING = "scaling"


class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    LATENCY_BASED = "latency_based"
    GEOLOCATION = "geolocation"
    LEAST_CONNECTIONS = "least_connections"
    ADAPTIVE = "adaptive"


@dataclass
class EdgeNode:
    node_id: str
    region: GlobalRegion
    zone: str
    capacity: int
    current_load: int
    status: DeploymentStatus
    endpoint: str
    capabilities: List[str]
    last_health_check: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentConfig:
    service_name: str
    regions: List[GlobalRegion]
    min_replicas_per_region: int
    max_replicas_per_region: int
    auto_scaling_enabled: bool
    load_balancing_strategy: LoadBalancingStrategy
    health_check_interval: int = 30
    failover_enabled: bool = True
    compliance_requirements: List[str] = field(default_factory=list)


@dataclass
class RoutingRule:
    rule_id: str
    priority: int
    condition: str  # JSON-encoded condition
    target_regions: List[GlobalRegion]
    weight_distribution: Dict[GlobalRegion, float]
    enabled: bool = True


@dataclass
class GlobalMetrics:
    timestamp: datetime
    total_requests: int
    requests_by_region: Dict[GlobalRegion, int]
    avg_latency_ms: float
    latency_by_region: Dict[GlobalRegion, float]
    error_rate_percent: float
    errors_by_region: Dict[GlobalRegion, int]
    active_nodes: int
    nodes_by_region: Dict[GlobalRegion, int]


class RegionManager(ABC):
    @abstractmethod
    async def deploy_service(self, config: DeploymentConfig) -> bool:
        pass
    
    @abstractmethod
    async def scale_service(self, service_name: str, replicas: int) -> bool:
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        pass


class CloudRegionManager(RegionManager):
    def __init__(self, region: GlobalRegion):
        self.region = region
        self.deployed_services: Dict[str, DeploymentConfig] = {}
        self.edge_nodes: List[EdgeNode] = []
        self.resource_utilization = 0.0
        
    async def deploy_service(self, config: DeploymentConfig) -> bool:
        try:
            logger.info(f"Deploying {config.service_name} to {self.region.value}")
            
            # Simulate deployment process
            await asyncio.sleep(0.1)  # Simulate deployment time
            
            # Create edge nodes for the service
            for i in range(config.min_replicas_per_region):
                node = EdgeNode(
                    node_id=f"{config.service_name}-{self.region.value}-{i}",
                    region=self.region,
                    zone=f"zone-{i % 3 + 1}",  # Distribute across 3 zones
                    capacity=100,
                    current_load=0,
                    status=DeploymentStatus.ACTIVE,
                    endpoint=f"https://{config.service_name}-{i}.{self.region.value}.example.com",
                    capabilities=["self-healing", "monitoring", "scaling"]
                )
                self.edge_nodes.append(node)
            
            self.deployed_services[config.service_name] = config
            logger.info(f"Successfully deployed {config.service_name} to {self.region.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy {config.service_name} to {self.region.value}: {str(e)}")
            return False
    
    async def scale_service(self, service_name: str, replicas: int) -> bool:
        try:
            if service_name not in self.deployed_services:
                return False
            
            current_nodes = [n for n in self.edge_nodes if service_name in n.node_id]
            current_count = len(current_nodes)
            
            if replicas > current_count:
                # Scale up
                for i in range(current_count, replicas):
                    node = EdgeNode(
                        node_id=f"{service_name}-{self.region.value}-{i}",
                        region=self.region,
                        zone=f"zone-{i % 3 + 1}",
                        capacity=100,
                        current_load=0,
                        status=DeploymentStatus.ACTIVE,
                        endpoint=f"https://{service_name}-{i}.{self.region.value}.example.com",
                        capabilities=["self-healing", "monitoring", "scaling"]
                    )
                    self.edge_nodes.append(node)
                    
            elif replicas < current_count:
                # Scale down
                nodes_to_remove = current_nodes[replicas:]
                for node in nodes_to_remove:
                    self.edge_nodes.remove(node)
            
            logger.info(f"Scaled {service_name} in {self.region.value} to {replicas} replicas")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale {service_name} in {self.region.value}: {str(e)}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        try:
            # Simulate health check
            await asyncio.sleep(0.05)
            
            healthy_nodes = len([n for n in self.edge_nodes if n.status == DeploymentStatus.ACTIVE])
            total_nodes = len(self.edge_nodes)
            
            # Simulate some variation in resource utilization
            self.resource_utilization = min(95.0, max(10.0, 
                self.resource_utilization + random.uniform(-5, 5)
            ))
            
            return {
                "region": self.region.value,
                "healthy": True,
                "total_nodes": total_nodes,
                "healthy_nodes": healthy_nodes,
                "resource_utilization": self.resource_utilization,
                "avg_latency_ms": random.uniform(10, 50),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed for {self.region.value}: {str(e)}")
            return {
                "region": self.region.value,
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


class GlobalLoadBalancer:
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE):
        self.strategy = strategy
        self.routing_rules: List[RoutingRule] = []
        self.region_weights: Dict[GlobalRegion, float] = {}
        self.latency_cache: Dict[GlobalRegion, float] = {}
        self.connection_counts: Dict[GlobalRegion, int] = {}
        
    def add_routing_rule(self, rule: RoutingRule) -> None:
        self.routing_rules.append(rule)
        self.routing_rules.sort(key=lambda r: r.priority)
        logger.info(f"Added routing rule: {rule.rule_id}")
    
    def set_region_weights(self, weights: Dict[GlobalRegion, float]) -> None:
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.region_weights = {region: weight / total_weight for region, weight in weights.items()}
        else:
            self.region_weights = weights
    
    def update_latency(self, region: GlobalRegion, latency_ms: float) -> None:
        self.latency_cache[region] = latency_ms
    
    def select_region(self, available_regions: List[GlobalRegion], 
                     client_context: Optional[Dict[str, Any]] = None) -> GlobalRegion:
        """Select optimal region based on load balancing strategy"""
        
        if not available_regions:
            raise ValueError("No available regions")
        
        if len(available_regions) == 1:
            return available_regions[0]
        
        client_context = client_context or {}
        
        # Check routing rules first
        for rule in self.routing_rules:
            if not rule.enabled:
                continue
                
            if self._evaluate_routing_condition(rule.condition, client_context):
                candidate_regions = [r for r in rule.target_regions if r in available_regions]
                if candidate_regions:
                    return self._select_by_weight(candidate_regions, rule.weight_distribution)
        
        # Apply load balancing strategy
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(available_regions)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED:
            return self._weighted_selection(available_regions)
        elif self.strategy == LoadBalancingStrategy.LATENCY_BASED:
            return self._latency_based_selection(available_regions)
        elif self.strategy == LoadBalancingStrategy.GEOLOCATION:
            return self._geolocation_selection(available_regions, client_context)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(available_regions)
        elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
            return self._adaptive_selection(available_regions, client_context)
        else:
            return available_regions[0]  # Fallback
    
    def _evaluate_routing_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate routing rule condition"""
        try:
            condition_obj = json.loads(condition)
            
            # Simple condition evaluation
            if "country" in condition_obj and "country" in context:
                return context["country"] in condition_obj["country"]
            
            if "user_type" in condition_obj and "user_type" in context:
                return context["user_type"] == condition_obj["user_type"]
            
            if "time_of_day" in condition_obj:
                current_hour = datetime.now().hour
                return condition_obj["time_of_day"]["start"] <= current_hour <= condition_obj["time_of_day"]["end"]
            
            return True  # Default to true if no conditions match
            
        except Exception:
            return False
    
    def _select_by_weight(self, regions: List[GlobalRegion], 
                         weights: Dict[GlobalRegion, float]) -> GlobalRegion:
        """Select region based on weights"""
        weighted_regions = [(region, weights.get(region, 1.0)) for region in regions]
        total_weight = sum(weight for _, weight in weighted_regions)
        
        if total_weight == 0:
            return regions[0]
        
        rand_val = random.uniform(0, total_weight)
        current_weight = 0
        
        for region, weight in weighted_regions:
            current_weight += weight
            if rand_val <= current_weight:
                return region
        
        return regions[-1]  # Fallback
    
    def _round_robin_selection(self, regions: List[GlobalRegion]) -> GlobalRegion:
        """Round-robin selection"""
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0
        
        selected_region = regions[self._round_robin_index % len(regions)]
        self._round_robin_index += 1
        return selected_region
    
    def _weighted_selection(self, regions: List[GlobalRegion]) -> GlobalRegion:
        """Weighted selection based on region weights"""
        return self._select_by_weight(regions, self.region_weights)
    
    def _latency_based_selection(self, regions: List[GlobalRegion]) -> GlobalRegion:
        """Select region with lowest latency"""
        region_latencies = [(region, self.latency_cache.get(region, 100.0)) for region in regions]
        return min(region_latencies, key=lambda x: x[1])[0]
    
    def _geolocation_selection(self, regions: List[GlobalRegion], 
                             context: Dict[str, Any]) -> GlobalRegion:
        """Select region based on geographic proximity"""
        client_country = context.get("country", "US")
        
        # Simple geographic mapping
        region_mapping = {
            "US": [GlobalRegion.NORTH_AMERICA_EAST, GlobalRegion.NORTH_AMERICA_WEST],
            "CA": [GlobalRegion.NORTH_AMERICA_EAST, GlobalRegion.NORTH_AMERICA_WEST],
            "GB": [GlobalRegion.EUROPE_WEST, GlobalRegion.EUROPE_CENTRAL],
            "DE": [GlobalRegion.EUROPE_CENTRAL, GlobalRegion.EUROPE_WEST],
            "FR": [GlobalRegion.EUROPE_WEST, GlobalRegion.EUROPE_CENTRAL],
            "JP": [GlobalRegion.ASIA_PACIFIC_NORTHEAST, GlobalRegion.ASIA_PACIFIC_SOUTHEAST],
            "CN": [GlobalRegion.ASIA_PACIFIC_NORTHEAST, GlobalRegion.ASIA_PACIFIC_SOUTHEAST],
            "SG": [GlobalRegion.ASIA_PACIFIC_SOUTHEAST, GlobalRegion.ASIA_PACIFIC_NORTHEAST],
            "AU": [GlobalRegion.OCEANIA, GlobalRegion.ASIA_PACIFIC_SOUTHEAST]
        }
        
        preferred_regions = region_mapping.get(client_country, [GlobalRegion.NORTH_AMERICA_EAST])
        
        for preferred in preferred_regions:
            if preferred in regions:
                return preferred
        
        return regions[0]  # Fallback
    
    def _least_connections_selection(self, regions: List[GlobalRegion]) -> GlobalRegion:
        """Select region with least connections"""
        region_connections = [(region, self.connection_counts.get(region, 0)) for region in regions]
        return min(region_connections, key=lambda x: x[1])[0]
    
    def _adaptive_selection(self, regions: List[GlobalRegion], 
                          context: Dict[str, Any]) -> GlobalRegion:
        """Adaptive selection combining multiple factors"""
        scores = {}
        
        for region in regions:
            score = 0.0
            
            # Latency factor (lower is better)
            latency = self.latency_cache.get(region, 100.0)
            latency_score = max(0, 100 - latency)  # Invert latency
            score += latency_score * 0.4
            
            # Connection count factor (lower is better)
            connections = self.connection_counts.get(region, 0)
            connection_score = max(0, 100 - connections)
            score += connection_score * 0.3
            
            # Weight factor
            weight = self.region_weights.get(region, 1.0)
            score += weight * 20  # Scale weight contribution
            
            # Geographic proximity factor
            if context.get("country"):
                geo_score = 30 if self._is_geographically_close(region, context["country"]) else 0
                score += geo_score * 0.3
            
            scores[region] = score
        
        # Select region with highest score
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _is_geographically_close(self, region: GlobalRegion, country: str) -> bool:
        """Check if region is geographically close to country"""
        proximity_map = {
            GlobalRegion.NORTH_AMERICA_EAST: ["US", "CA", "MX"],
            GlobalRegion.NORTH_AMERICA_WEST: ["US", "CA"],
            GlobalRegion.EUROPE_WEST: ["GB", "FR", "ES", "PT", "IE"],
            GlobalRegion.EUROPE_CENTRAL: ["DE", "AT", "CH", "NL", "BE"],
            GlobalRegion.ASIA_PACIFIC_NORTHEAST: ["JP", "CN", "KR"],
            GlobalRegion.ASIA_PACIFIC_SOUTHEAST: ["SG", "MY", "TH", "ID", "VN"],
            GlobalRegion.OCEANIA: ["AU", "NZ"]
        }
        
        return country in proximity_map.get(region, [])


class MultiRegionOrchestrator:
    def __init__(self):
        self.region_managers: Dict[GlobalRegion, RegionManager] = {}
        self.load_balancer = GlobalLoadBalancer()
        self.deployment_configs: Dict[str, DeploymentConfig] = {}
        
        # Monitoring and metrics
        self.metrics_history: List[GlobalMetrics] = []
        self.health_check_interval = 30
        self.auto_scaling_enabled = True
        
        # Circuit breaker for failed regions
        self.failed_regions: Set[GlobalRegion] = set()
        self.region_failure_counts: Dict[GlobalRegion, int] = {}
        
        # Initialize region managers
        self._initialize_region_managers()
    
    def _initialize_region_managers(self) -> None:
        """Initialize region managers for all global regions"""
        for region in GlobalRegion:
            self.region_managers[region] = CloudRegionManager(region)
            self.region_failure_counts[region] = 0
        
        logger.info(f"Initialized {len(self.region_managers)} region managers")
    
    async def deploy_globally(self, config: DeploymentConfig) -> Dict[GlobalRegion, bool]:
        """Deploy service across specified regions"""
        results = {}
        
        deployment_tasks = []
        for region in config.regions:
            if region in self.region_managers and region not in self.failed_regions:
                task = asyncio.create_task(
                    self.region_managers[region].deploy_service(config)
                )
                deployment_tasks.append((region, task))
        
        # Wait for all deployments
        for region, task in deployment_tasks:
            try:
                success = await task
                results[region] = success
                
                if success:
                    logger.info(f"Deployment successful in {region.value}")
                else:
                    logger.error(f"Deployment failed in {region.value}")
                    self._handle_region_failure(region)
                    
            except Exception as e:
                logger.error(f"Deployment error in {region.value}: {str(e)}")
                results[region] = False
                self._handle_region_failure(region)
        
        # Store deployment config
        self.deployment_configs[config.service_name] = config
        
        # Setup load balancing weights
        successful_regions = [region for region, success in results.items() if success]
        if successful_regions:
            equal_weight = 1.0 / len(successful_regions)
            weights = {region: equal_weight for region in successful_regions}
            self.load_balancer.set_region_weights(weights)
        
        return results
    
    async def route_request(self, service_name: str, 
                          client_context: Optional[Dict[str, Any]] = None) -> Tuple[GlobalRegion, str]:
        """Route request to optimal region"""
        
        if service_name not in self.deployment_configs:
            raise ValueError(f"Service {service_name} not deployed")
        
        config = self.deployment_configs[service_name]
        available_regions = [
            region for region in config.regions 
            if region not in self.failed_regions
        ]
        
        if not available_regions:
            raise RuntimeError("No healthy regions available")
        
        # Select optimal region
        selected_region = self.load_balancer.select_region(available_regions, client_context)
        
        # Get endpoint from region manager
        region_manager = self.region_managers[selected_region]
        if hasattr(region_manager, 'edge_nodes') and region_manager.edge_nodes:
            # Simple round-robin within region
            healthy_nodes = [
                node for node in region_manager.edge_nodes 
                if node.status == DeploymentStatus.ACTIVE and service_name in node.node_id
            ]
            
            if healthy_nodes:
                selected_node = healthy_nodes[0]  # Could implement more sophisticated selection
                endpoint = selected_node.endpoint
                
                # Update connection count
                if selected_region not in self.load_balancer.connection_counts:
                    self.load_balancer.connection_counts[selected_region] = 0
                self.load_balancer.connection_counts[selected_region] += 1
                
                return selected_region, endpoint
        
        # Fallback
        return selected_region, f"https://{service_name}.{selected_region.value}.example.com"
    
    async def auto_scale_service(self, service_name: str) -> Dict[GlobalRegion, bool]:
        """Auto-scale service based on current metrics"""
        
        if not self.auto_scaling_enabled or service_name not in self.deployment_configs:
            return {}
        
        config = self.deployment_configs[service_name]
        if not config.auto_scaling_enabled:
            return {}
        
        results = {}
        scaling_tasks = []
        
        for region in config.regions:
            if region in self.failed_regions:
                continue
            
            region_manager = self.region_managers[region]
            
            # Get current load and determine scaling decision
            if hasattr(region_manager, 'resource_utilization'):
                utilization = region_manager.resource_utilization
                current_replicas = len([
                    node for node in getattr(region_manager, 'edge_nodes', [])
                    if service_name in node.node_id and node.status == DeploymentStatus.ACTIVE
                ])
                
                target_replicas = current_replicas
                
                # Scale up if utilization is high
                if utilization > 80 and current_replicas < config.max_replicas_per_region:
                    target_replicas = min(current_replicas + 1, config.max_replicas_per_region)
                
                # Scale down if utilization is low
                elif utilization < 30 and current_replicas > config.min_replicas_per_region:
                    target_replicas = max(current_replicas - 1, config.min_replicas_per_region)
                
                if target_replicas != current_replicas:
                    task = asyncio.create_task(
                        region_manager.scale_service(service_name, target_replicas)
                    )
                    scaling_tasks.append((region, task, current_replicas, target_replicas))
        
        # Execute scaling operations
        for region, task, old_count, new_count in scaling_tasks:
            try:
                success = await task
                results[region] = success
                
                if success:
                    logger.info(f"Scaled {service_name} in {region.value} from {old_count} to {new_count}")
                else:
                    logger.error(f"Failed to scale {service_name} in {region.value}")
            
            except Exception as e:
                logger.error(f"Scaling error in {region.value}: {str(e)}")
                results[region] = False
        
        return results
    
    async def health_check_all_regions(self) -> Dict[GlobalRegion, Dict[str, Any]]:
        """Perform health checks across all regions"""
        
        health_tasks = []
        for region, manager in self.region_managers.items():
            task = asyncio.create_task(manager.health_check())
            health_tasks.append((region, task))
        
        results = {}
        for region, task in health_tasks:
            try:
                health_data = await task
                results[region] = health_data
                
                # Update load balancer with latency data
                if "avg_latency_ms" in health_data:
                    self.load_balancer.update_latency(region, health_data["avg_latency_ms"])
                
                # Handle failed regions
                if not health_data.get("healthy", False):
                    self._handle_region_failure(region)
                else:
                    self._handle_region_recovery(region)
                    
            except Exception as e:
                logger.error(f"Health check failed for {region.value}: {str(e)}")
                results[region] = {"healthy": False, "error": str(e)}
                self._handle_region_failure(region)
        
        return results
    
    def _handle_region_failure(self, region: GlobalRegion) -> None:
        """Handle region failure"""
        self.region_failure_counts[region] += 1
        
        if self.region_failure_counts[region] >= 3:  # Failure threshold
            if region not in self.failed_regions:
                self.failed_regions.add(region)
                logger.warning(f"Region {region.value} marked as failed after {self.region_failure_counts[region]} failures")
                
                # Redistribute load balancing weights
                self._rebalance_after_failure()
    
    def _handle_region_recovery(self, region: GlobalRegion) -> None:
        """Handle region recovery"""
        if region in self.failed_regions:
            self.failed_regions.remove(region)
            self.region_failure_counts[region] = 0
            logger.info(f"Region {region.value} recovered and marked as healthy")
            
            # Redistribute load balancing weights
            self._rebalance_after_recovery()
    
    def _rebalance_after_failure(self) -> None:
        """Rebalance load after region failure"""
        healthy_regions = [
            region for region in self.region_managers.keys()
            if region not in self.failed_regions
        ]
        
        if healthy_regions:
            equal_weight = 1.0 / len(healthy_regions)
            weights = {region: equal_weight for region in healthy_regions}
            self.load_balancer.set_region_weights(weights)
    
    def _rebalance_after_recovery(self) -> None:
        """Rebalance load after region recovery"""
        self._rebalance_after_failure()  # Same logic for now
    
    async def collect_global_metrics(self) -> GlobalMetrics:
        """Collect metrics from all regions"""
        
        health_data = await self.health_check_all_regions()
        
        total_requests = 0
        requests_by_region = {}
        total_latency = 0
        latency_by_region = {}
        total_errors = 0
        errors_by_region = {}
        active_nodes = 0
        nodes_by_region = {}
        
        healthy_regions = 0
        
        for region, data in health_data.items():
            if data.get("healthy", False):
                healthy_regions += 1
                
                # Simulate request metrics
                region_requests = random.randint(100, 1000)
                total_requests += region_requests
                requests_by_region[region] = region_requests
                
                # Latency metrics
                region_latency = data.get("avg_latency_ms", 50.0)
                total_latency += region_latency
                latency_by_region[region] = region_latency
                
                # Error metrics (simulate)
                region_errors = random.randint(0, 10)
                total_errors += region_errors
                errors_by_region[region] = region_errors
                
                # Node metrics
                region_nodes = data.get("total_nodes", 0)
                active_nodes += region_nodes
                nodes_by_region[region] = region_nodes
        
        avg_latency = total_latency / max(1, healthy_regions)
        error_rate = (total_errors / max(1, total_requests)) * 100
        
        metrics = GlobalMetrics(
            timestamp=datetime.now(),
            total_requests=total_requests,
            requests_by_region=requests_by_region,
            avg_latency_ms=avg_latency,
            latency_by_region=latency_by_region,
            error_rate_percent=error_rate,
            errors_by_region=errors_by_region,
            active_nodes=active_nodes,
            nodes_by_region=nodes_by_region
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        # Maintain metrics history size
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-500:]
        
        return metrics
    
    def get_global_status(self) -> Dict[str, Any]:
        """Get comprehensive global status"""
        
        healthy_regions = [
            region for region in self.region_managers.keys()
            if region not in self.failed_regions
        ]
        
        recent_metrics = self.metrics_history[-10:] if self.metrics_history else []
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_regions": len(self.region_managers),
            "healthy_regions": len(healthy_regions),
            "failed_regions": len(self.failed_regions),
            "deployed_services": len(self.deployment_configs),
            "auto_scaling_enabled": self.auto_scaling_enabled,
            "load_balancing_strategy": self.load_balancer.strategy.value,
            "healthy_region_list": [r.value for r in healthy_regions],
            "failed_region_list": [r.value for r in self.failed_regions],
            "recent_metrics": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "total_requests": m.total_requests,
                    "avg_latency_ms": m.avg_latency_ms,
                    "error_rate_percent": m.error_rate_percent,
                    "active_nodes": m.active_nodes
                }
                for m in recent_metrics
            ]
        }
    
    async def start_monitoring(self) -> None:
        """Start continuous monitoring and auto-scaling"""
        logger.info("Starting multi-region monitoring and auto-scaling")
        
        while True:
            try:
                # Collect metrics
                await self.collect_global_metrics()
                
                # Auto-scale services
                for service_name in self.deployment_configs.keys():
                    await self.auto_scale_service(service_name)
                
                # Wait for next cycle
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                logger.info("Monitoring cancelled")
                break
            except Exception as e:
                logger.error(f"Error in monitoring cycle: {str(e)}")
                await asyncio.sleep(5)  # Short sleep on error


# Global multi-region orchestrator instance
_global_orchestrator: Optional[MultiRegionOrchestrator] = None


def get_multi_region_orchestrator() -> MultiRegionOrchestrator:
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = MultiRegionOrchestrator()
    return _global_orchestrator


async def deploy_service_globally(service_name: str, regions: List[str],
                                 min_replicas: int = 1, max_replicas: int = 5) -> Dict[str, bool]:
    """Deploy service globally across specified regions"""
    orchestrator = get_multi_region_orchestrator()
    
    region_enums = [GlobalRegion(region) for region in regions]
    
    config = DeploymentConfig(
        service_name=service_name,
        regions=region_enums,
        min_replicas_per_region=min_replicas,
        max_replicas_per_region=max_replicas,
        auto_scaling_enabled=True,
        load_balancing_strategy=LoadBalancingStrategy.ADAPTIVE
    )
    
    results = await orchestrator.deploy_globally(config)
    
    return {region.value: success for region, success in results.items()}


async def route_request_globally(service_name: str, 
                               client_country: Optional[str] = None) -> Dict[str, str]:
    """Route request to optimal global region"""
    orchestrator = get_multi_region_orchestrator()
    
    context = {}
    if client_country:
        context["country"] = client_country
    
    try:
        region, endpoint = await orchestrator.route_request(service_name, context)
        return {
            "selected_region": region.value,
            "endpoint": endpoint,
            "routing_successful": True
        }
    except Exception as e:
        return {
            "selected_region": None,
            "endpoint": None,
            "routing_successful": False,
            "error": str(e)
        }


def get_global_deployment_status() -> Dict[str, Any]:
    """Get global deployment status"""
    orchestrator = get_multi_region_orchestrator()
    return orchestrator.get_global_status()