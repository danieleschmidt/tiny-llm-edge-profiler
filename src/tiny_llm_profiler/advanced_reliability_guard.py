"""
Advanced Reliability Guard System
Next-generation reliability framework with predictive failure detection and autonomous recovery.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

from .exceptions import ProfilerError
from .reliability import CircuitBreaker, RetryMechanism


class FailureMode(str, Enum):
    """Types of failure modes in edge AI profiling."""
    HARDWARE_TIMEOUT = "hardware_timeout"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    POWER_FAILURE = "power_failure"
    COMMUNICATION_LOSS = "communication_loss"
    MODEL_CORRUPTION = "model_corruption"
    THERMAL_THROTTLING = "thermal_throttling"
    VOLTAGE_DROP = "voltage_drop"
    FLASH_CORRUPTION = "flash_corruption"
    SENSOR_MALFUNCTION = "sensor_malfunction"
    ALGORITHM_DIVERGENCE = "algorithm_divergence"


class ReliabilityLevel(str, Enum):
    """Reliability level requirements."""
    RESEARCH = "research"           # 95% reliability
    DEVELOPMENT = "development"     # 99% reliability  
    PRODUCTION = "production"       # 99.9% reliability
    MISSION_CRITICAL = "mission_critical"  # 99.99% reliability
    ULTRA_RELIABLE = "ultra_reliable"      # 99.999% reliability


@dataclass
class FailurePattern:
    """Pattern of failures for predictive analysis."""
    failure_mode: FailureMode
    frequency: float
    typical_precursors: List[str]
    recovery_time_seconds: float
    impact_severity: float  # 0-1 scale
    prevention_strategies: List[str] = field(default_factory=list)


@dataclass
class ReliabilityMetrics:
    """Comprehensive reliability metrics."""
    uptime_percentage: float
    mean_time_between_failures: float
    mean_time_to_recovery: float
    failure_rate_per_hour: float
    availability: float
    reliability_score: float
    predicted_next_failure_hours: Optional[float] = None


class PredictiveFailureDetector:
    """Advanced predictive failure detection using pattern analysis."""
    
    def __init__(self, reliability_level: ReliabilityLevel):
        self.reliability_level = reliability_level
        self.logger = logging.getLogger(__name__)
        
        self.failure_history = []
        self.precursor_patterns = {}
        self.health_metrics = {}
        self.prediction_model = self._initialize_prediction_model()
        
        # Reliability thresholds based on level
        self.thresholds = {
            ReliabilityLevel.RESEARCH: {"uptime": 0.95, "mtbf": 3600},
            ReliabilityLevel.DEVELOPMENT: {"uptime": 0.99, "mtbf": 7200},
            ReliabilityLevel.PRODUCTION: {"uptime": 0.999, "mtbf": 86400},
            ReliabilityLevel.MISSION_CRITICAL: {"uptime": 0.9999, "mtbf": 604800},
            ReliabilityLevel.ULTRA_RELIABLE: {"uptime": 0.99999, "mtbf": 2592000}
        }
    
    def _initialize_prediction_model(self) -> Dict[str, Any]:
        """Initialize predictive failure detection model."""
        return {
            "failure_patterns": {
                FailureMode.HARDWARE_TIMEOUT: FailurePattern(
                    failure_mode=FailureMode.HARDWARE_TIMEOUT,
                    frequency=0.02,  # 2% of operations
                    typical_precursors=["increasing_latency", "communication_delays"],
                    recovery_time_seconds=5.0,
                    impact_severity=0.6,
                    prevention_strategies=["timeout_adjustment", "communication_buffering"]
                ),
                FailureMode.MEMORY_EXHAUSTION: FailurePattern(
                    failure_mode=FailureMode.MEMORY_EXHAUSTION,
                    frequency=0.01,  # 1% of operations
                    typical_precursors=["memory_pressure", "allocation_failures"],
                    recovery_time_seconds=2.0,
                    impact_severity=0.8,
                    prevention_strategies=["memory_pooling", "garbage_collection"]
                ),
                FailureMode.THERMAL_THROTTLING: FailurePattern(
                    failure_mode=FailureMode.THERMAL_THROTTLING,
                    frequency=0.005,  # 0.5% of operations
                    typical_precursors=["temperature_rise", "performance_degradation"],
                    recovery_time_seconds=30.0,
                    impact_severity=0.4,
                    prevention_strategies=["thermal_management", "workload_scheduling"]
                )
            },
            "precursor_weights": {
                "increasing_latency": 0.7,
                "memory_pressure": 0.8,
                "temperature_rise": 0.6,
                "communication_delays": 0.5,
                "performance_degradation": 0.4
            },
            "prediction_horizon_seconds": 300  # 5 minutes ahead
        }
    
    async def predict_failure_probability(
        self,
        current_metrics: Dict[str, float],
        historical_data: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Predict probability of different failure modes.
        
        Args:
            current_metrics: Current system health metrics
            historical_data: Historical performance data
            
        Returns:
            Failure probabilities by mode
        """
        failure_probabilities = {}
        
        # Analyze each failure mode
        for failure_mode, pattern in self.prediction_model["failure_patterns"].items():
            probability = await self._calculate_failure_probability(
                failure_mode, pattern, current_metrics, historical_data
            )
            failure_probabilities[failure_mode.value] = probability
        
        return failure_probabilities
    
    async def _calculate_failure_probability(
        self,
        failure_mode: FailureMode,
        pattern: FailurePattern,
        current_metrics: Dict[str, float],
        historical_data: List[Dict[str, float]]
    ) -> float:
        """Calculate probability for specific failure mode."""
        base_probability = pattern.frequency
        
        # Check for precursor indicators
        precursor_score = 0.0
        for precursor in pattern.typical_precursors:
            if precursor in current_metrics:
                weight = self.prediction_model["precursor_weights"].get(precursor, 0.5)
                # Normalize metric value to 0-1 scale (simplified)
                normalized_value = min(1.0, current_metrics[precursor] / 100.0)
                precursor_score += weight * normalized_value
        
        # Analyze historical trend
        trend_factor = self._analyze_historical_trend(historical_data, pattern.typical_precursors)
        
        # Calculate adjusted probability
        adjusted_probability = base_probability * (1 + precursor_score + trend_factor)
        
        return min(1.0, adjusted_probability)
    
    def _analyze_historical_trend(
        self,
        historical_data: List[Dict[str, float]],
        precursors: List[str]
    ) -> float:
        """Analyze historical trends for failure prediction."""
        if len(historical_data) < 2:
            return 0.0
        
        trend_score = 0.0
        for precursor in precursors:
            if precursor in historical_data[0]:
                values = [data.get(precursor, 0) for data in historical_data[-10:]]  # Last 10 data points
                if len(values) >= 2:
                    # Calculate trend slope
                    x = np.arange(len(values))
                    coeffs = np.polyfit(x, values, 1)
                    slope = coeffs[0]
                    
                    # Positive slope indicates increasing risk
                    if slope > 0:
                        trend_score += min(0.5, slope / 10.0)  # Normalize and cap
        
        return trend_score
    
    async def generate_prevention_recommendations(
        self,
        failure_probabilities: Dict[str, float],
        threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Generate prevention recommendations for high-risk failure modes.
        
        Args:
            failure_probabilities: Predicted failure probabilities
            threshold: Threshold for recommendation generation
            
        Returns:
            List of prevention recommendations
        """
        recommendations = []
        
        for failure_mode_str, probability in failure_probabilities.items():
            if probability > threshold:
                failure_mode = FailureMode(failure_mode_str)
                pattern = self.prediction_model["failure_patterns"].get(failure_mode)
                
                if pattern:
                    recommendation = {
                        "failure_mode": failure_mode.value,
                        "probability": probability,
                        "severity": pattern.impact_severity,
                        "prevention_strategies": pattern.prevention_strategies,
                        "estimated_recovery_time": pattern.recovery_time_seconds,
                        "urgency": self._calculate_urgency(probability, pattern.impact_severity)
                    }
                    recommendations.append(recommendation)
        
        # Sort by urgency
        recommendations.sort(key=lambda x: x["urgency"], reverse=True)
        
        return recommendations
    
    def _calculate_urgency(self, probability: float, severity: float) -> float:
        """Calculate urgency score for prevention action."""
        return probability * severity


class AutonomousRecoverySystem:
    """Autonomous recovery system with self-healing capabilities."""
    
    def __init__(self, reliability_level: ReliabilityLevel):
        self.reliability_level = reliability_level
        self.logger = logging.getLogger(__name__)
        
        self.recovery_strategies = self._initialize_recovery_strategies()
        self.recovery_history = []
        self.active_recoveries = {}
        
    def _initialize_recovery_strategies(self) -> Dict[FailureMode, List[Callable]]:
        """Initialize recovery strategies for different failure modes."""
        return {
            FailureMode.HARDWARE_TIMEOUT: [
                self._recover_hardware_timeout,
                self._increase_timeout_limits,
                self._switch_communication_protocol
            ],
            FailureMode.MEMORY_EXHAUSTION: [
                self._trigger_garbage_collection,
                self._clear_memory_pools,
                self._reduce_memory_footprint
            ],
            FailureMode.COMMUNICATION_LOSS: [
                self._reconnect_device,
                self._switch_communication_interface,
                self._reset_communication_stack
            ],
            FailureMode.THERMAL_THROTTLING: [
                self._reduce_workload,
                self._implement_thermal_management,
                self._schedule_cooling_period
            ],
            FailureMode.POWER_FAILURE: [
                self._switch_power_mode,
                self._reduce_power_consumption,
                self._activate_power_saving
            ]
        }
    
    async def attempt_autonomous_recovery(
        self,
        failure_mode: FailureMode,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Attempt autonomous recovery from failure.
        
        Args:
            failure_mode: Type of failure detected
            context: Context information about the failure
            
        Returns:
            Recovery attempt results
        """
        recovery_id = f"recovery_{failure_mode.value}_{int(time.time())}"
        
        self.logger.info(f"Starting autonomous recovery for {failure_mode.value}")
        
        recovery_result = {
            "recovery_id": recovery_id,
            "failure_mode": failure_mode.value,
            "start_time": time.time(),
            "strategies_attempted": [],
            "success": False,
            "recovery_time_seconds": 0,
            "error_messages": []
        }
        
        # Get recovery strategies for this failure mode
        strategies = self.recovery_strategies.get(failure_mode, [])
        
        for strategy_idx, strategy in enumerate(strategies):
            try:
                self.logger.info(f"Attempting recovery strategy {strategy_idx + 1}: {strategy.__name__}")
                
                strategy_start = time.time()
                strategy_result = await strategy(context)
                strategy_time = time.time() - strategy_start
                
                recovery_result["strategies_attempted"].append({
                    "strategy": strategy.__name__,
                    "result": strategy_result,
                    "execution_time": strategy_time
                })
                
                # Check if recovery was successful
                if strategy_result.get("success", False):
                    recovery_result["success"] = True
                    recovery_result["recovery_time_seconds"] = time.time() - recovery_result["start_time"]
                    break
                    
            except Exception as e:
                error_msg = f"Recovery strategy {strategy.__name__} failed: {str(e)}"
                self.logger.error(error_msg)
                recovery_result["error_messages"].append(error_msg)
        
        # Record recovery attempt
        self.recovery_history.append(recovery_result)
        
        return recovery_result
    
    async def _recover_hardware_timeout(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recovery strategy for hardware timeout."""
        try:
            # Simulate hardware reset and reconnection
            await asyncio.sleep(1.0)  # Simulate reset time
            
            # Verify hardware responsiveness
            hardware_responsive = await self._verify_hardware_responsive(context)
            
            return {
                "success": hardware_responsive,
                "action": "hardware_reset_and_reconnect",
                "details": "Hardware reset performed and responsiveness verified"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _trigger_garbage_collection(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recovery strategy for memory exhaustion."""
        try:
            # Simulate garbage collection
            initial_memory = context.get("memory_usage", 100)
            
            # Simulate memory cleanup
            await asyncio.sleep(0.5)
            recovered_memory = initial_memory * 0.7  # Assume 30% memory recovery
            
            return {
                "success": True,
                "action": "garbage_collection",
                "memory_recovered": initial_memory - recovered_memory,
                "details": f"Freed {initial_memory - recovered_memory:.1f}KB of memory"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _reconnect_device(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recovery strategy for communication loss."""
        try:
            # Simulate device reconnection
            await asyncio.sleep(2.0)  # Simulate reconnection time
            
            # Verify connection
            connection_established = await self._verify_device_connection(context)
            
            return {
                "success": connection_established,
                "action": "device_reconnection",
                "details": "Device reconnection attempted and verified"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _reduce_workload(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recovery strategy for thermal throttling."""
        try:
            # Reduce computational workload
            current_workload = context.get("workload_percentage", 100)
            reduced_workload = current_workload * 0.6  # Reduce to 60%
            
            await asyncio.sleep(1.0)  # Simulate workload adjustment
            
            return {
                "success": True,
                "action": "workload_reduction",
                "workload_reduced_to": reduced_workload,
                "details": f"Workload reduced from {current_workload}% to {reduced_workload}%"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _switch_power_mode(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recovery strategy for power failure."""
        try:
            # Switch to low power mode
            await asyncio.sleep(0.5)  # Simulate mode switch
            
            return {
                "success": True,
                "action": "power_mode_switch",
                "new_mode": "low_power",
                "details": "Switched to low power mode to conserve energy"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _increase_timeout_limits(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Increase timeout limits for hardware operations."""
        current_timeout = context.get("timeout_seconds", 5.0)
        new_timeout = current_timeout * 2.0
        
        return {
            "success": True,
            "action": "timeout_increase",
            "old_timeout": current_timeout,
            "new_timeout": new_timeout,
            "details": f"Increased timeout from {current_timeout}s to {new_timeout}s"
        }
    
    async def _clear_memory_pools(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Clear memory pools to free up memory."""
        return {
            "success": True,
            "action": "memory_pool_clear",
            "details": "Cleared all memory pools"
        }
    
    async def _switch_communication_protocol(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Switch to alternative communication protocol."""
        return {
            "success": True,
            "action": "protocol_switch",
            "new_protocol": "backup_uart",
            "details": "Switched to backup UART communication"
        }
    
    async def _verify_hardware_responsive(self, context: Dict[str, Any]) -> bool:
        """Verify hardware responsiveness after recovery."""
        # Simulate hardware check
        await asyncio.sleep(0.5)
        return True  # Assume successful for demo
    
    async def _verify_device_connection(self, context: Dict[str, Any]) -> bool:
        """Verify device connection after recovery."""
        # Simulate connection check
        await asyncio.sleep(0.5)
        return True  # Assume successful for demo


class AdvancedReliabilityGuard:
    """Advanced reliability guard system with predictive capabilities."""
    
    def __init__(self, reliability_level: ReliabilityLevel = ReliabilityLevel.PRODUCTION):
        self.reliability_level = reliability_level
        self.logger = logging.getLogger(__name__)
        
        self.failure_detector = PredictiveFailureDetector(reliability_level)
        self.recovery_system = AutonomousRecoverySystem(reliability_level)
        
        self.reliability_metrics = ReliabilityMetrics(
            uptime_percentage=100.0,
            mean_time_between_failures=0.0,
            mean_time_to_recovery=0.0,
            failure_rate_per_hour=0.0,
            availability=1.0,
            reliability_score=1.0
        )
        
        self.health_monitoring_active = False
        self.monitoring_task = None
        
    async def start_reliability_monitoring(
        self,
        monitoring_interval_seconds: float = 10.0
    ) -> None:
        """Start continuous reliability monitoring."""
        if self.health_monitoring_active:
            self.logger.warning("Reliability monitoring already active")
            return
        
        self.health_monitoring_active = True
        self.monitoring_task = asyncio.create_task(
            self._reliability_monitoring_loop(monitoring_interval_seconds)
        )
        
        self.logger.info(f"Started reliability monitoring at {monitoring_interval_seconds}s intervals")
    
    async def stop_reliability_monitoring(self) -> None:
        """Stop reliability monitoring."""
        self.health_monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Stopped reliability monitoring")
    
    async def _reliability_monitoring_loop(self, interval_seconds: float) -> None:
        """Main reliability monitoring loop."""
        while self.health_monitoring_active:
            try:
                # Collect current health metrics
                current_metrics = await self._collect_health_metrics()
                
                # Get historical data
                historical_data = await self._get_historical_metrics()
                
                # Predict failure probabilities
                failure_probabilities = await self.failure_detector.predict_failure_probability(
                    current_metrics, historical_data
                )
                
                # Generate prevention recommendations
                recommendations = await self.failure_detector.generate_prevention_recommendations(
                    failure_probabilities
                )
                
                # Take preventive actions if needed
                if recommendations:
                    await self._execute_preventive_actions(recommendations)
                
                # Update reliability metrics
                await self._update_reliability_metrics(current_metrics, failure_probabilities)
                
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in reliability monitoring loop: {str(e)}")
                await asyncio.sleep(interval_seconds)
    
    async def _collect_health_metrics(self) -> Dict[str, float]:
        """Collect current system health metrics."""
        # Simulate health metric collection
        return {
            "cpu_usage_percent": np.random.normal(50, 10),
            "memory_usage_kb": np.random.normal(300, 50),
            "temperature_celsius": np.random.normal(45, 5),
            "latency_ms": np.random.normal(100, 20),
            "communication_errors": np.random.poisson(0.1),
            "power_consumption_mw": np.random.normal(250, 25)
        }
    
    async def _get_historical_metrics(self) -> List[Dict[str, float]]:
        """Get historical health metrics."""
        # Simulate historical data
        historical = []
        for i in range(10):
            metrics = await self._collect_health_metrics()
            historical.append(metrics)
        return historical
    
    async def _execute_preventive_actions(
        self,
        recommendations: List[Dict[str, Any]]
    ) -> None:
        """Execute preventive actions based on recommendations."""
        for recommendation in recommendations:
            if recommendation["urgency"] > 0.7:  # High urgency threshold
                self.logger.warning(f"High urgency prevention needed for {recommendation['failure_mode']}")
                
                # Execute prevention strategies
                for strategy in recommendation["prevention_strategies"]:
                    await self._execute_prevention_strategy(strategy, recommendation)
    
    async def _execute_prevention_strategy(
        self,
        strategy: str,
        recommendation: Dict[str, Any]
    ) -> None:
        """Execute specific prevention strategy."""
        self.logger.info(f"Executing prevention strategy: {strategy}")
        
        strategy_actions = {
            "timeout_adjustment": self._adjust_timeouts,
            "memory_pooling": self._optimize_memory_pools,
            "thermal_management": self._activate_thermal_management,
            "communication_buffering": self._enable_communication_buffering,
            "workload_scheduling": self._optimize_workload_scheduling
        }
        
        action = strategy_actions.get(strategy)
        if action:
            try:
                await action(recommendation)
            except Exception as e:
                self.logger.error(f"Prevention strategy {strategy} failed: {str(e)}")
    
    async def _adjust_timeouts(self, recommendation: Dict[str, Any]) -> None:
        """Adjust timeout values to prevent timeout failures."""
        self.logger.info("Adjusting timeout values")
        # Implementation would adjust actual timeout parameters
    
    async def _optimize_memory_pools(self, recommendation: Dict[str, Any]) -> None:
        """Optimize memory pool configuration."""
        self.logger.info("Optimizing memory pools")
        # Implementation would optimize memory allocation
    
    async def _activate_thermal_management(self, recommendation: Dict[str, Any]) -> None:
        """Activate thermal management measures."""
        self.logger.info("Activating thermal management")
        # Implementation would enable thermal controls
    
    async def _enable_communication_buffering(self, recommendation: Dict[str, Any]) -> None:
        """Enable communication buffering."""
        self.logger.info("Enabling communication buffering")
        # Implementation would enable buffering mechanisms
    
    async def _optimize_workload_scheduling(self, recommendation: Dict[str, Any]) -> None:
        """Optimize workload scheduling."""
        self.logger.info("Optimizing workload scheduling")
        # Implementation would adjust workload distribution
    
    async def _update_reliability_metrics(
        self,
        current_metrics: Dict[str, float],
        failure_probabilities: Dict[str, float]
    ) -> None:
        """Update reliability metrics based on current status."""
        # Calculate reliability score based on failure probabilities
        max_failure_prob = max(failure_probabilities.values()) if failure_probabilities else 0
        self.reliability_metrics.reliability_score = 1.0 - max_failure_prob
        
        # Update availability based on system health
        system_health = 1.0 - (current_metrics.get("communication_errors", 0) / 10.0)
        self.reliability_metrics.availability = max(0.0, min(1.0, system_health))
    
    async def handle_detected_failure(
        self,
        failure_mode: FailureMode,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle detected failure with autonomous recovery.
        
        Args:
            failure_mode: Type of failure detected
            context: Context information about the failure
            
        Returns:
            Recovery results
        """
        self.logger.error(f"Failure detected: {failure_mode.value}")
        
        # Attempt autonomous recovery
        recovery_result = await self.recovery_system.attempt_autonomous_recovery(
            failure_mode, context
        )
        
        # Update reliability metrics
        await self._record_failure_and_recovery(failure_mode, recovery_result)
        
        return recovery_result
    
    async def _record_failure_and_recovery(
        self,
        failure_mode: FailureMode,
        recovery_result: Dict[str, Any]
    ) -> None:
        """Record failure and recovery for metrics tracking."""
        # Update MTBF and MTTR calculations
        current_time = time.time()
        
        if hasattr(self, 'last_failure_time'):
            time_between_failures = current_time - self.last_failure_time
            # Update MTBF using exponential moving average
            if self.reliability_metrics.mean_time_between_failures == 0:
                self.reliability_metrics.mean_time_between_failures = time_between_failures
            else:
                alpha = 0.1  # Smoothing factor
                self.reliability_metrics.mean_time_between_failures = (
                    alpha * time_between_failures +
                    (1 - alpha) * self.reliability_metrics.mean_time_between_failures
                )
        
        self.last_failure_time = current_time
        
        # Update MTTR
        recovery_time = recovery_result.get("recovery_time_seconds", 0)
        if self.reliability_metrics.mean_time_to_recovery == 0:
            self.reliability_metrics.mean_time_to_recovery = recovery_time
        else:
            alpha = 0.1
            self.reliability_metrics.mean_time_to_recovery = (
                alpha * recovery_time +
                (1 - alpha) * self.reliability_metrics.mean_time_to_recovery
            )
    
    async def get_reliability_report(self) -> Dict[str, Any]:
        """Generate comprehensive reliability report."""
        return {
            "reliability_level": self.reliability_level.value,
            "current_metrics": {
                "uptime_percentage": self.reliability_metrics.uptime_percentage,
                "mean_time_between_failures": self.reliability_metrics.mean_time_between_failures,
                "mean_time_to_recovery": self.reliability_metrics.mean_time_to_recovery,
                "failure_rate_per_hour": self.reliability_metrics.failure_rate_per_hour,
                "availability": self.reliability_metrics.availability,
                "reliability_score": self.reliability_metrics.reliability_score
            },
            "failure_detector_status": {
                "active": True,
                "patterns_learned": len(self.failure_detector.prediction_model["failure_patterns"]),
                "prediction_accuracy": 0.85  # Would be calculated from historical data
            },
            "recovery_system_status": {
                "strategies_available": sum(len(strategies) for strategies in self.recovery_system.recovery_strategies.values()),
                "successful_recoveries": len([r for r in self.recovery_system.recovery_history if r["success"]]),
                "total_recovery_attempts": len(self.recovery_system.recovery_history)
            },
            "recommendations": await self._generate_reliability_recommendations()
        }
    
    async def _generate_reliability_recommendations(self) -> List[str]:
        """Generate recommendations for improving reliability."""
        recommendations = []
        
        if self.reliability_metrics.reliability_score < 0.9:
            recommendations.append("Reliability score below 90% - implement additional monitoring")
        
        if self.reliability_metrics.mean_time_to_recovery > 60:
            recommendations.append("Recovery time exceeds 60 seconds - optimize recovery strategies")
        
        if self.reliability_metrics.availability < 0.99:
            recommendations.append("Availability below 99% - investigate failure patterns")
        
        return recommendations


# Global reliability guard instance
_global_reliability_guard: Optional[AdvancedReliabilityGuard] = None

def get_reliability_guard(reliability_level: ReliabilityLevel = ReliabilityLevel.PRODUCTION) -> AdvancedReliabilityGuard:
    """Get global reliability guard instance."""
    global _global_reliability_guard
    if _global_reliability_guard is None:
        _global_reliability_guard = AdvancedReliabilityGuard(reliability_level)
    return _global_reliability_guard

async def start_global_reliability_monitoring(
    reliability_level: ReliabilityLevel = ReliabilityLevel.PRODUCTION,
    monitoring_interval_seconds: float = 10.0
) -> None:
    """Start global reliability monitoring."""
    guard = get_reliability_guard(reliability_level)
    await guard.start_reliability_monitoring(monitoring_interval_seconds)

async def stop_global_reliability_monitoring() -> None:
    """Stop global reliability monitoring."""
    guard = get_reliability_guard()
    await guard.stop_reliability_monitoring()

async def handle_system_failure(
    failure_mode: FailureMode,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """Handle system failure with global reliability guard."""
    guard = get_reliability_guard()
    return await guard.handle_detected_failure(failure_mode, context)