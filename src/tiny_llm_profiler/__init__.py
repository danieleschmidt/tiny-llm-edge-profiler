"""
Tiny LLM Edge Profiler - Generation 4 with Quantum Leap AI and Global Deployment

A comprehensive profiling toolkit for running quantized LLMs on microcontrollers
and edge devices. Measure real-world performance on ARM Cortex-M, RISC-V, ESP32.

Generation 1 Features:
- Basic profiling and performance measurement
- Platform-specific optimizations for edge devices

Generation 2 Features:
- Comprehensive error handling and retry mechanisms
- Circuit breaker patterns for reliability
- Advanced monitoring and alerting
- Security enhancements for production use
- Resource management and leak prevention
- Graceful degradation and auto-recovery

Generation 3 Features:
- Advanced performance optimization with vectorization and CPU optimization
- Multi-level caching architecture (L1/L2/L3) with intelligent cache management
- Distributed profiling system with multi-device coordination
- Async/parallel processing with pipeline optimization and backpressure handling
- Resource optimization with memory pools and connection pooling
- Predictive auto-scaling infrastructure with ML-based forecasting
- Database and storage optimization with compression and efficient formats
- Real-time performance analytics with regression detection and anomaly detection
- Global optimization system for multi-region deployment coordination
- Comprehensive benchmarking and performance comparison tools

Generation 4 Features (NEW - QUANTUM LEAP):
- Quantum-inspired optimization algorithms for breakthrough performance
- Neuromorphic computing patterns for adaptive profiling
- Autonomous AI learning with meta-learning and transfer learning
- Multi-objective Pareto-optimal solutions with evolutionary algorithms
- Real-time autonomous parameter optimization and self-improvement
- Advanced online learning with catastrophic forgetting prevention
- Global-first deployment with 40+ language support and regional compliance
- Multi-region edge coordination with intelligent load balancing
- Cultural adaptation and accessibility features for worldwide deployment
"""

__version__ = "0.4.0"  # Generation 4 - Quantum Leap
__author__ = "Terragon Labs"
__email__ = "dev@terragon.dev"

# Core profiling components
from .profiler import EdgeProfiler
from .models import QuantizedModel
from .results import ProfileResults
from .platforms import PlatformManager

# Analysis and optimization tools
from .analyzer import MetricsAnalyzer, ComparativeAnalyzer
from .optimizer import (
    EnergyOptimizer,
    MemoryOptimizer,
    PlatformOptimizer,
)

# Power profiling
from .power import PowerProfiler

# Benchmarking
from .benchmarks import StandardBenchmarks

# Robustness and Reliability (Generation 2 enhancements)
from .reliability import (
    ReliabilityManager,
    RetryMechanism,
    CircuitBreaker,
    TimeoutManager,
    retry,
    circuit_breaker,
    with_timeout,
    managed_resource,
)
from .advanced_monitoring import (
    MonitoringSystem,
    AlertManager,
    MetricCollector,
    get_monitoring_system,
    start_monitoring,
    stop_monitoring,
    record_metric,
    get_health_summary,
)
from .security import (
    SecurityValidator,
    SecurityAuditor,
    validate_environment,
    validate_identifier,
    validate_file_path,
    sanitize_filename,
)

# Generation 3: Advanced Optimization and Scaling
from .performance_optimizer import (
    PerformanceOptimizer,
    AlgorithmicOptimizer,
    CPUOptimizer,
    MemoryOptimizer as AdvancedMemoryOptimizer,
    IOOptimizer,
    get_performance_optimizer,
)
from .advanced_cache import (
    MultiLevelCache,
    AdvancedCacheEntry,
    CompressionHandler,
    get_multilevel_cache,
    advanced_cached,
)
from .distributed_profiler import DistributedCoordinator, NetworkCommunicator, NodeInfo
from .async_pipeline import (
    AsyncPipeline,
    StreamProcessor,
    AdaptiveConcurrencyController,
)
from .resource_optimizer import (
    AdaptiveMemoryPool,
    AdaptiveConnectionPool,
    ResourceLeakDetector,
)
from .predictive_scaler import PredictiveScaler, TimeSeriesPredictor, CostOptimizer
from .storage_optimizer import OptimizedDatabase, DataLifecycleManager, DataCompressor
from .performance_analytics import (
    PerformanceAnalyticsEngine,
    RegressionDetector,
    AnomalyDetector,
)
from .global_optimizer import (
    GlobalOptimizer,
    RegionInfo,
    RegionType,
    OptimizationStrategy,
)
from .benchmarking import (
    BenchmarkSuite,
    BenchmarkConfiguration,
    BenchmarkType,
    ComparisonMethod,
)

# Generation 4: Quantum Leap AI and Autonomous Optimization
from .generation4_quantum_profiler import (
    QuantumLeapProfiler,
    QuantumProfilingResult,
    QuantumInspiredOptimizer,
    NeuromorphicProfiler,
    AIAutonomousLearningProfiler,
    QuantumOptimizationMethod,
    NeuromorphicPattern,
    get_quantum_leap_profiler,
    run_quantum_profiling_experiment,
    compare_quantum_vs_traditional,
)
from .autonomous_ai_optimizer import (
    AutonomousAIOptimizer,
    MetaLearner,
    OnlineLearner,
    MultiObjectiveOptimizer,
    HardwareProfile,
    ModelProfile,
    PerformanceTarget,
    LearningStrategy,
    OptimizationObjective,
    get_autonomous_ai_optimizer,
    run_autonomous_optimization_experiment,
)

# Global Deployment and Internationalization
from .i18n_manager import (
    InternationalizationManager,
    SupportedLanguage,
    RegionalCompliance,
    get_i18n_manager,
    init_i18n,
    set_language,
    get_supported_languages,
    _,
)
from .global_deployment_manager import (
    GlobalDeploymentManager,
    GlobalRegion,
    DataSovereigntyLevel,
    PlatformArchitecture,
    DeploymentConfiguration,
    EdgeLocation,
    GlobalLoadBalancer,
    ComplianceManager,
    get_global_deployment_manager,
    deploy_globally,
    route_profiling_request,
    get_global_status,
)

# Generation 5: Research and Academic Publication Framework
# Note: Import these components only when needed to avoid dependency issues
# from .research_framework import (
#     NovelAlgorithmProfiler, ComparativeStudyFramework, BenchmarkSuiteGenerator,
#     ResearchExperiment, ResearchResults, ResearchMetric, ExperimentalCondition
# )
# from .experimental_validation import (
#     ExperimentalValidationEngine, StatisticalValidator, CrossValidationFramework,
#     BootstrapValidator, ValidationConfiguration, ExperimentalResult
# )
# from .publication_pipeline import (
#     PublicationPipeline, PublicationFigureGenerator, LaTeXDocumentGenerator,
#     PublicationVenue, PublicationRequirements, FigureSpecification
# )

__all__ = [
    # Core API
    "EdgeProfiler",
    "QuantizedModel",
    "ProfileResults",
    "PlatformManager",
    # Analysis
    "MetricsAnalyzer",
    "ComparativeAnalyzer",
    # Optimization
    "EnergyOptimizer",
    "MemoryOptimizer",
    "PlatformOptimizer",
    # Power profiling
    "PowerProfiler",
    # Benchmarking
    "StandardBenchmarks",
    # Reliability & Robustness (Generation 2)
    "ReliabilityManager",
    "RetryMechanism",
    "CircuitBreaker",
    "TimeoutManager",
    "retry",
    "circuit_breaker",
    "with_timeout",
    "managed_resource",
    # Monitoring & Alerting
    "MonitoringSystem",
    "AlertManager",
    "MetricCollector",
    "get_monitoring_system",
    "start_monitoring",
    "stop_monitoring",
    "record_metric",
    "get_health_summary",
    # Security
    "SecurityValidator",
    "SecurityAuditor",
    "validate_environment",
    "validate_identifier",
    "validate_file_path",
    "sanitize_filename",
    # Generation 3: Advanced Optimization and Scaling
    # Performance Optimization
    "PerformanceOptimizer",
    "AlgorithmicOptimizer",
    "CPUOptimizer",
    "AdvancedMemoryOptimizer",
    "IOOptimizer",
    "get_performance_optimizer",
    # Advanced Caching
    "MultiLevelCache",
    "AdvancedCacheEntry",
    "CompressionHandler",
    "get_multilevel_cache",
    "advanced_cached",
    # Distributed Profiling
    "DistributedCoordinator",
    "NetworkCommunicator",
    "NodeInfo",
    "get_distributed_coordinator",
    "start_distributed_profiling",
    # Async Pipeline Processing
    "AsyncPipeline",
    "StreamProcessor",
    "AdaptiveConcurrencyController",
    "get_async_pipeline",
    "process_async_stream",
    # Resource Optimization
    "AdaptiveMemoryPool",
    "AdaptiveConnectionPool",
    "ResourceLeakDetector",
    "get_resource_optimizer",
    "optimize_resource_usage",
    # Predictive Scaling
    "PredictiveScaler",
    "TimeSeriesPredictor",
    "CostOptimizer",
    "get_predictive_scaler",
    "enable_auto_scaling",
    # Storage Optimization
    "OptimizedDatabase",
    "DataLifecycleManager",
    "DataCompressor",
    "get_storage_optimizer",
    "optimize_data_storage",
    # Performance Analytics
    "PerformanceAnalyticsEngine",
    "RegressionDetector",
    "AnomalyDetector",
    "get_analytics_engine",
    "start_performance_analytics",
    "record_performance_metric",
    "generate_performance_report",
    "add_performance_alert_callback",
    # Global Optimization
    "GlobalOptimizer",
    "RegionInfo",
    "RegionType",
    "OptimizationStrategy",
    "get_global_optimizer",
    "start_global_optimization",
    "register_deployment_region",
    "optimize_global_deployment",
    "get_global_optimization_status",
    # Benchmarking and Performance Comparison
    "BenchmarkSuite",
    "BenchmarkConfiguration",
    "BenchmarkType",
    "ComparisonMethod",
    "get_benchmark_suite",
    "run_standard_benchmarks",
    "compare_benchmark_performance",
    "generate_benchmark_report",
    "get_benchmark_trends",
    # Generation 4: Quantum Leap AI and Autonomous Optimization
    "QuantumLeapProfiler",
    "QuantumProfilingResult",
    "QuantumInspiredOptimizer",
    "NeuromorphicProfiler",
    "AIAutonomousLearningProfiler",
    "QuantumOptimizationMethod",
    "NeuromorphicPattern",
    "get_quantum_leap_profiler",
    "run_quantum_profiling_experiment",
    "compare_quantum_vs_traditional",
    # Autonomous AI Optimization
    "AutonomousAIOptimizer",
    "MetaLearner",
    "OnlineLearner",
    "MultiObjectiveOptimizer",
    "HardwareProfile",
    "ModelProfile",
    "PerformanceTarget",
    "LearningStrategy",
    "OptimizationObjective",
    "get_autonomous_ai_optimizer",
    "run_autonomous_optimization_experiment",
    # Global Deployment and Internationalization
    "InternationalizationManager",
    "SupportedLanguage",
    "RegionalCompliance",
    "get_i18n_manager",
    "init_i18n",
    "set_language",
    "get_supported_languages",
    "_",  # Translation shorthand
    # Global Deployment Management
    "GlobalDeploymentManager",
    "GlobalRegion",
    "DataSovereigntyLevel",
    "PlatformArchitecture",
    "DeploymentConfiguration",
    "EdgeLocation",
    "GlobalLoadBalancer",
    "ComplianceManager",
    "get_global_deployment_manager",
    "deploy_globally",
    "route_profiling_request",
    "get_global_status",
    # Generation 5: Research and Academic Publication Framework
    "ResearchFramework",
    "NovelAlgorithmProfiler",
    "ComparativeStudyFramework",
    "BenchmarkSuiteGenerator",
    "ExperimentalValidationEngine",
    "StatisticalValidator",
    "CrossValidationFramework",
    "BootstrapValidator",
    "PublicationPipeline",
    "PublicationFigureGenerator",
    "LaTeXDocumentGenerator",
    "ResearchExperiment",
    "ResearchResults",
    "ValidationConfiguration",
    "PublicationVenue",
]
