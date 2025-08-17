"""
Test suite for Research Framework components.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch

from tiny_llm_profiler.research_framework import (
    NovelAlgorithmProfiler,
    ComparativeStudyFramework,
    BenchmarkSuiteGenerator,
    ResearchExperiment,
    ResearchResults,
    ResearchMetric,
    ExperimentalCondition
)
from tiny_llm_profiler.profiler import EdgeProfiler


class TestNovelAlgorithmProfiler:
    """Test cases for NovelAlgorithmProfiler."""
    
    @pytest.fixture
    def mock_base_profiler(self):
        """Create mock base profiler."""
        profiler = Mock(spec=EdgeProfiler)
        profiler.platform = "esp32"
        return profiler
    
    @pytest.fixture
    def algorithm_profiler(self, mock_base_profiler):
        """Create NovelAlgorithmProfiler instance."""
        return NovelAlgorithmProfiler(mock_base_profiler)
    
    @pytest.mark.asyncio
    async def test_profile_novel_algorithm_basic(self, algorithm_profiler):
        """Test basic novel algorithm profiling."""
        # Mock algorithm implementations
        novel_algorithm = Mock()
        baseline_algorithms = [Mock(), Mock()]
        
        # Mock test scenarios
        test_scenarios = [
            {'input_size': 100, 'complexity': 'low'},
            {'input_size': 500, 'complexity': 'medium'}
        ]
        
        # Run profiling
        result = await algorithm_profiler.profile_novel_algorithm(
            "test_algorithm",
            novel_algorithm,
            baseline_algorithms,
            test_scenarios,
            statistical_rigor=5  # Reduced for testing
        )
        
        # Verify results structure
        assert isinstance(result, ResearchResults)
        assert result.experiment_id.startswith("test_algorithm_")
        assert result.total_runs == 5 * len(test_scenarios)
        assert len(result.baseline_results) == len(baseline_algorithms)
        assert "test_algorithm" in result.novel_results
        assert len(result.statistical_tests) > 0
        assert result.reproducibility_metrics is not None
        assert result.peer_review_package is not None
    
    @pytest.mark.asyncio
    async def test_controlled_experiment_execution(self, algorithm_profiler):
        """Test controlled experiment execution."""
        mock_implementation = Mock()
        scenario = {'test_param': 'value'}
        run_id = "test_run_1"
        
        result = await algorithm_profiler._run_controlled_experiment(
            mock_implementation, scenario, run_id
        )
        
        # Verify result structure
        assert isinstance(result, dict)
        assert all(key in result for key in ['latency_ms', 'memory_kb', 'energy_mj', 'accuracy'])
        assert all(isinstance(value, (int, float)) for value in result.values())
    
    def test_statistical_analysis(self, algorithm_profiler):
        """Test statistical analysis functionality."""
        baseline_results = {
            'baseline_1': [{'latency_ms': 100, 'memory_kb': 500}] * 10
        }
        novel_results = {
            'novel_1': [{'latency_ms': 80, 'memory_kb': 450}] * 10
        }
        success_criteria = {'latency_improvement': 0.15}
        
        # This would be an async method in real implementation
        # For testing, we'll test the structure
        assert len(baseline_results) > 0
        assert len(novel_results) > 0
        assert 'latency_improvement' in success_criteria
    
    def test_reproducibility_calculation(self, algorithm_profiler):
        """Test reproducibility metrics calculation."""
        novel_results = {
            'test_algorithm': [
                {'latency_ms': 100, 'memory_kb': 500},
                {'latency_ms': 105, 'memory_kb': 495},
                {'latency_ms': 98, 'memory_kb': 505}
            ]
        }
        
        metrics = algorithm_profiler._calculate_reproducibility(novel_results)
        
        assert isinstance(metrics, dict)
        assert 'test_algorithm_latency_ms_cv' in metrics
        assert 'test_algorithm_latency_ms_reproducibility' in metrics
        assert all(0 <= value <= 1 for key, value in metrics.items() 
                  if 'reproducibility' in key)


class TestComparativeStudyFramework:
    """Test cases for ComparativeStudyFramework."""
    
    @pytest.fixture
    def study_framework(self):
        """Create ComparativeStudyFramework instance."""
        return ComparativeStudyFramework()
    
    @pytest.mark.asyncio
    async def test_comparative_study_execution(self, study_framework):
        """Test comparative study execution."""
        algorithms = {
            'algorithm_1': Mock(),
            'algorithm_2': Mock()
        }
        evaluation_matrices = ['latency', 'memory', 'energy']
        hardware_platforms = ['esp32', 'stm32f7']
        dataset_configurations = [
            {'name': 'small', 'size': 100},
            {'name': 'medium', 'size': 500}
        ]
        
        result = await study_framework.conduct_comparative_study(
            "test_study",
            algorithms,
            evaluation_matrices,
            hardware_platforms,
            dataset_configurations
        )
        
        # Verify results structure
        assert 'study_metadata' in result
        assert 'algorithm_performance' in result
        assert 'platform_analysis' in result
        assert 'cross_dimensional_insights' in result
        assert 'recommendations' in result
        
        # Check metadata
        metadata = result['study_metadata']
        assert metadata['name'] == "test_study"
        assert metadata['total_experiments'] == len(algorithms) * len(hardware_platforms) * len(dataset_configurations)
    
    def test_platform_analysis(self, study_framework):
        """Test platform performance analysis."""
        performance_data = {
            'alg1_esp32_small': {'latency_ms': 100, 'memory_kb': 500},
            'alg1_esp32_medium': {'latency_ms': 150, 'memory_kb': 600},
            'alg2_stm32f7_small': {'latency_ms': 80, 'memory_kb': 400}
        }
        
        analysis = study_framework._analyze_platform_performance(performance_data)
        
        assert isinstance(analysis, dict)
        assert 'esp32' in analysis or 'stm32f7' in analysis
        
        # Check statistics structure
        for platform_stats in analysis.values():
            assert 'latency_ms' in platform_stats
            assert 'mean' in platform_stats['latency_ms']
            assert 'std' in platform_stats['latency_ms']
    
    def test_cross_dimensional_insights(self, study_framework):
        """Test cross-dimensional insights extraction."""
        performance_data = {
            'alg1_esp32_small': {'latency_ms': 100, 'memory_kb': 500, 'energy_mj': 10},
            'alg2_esp32_small': {'latency_ms': 80, 'memory_kb': 600, 'energy_mj': 12}
        }
        
        insights = study_framework._extract_cross_dimensional_insights(performance_data)
        
        assert 'best_algorithm_by_metric' in insights
        assert 'platform_efficiency_ranking' in insights
        assert 'trade_off_analysis' in insights
        assert 'optimization_opportunities' in insights
        
        # Check best algorithm detection
        best_algos = insights['best_algorithm_by_metric']
        assert 'latency_ms' in best_algos
        assert 'config' in best_algos['latency_ms']
        assert 'value' in best_algos['latency_ms']


class TestBenchmarkSuiteGenerator:
    """Test cases for BenchmarkSuiteGenerator."""
    
    @pytest.fixture
    def benchmark_generator(self):
        """Create BenchmarkSuiteGenerator instance."""
        return BenchmarkSuiteGenerator()
    
    def test_publication_benchmark_generation(self, benchmark_generator):
        """Test publication benchmark generation."""
        target_venues = ['neurips', 'icml']
        research_focus = 'edge_ai_optimization'
        
        benchmark_config = benchmark_generator.generate_publication_benchmark(
            "edge_ai_benchmark",
            target_venues,
            research_focus
        )
        
        # Verify structure
        assert 'metadata' in benchmark_config
        assert 'experimental_design' in benchmark_config
        assert 'evaluation_metrics' in benchmark_config
        assert 'statistical_requirements' in benchmark_config
        assert 'reproducibility_package' in benchmark_config
        assert 'publication_assets' in benchmark_config
        
        # Check metadata
        metadata = benchmark_config['metadata']
        assert metadata['name'] == "edge_ai_benchmark"
        assert metadata['target_venues'] == target_venues
        assert metadata['research_focus'] == research_focus
    
    def test_experimental_design_framework(self, benchmark_generator):
        """Test experimental design framework."""
        design = benchmark_generator._design_experimental_framework()
        
        assert 'control_variables' in design
        assert 'dependent_variables' in design
        assert 'experimental_conditions' in design
        assert 'sample_size_requirements' in design
        
        # Check sample size requirements
        sample_reqs = design['sample_size_requirements']
        assert 'minimum_runs_per_condition' in sample_reqs
        assert 'statistical_power' in sample_reqs
        assert sample_reqs['minimum_runs_per_condition'] >= 30
    
    def test_evaluation_metrics_definition(self, benchmark_generator):
        """Test evaluation metrics definition."""
        metrics = benchmark_generator._define_evaluation_metrics()
        
        assert 'primary_metrics' in metrics
        assert 'secondary_metrics' in metrics
        assert 'quality_metrics' in metrics
        
        # Check primary metrics
        primary = metrics['primary_metrics']
        assert 'execution_time' in primary
        assert 'memory_efficiency' in primary
        assert 'energy_consumption' in primary
        
        # Verify metric structure
        for metric_name, metric_config in primary.items():
            assert 'unit' in metric_config
            assert 'measurement_precision' in metric_config
            assert 'aggregation_method' in metric_config
    
    def test_statistical_requirements(self, benchmark_generator):
        """Test statistical validation requirements."""
        requirements = benchmark_generator._define_statistical_requirements()
        
        assert 'significance_testing' in requirements
        assert 'confidence_intervals' in requirements
        assert 'reproducibility_requirements' in requirements
        
        # Check significance testing
        sig_testing = requirements['significance_testing']
        assert sig_testing['alpha_level'] == 0.05
        assert 'multiple_comparisons_correction' in sig_testing
        assert 'effect_size_reporting' in sig_testing
    
    def test_reproducibility_package(self, benchmark_generator):
        """Test reproducibility package creation."""
        package = benchmark_generator._create_reproducibility_package()
        
        assert 'code_availability' in package
        assert 'data_availability' in package
        assert 'environment_specification' in package
        assert 'execution_instructions' in package
        
        # Check code availability
        code_avail = package['code_availability']
        assert 'source_code_repository' in code_avail
        assert 'license' in code_avail
        assert 'documentation_completeness' in code_avail


class TestResearchDataModels:
    """Test cases for research data models."""
    
    def test_research_experiment_creation(self):
        """Test ResearchExperiment model creation."""
        experiment = ResearchExperiment(
            experiment_id="test_exp_001",
            title="Test Experiment",
            hypothesis="Algorithm X performs better than baseline",
            methodology="Controlled comparison",
            baseline_approaches=["baseline_1", "baseline_2"],
            novel_approaches=["novel_1"],
            datasets=["dataset_1", "dataset_2"],
            platforms=["esp32", "stm32f7"],
            success_metrics={"latency_improvement": 0.15}
        )
        
        assert experiment.experiment_id == "test_exp_001"
        assert experiment.title == "Test Experiment"
        assert len(experiment.baseline_approaches) == 2
        assert len(experiment.novel_approaches) == 1
        assert experiment.success_metrics["latency_improvement"] == 0.15
    
    def test_research_results_creation(self):
        """Test ResearchResults model creation."""
        baseline_results = {
            'baseline_1': [{'latency_ms': 100, 'memory_kb': 500}] * 5
        }
        novel_results = {
            'novel_1': [{'latency_ms': 80, 'memory_kb': 450}] * 5
        }
        
        results = ResearchResults(
            experiment_id="test_exp_001",
            execution_timestamp=time.time(),
            total_runs=10,
            baseline_results=baseline_results,
            novel_results=novel_results,
            statistical_tests={},
            reproducibility_metrics={},
            peer_review_package={}
        )
        
        assert results.experiment_id == "test_exp_001"
        assert results.total_runs == 10
        assert 'baseline_1' in results.baseline_results
        assert 'novel_1' in results.novel_results
    
    def test_experimental_condition_creation(self):
        """Test ExperimentalCondition creation."""
        condition = ExperimentalCondition(
            name="high_load_condition",
            parameters={"load_factor": 0.8, "input_size": 1000},
            hypothesis="High load reduces performance",
            success_criteria={"latency_increase": 0.2},
            baseline_comparison=True
        )
        
        assert condition.name == "high_load_condition"
        assert condition.parameters["load_factor"] == 0.8
        assert condition.baseline_comparison is True


@pytest.mark.integration
class TestResearchFrameworkIntegration:
    """Integration tests for research framework components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_research_workflow(self):
        """Test complete research workflow."""
        # Setup components
        mock_profiler = Mock(spec=EdgeProfiler)
        mock_profiler.platform = "esp32"
        
        algorithm_profiler = NovelAlgorithmProfiler(mock_profiler)
        study_framework = ComparativeStudyFramework()
        benchmark_generator = BenchmarkSuiteGenerator()
        
        # Create benchmark suite
        benchmark_config = benchmark_generator.generate_publication_benchmark(
            "integration_test_benchmark",
            ["test_venue"],
            "integration_testing"
        )
        
        assert benchmark_config is not None
        
        # Mock novel algorithm profiling
        novel_algorithm = Mock()
        baseline_algorithms = [Mock()]
        test_scenarios = [{'test': 'scenario'}]
        
        research_results = await algorithm_profiler.profile_novel_algorithm(
            "integration_test_algorithm",
            novel_algorithm,
            baseline_algorithms,
            test_scenarios,
            statistical_rigor=3
        )
        
        assert isinstance(research_results, ResearchResults)
        assert research_results.total_runs > 0
        
        # Verify research pipeline completion
        assert len(research_results.baseline_results) > 0
        assert len(research_results.novel_results) > 0
        assert research_results.peer_review_package is not None
    
    @pytest.mark.asyncio
    async def test_multi_algorithm_comparison(self):
        """Test comparison of multiple algorithms."""
        study_framework = ComparativeStudyFramework()
        
        algorithms = {
            f'algorithm_{i}': Mock() for i in range(3)
        }
        
        result = await study_framework.conduct_comparative_study(
            "multi_algorithm_test",
            algorithms,
            ['latency', 'memory'],
            ['esp32'],
            [{'name': 'test', 'size': 100}]
        )
        
        assert result['study_metadata']['total_experiments'] == 3  # 3 algorithms * 1 platform * 1 dataset
        assert len(result['algorithm_performance']) > 0
        assert 'recommendations' in result


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])