"""
Experimental Validation Engine for Research Framework

Provides comprehensive experimental validation, statistical analysis,
and reproducibility testing for breakthrough research algorithms.

Features:
- Cross-validation frameworks
- Bootstrap validation
- Statistical significance testing
- Effect size calculations
- Reproducibility metrics
- Research quality gates
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor

from .exceptions import ProfilerError
from .breakthrough_research_algorithms import (
    HardwareProfile, 
    BreakthroughProfilingEngine,
    run_breakthrough_research_experiment
)


class ValidationMethod(str, Enum):
    """Validation method types."""
    CROSS_VALIDATION = "cross_validation"
    BOOTSTRAP = "bootstrap"
    HOLDOUT = "holdout"
    TIME_SERIES_SPLIT = "time_series_split"
    MONTE_CARLO = "monte_carlo"


class StatisticalTest(str, Enum):
    """Statistical test types."""
    T_TEST_INDEPENDENT = "t_test_independent"
    T_TEST_PAIRED = "t_test_paired"
    WILCOXON_SIGNED_RANK = "wilcoxon_signed_rank"
    MANN_WHITNEY_U = "mann_whitney_u"
    KRUSKAL_WALLIS = "kruskal_wallis"
    ANOVA_ONE_WAY = "anova_one_way"
    CHI_SQUARE = "chi_square"


class EffectSizeMetric(str, Enum):
    """Effect size metrics."""
    COHENS_D = "cohens_d"
    HEDGES_G = "hedges_g"
    GLASS_DELTA = "glass_delta"
    ETA_SQUARED = "eta_squared"
    OMEGA_SQUARED = "omega_squared"
    CLIFF_DELTA = "cliff_delta"


@dataclass
class ValidationConfiguration:
    """Configuration for experimental validation."""
    validation_method: ValidationMethod
    n_folds: int = 5
    n_bootstrap_samples: int = 1000
    test_size: float = 0.2
    random_state: int = 42
    confidence_level: float = 0.95
    statistical_tests: List[StatisticalTest] = field(default_factory=lambda: [
        StatisticalTest.T_TEST_INDEPENDENT,
        StatisticalTest.WILCOXON_SIGNED_RANK
    ])
    effect_size_metrics: List[EffectSizeMetric] = field(default_factory=lambda: [
        EffectSizeMetric.COHENS_D,
        EffectSizeMetric.HEDGES_G
    ])
    min_sample_size: int = 30
    max_p_value: float = 0.05
    min_effect_size: float = 0.5


@dataclass
class ExperimentalCondition:
    """Experimental condition specification."""
    name: str
    parameters: Dict[str, Any]
    control_group: bool = False
    expected_improvement: Optional[float] = None
    replication_count: int = 10


@dataclass
class StatisticalResult:
    """Statistical test result."""
    test_type: StatisticalTest
    statistic: float
    p_value: float
    effect_size: float
    effect_size_metric: EffectSizeMetric
    confidence_interval: Tuple[float, float]
    is_significant: bool
    power: float
    sample_size: int


@dataclass
class ValidationResult:
    """Validation result with metrics."""
    method: ValidationMethod
    mean_score: float
    std_score: float
    scores: List[float]
    confidence_interval: Tuple[float, float]
    cross_validation_scores: Optional[List[float]] = None
    bootstrap_scores: Optional[List[float]] = None


class ExperimentalValidationEngine:
    """
    Comprehensive experimental validation engine for research experiments.
    
    Provides statistical validation, reproducibility testing, and quality gates
    for breakthrough research algorithms in edge AI profiling.
    """
    
    def __init__(self, config: ValidationConfiguration):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.validation_history = []
        self.statistical_cache = {}
        
    async def validate_experimental_results(
        self,
        experimental_data: Dict[str, List[float]],
        baseline_data: Dict[str, List[float]],
        conditions: List[ExperimentalCondition]
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of experimental results.
        
        Args:
            experimental_data: Results from experimental conditions
            baseline_data: Baseline/control results
            conditions: Experimental conditions tested
            
        Returns:
            Comprehensive validation report with statistical analysis
        """
        validation_report = {
            'validation_configuration': self.config,
            'experimental_conditions': conditions,
            'statistical_tests': {},
            'effect_sizes': {},
            'validation_results': {},
            'quality_gates': {},
            'reproducibility_metrics': {},
            'research_recommendations': []
        }
        
        # Perform statistical tests
        for condition_name, exp_values in experimental_data.items():
            if condition_name in baseline_data:
                baseline_values = baseline_data[condition_name]
                
                statistical_results = await self._perform_statistical_tests(
                    exp_values, baseline_values, condition_name
                )
                validation_report['statistical_tests'][condition_name] = statistical_results
        
        # Calculate effect sizes
        validation_report['effect_sizes'] = self._calculate_effect_sizes(
            experimental_data, baseline_data
        )
        
        # Cross-validation and bootstrap validation
        validation_report['validation_results'] = await self._perform_validation_methods(
            experimental_data, baseline_data
        )
        
        # Quality gate assessment
        validation_report['quality_gates'] = self._assess_quality_gates(
            validation_report['statistical_tests'],
            validation_report['effect_sizes']
        )
        
        # Reproducibility metrics
        validation_report['reproducibility_metrics'] = self._calculate_reproducibility_metrics(
            experimental_data
        )
        
        # Research recommendations
        validation_report['research_recommendations'] = self._generate_research_recommendations(
            validation_report
        )
        
        return validation_report
    
    async def _perform_statistical_tests(
        self,
        experimental_values: List[float],
        baseline_values: List[float],
        condition_name: str
    ) -> Dict[StatisticalTest, StatisticalResult]:
        """Perform comprehensive statistical testing."""
        results = {}
        
        for test_type in self.config.statistical_tests:
            try:
                result = await self._execute_statistical_test(
                    test_type, experimental_values, baseline_values, condition_name
                )
                results[test_type] = result
            except Exception as e:
                self.logger.warning(f"Statistical test {test_type} failed for {condition_name}: {e}")
        
        return results
    
    async def _execute_statistical_test(
        self,
        test_type: StatisticalTest,
        experimental_values: List[float],
        baseline_values: List[float],
        condition_name: str
    ) -> StatisticalResult:
        """Execute a specific statistical test."""
        exp_array = np.array(experimental_values)
        baseline_array = np.array(baseline_values)
        
        if test_type == StatisticalTest.T_TEST_INDEPENDENT:
            statistic, p_value = stats.ttest_ind(exp_array, baseline_array)
            
        elif test_type == StatisticalTest.T_TEST_PAIRED:
            # Ensure same length for paired test
            min_len = min(len(exp_array), len(baseline_array))
            statistic, p_value = stats.ttest_rel(
                exp_array[:min_len], baseline_array[:min_len]
            )
            
        elif test_type == StatisticalTest.WILCOXON_SIGNED_RANK:
            min_len = min(len(exp_array), len(baseline_array))
            statistic, p_value = stats.wilcoxon(
                exp_array[:min_len], baseline_array[:min_len]
            )
            
        elif test_type == StatisticalTest.MANN_WHITNEY_U:
            statistic, p_value = stats.mannwhitneyu(
                exp_array, baseline_array, alternative='two-sided'
            )
            
        elif test_type == StatisticalTest.KRUSKAL_WALLIS:
            statistic, p_value = stats.kruskal(exp_array, baseline_array)
            
        elif test_type == StatisticalTest.ANOVA_ONE_WAY:
            statistic, p_value = stats.f_oneway(exp_array, baseline_array)
            
        else:
            raise ValueError(f"Unsupported statistical test: {test_type}")
        
        # Calculate effect size
        effect_size = self._calculate_cohens_d(exp_array, baseline_array)
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(
            exp_array, self.config.confidence_level
        )
        
        # Calculate statistical power
        power = self._calculate_statistical_power(
            effect_size, len(exp_array), len(baseline_array), self.config.max_p_value
        )
        
        return StatisticalResult(
            test_type=test_type,
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=effect_size,
            effect_size_metric=EffectSizeMetric.COHENS_D,
            confidence_interval=confidence_interval,
            is_significant=p_value < self.config.max_p_value,
            power=power,
            sample_size=len(exp_array)
        )
    
    def _calculate_effect_sizes(
        self,
        experimental_data: Dict[str, List[float]],
        baseline_data: Dict[str, List[float]]
    ) -> Dict[str, Dict[EffectSizeMetric, float]]:
        """Calculate multiple effect size metrics."""
        effect_sizes = {}
        
        for condition_name, exp_values in experimental_data.items():
            if condition_name in baseline_data:
                baseline_values = baseline_data[condition_name]
                condition_effects = {}
                
                for metric in self.config.effect_size_metrics:
                    try:
                        effect_size = self._calculate_effect_size_metric(
                            metric, exp_values, baseline_values
                        )
                        condition_effects[metric] = effect_size
                    except Exception as e:
                        self.logger.warning(f"Effect size calculation failed for {metric}: {e}")
                
                effect_sizes[condition_name] = condition_effects
        
        return effect_sizes
    
    def _calculate_effect_size_metric(
        self,
        metric: EffectSizeMetric,
        experimental_values: List[float],
        baseline_values: List[float]
    ) -> float:
        """Calculate specific effect size metric."""
        exp_array = np.array(experimental_values)
        baseline_array = np.array(baseline_values)
        
        if metric == EffectSizeMetric.COHENS_D:
            return self._calculate_cohens_d(exp_array, baseline_array)
            
        elif metric == EffectSizeMetric.HEDGES_G:
            return self._calculate_hedges_g(exp_array, baseline_array)
            
        elif metric == EffectSizeMetric.GLASS_DELTA:
            return self._calculate_glass_delta(exp_array, baseline_array)
            
        elif metric == EffectSizeMetric.CLIFF_DELTA:
            return self._calculate_cliff_delta(exp_array, baseline_array)
            
        else:
            raise ValueError(f"Unsupported effect size metric: {metric}")
    
    def _calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def _calculate_hedges_g(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Hedges' g effect size (bias-corrected Cohen's d)."""
        cohens_d = self._calculate_cohens_d(group1, group2)
        n1, n2 = len(group1), len(group2)
        df = n1 + n2 - 2
        
        # Bias correction factor
        correction_factor = 1 - (3 / (4 * df - 1))
        
        return cohens_d * correction_factor
    
    def _calculate_glass_delta(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Glass's delta effect size."""
        control_std = np.std(group2, ddof=1)
        
        if control_std == 0:
            return 0.0
        
        return (np.mean(group1) - np.mean(group2)) / control_std
    
    def _calculate_cliff_delta(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cliff's delta (non-parametric effect size)."""
        n1, n2 = len(group1), len(group2)
        
        # Count favorable, unfavorable, and tied comparisons
        favorable = 0
        unfavorable = 0
        
        for x in group1:
            for y in group2:
                if x > y:
                    favorable += 1
                elif x < y:
                    unfavorable += 1
        
        total_comparisons = n1 * n2
        
        if total_comparisons == 0:
            return 0.0
        
        return (favorable - unfavorable) / total_comparisons
    
    def _calculate_confidence_interval(
        self,
        data: np.ndarray,
        confidence_level: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval for the mean."""
        alpha = 1 - confidence_level
        df = len(data) - 1
        
        mean = np.mean(data)
        std_error = stats.sem(data)
        
        t_critical = stats.t.ppf(1 - alpha/2, df)
        margin_error = t_critical * std_error
        
        return (mean - margin_error, mean + margin_error)
    
    def _calculate_statistical_power(
        self,
        effect_size: float,
        n1: int,
        n2: int,
        alpha: float
    ) -> float:
        """Calculate statistical power for the test."""
        # Simplified power calculation for t-test
        # This is an approximation; more sophisticated methods exist
        
        # Pooled sample size
        n_harmonic = 2 / (1/n1 + 1/n2)
        
        # Non-centrality parameter
        delta = effect_size * np.sqrt(n_harmonic / 2)
        
        # Critical t-value
        df = n1 + n2 - 2
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        # Power calculation (approximation)
        power = 1 - stats.t.cdf(t_critical - delta, df) + stats.t.cdf(-t_critical - delta, df)
        
        return max(0.0, min(1.0, power))
    
    async def _perform_validation_methods(
        self,
        experimental_data: Dict[str, List[float]],
        baseline_data: Dict[str, List[float]]
    ) -> Dict[str, ValidationResult]:
        """Perform cross-validation and bootstrap validation."""
        validation_results = {}
        
        for condition_name, exp_values in experimental_data.items():
            if condition_name in baseline_data:
                baseline_values = baseline_data[condition_name]
                
                # Combine data for validation
                combined_data = exp_values + baseline_values
                labels = [1] * len(exp_values) + [0] * len(baseline_values)
                
                if self.config.validation_method == ValidationMethod.CROSS_VALIDATION:
                    result = await self._cross_validation(combined_data, labels, condition_name)
                    
                elif self.config.validation_method == ValidationMethod.BOOTSTRAP:
                    result = await self._bootstrap_validation(exp_values, baseline_values, condition_name)
                    
                elif self.config.validation_method == ValidationMethod.MONTE_CARLO:
                    result = await self._monte_carlo_validation(exp_values, baseline_values, condition_name)
                    
                else:
                    self.logger.warning(f"Validation method {self.config.validation_method} not implemented")
                    continue
                
                validation_results[condition_name] = result
        
        return validation_results
    
    async def _cross_validation(
        self,
        data: List[float],
        labels: List[int],
        condition_name: str
    ) -> ValidationResult:
        """Perform k-fold cross-validation."""
        data_array = np.array(data).reshape(-1, 1)
        labels_array = np.array(labels)
        
        # Use a simple classifier for validation
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(random_state=self.config.random_state)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            classifier, data_array, labels_array, 
            cv=self.config.n_folds, scoring='accuracy'
        )
        
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(
            cv_scores, self.config.confidence_level
        )
        
        return ValidationResult(
            method=ValidationMethod.CROSS_VALIDATION,
            mean_score=mean_score,
            std_score=std_score,
            scores=cv_scores.tolist(),
            confidence_interval=confidence_interval,
            cross_validation_scores=cv_scores.tolist()
        )
    
    async def _bootstrap_validation(
        self,
        experimental_values: List[float],
        baseline_values: List[float],
        condition_name: str
    ) -> ValidationResult:
        """Perform bootstrap validation."""
        exp_array = np.array(experimental_values)
        baseline_array = np.array(baseline_values)
        
        bootstrap_scores = []
        
        for _ in range(self.config.n_bootstrap_samples):
            # Bootstrap sampling
            exp_bootstrap = np.random.choice(
                exp_array, size=len(exp_array), replace=True
            )
            baseline_bootstrap = np.random.choice(
                baseline_array, size=len(baseline_array), replace=True
            )
            
            # Calculate difference in means
            difference = np.mean(exp_bootstrap) - np.mean(baseline_bootstrap)
            bootstrap_scores.append(difference)
        
        bootstrap_scores = np.array(bootstrap_scores)
        mean_score = np.mean(bootstrap_scores)
        std_score = np.std(bootstrap_scores)
        
        # Calculate confidence interval
        alpha = 1 - self.config.confidence_level
        ci_lower = np.percentile(bootstrap_scores, 100 * alpha/2)
        ci_upper = np.percentile(bootstrap_scores, 100 * (1 - alpha/2))
        
        return ValidationResult(
            method=ValidationMethod.BOOTSTRAP,
            mean_score=mean_score,
            std_score=std_score,
            scores=bootstrap_scores.tolist(),
            confidence_interval=(ci_lower, ci_upper),
            bootstrap_scores=bootstrap_scores.tolist()
        )
    
    async def _monte_carlo_validation(
        self,
        experimental_values: List[float],
        baseline_values: List[float],
        condition_name: str
    ) -> ValidationResult:
        """Perform Monte Carlo validation."""
        exp_array = np.array(experimental_values)
        baseline_array = np.array(baseline_values)
        
        # Estimate distributions
        exp_mean, exp_std = np.mean(exp_array), np.std(exp_array)
        baseline_mean, baseline_std = np.mean(baseline_array), np.std(baseline_array)
        
        monte_carlo_scores = []
        
        for _ in range(self.config.n_bootstrap_samples):
            # Generate samples from estimated distributions
            exp_sample = np.random.normal(exp_mean, exp_std, len(exp_array))
            baseline_sample = np.random.normal(baseline_mean, baseline_std, len(baseline_array))
            
            # Calculate difference in means
            difference = np.mean(exp_sample) - np.mean(baseline_sample)
            monte_carlo_scores.append(difference)
        
        monte_carlo_scores = np.array(monte_carlo_scores)
        mean_score = np.mean(monte_carlo_scores)
        std_score = np.std(monte_carlo_scores)
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(
            monte_carlo_scores, self.config.confidence_level
        )
        
        return ValidationResult(
            method=ValidationMethod.MONTE_CARLO,
            mean_score=mean_score,
            std_score=std_score,
            scores=monte_carlo_scores.tolist(),
            confidence_interval=confidence_interval
        )
    
    def _assess_quality_gates(
        self,
        statistical_tests: Dict[str, Dict[StatisticalTest, StatisticalResult]],
        effect_sizes: Dict[str, Dict[EffectSizeMetric, float]]
    ) -> Dict[str, Any]:
        """Assess research quality gates."""
        quality_gates = {
            'statistical_significance': {},
            'effect_size_adequacy': {},
            'sample_size_adequacy': {},
            'reproducibility': {},
            'overall_quality': 'UNKNOWN'
        }
        
        # Check statistical significance
        significance_passed = 0
        total_tests = 0
        
        for condition, tests in statistical_tests.items():
            condition_significant = False
            for test_type, result in tests.items():
                total_tests += 1
                if result.is_significant:
                    significance_passed += 1
                    condition_significant = True
            
            quality_gates['statistical_significance'][condition] = condition_significant
        
        # Check effect size adequacy
        effect_size_passed = 0
        total_effect_sizes = 0
        
        for condition, effects in effect_sizes.items():
            condition_adequate = False
            for metric, effect_size in effects.items():
                total_effect_sizes += 1
                if abs(effect_size) >= self.config.min_effect_size:
                    effect_size_passed += 1
                    condition_adequate = True
            
            quality_gates['effect_size_adequacy'][condition] = condition_adequate
        
        # Check sample size adequacy
        for condition, tests in statistical_tests.items():
            adequate_sample = True
            for test_type, result in tests.items():
                if result.sample_size < self.config.min_sample_size:
                    adequate_sample = False
                    break
            
            quality_gates['sample_size_adequacy'][condition] = adequate_sample
        
        # Overall quality assessment
        significance_rate = significance_passed / max(1, total_tests)
        effect_size_rate = effect_size_passed / max(1, total_effect_sizes)
        
        if significance_rate >= 0.8 and effect_size_rate >= 0.8:
            quality_gates['overall_quality'] = 'EXCELLENT'
        elif significance_rate >= 0.6 and effect_size_rate >= 0.6:
            quality_gates['overall_quality'] = 'GOOD'
        elif significance_rate >= 0.4 and effect_size_rate >= 0.4:
            quality_gates['overall_quality'] = 'ACCEPTABLE'
        else:
            quality_gates['overall_quality'] = 'INSUFFICIENT'
        
        return quality_gates
    
    def _calculate_reproducibility_metrics(
        self,
        experimental_data: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """Calculate reproducibility metrics."""
        reproducibility_metrics = {}
        
        for condition_name, values in experimental_data.items():
            if len(values) >= 3:  # Need at least 3 replicates
                # Coefficient of variation
                cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else float('inf')
                
                # Intraclass correlation coefficient (simplified)
                icc = self._calculate_simple_icc(values)
                
                # Reproducibility score (0-1, higher is better)
                reproducibility_score = 1 / (1 + cv) if cv != float('inf') else 0
                
                reproducibility_metrics[condition_name] = {
                    'coefficient_of_variation': cv,
                    'intraclass_correlation': icc,
                    'reproducibility_score': reproducibility_score
                }
        
        return reproducibility_metrics
    
    def _calculate_simple_icc(self, values: List[float]) -> float:
        """Calculate simplified intraclass correlation coefficient."""
        values_array = np.array(values)
        
        # Simple ICC calculation (ICC(1,1) approximation)
        # This is a simplified version; full ICC requires more complex calculation
        
        mean_value = np.mean(values_array)
        between_variance = np.var([mean_value] * len(values_array))
        within_variance = np.var(values_array)
        
        if within_variance == 0:
            return 1.0
        
        icc = between_variance / (between_variance + within_variance)
        return max(0.0, min(1.0, icc))
    
    def _generate_research_recommendations(
        self,
        validation_report: Dict[str, Any]
    ) -> List[str]:
        """Generate research recommendations based on validation results."""
        recommendations = []
        
        quality_gates = validation_report['quality_gates']
        overall_quality = quality_gates['overall_quality']
        
        if overall_quality == 'INSUFFICIENT':
            recommendations.extend([
                "Increase sample size to improve statistical power",
                "Consider stronger experimental interventions to increase effect sizes",
                "Replicate experiments to improve reproducibility",
                "Review experimental design for potential confounding factors"
            ])
        
        elif overall_quality == 'ACCEPTABLE':
            recommendations.extend([
                "Consider additional validation methods to strengthen results",
                "Increase replication count for better reproducibility metrics",
                "Explore practical significance alongside statistical significance"
            ])
        
        elif overall_quality == 'GOOD':
            recommendations.extend([
                "Results are suitable for publication with minor improvements",
                "Consider meta-analysis if multiple studies are available",
                "Document limitations and potential confounding factors"
            ])
        
        elif overall_quality == 'EXCELLENT':
            recommendations.extend([
                "Results meet high standards for publication",
                "Consider extending to different hardware platforms",
                "Share data and methodology for reproducibility"
            ])
        
        # Specific recommendations based on validation results
        statistical_tests = validation_report['statistical_tests']
        for condition, tests in statistical_tests.items():
            for test_type, result in tests.items():
                if not result.is_significant:
                    recommendations.append(
                        f"Consider investigating why {condition} did not show significance in {test_type}"
                    )
                
                if result.power < 0.8:
                    recommendations.append(
                        f"Increase sample size for {condition} to achieve adequate power (current: {result.power:.2f})"
                    )
        
        return list(set(recommendations))  # Remove duplicates


class ResearchQualityGates:
    """Research-specific quality gates for breakthrough algorithms."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def validate_research_quality(
        self,
        validation_report: Dict[str, Any],
        requirements: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate research quality against specific requirements.
        
        Args:
            validation_report: Comprehensive validation report
            requirements: Optional custom requirements
            
        Returns:
            Quality gate assessment with pass/fail status
        """
        if requirements is None:
            requirements = self._get_default_research_requirements()
        
        quality_assessment = {
            'gates': {},
            'overall_status': 'UNKNOWN',
            'passed_gates': 0,
            'total_gates': 0,
            'recommendations': []
        }
        
        # Statistical significance gate
        quality_assessment['gates']['statistical_significance'] = self._check_statistical_significance_gate(
            validation_report, requirements
        )
        
        # Effect size gate
        quality_assessment['gates']['effect_size'] = self._check_effect_size_gate(
            validation_report, requirements
        )
        
        # Sample size gate
        quality_assessment['gates']['sample_size'] = self._check_sample_size_gate(
            validation_report, requirements
        )
        
        # Reproducibility gate
        quality_assessment['gates']['reproducibility'] = self._check_reproducibility_gate(
            validation_report, requirements
        )
        
        # Publication readiness gate
        quality_assessment['gates']['publication_readiness'] = self._check_publication_readiness_gate(
            validation_report, requirements
        )
        
        # Calculate overall status
        passed_gates = sum(1 for gate in quality_assessment['gates'].values() if gate['status'] == 'PASS')
        total_gates = len(quality_assessment['gates'])
        
        quality_assessment['passed_gates'] = passed_gates
        quality_assessment['total_gates'] = total_gates
        
        pass_rate = passed_gates / total_gates
        if pass_rate >= 0.9:
            quality_assessment['overall_status'] = 'EXCELLENT'
        elif pass_rate >= 0.8:
            quality_assessment['overall_status'] = 'GOOD'
        elif pass_rate >= 0.6:
            quality_assessment['overall_status'] = 'ACCEPTABLE'
        else:
            quality_assessment['overall_status'] = 'INSUFFICIENT'
        
        # Generate recommendations
        quality_assessment['recommendations'] = self._generate_quality_recommendations(
            quality_assessment['gates']
        )
        
        return quality_assessment
    
    def _get_default_research_requirements(self) -> Dict[str, Any]:
        """Get default research quality requirements."""
        return {
            'min_p_value': 0.05,
            'min_effect_size': 0.5,
            'min_sample_size': 30,
            'min_reproducibility_score': 0.7,
            'min_statistical_power': 0.8,
            'required_validation_methods': ['bootstrap', 'cross_validation'],
            'min_confidence_level': 0.95
        }
    
    def _check_statistical_significance_gate(
        self,
        validation_report: Dict[str, Any],
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check statistical significance quality gate."""
        gate_result = {
            'status': 'UNKNOWN',
            'details': {},
            'recommendations': []
        }
        
        statistical_tests = validation_report.get('statistical_tests', {})
        significant_tests = 0
        total_tests = 0
        
        for condition, tests in statistical_tests.items():
            for test_type, result in tests.items():
                total_tests += 1
                if result.p_value < requirements['min_p_value']:
                    significant_tests += 1
        
        significance_rate = significant_tests / max(1, total_tests)
        gate_result['details']['significance_rate'] = significance_rate
        gate_result['details']['significant_tests'] = significant_tests
        gate_result['details']['total_tests'] = total_tests
        
        if significance_rate >= 0.8:
            gate_result['status'] = 'PASS'
        elif significance_rate >= 0.6:
            gate_result['status'] = 'WARNING'
            gate_result['recommendations'].append("Some tests lack statistical significance")
        else:
            gate_result['status'] = 'FAIL'
            gate_result['recommendations'].append("Insufficient statistical significance across tests")
        
        return gate_result
    
    def _check_effect_size_gate(
        self,
        validation_report: Dict[str, Any],
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check effect size quality gate."""
        gate_result = {
            'status': 'UNKNOWN',
            'details': {},
            'recommendations': []
        }
        
        effect_sizes = validation_report.get('effect_sizes', {})
        adequate_effects = 0
        total_effects = 0
        
        for condition, effects in effect_sizes.items():
            for metric, effect_size in effects.items():
                total_effects += 1
                if abs(effect_size) >= requirements['min_effect_size']:
                    adequate_effects += 1
        
        effect_adequacy_rate = adequate_effects / max(1, total_effects)
        gate_result['details']['effect_adequacy_rate'] = effect_adequacy_rate
        gate_result['details']['adequate_effects'] = adequate_effects
        gate_result['details']['total_effects'] = total_effects
        
        if effect_adequacy_rate >= 0.8:
            gate_result['status'] = 'PASS'
        elif effect_adequacy_rate >= 0.6:
            gate_result['status'] = 'WARNING'
            gate_result['recommendations'].append("Some effect sizes are below practical significance threshold")
        else:
            gate_result['status'] = 'FAIL'
            gate_result['recommendations'].append("Effect sizes indicate limited practical significance")
        
        return gate_result
    
    def _check_sample_size_gate(
        self,
        validation_report: Dict[str, Any],
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check sample size adequacy gate."""
        gate_result = {
            'status': 'UNKNOWN',
            'details': {},
            'recommendations': []
        }
        
        statistical_tests = validation_report.get('statistical_tests', {})
        adequate_samples = 0
        total_conditions = 0
        
        for condition, tests in statistical_tests.items():
            total_conditions += 1
            condition_adequate = True
            
            for test_type, result in tests.items():
                if result.sample_size < requirements['min_sample_size']:
                    condition_adequate = False
                    break
            
            if condition_adequate:
                adequate_samples += 1
        
        sample_adequacy_rate = adequate_samples / max(1, total_conditions)
        gate_result['details']['sample_adequacy_rate'] = sample_adequacy_rate
        gate_result['details']['adequate_samples'] = adequate_samples
        gate_result['details']['total_conditions'] = total_conditions
        
        if sample_adequacy_rate >= 0.9:
            gate_result['status'] = 'PASS'
        elif sample_adequacy_rate >= 0.7:
            gate_result['status'] = 'WARNING'
            gate_result['recommendations'].append("Some conditions have limited sample sizes")
        else:
            gate_result['status'] = 'FAIL'
            gate_result['recommendations'].append("Insufficient sample sizes for reliable conclusions")
        
        return gate_result
    
    def _check_reproducibility_gate(
        self,
        validation_report: Dict[str, Any],
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check reproducibility quality gate."""
        gate_result = {
            'status': 'UNKNOWN',
            'details': {},
            'recommendations': []
        }
        
        reproducibility_metrics = validation_report.get('reproducibility_metrics', {})
        reproducible_conditions = 0
        total_conditions = len(reproducibility_metrics)
        
        for condition, metrics in reproducibility_metrics.items():
            if isinstance(metrics, dict) and 'reproducibility_score' in metrics:
                if metrics['reproducibility_score'] >= requirements['min_reproducibility_score']:
                    reproducible_conditions += 1
        
        reproducibility_rate = reproducible_conditions / max(1, total_conditions)
        gate_result['details']['reproducibility_rate'] = reproducibility_rate
        gate_result['details']['reproducible_conditions'] = reproducible_conditions
        gate_result['details']['total_conditions'] = total_conditions
        
        if reproducibility_rate >= 0.8:
            gate_result['status'] = 'PASS'
        elif reproducibility_rate >= 0.6:
            gate_result['status'] = 'WARNING'
            gate_result['recommendations'].append("Some conditions show limited reproducibility")
        else:
            gate_result['status'] = 'FAIL'
            gate_result['recommendations'].append("Results show poor reproducibility across conditions")
        
        return gate_result
    
    def _check_publication_readiness_gate(
        self,
        validation_report: Dict[str, Any],
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check publication readiness gate."""
        gate_result = {
            'status': 'UNKNOWN',
            'details': {},
            'recommendations': []
        }
        
        # Check multiple criteria for publication readiness
        criteria_met = 0
        total_criteria = 4
        
        # 1. Statistical significance
        statistical_tests = validation_report.get('statistical_tests', {})
        has_significant_results = any(
            result.is_significant
            for tests in statistical_tests.values()
            for result in tests.values()
        )
        if has_significant_results:
            criteria_met += 1
        else:
            gate_result['recommendations'].append("Lacks statistically significant results")
        
        # 2. Adequate effect sizes
        effect_sizes = validation_report.get('effect_sizes', {})
        has_adequate_effects = any(
            abs(effect_size) >= requirements['min_effect_size']
            for effects in effect_sizes.values()
            for effect_size in effects.values()
        )
        if has_adequate_effects:
            criteria_met += 1
        else:
            gate_result['recommendations'].append("Effect sizes below practical significance threshold")
        
        # 3. Validation methods
        validation_results = validation_report.get('validation_results', {})
        has_validation = len(validation_results) > 0
        if has_validation:
            criteria_met += 1
        else:
            gate_result['recommendations'].append("Lacks comprehensive validation methods")
        
        # 4. Quality gates
        quality_gates = validation_report.get('quality_gates', {})
        overall_quality = quality_gates.get('overall_quality', 'UNKNOWN')
        quality_acceptable = overall_quality in ['GOOD', 'EXCELLENT']
        if quality_acceptable:
            criteria_met += 1
        else:
            gate_result['recommendations'].append("Overall research quality needs improvement")
        
        readiness_score = criteria_met / total_criteria
        gate_result['details']['readiness_score'] = readiness_score
        gate_result['details']['criteria_met'] = criteria_met
        gate_result['details']['total_criteria'] = total_criteria
        
        if readiness_score >= 0.9:
            gate_result['status'] = 'PASS'
        elif readiness_score >= 0.7:
            gate_result['status'] = 'WARNING'
        else:
            gate_result['status'] = 'FAIL'
        
        return gate_result
    
    def _generate_quality_recommendations(
        self,
        gates: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        for gate_name, gate_result in gates.items():
            if gate_result['status'] in ['FAIL', 'WARNING']:
                recommendations.extend(gate_result.get('recommendations', []))
        
        # Add general recommendations
        failed_gates = [name for name, result in gates.items() if result['status'] == 'FAIL']
        if failed_gates:
            recommendations.append(f"Priority: Address failing quality gates: {', '.join(failed_gates)}")
        
        return list(set(recommendations))  # Remove duplicates


# Global functions for easy access

def get_experimental_validation_engine(
    config: Optional[ValidationConfiguration] = None
) -> ExperimentalValidationEngine:
    """Get instance of experimental validation engine."""
    if config is None:
        config = ValidationConfiguration(validation_method=ValidationMethod.BOOTSTRAP)
    return ExperimentalValidationEngine(config)


async def validate_breakthrough_experiment(
    experimental_results: Dict[str, Any],
    baseline_results: Dict[str, Any],
    validation_config: Optional[ValidationConfiguration] = None
) -> Dict[str, Any]:
    """
    Validate breakthrough experiment with comprehensive statistical analysis.
    
    Args:
        experimental_results: Results from breakthrough algorithms
        baseline_results: Baseline comparison results
        validation_config: Optional validation configuration
        
    Returns:
        Comprehensive validation report
    """
    if validation_config is None:
        validation_config = ValidationConfiguration(
            validation_method=ValidationMethod.BOOTSTRAP,
            n_bootstrap_samples=1000,
            confidence_level=0.95
        )
    
    engine = ExperimentalValidationEngine(validation_config)
    
    # Extract experimental data from results
    experimental_data = {}
    baseline_data = {}
    
    # Process algorithm results
    if 'algorithm_results' in experimental_results:
        for algorithm, results in experimental_results['algorithm_results'].items():
            if 'optimal_score' in results:
                experimental_data[algorithm] = [results['optimal_score']]
            if 'execution_time' in results:
                experimental_data[f"{algorithm}_time"] = [results['execution_time']]
    
    # Process baseline data
    if 'baseline_score' in baseline_results:
        for key in experimental_data:
            if not key.endswith('_time'):
                baseline_data[key] = [baseline_results['baseline_score']]
    
    if 'baseline_time' in baseline_results:
        for key in experimental_data:
            if key.endswith('_time'):
                baseline_data[key] = [baseline_results['baseline_time']]
    
    # Create experimental conditions
    conditions = [
        ExperimentalCondition(
            name=condition_name,
            parameters={},
            control_group=False
        ) for condition_name in experimental_data.keys()
    ]
    
    # Perform validation
    validation_report = await engine.validate_experimental_results(
        experimental_data, baseline_data, conditions
    )
    
    return validation_report


def assess_research_quality_gates(validation_report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess research quality gates for validation report.
    
    Args:
        validation_report: Comprehensive validation report
        
    Returns:
        Quality gate assessment
    """
    quality_gates = ResearchQualityGates()
    assessment = quality_gates.validate_research_quality(validation_report)
    
    return assessment