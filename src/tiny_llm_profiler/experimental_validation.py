"""
Experimental Validation Framework
Advanced validation system for edge AI research with statistical rigor.
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from .exceptions import ProfilerError
from .research_framework import ResearchExperiment, ResearchResults


@dataclass
class ValidationConfiguration:
    """Configuration for experimental validation."""
    statistical_power: float = 0.8
    significance_level: float = 0.05
    effect_size_threshold: float = 0.3
    minimum_sample_size: int = 30
    bootstrap_iterations: int = 10000
    cross_validation_folds: int = 5
    reproducibility_threshold: float = 0.95


@dataclass
class ExperimentalResult:
    """Single experimental result with metadata."""
    experiment_id: str
    condition_id: str
    timestamp: float
    measurements: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_status: str = "pending"


class StatisticalValidator:
    """Comprehensive statistical validation for experimental results."""
    
    def __init__(self, config: ValidationConfiguration):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def validate_statistical_significance(
        self,
        control_group: List[float],
        treatment_group: List[float],
        test_type: str = "welch_t_test"
    ) -> Dict[str, Any]:
        """
        Validate statistical significance between groups.
        
        Args:
            control_group: Control group measurements
            treatment_group: Treatment group measurements  
            test_type: Type of statistical test to perform
            
        Returns:
            Statistical test results with validation status
        """
        if len(control_group) < self.config.minimum_sample_size:
            return {
                "valid": False,
                "reason": f"Control group size {len(control_group)} below minimum {self.config.minimum_sample_size}"
            }
        
        if len(treatment_group) < self.config.minimum_sample_size:
            return {
                "valid": False,
                "reason": f"Treatment group size {len(treatment_group)} below minimum {self.config.minimum_sample_size}"
            }
        
        # Perform statistical test
        if test_type == "welch_t_test":
            statistic, p_value = stats.ttest_ind(
                treatment_group, control_group, equal_var=False
            )
        elif test_type == "mann_whitney":
            statistic, p_value = stats.mannwhitneyu(
                treatment_group, control_group, alternative='two-sided'
            )
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(control_group) - 1) * np.var(control_group, ddof=1) +
             (len(treatment_group) - 1) * np.var(treatment_group, ddof=1)) /
            (len(control_group) + len(treatment_group) - 2)
        )
        
        effect_size = (np.mean(treatment_group) - np.mean(control_group)) / pooled_std
        
        # Calculate confidence intervals
        confidence_interval = stats.t.interval(
            1 - self.config.significance_level,
            len(treatment_group) + len(control_group) - 2,
            loc=np.mean(treatment_group) - np.mean(control_group),
            scale=pooled_std * np.sqrt(1/len(treatment_group) + 1/len(control_group))
        )
        
        # Determine validation status
        is_significant = p_value < self.config.significance_level
        has_meaningful_effect = abs(effect_size) >= self.config.effect_size_threshold
        
        return {
            "valid": is_significant and has_meaningful_effect,
            "p_value": p_value,
            "effect_size": effect_size,
            "confidence_interval": confidence_interval,
            "statistic": statistic,
            "test_type": test_type,
            "is_significant": is_significant,
            "has_meaningful_effect": has_meaningful_effect,
            "sample_sizes": {
                "control": len(control_group),
                "treatment": len(treatment_group)
            }
        }
    
    def validate_reproducibility(
        self,
        experimental_runs: List[List[float]],
        metric_name: str
    ) -> Dict[str, Any]:
        """
        Validate reproducibility across multiple experimental runs.
        
        Args:
            experimental_runs: List of measurement lists from different runs
            metric_name: Name of the metric being validated
            
        Returns:
            Reproducibility validation results
        """
        if len(experimental_runs) < 3:
            return {
                "valid": False,
                "reason": "Minimum 3 experimental runs required for reproducibility validation"
            }
        
        # Calculate statistics for each run
        run_means = [np.mean(run) for run in experimental_runs]
        run_stds = [np.std(run, ddof=1) for run in experimental_runs]
        
        # Calculate overall statistics
        overall_mean = np.mean(run_means)
        between_run_variance = np.var(run_means, ddof=1)
        within_run_variance = np.mean([std**2 for std in run_stds])
        
        # Calculate intraclass correlation coefficient (ICC)
        total_variance = between_run_variance + within_run_variance
        icc = between_run_variance / total_variance if total_variance > 0 else 0
        
        # Calculate coefficient of variation
        cv = np.std(run_means, ddof=1) / overall_mean if overall_mean != 0 else float('inf')
        
        # Determine reproducibility status
        is_reproducible = icc >= self.config.reproducibility_threshold and cv <= 0.1
        
        return {
            "valid": is_reproducible,
            "icc": icc,
            "coefficient_of_variation": cv,
            "between_run_variance": between_run_variance,
            "within_run_variance": within_run_variance,
            "run_means": run_means,
            "run_stds": run_stds,
            "is_reproducible": is_reproducible,
            "reproducibility_threshold": self.config.reproducibility_threshold
        }
    
    def perform_power_analysis(
        self,
        effect_size: float,
        sample_size: int,
        significance_level: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Perform statistical power analysis.
        
        Args:
            effect_size: Expected effect size
            sample_size: Sample size per group
            significance_level: Significance level (alpha)
            
        Returns:
            Power analysis results
        """
        alpha = significance_level or self.config.significance_level
        
        # Calculate statistical power using approximation
        # This is a simplified calculation - in practice, you'd use specialized libraries
        critical_value = stats.norm.ppf(1 - alpha/2)
        power = 1 - stats.norm.cdf(
            critical_value - effect_size * np.sqrt(sample_size / 2)
        ) + stats.norm.cdf(
            -critical_value - effect_size * np.sqrt(sample_size / 2)
        )
        
        # Calculate required sample size for desired power
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(self.config.statistical_power)
        required_sample_size = int(np.ceil(
            2 * ((z_alpha + z_beta) / effect_size) ** 2
        ))
        
        return {
            "calculated_power": power,
            "required_power": self.config.statistical_power,
            "meets_power_requirement": power >= self.config.statistical_power,
            "current_sample_size": sample_size,
            "required_sample_size": required_sample_size,
            "effect_size": effect_size,
            "significance_level": alpha
        }


class CrossValidationFramework:
    """Cross-validation framework for robust experimental validation."""
    
    def __init__(self, config: ValidationConfiguration):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def perform_k_fold_validation(
        self,
        experiment_function: Callable,
        experimental_conditions: List[Dict[str, Any]],
        k_folds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation of experimental results.
        
        Args:
            experiment_function: Function that executes the experiment
            experimental_conditions: List of experimental conditions
            k_folds: Number of folds (default from config)
            
        Returns:
            Cross-validation results
        """
        folds = k_folds or self.config.cross_validation_folds
        
        # Split conditions into folds
        fold_size = len(experimental_conditions) // folds
        fold_results = []
        
        for fold_idx in range(folds):
            start_idx = fold_idx * fold_size
            end_idx = start_idx + fold_size if fold_idx < folds - 1 else len(experimental_conditions)
            
            test_conditions = experimental_conditions[start_idx:end_idx]
            train_conditions = (
                experimental_conditions[:start_idx] + 
                experimental_conditions[end_idx:]
            )
            
            # Execute fold
            fold_result = await self._execute_fold(
                experiment_function, train_conditions, test_conditions, fold_idx
            )
            fold_results.append(fold_result)
        
        # Aggregate results
        aggregated_results = self._aggregate_fold_results(fold_results)
        
        return {
            "fold_results": fold_results,
            "aggregated_results": aggregated_results,
            "cross_validation_score": aggregated_results.get("mean_performance", 0),
            "score_variance": aggregated_results.get("performance_variance", 0),
            "validation_stability": self._calculate_validation_stability(fold_results)
        }
    
    async def _execute_fold(
        self,
        experiment_function: Callable,
        train_conditions: List[Dict[str, Any]],
        test_conditions: List[Dict[str, Any]],
        fold_idx: int
    ) -> Dict[str, Any]:
        """Execute a single fold of cross-validation."""
        self.logger.info(f"Executing fold {fold_idx + 1}")
        
        # Train on training conditions
        train_results = []
        for condition in train_conditions:
            result = await experiment_function(condition)
            train_results.append(result)
        
        # Test on test conditions
        test_results = []
        for condition in test_conditions:
            result = await experiment_function(condition)
            test_results.append(result)
        
        return {
            "fold_index": fold_idx,
            "train_results": train_results,
            "test_results": test_results,
            "train_performance": np.mean([r.get('performance', 0) for r in train_results]),
            "test_performance": np.mean([r.get('performance', 0) for r in test_results])
        }
    
    def _aggregate_fold_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across all folds."""
        train_performances = [fold['train_performance'] for fold in fold_results]
        test_performances = [fold['test_performance'] for fold in fold_results]
        
        return {
            "mean_train_performance": np.mean(train_performances),
            "mean_test_performance": np.mean(test_performances),
            "train_performance_std": np.std(train_performances, ddof=1),
            "test_performance_std": np.std(test_performances, ddof=1),
            "performance_variance": np.var(test_performances, ddof=1),
            "mean_performance": np.mean(test_performances)
        }
    
    def _calculate_validation_stability(self, fold_results: List[Dict[str, Any]]) -> float:
        """Calculate stability of cross-validation results."""
        test_performances = [fold['test_performance'] for fold in fold_results]
        cv = np.std(test_performances, ddof=1) / np.mean(test_performances)
        return 1.0 - min(1.0, cv)  # Higher values indicate more stable validation


class BootstrapValidator:
    """Bootstrap validation for confidence interval estimation."""
    
    def __init__(self, config: ValidationConfiguration):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def bootstrap_confidence_interval(
        self,
        data: List[float],
        statistic_function: Callable[[List[float]], float] = np.mean,
        confidence_level: float = 0.95,
        n_bootstrap: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calculate bootstrap confidence intervals.
        
        Args:
            data: Original data sample
            statistic_function: Function to calculate statistic
            confidence_level: Confidence level for interval
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Bootstrap confidence interval results
        """
        n_bootstrap = n_bootstrap or self.config.bootstrap_iterations
        
        # Generate bootstrap samples
        bootstrap_statistics = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            statistic = statistic_function(bootstrap_sample)
            bootstrap_statistics.append(statistic)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        confidence_interval = (
            np.percentile(bootstrap_statistics, lower_percentile),
            np.percentile(bootstrap_statistics, upper_percentile)
        )
        
        # Calculate bias-corrected interval
        original_statistic = statistic_function(data)
        bias_correction = 2 * original_statistic - np.mean(bootstrap_statistics)
        
        return {
            "confidence_interval": confidence_interval,
            "original_statistic": original_statistic,
            "bootstrap_mean": np.mean(bootstrap_statistics),
            "bootstrap_std": np.std(bootstrap_statistics, ddof=1),
            "bias_correction": bias_correction,
            "confidence_level": confidence_level,
            "n_bootstrap": n_bootstrap
        }
    
    def bootstrap_hypothesis_test(
        self,
        group1: List[float],
        group2: List[float],
        test_statistic: Callable[[List[float], List[float]], float],
        n_bootstrap: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform bootstrap hypothesis test.
        
        Args:
            group1: First group data
            group2: Second group data
            test_statistic: Function to calculate test statistic
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Bootstrap hypothesis test results
        """
        n_bootstrap = n_bootstrap or self.config.bootstrap_iterations
        
        # Calculate observed test statistic
        observed_statistic = test_statistic(group1, group2)
        
        # Combine groups under null hypothesis
        combined_data = group1 + group2
        n1, n2 = len(group1), len(group2)
        
        # Bootstrap under null hypothesis
        bootstrap_statistics = []
        for _ in range(n_bootstrap):
            # Resample from combined data
            resampled = np.random.choice(combined_data, size=len(combined_data), replace=True)
            
            # Split into groups of original sizes
            bootstrap_group1 = resampled[:n1].tolist()
            bootstrap_group2 = resampled[n1:n1+n2].tolist()
            
            # Calculate test statistic
            bootstrap_stat = test_statistic(bootstrap_group1, bootstrap_group2)
            bootstrap_statistics.append(bootstrap_stat)
        
        # Calculate p-value
        extreme_count = sum(1 for stat in bootstrap_statistics if abs(stat) >= abs(observed_statistic))
        p_value = extreme_count / n_bootstrap
        
        return {
            "observed_statistic": observed_statistic,
            "bootstrap_distribution": bootstrap_statistics,
            "p_value": p_value,
            "is_significant": p_value < self.config.significance_level,
            "n_bootstrap": n_bootstrap
        }


class ExperimentalValidationEngine:
    """Comprehensive experimental validation engine."""
    
    def __init__(self, config: Optional[ValidationConfiguration] = None):
        self.config = config or ValidationConfiguration()
        self.statistical_validator = StatisticalValidator(self.config)
        self.cross_validator = CrossValidationFramework(self.config)
        self.bootstrap_validator = BootstrapValidator(self.config)
        self.logger = logging.getLogger(__name__)
        
        self.validation_history = []
        self.validation_cache = {}
    
    async def validate_experimental_results(
        self,
        experiment_results: ResearchResults,
        validation_requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive validation of experimental results.
        
        Args:
            experiment_results: Results from research experiment
            validation_requirements: Specific validation requirements
            
        Returns:
            Comprehensive validation report
        """
        self.logger.info(f"Validating experiment: {experiment_results.experiment_id}")
        
        validation_report = {
            "experiment_id": experiment_results.experiment_id,
            "validation_timestamp": time.time(),
            "validation_config": self.config.__dict__,
            "statistical_validation": {},
            "reproducibility_validation": {},
            "cross_validation": {},
            "bootstrap_validation": {},
            "overall_validity": False,
            "validation_summary": {}
        }
        
        # Statistical significance validation
        validation_report["statistical_validation"] = await self._validate_statistical_significance(
            experiment_results, validation_requirements.get("statistical", {})
        )
        
        # Reproducibility validation
        validation_report["reproducibility_validation"] = await self._validate_reproducibility(
            experiment_results, validation_requirements.get("reproducibility", {})
        )
        
        # Cross-validation if required
        if validation_requirements.get("cross_validation", {}).get("enabled", False):
            validation_report["cross_validation"] = await self._perform_cross_validation(
                experiment_results, validation_requirements["cross_validation"]
            )
        
        # Bootstrap validation if required
        if validation_requirements.get("bootstrap", {}).get("enabled", False):
            validation_report["bootstrap_validation"] = await self._perform_bootstrap_validation(
                experiment_results, validation_requirements["bootstrap"]
            )
        
        # Determine overall validity
        validation_report["overall_validity"] = self._determine_overall_validity(validation_report)
        
        # Generate validation summary
        validation_report["validation_summary"] = self._generate_validation_summary(validation_report)
        
        # Store validation history
        self.validation_history.append(validation_report)
        
        return validation_report
    
    async def _validate_statistical_significance(
        self,
        experiment_results: ResearchResults,
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate statistical significance of experimental results."""
        validation_results = {}
        
        for novel_name, novel_data in experiment_results.novel_results.items():
            for baseline_name, baseline_data in experiment_results.baseline_results.items():
                comparison_key = f"{novel_name}_vs_{baseline_name}"
                
                # Extract metrics for comparison
                for metric in ['latency_ms', 'memory_kb', 'energy_mj']:
                    if metric in novel_data[0] and metric in baseline_data[0]:
                        novel_values = [result[metric] for result in novel_data]
                        baseline_values = [result[metric] for result in baseline_data]
                        
                        test_key = f"{comparison_key}_{metric}"
                        validation_results[test_key] = self.statistical_validator.validate_statistical_significance(
                            baseline_values, novel_values, requirements.get("test_type", "welch_t_test")
                        )
        
        return validation_results
    
    async def _validate_reproducibility(
        self,
        experiment_results: ResearchResults,
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate reproducibility of experimental results."""
        validation_results = {}
        
        # Group results by runs
        for algorithm_name, results in experiment_results.novel_results.items():
            # Simulate multiple runs by grouping results
            runs_per_metric = {}
            for metric in ['latency_ms', 'memory_kb', 'energy_mj']:
                if results and metric in results[0]:
                    # Split results into runs (simplified)
                    run_size = len(results) // 3  # Assume 3 runs
                    runs = [
                        [results[i][metric] for i in range(j*run_size, (j+1)*run_size)]
                        for j in range(3)
                        if (j+1)*run_size <= len(results)
                    ]
                    
                    if len(runs) >= 3:
                        runs_per_metric[metric] = runs
            
            # Validate reproducibility for each metric
            for metric, runs in runs_per_metric.items():
                validation_key = f"{algorithm_name}_{metric}"
                validation_results[validation_key] = self.statistical_validator.validate_reproducibility(
                    runs, metric
                )
        
        return validation_results
    
    async def _perform_cross_validation(
        self,
        experiment_results: ResearchResults,
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform cross-validation if specified in requirements."""
        # This would typically require re-running experiments
        # For now, return a placeholder
        return {
            "enabled": True,
            "note": "Cross-validation would require re-execution of experiments",
            "recommendation": "Implement experiment re-execution for full cross-validation"
        }
    
    async def _perform_bootstrap_validation(
        self,
        experiment_results: ResearchResults,
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform bootstrap validation if specified in requirements."""
        validation_results = {}
        
        for algorithm_name, results in experiment_results.novel_results.items():
            for metric in ['latency_ms', 'memory_kb', 'energy_mj']:
                if results and metric in results[0]:
                    values = [result[metric] for result in results]
                    
                    validation_key = f"{algorithm_name}_{metric}_bootstrap"
                    validation_results[validation_key] = self.bootstrap_validator.bootstrap_confidence_interval(
                        values, confidence_level=requirements.get("confidence_level", 0.95)
                    )
        
        return validation_results
    
    def _determine_overall_validity(self, validation_report: Dict[str, Any]) -> bool:
        """Determine overall validity based on all validation results."""
        # Check statistical significance
        stat_valid = all(
            result.get("valid", False)
            for result in validation_report["statistical_validation"].values()
        )
        
        # Check reproducibility
        repro_valid = all(
            result.get("valid", False)
            for result in validation_report["reproducibility_validation"].values()
        )
        
        # Overall validity requires both statistical significance and reproducibility
        return stat_valid and repro_valid
    
    def _generate_validation_summary(self, validation_report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of validation results."""
        stat_results = validation_report["statistical_validation"]
        repro_results = validation_report["reproducibility_validation"]
        
        return {
            "total_statistical_tests": len(stat_results),
            "significant_tests": sum(1 for r in stat_results.values() if r.get("is_significant", False)),
            "total_reproducibility_tests": len(repro_results),
            "reproducible_tests": sum(1 for r in repro_results.values() if r.get("is_reproducible", False)),
            "overall_valid": validation_report["overall_validity"],
            "validation_score": self._calculate_validation_score(validation_report),
            "recommendations": self._generate_validation_recommendations(validation_report)
        }
    
    def _calculate_validation_score(self, validation_report: Dict[str, Any]) -> float:
        """Calculate overall validation score (0-1)."""
        stat_results = validation_report["statistical_validation"]
        repro_results = validation_report["reproducibility_validation"]
        
        if not stat_results and not repro_results:
            return 0.0
        
        stat_score = sum(1 for r in stat_results.values() if r.get("valid", False)) / len(stat_results) if stat_results else 0
        repro_score = sum(1 for r in repro_results.values() if r.get("valid", False)) / len(repro_results) if repro_results else 0
        
        return (stat_score + repro_score) / 2
    
    def _generate_validation_recommendations(self, validation_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if not validation_report["overall_validity"]:
            recommendations.append("Results do not meet validation criteria for publication")
        
        stat_score = validation_report["validation_summary"]["validation_score"]
        if stat_score < 0.8:
            recommendations.append("Consider increasing sample size or effect size")
        
        if not validation_report["statistical_validation"]:
            recommendations.append("Statistical significance validation required")
        
        if not validation_report["reproducibility_validation"]:
            recommendations.append("Reproducibility validation strongly recommended")
        
        return recommendations