"""
Global Deployment Manager for Multi-Region Edge AI Profiling

Comprehensive global deployment system with:
1. Multi-region coordination and optimization
2. Edge location selection and load balancing
3. Data sovereignty and compliance management
4. Cross-platform compatibility layer
5. Intelligent regional failover and recovery
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import logging
import socket
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

from .i18n_manager import SupportedLanguage, RegionalCompliance


class GlobalRegion(Enum):
    """Global regions for deployment"""
    NORTH_AMERICA_EAST = "na-east"
    NORTH_AMERICA_WEST = "na-west" 
    EUROPE_WEST = "eu-west"
    EUROPE_CENTRAL = "eu-central"
    ASIA_PACIFIC_EAST = "ap-east"
    ASIA_PACIFIC_SOUTH = "ap-south"
    ASIA_PACIFIC_SOUTHEAST = "ap-southeast"
    SOUTH_AMERICA = "sa-east"
    MIDDLE_EAST = "me-central"
    AFRICA_SOUTH = "af-south"
    OCEANIA = "oc-southeast"
    
    @property
    def display_name(self) -> str:
        names = {
            "na-east": "North America East (Virginia)",
            "na-west": "North America West (California)",
            "eu-west": "Europe West (Ireland)",
            "eu-central": "Europe Central (Frankfurt)",
            "ap-east": "Asia Pacific East (Tokyo)",
            "ap-south": "Asia Pacific South (Mumbai)",
            "ap-southeast": "Asia Pacific Southeast (Singapore)",
            "sa-east": "South America (São Paulo)",
            "me-central": "Middle East (Bahrain)",
            "af-south": "Africa South (Cape Town)",
            "oc-southeast": "Oceania (Sydney)"
        }
        return names.get(self.value, self.value)


class DataSovereigntyLevel(Enum):
    """Data sovereignty requirements"""
    NONE = "none"
    REGIONAL_ONLY = "regional_only"
    COUNTRY_ONLY = "country_only"
    ON_PREMISES_ONLY = "on_premises_only"


class PlatformArchitecture(Enum):
    """Supported platform architectures"""
    X86_64 = "x86_64"
    ARM64 = "arm64"
    ARM32 = "arm32"
    RISC_V_64 = "riscv64"
    RISC_V_32 = "riscv32"
    XTENSA = "xtensa"
    MIPS = "mips"
    
    @property
    def display_name(self) -> str:
        names = {
            "x86_64": "Intel/AMD 64-bit",
            "arm64": "ARM 64-bit (AArch64)",
            "arm32": "ARM 32-bit",
            "riscv64": "RISC-V 64-bit",
            "riscv32": "RISC-V 32-bit", 
            "xtensa": "Xtensa (ESP32)",
            "mips": "MIPS"
        }
        return names.get(self.value, self.value)


@dataclass
class RegionInfo:
    """Information about a deployment region"""
    region: GlobalRegion
    primary_languages: List[SupportedLanguage]
    compliance_requirements: List[RegionalCompliance]
    data_sovereignty: DataSovereigntyLevel
    supported_platforms: List[PlatformArchitecture]
    edge_locations: List[str]
    latency_to_regions: Dict[GlobalRegion, float] = field(default_factory=dict)
    regulatory_notes: Optional[str] = None
    local_partnerships: List[str] = field(default_factory=list)
    timezone: str = "UTC"
    currency: str = "USD"
    
    def calculate_affinity_score(self, user_location: Tuple[float, float],
                                preferred_language: SupportedLanguage,
                                required_compliance: List[RegionalCompliance]) -> float:
        """Calculate region affinity score for user"""
        
        score = 0.0
        
        # Language affinity (30% weight)
        if preferred_language in self.primary_languages:
            score += 0.3
        
        # Compliance match (40% weight)
        compliance_matches = len(set(required_compliance) & set(self.compliance_requirements))
        total_required = len(required_compliance)
        if total_required > 0:
            compliance_score = compliance_matches / total_required
            score += 0.4 * compliance_score
        
        # Geographic proximity (30% weight) - simplified calculation
        region_centers = {
            GlobalRegion.NORTH_AMERICA_EAST: (39.0458, -76.6413),  # Virginia
            GlobalRegion.NORTH_AMERICA_WEST: (37.4419, -122.1430),  # California
            GlobalRegion.EUROPE_WEST: (53.4084, -6.1917),  # Ireland
            GlobalRegion.EUROPE_CENTRAL: (50.1109, 8.6821),  # Frankfurt
            GlobalRegion.ASIA_PACIFIC_EAST: (35.6762, 139.6503),  # Tokyo
            GlobalRegion.ASIA_PACIFIC_SOUTH: (19.0760, 72.8777),  # Mumbai
            GlobalRegion.ASIA_PACIFIC_SOUTHEAST: (1.3521, 103.8198),  # Singapore
            GlobalRegion.SOUTH_AMERICA: (-23.5505, -46.6333),  # São Paulo
            GlobalRegion.MIDDLE_EAST: (26.0667, 50.5577),  # Bahrain
            GlobalRegion.AFRICA_SOUTH: (-33.9249, 18.4241),  # Cape Town
            GlobalRegion.OCEANIA: (-33.8688, 151.2093)  # Sydney
        }
        
        if self.region in region_centers:
            region_lat, region_lon = region_centers[self.region]
            user_lat, user_lon = user_location
            
            # Simplified distance calculation (haversine approximation)
            lat_diff = abs(region_lat - user_lat)
            lon_diff = abs(region_lon - user_lon)
            distance = (lat_diff ** 2 + lon_diff ** 2) ** 0.5
            
            # Convert to proximity score (closer = higher score)
            proximity_score = max(0, 1 - distance / 180)  # Normalize by max distance
            score += 0.3 * proximity_score
        
        return min(1.0, score)


@dataclass
class DeploymentConfiguration:
    """Configuration for global deployment"""
    primary_region: GlobalRegion
    fallback_regions: List[GlobalRegion]
    supported_platforms: List[PlatformArchitecture]
    language_priorities: List[SupportedLanguage]
    compliance_requirements: List[RegionalCompliance]
    data_sovereignty: DataSovereigntyLevel
    performance_targets: Dict[str, float]
    cross_region_replication: bool = True
    auto_failover: bool = True
    load_balancing_strategy: str = "latency_weighted"
    edge_caching_enabled: bool = True


@dataclass  
class EdgeLocation:
    """Edge computing location information"""
    id: str
    region: GlobalRegion
    city: str
    country: str
    coordinates: Tuple[float, float]
    platforms: List[PlatformArchitecture]
    capacity_score: float  # 0-1 scale
    current_load: float  # 0-1 scale
    latency_to_users: Dict[str, float] = field(default_factory=dict)
    health_status: str = "healthy"  # healthy, degraded, unavailable
    compliance_certifications: List[RegionalCompliance] = field(default_factory=list)
    last_health_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class GlobalLoadBalancer:
    """Intelligent global load balancer for edge profiling"""
    
    def __init__(self):
        self.edge_locations: Dict[str, EdgeLocation] = {}
        self.routing_table: Dict[str, List[str]] = {}  # user_id -> ordered edge location IDs
        self.performance_metrics: Dict[str, Dict[str, float]] = {}
        self.health_check_interval = 30.0  # seconds
        
        # Initialize edge locations
        self._initialize_edge_locations()
    
    def _initialize_edge_locations(self):
        """Initialize global edge locations"""
        
        edge_configs = [
            # North America
            {
                "id": "na-east-1", "region": GlobalRegion.NORTH_AMERICA_EAST,
                "city": "Ashburn", "country": "USA", "coordinates": (39.0458, -77.4874),
                "platforms": [PlatformArchitecture.X86_64, PlatformArchitecture.ARM64],
                "capacity": 0.85, "compliance": [RegionalCompliance.CCPA]
            },
            {
                "id": "na-west-1", "region": GlobalRegion.NORTH_AMERICA_WEST,
                "city": "Palo Alto", "country": "USA", "coordinates": (37.4419, -122.1430),
                "platforms": [PlatformArchitecture.X86_64, PlatformArchitecture.ARM64],
                "capacity": 0.90, "compliance": [RegionalCompliance.CCPA]
            },
            
            # Europe
            {
                "id": "eu-west-1", "region": GlobalRegion.EUROPE_WEST,
                "city": "Dublin", "country": "Ireland", "coordinates": (53.3498, -6.2603),
                "platforms": [PlatformArchitecture.X86_64, PlatformArchitecture.ARM64],
                "capacity": 0.80, "compliance": [RegionalCompliance.GDPR]
            },
            {
                "id": "eu-central-1", "region": GlobalRegion.EUROPE_CENTRAL,
                "city": "Frankfurt", "country": "Germany", "coordinates": (50.1109, 8.6821),
                "platforms": [PlatformArchitecture.X86_64, PlatformArchitecture.ARM64, PlatformArchitecture.RISC_V_64],
                "capacity": 0.88, "compliance": [RegionalCompliance.GDPR]
            },
            
            # Asia Pacific
            {
                "id": "ap-east-1", "region": GlobalRegion.ASIA_PACIFIC_EAST,
                "city": "Tokyo", "country": "Japan", "coordinates": (35.6762, 139.6503),
                "platforms": [PlatformArchitecture.X86_64, PlatformArchitecture.ARM64],
                "capacity": 0.92, "compliance": []
            },
            {
                "id": "ap-southeast-1", "region": GlobalRegion.ASIA_PACIFIC_SOUTHEAST,
                "city": "Singapore", "country": "Singapore", "coordinates": (1.3521, 103.8198),
                "platforms": [PlatformArchitecture.X86_64, PlatformArchitecture.ARM64, PlatformArchitecture.ARM32],
                "capacity": 0.87, "compliance": [RegionalCompliance.PDPA]
            },
            {
                "id": "ap-south-1", "region": GlobalRegion.ASIA_PACIFIC_SOUTH,
                "city": "Mumbai", "country": "India", "coordinates": (19.0760, 72.8777),
                "platforms": [PlatformArchitecture.X86_64, PlatformArchitecture.ARM64, PlatformArchitecture.ARM32],
                "capacity": 0.75, "compliance": []
            },
            
            # Other regions
            {
                "id": "sa-east-1", "region": GlobalRegion.SOUTH_AMERICA,
                "city": "São Paulo", "country": "Brazil", "coordinates": (-23.5505, -46.6333),
                "platforms": [PlatformArchitecture.X86_64, PlatformArchitecture.ARM64],
                "capacity": 0.70, "compliance": [RegionalCompliance.LGPD]
            },
        ]
        
        for config in edge_configs:
            location = EdgeLocation(
                id=config["id"],
                region=config["region"],
                city=config["city"],
                country=config["country"],
                coordinates=config["coordinates"],
                platforms=config["platforms"],
                capacity_score=config["capacity"],
                current_load=0.1,  # Start with low load
                compliance_certifications=config["compliance"]
            )
            self.edge_locations[location.id] = location
    
    async def select_optimal_edge_location(self, 
                                         user_location: Optional[Tuple[float, float]] = None,
                                         platform_requirement: Optional[PlatformArchitecture] = None,
                                         compliance_requirements: Optional[List[RegionalCompliance]] = None,
                                         performance_priority: str = "latency") -> Optional[str]:
        """Select optimal edge location based on requirements"""
        
        candidates = []
        
        for location_id, location in self.edge_locations.items():
            # Health check
            if location.health_status != "healthy":
                continue
            
            # Platform compatibility
            if platform_requirement and platform_requirement not in location.platforms:
                continue
            
            # Compliance requirements
            if compliance_requirements:
                location_compliance = set(location.compliance_certifications)
                required_compliance = set(compliance_requirements)
                if not required_compliance.issubset(location_compliance):
                    continue
            
            # Calculate score based on priority
            score = await self._calculate_location_score(
                location, user_location, performance_priority
            )
            
            candidates.append((location_id, score))
        
        if not candidates:
            logging.warning("No suitable edge locations found")
            return None
        
        # Sort by score (higher is better)
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected_location = candidates[0][0]
        
        logging.info(f"Selected edge location: {selected_location} (score: {candidates[0][1]:.3f})")
        return selected_location
    
    async def _calculate_location_score(self, 
                                      location: EdgeLocation,
                                      user_location: Optional[Tuple[float, float]],
                                      performance_priority: str) -> float:
        """Calculate location score based on various factors"""
        
        score = 0.0
        
        # Capacity score (40% weight)
        available_capacity = location.capacity_score - location.current_load
        capacity_score = max(0, available_capacity)
        score += 0.4 * capacity_score
        
        # Geographic proximity (35% weight)
        if user_location:
            lat1, lon1 = user_location
            lat2, lon2 = location.coordinates
            
            # Simplified distance calculation
            distance = ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5
            proximity_score = max(0, 1 - distance / 180)  # Normalize
            score += 0.35 * proximity_score
        else:
            # Default proximity bonus for major locations
            major_locations = ["na-east-1", "eu-west-1", "ap-east-1"]
            if location.id in major_locations:
                score += 0.35 * 0.8
        
        # Performance history (25% weight)
        if location.id in self.performance_metrics:
            perf_data = self.performance_metrics[location.id]
            avg_latency = perf_data.get("avg_latency", 100)  # ms
            latency_score = max(0, 1 - avg_latency / 500)  # Normalize by 500ms
            score += 0.25 * latency_score
        else:
            score += 0.25 * 0.5  # Default score
        
        return min(1.0, score)
    
    async def update_location_load(self, location_id: str, load_change: float):
        """Update location load"""
        if location_id in self.edge_locations:
            location = self.edge_locations[location_id]
            location.current_load = max(0, min(1, location.current_load + load_change))
            
            # Log if location is getting overloaded
            if location.current_load > 0.9:
                logging.warning(f"Edge location {location_id} is overloaded: {location.current_load:.2%}")
    
    async def health_check_all_locations(self):
        """Perform health check on all edge locations"""
        
        async def check_location(location_id: str, location: EdgeLocation):
            try:
                # Simulate health check (replace with actual implementation)
                await asyncio.sleep(0.1)
                
                # Simple health simulation based on load
                if location.current_load > 0.95:
                    location.health_status = "unavailable"
                elif location.current_load > 0.85:
                    location.health_status = "degraded"
                else:
                    location.health_status = "healthy"
                
                location.last_health_check = datetime.now(timezone.utc)
                
                logging.debug(f"Health check for {location_id}: {location.health_status}")
                
            except Exception as e:
                logging.error(f"Health check failed for {location_id}: {e}")
                location.health_status = "unavailable"
        
        # Run health checks in parallel
        tasks = []
        for location_id, location in self.edge_locations.items():
            task = check_location(location_id, location)
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_region_status(self) -> Dict[str, Any]:
        """Get status of all regions"""
        
        status = {}
        
        for location_id, location in self.edge_locations.items():
            status[location_id] = {
                "region": location.region.value,
                "city": location.city,
                "country": location.country,
                "health": location.health_status,
                "load": f"{location.current_load:.1%}",
                "capacity": f"{location.capacity_score:.1%}",
                "platforms": [p.value for p in location.platforms],
                "compliance": [c.value for c in location.compliance_certifications],
                "last_check": location.last_health_check.isoformat()
            }
        
        return status


class ComplianceManager:
    """Manages regional compliance requirements"""
    
    def __init__(self):
        self.compliance_rules: Dict[RegionalCompliance, Dict[str, Any]] = {}
        self._initialize_compliance_rules()
    
    def _initialize_compliance_rules(self):
        """Initialize compliance rules for different regulations"""
        
        self.compliance_rules = {
            RegionalCompliance.GDPR: {
                "data_retention_days": 90,
                "consent_required": True,
                "data_minimization": True,
                "right_to_erasure": True,
                "data_portability": True,
                "privacy_by_design": True,
                "dpo_required": True,
                "breach_notification_hours": 72,
                "allowed_data_transfers": ["eu", "adequacy_decision_countries"],
                "lawful_basis_required": True
            },
            RegionalCompliance.CCPA: {
                "data_retention_days": 365,
                "opt_out_rights": True,
                "data_disclosure_required": True,
                "consumer_request_response_days": 45,
                "non_discrimination": True,
                "sale_opt_out": True,
                "data_categories_disclosure": True,
                "third_party_disclosure": True
            },
            RegionalCompliance.PDPA: {
                "data_retention_days": 365,
                "consent_required": True,
                "data_breach_notification": True,
                "data_protection_officer": False,
                "cross_border_transfer_restrictions": True,
                "data_subject_rights": True
            },
            RegionalCompliance.LGPD: {
                "data_retention_days": 180,
                "consent_required": True,
                "data_minimization": True,
                "data_protection_officer": True,
                "breach_notification_hours": 72,
                "data_subject_rights": True,
                "international_transfer_restrictions": True
            }
        }
    
    def check_compliance(self, 
                        region: GlobalRegion,
                        data_operation: str,
                        user_consent: bool = False) -> Dict[str, Any]:
        """Check compliance requirements for a data operation"""
        
        # Map regions to compliance requirements (simplified)
        region_compliance = {
            GlobalRegion.EUROPE_WEST: [RegionalCompliance.GDPR],
            GlobalRegion.EUROPE_CENTRAL: [RegionalCompliance.GDPR],
            GlobalRegion.NORTH_AMERICA_WEST: [RegionalCompliance.CCPA],
            GlobalRegion.NORTH_AMERICA_EAST: [RegionalCompliance.CCPA],
            GlobalRegion.ASIA_PACIFIC_SOUTHEAST: [RegionalCompliance.PDPA],
            GlobalRegion.SOUTH_AMERICA: [RegionalCompliance.LGPD]
        }
        
        applicable_rules = region_compliance.get(region, [])
        compliance_check = {
            "compliant": True,
            "violations": [],
            "recommendations": [],
            "required_actions": []
        }
        
        for rule in applicable_rules:
            if rule in self.compliance_rules:
                rules = self.compliance_rules[rule]
                
                # Check consent requirements
                if rules.get("consent_required", False) and not user_consent:
                    compliance_check["compliant"] = False
                    compliance_check["violations"].append(f"{rule.value}: Consent required but not obtained")
                    compliance_check["required_actions"].append("Obtain explicit user consent")
                
                # Check data retention
                if data_operation == "long_term_storage":
                    retention_days = rules.get("data_retention_days", 365)
                    compliance_check["recommendations"].append(
                        f"Data retention limit: {retention_days} days for {rule.value}"
                    )
                
                # Check data minimization
                if rules.get("data_minimization", False):
                    compliance_check["recommendations"].append(
                        f"Apply data minimization principles per {rule.value}"
                    )
        
        return compliance_check
    
    def get_data_retention_period(self, region: GlobalRegion) -> int:
        """Get maximum data retention period for region"""
        
        region_compliance = {
            GlobalRegion.EUROPE_WEST: [RegionalCompliance.GDPR],
            GlobalRegion.EUROPE_CENTRAL: [RegionalCompliance.GDPR],
            GlobalRegion.NORTH_AMERICA_WEST: [RegionalCompliance.CCPA],
            GlobalRegion.NORTH_AMERICA_EAST: [RegionalCompliance.CCPA],
            GlobalRegion.ASIA_PACIFIC_SOUTHEAST: [RegionalCompliance.PDPA],
            GlobalRegion.SOUTH_AMERICA: [RegionalCompliance.LGPD]
        }
        
        applicable_rules = region_compliance.get(region, [])
        
        if not applicable_rules:
            return 365  # Default 1 year
        
        # Return most restrictive retention period
        min_retention = float('inf')
        for rule in applicable_rules:
            if rule in self.compliance_rules:
                retention = self.compliance_rules[rule].get("data_retention_days", 365)
                min_retention = min(min_retention, retention)
        
        return int(min_retention) if min_retention != float('inf') else 365


class GlobalDeploymentManager:
    """Main manager for global deployment operations"""
    
    def __init__(self):
        self.load_balancer = GlobalLoadBalancer()
        self.compliance_manager = ComplianceManager()
        self.region_info: Dict[GlobalRegion, RegionInfo] = {}
        self.active_deployments: Dict[str, DeploymentConfiguration] = {}
        self.performance_monitor = GlobalPerformanceMonitor()
        
        # Initialize regions
        self._initialize_regions()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _initialize_regions(self):
        """Initialize region information"""
        
        regions_config = [
            {
                "region": GlobalRegion.NORTH_AMERICA_EAST,
                "languages": [SupportedLanguage.ENGLISH, SupportedLanguage.SPANISH, SupportedLanguage.FRENCH],
                "compliance": [RegionalCompliance.CCPA],
                "sovereignty": DataSovereigntyLevel.REGIONAL_ONLY,
                "platforms": [PlatformArchitecture.X86_64, PlatformArchitecture.ARM64],
                "edge_locations": ["na-east-1", "na-east-2"],
                "timezone": "America/New_York",
                "currency": "USD"
            },
            {
                "region": GlobalRegion.EUROPE_WEST,
                "languages": [SupportedLanguage.ENGLISH, SupportedLanguage.FRENCH, SupportedLanguage.GERMAN, SupportedLanguage.SPANISH],
                "compliance": [RegionalCompliance.GDPR],
                "sovereignty": DataSovereigntyLevel.REGIONAL_ONLY,
                "platforms": [PlatformArchitecture.X86_64, PlatformArchitecture.ARM64, PlatformArchitecture.RISC_V_64],
                "edge_locations": ["eu-west-1", "eu-west-2"],
                "timezone": "Europe/Dublin",
                "currency": "EUR"
            },
            {
                "region": GlobalRegion.ASIA_PACIFIC_EAST,
                "languages": [SupportedLanguage.JAPANESE, SupportedLanguage.KOREAN, SupportedLanguage.CHINESE_SIMPLIFIED],
                "compliance": [],
                "sovereignty": DataSovereigntyLevel.NONE,
                "platforms": [PlatformArchitecture.X86_64, PlatformArchitecture.ARM64],
                "edge_locations": ["ap-east-1", "ap-east-2"],
                "timezone": "Asia/Tokyo",
                "currency": "JPY"
            },
            {
                "region": GlobalRegion.ASIA_PACIFIC_SOUTHEAST,
                "languages": [SupportedLanguage.ENGLISH, SupportedLanguage.CHINESE_SIMPLIFIED, SupportedLanguage.MALAY, SupportedLanguage.THAI],
                "compliance": [RegionalCompliance.PDPA],
                "sovereignty": DataSovereigntyLevel.COUNTRY_ONLY,
                "platforms": [PlatformArchitecture.X86_64, PlatformArchitecture.ARM64, PlatformArchitecture.ARM32],
                "edge_locations": ["ap-southeast-1", "ap-southeast-2"],
                "timezone": "Asia/Singapore",
                "currency": "SGD"
            }
        ]
        
        for config in regions_config:
            region_info = RegionInfo(
                region=config["region"],
                primary_languages=config["languages"],
                compliance_requirements=config["compliance"],
                data_sovereignty=config["sovereignty"],
                supported_platforms=config["platforms"],
                edge_locations=config["edge_locations"],
                timezone=config["timezone"],
                currency=config["currency"]
            )
            
            self.region_info[config["region"]] = region_info
    
    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        
        async def background_health_monitor():
            while True:
                try:
                    await self.load_balancer.health_check_all_locations()
                    await asyncio.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    logging.error(f"Background health monitor error: {e}")
                    await asyncio.sleep(60)  # Wait longer on error
        
        # Schedule background task
        asyncio.create_task(background_health_monitor())
    
    async def create_global_deployment(self, 
                                     deployment_id: str,
                                     config: DeploymentConfiguration) -> bool:
        """Create a new global deployment"""
        
        try:
            # Validate configuration
            validation_result = await self._validate_deployment_config(config)
            if not validation_result["valid"]:
                logging.error(f"Deployment configuration invalid: {validation_result['errors']}")
                return False
            
            # Select optimal regions
            selected_regions = await self._select_optimal_regions(config)
            
            # Set up regional deployments
            deployment_tasks = []
            for region in selected_regions:
                task = self._deploy_to_region(deployment_id, region, config)
                deployment_tasks.append(task)
            
            results = await asyncio.gather(*deployment_tasks, return_exceptions=True)
            
            # Check if all deployments succeeded
            success_count = sum(1 for r in results if r is True)
            
            if success_count > 0:
                self.active_deployments[deployment_id] = config
                logging.info(f"Global deployment {deployment_id} created successfully in {success_count} regions")
                return True
            else:
                logging.error(f"Global deployment {deployment_id} failed in all regions")
                return False
                
        except Exception as e:
            logging.error(f"Failed to create global deployment {deployment_id}: {e}")
            return False
    
    async def _validate_deployment_config(self, config: DeploymentConfiguration) -> Dict[str, Any]:
        """Validate deployment configuration"""
        
        validation = {"valid": True, "errors": [], "warnings": []}
        
        # Check if primary region exists
        if config.primary_region not in self.region_info:
            validation["valid"] = False
            validation["errors"].append(f"Primary region {config.primary_region.value} not supported")
        
        # Check fallback regions
        for region in config.fallback_regions:
            if region not in self.region_info:
                validation["warnings"].append(f"Fallback region {region.value} not supported")
        
        # Check platform compatibility
        primary_region_info = self.region_info.get(config.primary_region)
        if primary_region_info:
            unsupported_platforms = set(config.supported_platforms) - set(primary_region_info.supported_platforms)
            if unsupported_platforms:
                validation["warnings"].append(f"Some platforms not supported in primary region: {unsupported_platforms}")
        
        # Check compliance requirements
        if config.compliance_requirements and primary_region_info:
            region_compliance = set(primary_region_info.compliance_requirements)
            required_compliance = set(config.compliance_requirements)
            missing_compliance = required_compliance - region_compliance
            if missing_compliance:
                validation["valid"] = False
                validation["errors"].append(f"Primary region doesn't meet compliance requirements: {missing_compliance}")
        
        return validation
    
    async def _select_optimal_regions(self, config: DeploymentConfiguration) -> List[GlobalRegion]:
        """Select optimal regions for deployment"""
        
        selected = [config.primary_region]
        
        # Add fallback regions based on requirements
        for region in config.fallback_regions:
            if region in self.region_info:
                region_info = self.region_info[region]
                
                # Check compliance compatibility
                if config.compliance_requirements:
                    region_compliance = set(region_info.compliance_requirements)
                    required_compliance = set(config.compliance_requirements)
                    if not required_compliance.issubset(region_compliance):
                        continue
                
                # Check platform support
                region_platforms = set(region_info.supported_platforms)
                required_platforms = set(config.supported_platforms)
                if not required_platforms.issubset(region_platforms):
                    continue
                
                selected.append(region)
        
        return selected
    
    async def _deploy_to_region(self, 
                              deployment_id: str,
                              region: GlobalRegion,
                              config: DeploymentConfiguration) -> bool:
        """Deploy to a specific region"""
        
        try:
            # Select edge location in region
            region_info = self.region_info[region]
            edge_location = await self.load_balancer.select_optimal_edge_location(
                platform_requirement=config.supported_platforms[0] if config.supported_platforms else None,
                compliance_requirements=config.compliance_requirements
            )
            
            if not edge_location:
                logging.warning(f"No suitable edge location found in region {region.value}")
                return False
            
            # Simulate deployment (replace with actual deployment logic)
            await asyncio.sleep(0.1)
            
            # Update load balancer
            await self.load_balancer.update_location_load(edge_location, 0.1)
            
            logging.info(f"Deployed {deployment_id} to region {region.value} at location {edge_location}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to deploy {deployment_id} to region {region.value}: {e}")
            return False
    
    async def route_profiling_request(self, 
                                    request_info: Dict[str, Any]) -> Optional[str]:
        """Route profiling request to optimal edge location"""
        
        user_location = request_info.get("user_location")  # (lat, lon) tuple
        platform_requirement = request_info.get("platform")
        compliance_requirements = request_info.get("compliance", [])
        
        # Select optimal edge location
        selected_location = await self.load_balancer.select_optimal_edge_location(
            user_location=user_location,
            platform_requirement=platform_requirement,
            compliance_requirements=compliance_requirements
        )
        
        if selected_location:
            # Update load
            await self.load_balancer.update_location_load(selected_location, 0.05)
            
            # Record routing decision
            logging.info(f"Routed profiling request to {selected_location}")
        
        return selected_location
    
    def get_global_status(self) -> Dict[str, Any]:
        """Get global deployment status"""
        
        return {
            "regions": {
                region.value: {
                    "name": region.display_name,
                    "languages": [lang.native_name for lang in info.primary_languages],
                    "compliance": [comp.value for comp in info.compliance_requirements],
                    "platforms": [plat.display_name for plat in info.supported_platforms],
                    "edge_locations": info.edge_locations,
                    "data_sovereignty": info.data_sovereignty.value
                }
                for region, info in self.region_info.items()
            },
            "edge_locations": self.load_balancer.get_region_status(),
            "active_deployments": len(self.active_deployments),
            "total_capacity": sum(loc.capacity_score for loc in self.load_balancer.edge_locations.values()),
            "average_load": sum(loc.current_load for loc in self.load_balancer.edge_locations.values()) / len(self.load_balancer.edge_locations) if self.load_balancer.edge_locations else 0
        }


class GlobalPerformanceMonitor:
    """Monitor global performance across regions"""
    
    def __init__(self):
        self.performance_data: Dict[str, List[Dict[str, Any]]] = {}
        self.alert_thresholds = {
            "latency_ms": 500,
            "error_rate": 0.05,
            "availability": 0.99
        }
    
    async def record_performance_metric(self, 
                                      location_id: str,
                                      metric_type: str,
                                      value: float,
                                      timestamp: Optional[datetime] = None):
        """Record performance metric for location"""
        
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        if location_id not in self.performance_data:
            self.performance_data[location_id] = []
        
        metric_entry = {
            "timestamp": timestamp.isoformat(),
            "type": metric_type,
            "value": value
        }
        
        self.performance_data[location_id].append(metric_entry)
        
        # Keep only recent data (last 24 hours)
        cutoff_time = datetime.now(timezone.utc).timestamp() - 86400
        self.performance_data[location_id] = [
            entry for entry in self.performance_data[location_id]
            if datetime.fromisoformat(entry["timestamp"].replace('Z', '+00:00')).timestamp() > cutoff_time
        ]
        
        # Check for alerts
        await self._check_performance_alerts(location_id, metric_type, value)
    
    async def _check_performance_alerts(self, location_id: str, metric_type: str, value: float):
        """Check if performance metric triggers alert"""
        
        threshold = self.alert_thresholds.get(metric_type)
        if threshold is None:
            return
        
        alert_triggered = False
        
        if metric_type == "latency_ms" and value > threshold:
            alert_triggered = True
        elif metric_type == "error_rate" and value > threshold:
            alert_triggered = True
        elif metric_type == "availability" and value < threshold:
            alert_triggered = True
        
        if alert_triggered:
            logging.warning(f"Performance alert for {location_id}: {metric_type} = {value} (threshold: {threshold})")
    
    def get_performance_summary(self, location_id: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary for location or globally"""
        
        if location_id:
            return self._get_location_performance(location_id)
        else:
            return self._get_global_performance()
    
    def _get_location_performance(self, location_id: str) -> Dict[str, Any]:
        """Get performance summary for specific location"""
        
        if location_id not in self.performance_data:
            return {"location": location_id, "metrics": {}, "status": "no_data"}
        
        data = self.performance_data[location_id]
        metrics = {}
        
        # Group by metric type and calculate averages
        metric_groups = {}
        for entry in data:
            metric_type = entry["type"]
            if metric_type not in metric_groups:
                metric_groups[metric_type] = []
            metric_groups[metric_type].append(entry["value"])
        
        for metric_type, values in metric_groups.items():
            if values:
                metrics[metric_type] = {
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
        
        return {
            "location": location_id,
            "metrics": metrics,
            "status": "active",
            "last_updated": data[-1]["timestamp"] if data else None
        }
    
    def _get_global_performance(self) -> Dict[str, Any]:
        """Get global performance summary"""
        
        global_metrics = {}
        location_count = len(self.performance_data)
        
        if location_count == 0:
            return {"global": True, "metrics": {}, "locations": 0}
        
        # Aggregate metrics across all locations
        all_metrics = {}
        for location_data in self.performance_data.values():
            for entry in location_data:
                metric_type = entry["type"]
                if metric_type not in all_metrics:
                    all_metrics[metric_type] = []
                all_metrics[metric_type].append(entry["value"])
        
        for metric_type, values in all_metrics.items():
            if values:
                global_metrics[metric_type] = {
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                    "locations": location_count
                }
        
        return {
            "global": True,
            "metrics": global_metrics,
            "locations": location_count,
            "status": "active"
        }


# Global instance for easy access
_global_deployment_manager: Optional[GlobalDeploymentManager] = None

def get_global_deployment_manager() -> GlobalDeploymentManager:
    """Get global deployment manager instance"""
    global _global_deployment_manager
    if _global_deployment_manager is None:
        _global_deployment_manager = GlobalDeploymentManager()
    return _global_deployment_manager

async def deploy_globally(deployment_id: str, config: DeploymentConfiguration) -> bool:
    """Deploy profiler globally with given configuration"""
    manager = get_global_deployment_manager()
    return await manager.create_global_deployment(deployment_id, config)

async def route_profiling_request(request_info: Dict[str, Any]) -> Optional[str]:
    """Route profiling request to optimal location"""
    manager = get_global_deployment_manager()
    return await manager.route_profiling_request(request_info)

def get_global_status() -> Dict[str, Any]:
    """Get global deployment status"""
    manager = get_global_deployment_manager()
    return manager.get_global_status()