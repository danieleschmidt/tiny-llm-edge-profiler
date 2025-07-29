#!/usr/bin/env python3
"""
Software Bill of Materials (SBOM) generation script for tiny-llm-edge-profiler.

This script generates comprehensive SBOM reports in multiple formats
for supply chain security and compliance.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from cyclonedx.builder.this import this_component
from cyclonedx.factory.license import LicenseFactory
from cyclonedx.model import (
    Bom,
    Component,
    ComponentType,
    HashType,
    License,
    LicenseAcknowledgement,
    Property,
    Tool,
)
from cyclonedx.model.component import ComponentScope
from cyclonedx.output import get_instance as get_outputter
from cyclonedx.schema import SchemaVersion


class SBOMGenerator:
    """Generate SBOM for tiny-llm-edge-profiler project."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.license_factory = LicenseFactory()
        
    def get_project_metadata(self) -> Dict[str, Any]:
        """Extract project metadata from pyproject.toml."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
            
        with open(self.project_root / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)
            
        return data.get("project", {})
    
    def get_installed_packages(self) -> List[Dict[str, str]]:
        """Get list of installed Python packages with versions."""
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            check=True
        )
        return json.loads(result.stdout)
    
    def get_package_info(self, package_name: str) -> Dict[str, Any]:
        """Get detailed package information."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", "--verbose", package_name],
                capture_output=True,
                text=True,
                check=True
            )
            
            info = {}
            for line in result.stdout.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    info[key.strip().lower()] = value.strip()
            
            return info
        except subprocess.CalledProcessError:
            return {}
    
    def create_component_from_package(self, package: Dict[str, str]) -> Component:
        """Create CycloneDX component from package info."""
        name = package["name"]
        version = package["version"]
        
        # Get detailed package information
        info = self.get_package_info(name)
        
        component = Component(
            type=ComponentType.LIBRARY,
            name=name,
            version=version,
            scope=ComponentScope.REQUIRED,
        )
        
        # Add description if available
        if "summary" in info:
            component.description = info["summary"]
        
        # Add homepage if available
        if "home-page" in info:
            component.external_references.add_website(info["home-page"])
        
        # Add license information
        if "license" in info and info["license"]:
            try:
                license_obj = self.license_factory.make_from_string(info["license"])
                component.licenses.add(license_obj)
            except Exception:
                # Fallback for unknown license strings
                component.licenses.add(License(license_name=info["license"]))
        
        # Add author information as properties
        if "author" in info:
            component.properties.add(Property(name="author", value=info["author"]))
        
        if "author-email" in info:
            component.properties.add(Property(name="author-email", value=info["author-email"]))
        
        return component
    
    def create_main_component(self) -> Component:
        """Create the main component representing this project."""
        metadata = self.get_project_metadata()
        
        component = Component(
            type=ComponentType.APPLICATION,
            name=metadata.get("name", "tiny-llm-edge-profiler"),
            version=metadata.get("version", "0.1.0"),
            description=metadata.get("description", ""),
            scope=ComponentScope.REQUIRED,
        )
        
        # Add project URLs
        urls = metadata.get("urls", {})
        if "homepage" in urls:
            component.external_references.add_website(urls["homepage"])
        if "repository" in urls:
            component.external_references.add_vcs(urls["repository"])
        if "documentation" in urls:
            component.external_references.add_documentation(urls["documentation"])
        
        # Add license
        try:
            with open(self.project_root / "LICENSE", "r") as f:
                license_text = f.read()
            if "Apache License" in license_text:
                component.licenses.add(self.license_factory.make_from_string("Apache-2.0"))
        except FileNotFoundError:
            pass
        
        # Add authors as properties
        authors = metadata.get("authors", [])
        for i, author in enumerate(authors):
            if "name" in author:
                component.properties.add(Property(
                    name=f"author-{i+1}-name", 
                    value=author["name"]
                ))
            if "email" in author:
                component.properties.add(Property(
                    name=f"author-{i+1}-email", 
                    value=author["email"]
                ))
        
        # Add Python version requirement
        if "requires-python" in metadata:
            component.properties.add(Property(
                name="python-version-required",
                value=metadata["requires-python"]
            ))
        
        return component
    
    def create_tool_info(self) -> Tool:
        """Create tool information for SBOM generation."""
        return Tool(
            vendor="Terragon Labs",
            name="tiny-llm-profiler-sbom-generator",
            version="1.0.0"
        )
    
    def generate_bom(self) -> Bom:
        """Generate the complete Bill of Materials."""
        # Create main component
        main_component = self.create_main_component()
        
        # Create BOM with main component
        bom = Bom(
            main_component=main_component,
            schema_version=SchemaVersion.V1_5,
        )
        
        # Add tool information
        bom.metadata.tools.add(self.create_tool_info())
        
        # Add timestamp
        bom.metadata.timestamp = time.time()
        
        # Get all installed packages
        packages = self.get_installed_packages()
        
        # Add components for each package
        for package in packages:
            # Skip the main package itself
            if package["name"] == main_component.name:
                continue
                
            component = self.create_component_from_package(package)
            bom.components.add(component)
        
        return bom
    
    def save_sbom(self, bom: Bom, output_dir: Path, formats: List[str]):
        """Save SBOM in specified formats."""
        output_dir.mkdir(exist_ok=True)
        
        for fmt in formats:
            if fmt == "json":
                outputter = get_outputter(bom, output_format="json", schema_version=SchemaVersion.V1_5)
                filename = output_dir / "sbom.json"
            elif fmt == "xml":
                outputter = get_outputter(bom, output_format="xml", schema_version=SchemaVersion.V1_5)
                filename = output_dir / "sbom.xml"
            else:
                click.echo(f"Unsupported format: {fmt}", err=True)
                continue
            
            with open(filename, "w") as f:
                f.write(outputter.output_as_string())
            
            click.echo(f"SBOM saved to: {filename}")


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default="./sbom",
    help="Output directory for SBOM files"
)
@click.option(
    "--format",
    "formats",
    multiple=True,
    type=click.Choice(["json", "xml"]),
    default=["json"],
    help="Output format(s) for SBOM"
)
@click.option(
    "--project-root",
    type=click.Path(exists=True, path_type=Path),
    default=".",
    help="Project root directory"
)
def main(output_dir: Path, formats: List[str], project_root: Path):
    """Generate Software Bill of Materials (SBOM) for tiny-llm-edge-profiler."""
    click.echo("🔍 Generating Software Bill of Materials (SBOM)...")
    
    try:
        generator = SBOMGenerator(project_root)
        bom = generator.generate_bom()
        generator.save_sbom(bom, output_dir, formats)
        
        click.echo("✅ SBOM generation completed successfully!")
        click.echo(f"📊 Components analyzed: {len(bom.components)}")
        
    except Exception as e:
        click.echo(f"❌ Error generating SBOM: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()