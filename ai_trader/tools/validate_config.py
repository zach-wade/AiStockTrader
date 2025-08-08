#!/usr/bin/env python3
"""
Configuration Validation Tool

Validates all configuration files in the AI Trader system.
Checks for syntax errors, missing required fields, and configuration consistency.
"""

import sys
import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main.config.config_validator import ConfigValidator
from main.config.config_manager import get_config


def validate_yaml_syntax(file_path: Path) -> Tuple[bool, str]:
    """Validate YAML file syntax."""
    try:
        with open(file_path, 'r') as f:
            yaml.safe_load(f)
        return True, "OK"
    except yaml.YAMLError as e:
        return False, f"YAML Error: {str(e)}"
    except Exception as e:
        return False, f"Error: {str(e)}"


def validate_json_syntax(file_path: Path) -> Tuple[bool, str]:
    """Validate JSON file syntax."""
    try:
        with open(file_path, 'r') as f:
            json.load(f)
        return True, "OK"
    except json.JSONDecodeError as e:
        return False, f"JSON Error: {str(e)}"
    except Exception as e:
        return False, f"Error: {str(e)}"


def find_config_files(root_dir: Path) -> Dict[str, List[Path]]:
    """Find all configuration files in the project."""
    config_files = {
        'yaml': [],
        'json': [],
        'py': []
    }
    
    config_dirs = ['config', 'environments']
    
    for dir_name in config_dirs:
        for config_dir in root_dir.rglob(dir_name):
            if '.git' in str(config_dir):
                continue
                
            for yaml_file in config_dir.glob('*.yaml'):
                config_files['yaml'].append(yaml_file)
            for yml_file in config_dir.glob('*.yml'):
                config_files['yaml'].append(yml_file)
            for json_file in config_dir.glob('*.json'):
                config_files['json'].append(json_file)
    
    return config_files


def validate_all_configs(verbose: bool = False) -> int:
    """Validate all configuration files."""
    root_dir = Path(__file__).parent.parent
    config_files = find_config_files(root_dir)
    
    total_files = sum(len(files) for files in config_files.values())
    errors = 0
    
    print(f"Found {total_files} configuration files to validate\n")
    
    # Validate YAML files
    print("Validating YAML files...")
    for yaml_file in config_files['yaml']:
        valid, message = validate_yaml_syntax(yaml_file)
        if valid:
            if verbose:
                print(f"  ✓ {yaml_file.relative_to(root_dir)}")
        else:
            print(f"  ✗ {yaml_file.relative_to(root_dir)}: {message}")
            errors += 1
    
    # Validate JSON files
    print("\nValidating JSON files...")
    for json_file in config_files['json']:
        valid, message = validate_json_syntax(json_file)
        if valid:
            if verbose:
                print(f"  ✓ {json_file.relative_to(root_dir)}")
        else:
            print(f"  ✗ {json_file.relative_to(root_dir)}: {message}")
            errors += 1
    
    # Validate main configuration using ConfigValidator
    print("\nValidating main configuration structure...")
    try:
        validator = ConfigValidator()
        config = get_config()
        
        # Validate for different environments
        environments = ['dev', 'staging', 'prod', 'paper']
        for env in environments:
            is_valid, warnings, errors_list = validator.validate_configuration(
                config, environment=env
            )
            
            if is_valid:
                print(f"  ✓ Configuration valid for environment: {env}")
                if warnings and verbose:
                    for warning in warnings:
                        print(f"    ⚠ {warning}")
            else:
                print(f"  ✗ Configuration invalid for environment: {env}")
                for error in errors_list:
                    print(f"    - {error}")
                errors += len(errors_list)
    
    except Exception as e:
        print(f"  ✗ Error validating main configuration: {str(e)}")
        errors += 1
    
    # Summary
    print(f"\n{'='*50}")
    if errors == 0:
        print(f"✓ All configuration files are valid!")
        return 0
    else:
        print(f"✗ Found {errors} error(s) in configuration files")
        return 1


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Validate AI Trader configuration files'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show all files being validated'
    )
    
    args = parser.parse_args()
    
    sys.exit(validate_all_configs(verbose=args.verbose))


if __name__ == '__main__':
    main()