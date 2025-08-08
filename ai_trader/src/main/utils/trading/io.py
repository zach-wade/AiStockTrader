"""
Trading Universe Import/Export

Tools for importing and exporting trading universe configurations and data.
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .types import UniverseConfig, UniverseType, Filter, FilterCriteria

logger = logging.getLogger(__name__)


class UniverseImportExport:
    """Tools for importing and exporting universe data."""
    
    def __init__(self, manager):
        """Initialize with universe manager."""
        self.manager = manager
    
    def export_universe(self, name: str, format: str = 'json') -> str:
        """Export universe configuration and current symbols."""
        if name not in self.manager.universes:
            raise ValueError(f"Universe '{name}' not found")
        
        config = self.manager.universes[name]
        symbols = self.manager.get_universe_symbols(name)
        
        export_data = {
            'config': config.to_dict(),
            'current_symbols': list(symbols),
            'symbol_count': len(symbols),
            'last_updated': datetime.now().isoformat(),
            'export_format_version': '1.0'
        }
        
        if format == 'json':
            return json.dumps(export_data, indent=2)
        elif format == 'csv':
            return '\n'.join(['symbol'] + list(symbols))
        elif format == 'txt':
            lines = [f"# Universe: {name}"]
            lines.append(f"# Symbols: {len(symbols)}")
            lines.append(f"# Last Updated: {export_data['last_updated']}")
            lines.append("")
            lines.extend(list(symbols))
            return '\n'.join(lines)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def import_universe(self, data: str, format: str = 'json') -> UniverseConfig:
        """Import universe configuration."""
        if format == 'json':
            return self._import_json_universe(data)
        elif format == 'csv':
            return self._import_csv_universe(data)
        elif format == 'txt':
            return self._import_txt_universe(data)
        else:
            raise ValueError(f"Unsupported import format: {format}")
    
    def _import_json_universe(self, data: str) -> UniverseConfig:
        """Import universe from JSON format."""
        import_data = json.loads(data)
        config_data = import_data['config']
        
        # Reconstruct filters
        filters = []
        for f_data in config_data.get('filters', []):
            filters.append(Filter(
                criteria=FilterCriteria(f_data['criteria']),
                operator=f_data['operator'],
                value=f_data['value'],
                weight=f_data.get('weight', 1.0),
                enabled=f_data.get('enabled', True)
            ))
        
        config = UniverseConfig(
            name=config_data['name'],
            universe_type=UniverseType(config_data['universe_type']),
            filters=filters,
            max_symbols=config_data.get('max_symbols'),
            min_symbols=config_data.get('min_symbols'),
            rebalance_frequency=config_data.get('rebalance_frequency', 'daily'),
            ranking_criteria=config_data.get('ranking_criteria'),
            ranking_ascending=config_data.get('ranking_ascending', False)
        )
        
        self.manager.add_universe(config)
        logger.info(f"Imported universe configuration: {config.name}")
        return config
    
    def _import_csv_universe(self, data: str) -> UniverseConfig:
        """Import universe from CSV format (symbols only)."""
        lines = data.strip().split('\n')
        
        # Skip header if present
        if lines and lines[0].lower() == 'symbol':
            lines = lines[1:]
        
        symbols = [line.strip() for line in lines if line.strip()]
        
        # Create a static universe configuration
        config = UniverseConfig(
            name=f"imported_csv_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            universe_type=UniverseType.STATIC,
            filters=[Filter(FilterCriteria.CUSTOM, 'in', symbols)]
        )
        
        self.manager.add_universe(config)
        logger.info(f"Imported CSV universe with {len(symbols)} symbols")
        return config
    
    def _import_txt_universe(self, data: str) -> UniverseConfig:
        """Import universe from TXT format."""
        lines = data.strip().split('\n')
        
        # Parse header information
        name = "imported_txt_universe"
        symbols = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('# Universe:'):
                name = line.split(':', 1)[1].strip()
            elif line and not line.startswith('#'):
                symbols.append(line)
        
        # Create a static universe configuration
        config = UniverseConfig(
            name=name,
            universe_type=UniverseType.STATIC,
            filters=[Filter(FilterCriteria.CUSTOM, 'in', symbols)]
        )
        
        self.manager.add_universe(config)
        logger.info(f"Imported TXT universe '{name}' with {len(symbols)} symbols")
        return config
    
    def export_universe_history(self, name: str, format: str = 'json') -> str:
        """Export universe history."""
        history = self.manager.get_universe_history(name)
        
        if not history:
            raise ValueError(f"No history available for universe '{name}'")
        
        export_data = {
            'universe': name,
            'history_count': len(history),
            'first_snapshot': history[0].timestamp.isoformat(),
            'last_snapshot': history[-1].timestamp.isoformat(),
            'snapshots': [snapshot.to_dict() for snapshot in history]
        }
        
        if format == 'json':
            return json.dumps(export_data, indent=2)
        else:
            raise ValueError(f"Unsupported format for history export: {format}")
    
    def export_multiple_universes(self, universe_names: list, format: str = 'json') -> str:
        """Export multiple universes."""
        if format != 'json':
            raise ValueError("Multiple universe export only supports JSON format")
        
        export_data = {
            'universes': {},
            'export_timestamp': datetime.now().isoformat(),
            'universe_count': len(universe_names)
        }
        
        for name in universe_names:
            if name not in self.manager.universes:
                logger.warning(f"Universe '{name}' not found, skipping")
                continue
            
            config = self.manager.universes[name]
            symbols = self.manager.get_universe_symbols(name)
            
            export_data['universes'][name] = {
                'config': config.to_dict(),
                'current_symbols': list(symbols),
                'symbol_count': len(symbols)
            }
        
        return json.dumps(export_data, indent=2)
    
    def import_multiple_universes(self, data: str, format: str = 'json') -> list:
        """Import multiple universes."""
        if format != 'json':
            raise ValueError("Multiple universe import only supports JSON format")
        
        import_data = json.loads(data)
        imported_configs = []
        
        for name, universe_data in import_data.get('universes', {}).items():
            try:
                config_data = universe_data['config']
                
                # Reconstruct filters
                filters = []
                for f_data in config_data.get('filters', []):
                    filters.append(Filter(
                        criteria=FilterCriteria(f_data['criteria']),
                        operator=f_data['operator'],
                        value=f_data['value'],
                        weight=f_data.get('weight', 1.0),
                        enabled=f_data.get('enabled', True)
                    ))
                
                config = UniverseConfig(
                    name=config_data['name'],
                    universe_type=UniverseType(config_data['universe_type']),
                    filters=filters,
                    max_symbols=config_data.get('max_symbols'),
                    min_symbols=config_data.get('min_symbols'),
                    rebalance_frequency=config_data.get('rebalance_frequency', 'daily'),
                    ranking_criteria=config_data.get('ranking_criteria'),
                    ranking_ascending=config_data.get('ranking_ascending', False)
                )
                
                self.manager.add_universe(config)
                imported_configs.append(config)
                logger.info(f"Imported universe: {config.name}")
                
            except Exception as e:
                logger.error(f"Failed to import universe '{name}': {e}")
        
        return imported_configs
    
    def backup_all_universes(self, format: str = 'json') -> str:
        """Create a backup of all universes."""
        all_universe_names = self.manager.list_universes()
        return self.export_multiple_universes(all_universe_names, format)
    
    def restore_from_backup(self, backup_data: str, format: str = 'json', overwrite: bool = False) -> Dict[str, Any]:
        """Restore universes from backup."""
        if not overwrite:
            # Check for conflicts
            import_data = json.loads(backup_data)
            existing_names = set(self.manager.list_universes())
            backup_names = set(import_data.get('universes', {}).keys())
            conflicts = existing_names & backup_names
            
            if conflicts:
                return {
                    'success': False,
                    'error': f"Conflicts detected with existing universes: {list(conflicts)}",
                    'suggestion': 'Use overwrite=True to replace existing universes'
                }
        
        # Perform restore
        imported_configs = self.import_multiple_universes(backup_data, format)
        
        return {
            'success': True,
            'imported_count': len(imported_configs),
            'imported_universes': [config.name for config in imported_configs]
        }