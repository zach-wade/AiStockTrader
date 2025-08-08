#!/usr/bin/env python3
"""
Analyze PostgreSQL Database Schema

This script connects to the PostgreSQL database and analyzes all tables,
generating a complete schema report and SQLAlchemy model definitions.
"""

import sys
import os
from pathlib import Path
import json
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

# Load environment variables
load_dotenv()


class PostgreSQLSchemaAnalyzer:
    """Analyzes PostgreSQL database schema and generates SQLAlchemy models."""
    
    # PostgreSQL to SQLAlchemy type mapping
    TYPE_MAPPING = {
        'character varying': 'String',
        'varchar': 'String',
        'text': 'Text',
        'integer': 'Integer',
        'bigint': 'BigInteger',
        'smallint': 'SmallInteger',
        'numeric': 'Numeric',
        'real': 'Float',
        'double precision': 'Float',
        'boolean': 'Boolean',
        'date': 'Date',
        'timestamp': 'DateTime',
        'timestamp with time zone': 'DateTime(timezone=True)',
        'timestamp without time zone': 'DateTime',
        'json': 'JSON',
        'jsonb': 'JSON',
        'uuid': 'String(36)',
        'interval': 'Interval',
        'time': 'Time',
        'time with time zone': 'Time(timezone=True)',
    }
    
    def __init__(self):
        """Initialize the schema analyzer with database connection."""
        self.connection_string = self._build_connection_string()
        self.engine = create_engine(self.connection_string)
        
    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string from environment variables."""
        host = os.getenv('DB_HOST', 'localhost')
        port = os.getenv('DB_PORT', '5432')
        database = os.getenv('DB_NAME', 'ai_trader')
        user = os.getenv('DB_USER', 'postgres')
        password = os.getenv('DB_PASSWORD', '')
        
        if password:
            return f"postgresql://{user}:{password}@{host}:{port}/{database}"
        else:
            return f"postgresql://{user}@{host}:{port}/{database}"
    
    def get_all_tables(self) -> List[str]:
        """Get list of all tables in the database."""
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_type = 'BASE TABLE'
        ORDER BY table_name;
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            return [row[0] for row in result]
    
    def get_table_columns(self, table_name: str) -> List[Dict[str, Any]]:
        """Get detailed column information for a table."""
        query = """
        SELECT 
            column_name,
            data_type,
            character_maximum_length,
            numeric_precision,
            numeric_scale,
            is_nullable,
            column_default,
            is_identity,
            identity_generation
        FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = :table_name
        ORDER BY ordinal_position;
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {'table_name': table_name})
            columns = []
            for row in result:
                col_info = {
                    'name': row[0],
                    'data_type': row[1],
                    'max_length': row[2],
                    'numeric_precision': row[3],
                    'numeric_scale': row[4],
                    'nullable': row[5] == 'YES',
                    'default': row[6],
                    'is_identity': row[7] == 'YES',
                    'identity_generation': row[8]
                }
                columns.append(col_info)
            return columns
    
    def get_primary_keys(self, table_name: str) -> List[str]:
        """Get primary key columns for a table."""
        query = """
        SELECT kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_schema = kcu.table_schema
        WHERE tc.constraint_type = 'PRIMARY KEY'
        AND tc.table_schema = 'public'
        AND tc.table_name = :table_name
        ORDER BY kcu.ordinal_position;
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {'table_name': table_name})
            return [row[0] for row in result]
    
    def get_foreign_keys(self, table_name: str) -> List[Dict[str, Any]]:
        """Get foreign key information for a table."""
        query = """
        SELECT
            kcu.column_name,
            ccu.table_name AS foreign_table_name,
            ccu.column_name AS foreign_column_name,
            tc.constraint_name
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage AS ccu
            ON ccu.constraint_name = tc.constraint_name
            AND ccu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
        AND tc.table_schema = 'public'
        AND tc.table_name = :table_name;
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {'table_name': table_name})
            foreign_keys = []
            for row in result:
                fk_info = {
                    'column': row[0],
                    'foreign_table': row[1],
                    'foreign_column': row[2],
                    'constraint_name': row[3]
                }
                foreign_keys.append(fk_info)
            return foreign_keys
    
    def get_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        """Get index information for a table."""
        query = """
        SELECT
            i.relname AS index_name,
            array_agg(a.attname ORDER BY array_position(ix.indkey, a.attnum)) AS column_names,
            ix.indisunique AS is_unique,
            ix.indisprimary AS is_primary
        FROM pg_class t
        JOIN pg_index ix ON t.oid = ix.indrelid
        JOIN pg_class i ON i.oid = ix.indexrelid
        JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
        WHERE t.relname = :table_name
        AND t.relnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
        GROUP BY i.relname, ix.indisunique, ix.indisprimary;
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {'table_name': table_name})
            indexes = []
            for row in result:
                index_info = {
                    'name': row[0],
                    'columns': row[1],
                    'unique': row[2],
                    'primary': row[3]
                }
                indexes.append(index_info)
            return indexes
    
    def analyze_table(self, table_name: str) -> Dict[str, Any]:
        """Analyze a single table and return complete schema information."""
        return {
            'name': table_name,
            'columns': self.get_table_columns(table_name),
            'primary_keys': self.get_primary_keys(table_name),
            'foreign_keys': self.get_foreign_keys(table_name),
            'indexes': self.get_indexes(table_name)
        }
    
    def generate_sqlalchemy_column(self, col_info: Dict[str, Any], is_primary: bool = False) -> str:
        """Generate SQLAlchemy column definition from column info."""
        col_name = col_info['name']
        data_type = col_info['data_type']
        
        # Map PostgreSQL type to SQLAlchemy type
        if data_type in self.TYPE_MAPPING:
            sa_type = self.TYPE_MAPPING[data_type]
        else:
            # Handle varchar/char with length
            if data_type.startswith('character varying') or data_type == 'varchar':
                if col_info['max_length']:
                    sa_type = f"String({col_info['max_length']})"
                else:
                    sa_type = "String"
            elif data_type == 'numeric':
                if col_info['numeric_precision'] and col_info['numeric_scale']:
                    sa_type = f"Numeric({col_info['numeric_precision']}, {col_info['numeric_scale']})"
                else:
                    sa_type = "Numeric"
            else:
                sa_type = "String"  # Default fallback
        
        # Build column definition
        col_def = f"    {col_name} = Column({sa_type}"
        
        # Add primary key
        if is_primary:
            col_def += ", primary_key=True"
        
        # Add nullable
        if not col_info['nullable'] and not is_primary:
            col_def += ", nullable=False"
        
        # Add default value (simplified)
        if col_info['default'] and not col_info['is_identity']:
            if 'now()' in col_info['default']:
                col_def += ", default=func.now()"
            elif col_info['default'].startswith("'") and col_info['default'].endswith("'"):
                # String default
                default_val = col_info['default'].strip("'")
                col_def += f", default='{default_val}'"
            elif col_info['default'].replace('.', '').replace('-', '').isdigit():
                # Numeric default
                col_def += f", default={col_info['default']}"
            elif col_info['default'] in ('true', 'false'):
                # Boolean default
                col_def += f", default={col_info['default'].capitalize()}"
        
        # Add index for common fields
        if col_name in ['symbol', 'timestamp', 'created_at', 'updated_at'] and not is_primary:
            col_def += ", index=True"
        
        col_def += ")"
        
        return col_def
    
    def generate_sqlalchemy_model(self, table_info: Dict[str, Any]) -> str:
        """Generate complete SQLAlchemy model class from table info."""
        table_name = table_info['name']
        class_name = self._table_name_to_class_name(table_name)
        
        # Start class definition
        model_code = [f"\nclass {class_name}(Base):"]
        model_code.append(f'    """Model for {table_name} table."""')
        model_code.append(f"    __tablename__ = '{table_name}'")
        
        # Add columns
        primary_keys = set(table_info['primary_keys'])
        for col in table_info['columns']:
            col_def = self.generate_sqlalchemy_column(col, col['name'] in primary_keys)
            model_code.append(col_def)
        
        # Add foreign key relationships
        if table_info['foreign_keys']:
            model_code.append("\n    # Relationships")
            for fk in table_info['foreign_keys']:
                rel_name = fk['foreign_table'].rstrip('s')  # Simple pluralization
                rel_class = self._table_name_to_class_name(fk['foreign_table'])
                model_code.append(f'    {rel_name} = relationship("{rel_class}")')
        
        # Add indexes
        non_primary_indexes = [idx for idx in table_info['indexes'] if not idx['primary']]
        if non_primary_indexes:
            model_code.append("\n    # Indexes")
            model_code.append("    __table_args__ = (")
            for idx in non_primary_indexes:
                if len(idx['columns']) == 1:
                    continue  # Single column indexes are handled in column definition
                idx_cols = ', '.join([f"'{col}'" for col in idx['columns']])
                unique_str = ", unique=True" if idx['unique'] else ""
                model_code.append(f"        Index('{idx['name']}', {idx_cols}{unique_str}),")
            model_code.append("    )")
        
        return '\n'.join(model_code)
    
    def _table_name_to_class_name(self, table_name: str) -> str:
        """Convert table_name to ClassName."""
        parts = table_name.split('_')
        return ''.join(word.capitalize() for word in parts)
    
    def analyze_all_tables(self) -> Dict[str, Any]:
        """Analyze all tables in the database."""
        tables = self.get_all_tables()
        analysis = {
            'database_info': {
                'total_tables': len(tables),
                'connection': self.connection_string.split('@')[1] if '@' in self.connection_string else 'localhost'
            },
            'tables': {}
        }
        
        for table in tables:
            print(f"Analyzing table: {table}")
            analysis['tables'][table] = self.analyze_table(table)
        
        return analysis
    
    def generate_models_code(self, tables_to_generate: List[str], analysis: Dict[str, Any]) -> str:
        """Generate SQLAlchemy model code for specified tables."""
        code_parts = []
        
        # Add imports
        code_parts.append("# Generated SQLAlchemy Models")
        code_parts.append("from sqlalchemy import Column, String, Integer, BigInteger, Float, Boolean, Date, DateTime, Text, JSON, ForeignKey, Index, Numeric, func")
        code_parts.append("from sqlalchemy.orm import relationship")
        code_parts.append("from ai_trader.data_pipeline.storage.database_models import Base")
        code_parts.append("")
        
        # Generate models
        for table_name in tables_to_generate:
            if table_name in analysis['tables']:
                model_code = self.generate_sqlalchemy_model(analysis['tables'][table_name])
                code_parts.append(model_code)
                code_parts.append("")
        
        return '\n'.join(code_parts)
    
    def save_analysis_report(self, analysis: Dict[str, Any], output_file: str = 'schema_analysis.json'):
        """Save analysis report to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"\nAnalysis report saved to: {output_file}")


def main():
    """Main function to run schema analysis."""
    print("PostgreSQL Schema Analyzer")
    print("=" * 50)
    
    analyzer = PostgreSQLSchemaAnalyzer()
    
    # Get all tables
    tables = analyzer.get_all_tables()
    print(f"\nFound {len(tables)} tables in database:")
    for table in tables:
        print(f"  - {table}")
    
    # Analyze all tables
    print("\nAnalyzing table schemas...")
    analysis = analyzer.analyze_all_tables()
    
    # Save analysis report
    analyzer.save_analysis_report(analysis)
    
    # Tables that need to be added to database_models.py
    missing_tables = [
        'ensemble_signals',
        'financials_data',
        'model_predictions',
        'order_book_snapshots',
        'pair_relationships',
        'strategy_performance'
    ]
    
    # Generate model code for missing tables
    print("\nGenerating SQLAlchemy models for missing tables...")
    model_code = analyzer.generate_models_code(missing_tables, analysis)
    
    # Save generated models
    with open('generated_models.py', 'w') as f:
        f.write(model_code)
    print("Generated models saved to: generated_models.py")
    
    # Print summary
    print("\n" + "=" * 50)
    print("Analysis Complete!")
    print(f"Total tables analyzed: {len(tables)}")
    print(f"Missing table models generated: {len(missing_tables)}")
    print("\nNext steps:")
    print("1. Review the generated_models.py file")
    print("2. Copy the model definitions to database_models.py")
    print("3. Update any existing models based on schema_analysis.json")
    print("4. Test the updated models with your application")


if __name__ == "__main__":
    main()