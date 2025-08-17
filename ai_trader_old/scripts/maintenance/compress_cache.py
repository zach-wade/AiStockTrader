#!/usr/bin/env python3
"""
Utility to compress existing cached data.

This script will:
1. Scan existing cache files
2. Load uncompressed data
3. Re-save with compression
4. Report space savings
"""

# Standard library imports
import asyncio
from datetime import datetime
import gzip
import logging
import os
from pathlib import Path
import pickle
from typing import Any, Dict, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import lz4
try:
    # Third-party imports
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False
    logger.warning("LZ4 not available. Install with: pip install lz4")


class CacheCompressor:
    """Compress existing cache files."""

    def __init__(self, cache_dir: str, compression_type: str = 'gzip'):
        self.cache_dir = Path(cache_dir)
        self.compression_type = compression_type.lower()
        self.stats = {
            'files_processed': 0,
            'files_compressed': 0,
            'files_skipped': 0,
            'original_size': 0,
            'compressed_size': 0,
            'errors': 0
        }

        # Compression functions
        self.compress_funcs = {
            'gzip': lambda data: gzip.compress(data, compresslevel=6),
            'lz4': lambda data: lz4.frame.compress(data, compression_level=9) if LZ4_AVAILABLE else data
        }

        if self.compression_type not in self.compress_funcs:
            raise ValueError(f"Unsupported compression type: {compression_type}")

    async def compress_all_files(self, pattern: str = "*.pkl"):
        """Compress all matching files in cache directory."""
        logger.info(f"Starting compression of {self.cache_dir} with {self.compression_type}")

        # Find all cache files
        cache_files = list(self.cache_dir.rglob(pattern))
        logger.info(f"Found {len(cache_files)} cache files")

        for file_path in cache_files:
            await self._compress_file(file_path)

        # Report results
        self._report_results()

    async def _compress_file(self, file_path: Path) -> bool:
        """Compress a single file."""
        self.stats['files_processed'] += 1

        try:
            # Check if already compressed
            if self._is_compressed(file_path):
                logger.debug(f"Skipping already compressed file: {file_path}")
                self.stats['files_skipped'] += 1
                return False

            # Read original file
            original_size = file_path.stat().st_size
            self.stats['original_size'] += original_size

            with open(file_path, 'rb') as f:
                data = f.read()

            # Try to unpickle and repickle for better compression
            try:
                obj = pickle.loads(data)
                data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            except (pickle.UnpicklingError, ValueError, TypeError) as e:
                # If unpickling fails, compress raw data
                logger.debug(f"Could not repickle {file_path}, using raw data: {e}")

            # Compress data
            compressed_data = self.compress_funcs[self.compression_type](data)
            compressed_size = len(compressed_data)

            # Only save if compression actually reduces size
            if compressed_size < original_size * 0.95:  # At least 5% reduction
                # Save compressed file with marker
                compressed_path = file_path.with_suffix(f'.{self.compression_type}{file_path.suffix}')
                with open(compressed_path, 'wb') as f:
                    # Write magic marker
                    f.write(b'COMPRESSED:' + self.compression_type.encode() + b':')
                    f.write(compressed_data)

                # Remove original file
                file_path.unlink()

                # Rename compressed file to original name
                compressed_path.rename(file_path)

                self.stats['compressed_size'] += compressed_size
                self.stats['files_compressed'] += 1

                savings_pct = (1 - compressed_size / original_size) * 100
                logger.info(f"Compressed {file_path.name}: {original_size:,} -> {compressed_size:,} bytes ({savings_pct:.1f}% savings)")
                return True
            else:
                logger.debug(f"Skipping {file_path.name}: compression not beneficial")
                self.stats['files_skipped'] += 1
                self.stats['compressed_size'] += original_size
                return False

        except Exception as e:
            logger.error(f"Error compressing {file_path}: {e}")
            self.stats['errors'] += 1
            return False

    def _is_compressed(self, file_path: Path) -> bool:
        """Check if file is already compressed."""
        try:
            with open(file_path, 'rb') as f:
                # Check for compression marker
                header = f.read(20)
                return header.startswith(b'COMPRESSED:')
        except (IOError, OSError):
            # File might not exist or be readable
            return False

    def _report_results(self):
        """Report compression results."""
        logger.info("\n" + "="*60)
        logger.info("COMPRESSION RESULTS")
        logger.info("="*60)
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Files compressed: {self.stats['files_compressed']}")
        logger.info(f"Files skipped: {self.stats['files_skipped']}")
        logger.info(f"Errors: {self.stats['errors']}")

        if self.stats['original_size'] > 0:
            total_savings = self.stats['original_size'] - self.stats['compressed_size']
            savings_pct = (total_savings / self.stats['original_size']) * 100

            logger.info(f"\nOriginal size: {self.stats['original_size']:,} bytes ({self.stats['original_size']/1024/1024:.1f} MB)")
            logger.info(f"Compressed size: {self.stats['compressed_size']:,} bytes ({self.stats['compressed_size']/1024/1024:.1f} MB)")
            logger.info(f"Space saved: {total_savings:,} bytes ({total_savings/1024/1024:.1f} MB)")
            logger.info(f"Compression ratio: {savings_pct:.1f}%")
        logger.info("="*60)


async def main():
    """Main entry point."""
    # Standard library imports
    import argparse

    parser = argparse.ArgumentParser(description="Compress existing cache files")
    parser.add_argument(
        "--cache-dir",
        default="cache/features",
        help="Cache directory path (default: cache/features)"
    )
    parser.add_argument(
        "--compression",1
        choices=['gzip', 'lz4'],
        default='gzip',
        help="Compression type (default: gzip)"
    )
    parser.add_argument(
        "--pattern",
        default="*.pkl",
        help="File pattern to match (default: *.pkl)"
    )

    args = parser.parse_args()

    # Check if cache directory exists
    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        logger.error(f"Cache directory not found: {cache_dir}")
        return 1

    # Create compressor and run
    compressor = CacheCompressor(cache_dir, args.compression)
    await compressor.compress_all_files(args.pattern)

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
