"""
Model artifact storage management for the registry.

This module handles storage and retrieval of model artifacts including:
- Model weights and checkpoints
- Model metadata and configurations
- Training artifacts and logs
- Version control and deduplication
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib
import json
import pickle
import gzip
from pathlib import Path

from main.utils.core import (
    get_logger,
    ErrorHandlingMixin,
    timer,
    run_in_executor
)
from main.utils.database import DatabasePool
from main.utils.monitoring import record_metric

logger = get_logger(__name__)


@dataclass
class ModelArtifact:
    """Model artifact metadata."""
    artifact_id: str
    model_name: str
    version: str
    artifact_type: str  # 'weights', 'config', 'metadata', 'checkpoint'
    file_path: str
    file_size_bytes: int
    checksum: str
    compression: Optional[str] = None
    created_at: datetime = None
    metadata: Dict[str, Any] = None


@dataclass
class StorageStats:
    """Storage statistics."""
    total_size_bytes: int
    artifact_count: int
    model_count: int
    compression_ratio: float
    deduplication_ratio: float
    oldest_artifact: Optional[datetime] = None
    newest_artifact: Optional[datetime] = None


class RegistryStorageManager(ErrorHandlingMixin):
    """
    Manages storage of model artifacts in the registry.
    
    Features:
    - Efficient storage with compression and deduplication
    - Versioned artifact management
    - Garbage collection and retention policies
    - Storage optimization
    - Backup and recovery
    """
    
    def __init__(
        self,
        db_pool: DatabasePool,
        storage_path: str,
        max_storage_gb: float = 100.0
    ):
        """Initialize storage manager."""
        super().__init__()
        self.db_pool = db_pool
        self.storage_path = Path(storage_path)
        self.max_storage_bytes = int(max_storage_gb * 1024 * 1024 * 1024)
        
        # Create storage directories
        self.storage_path.mkdir(parents=True, exist_ok=True)
        (self.storage_path / 'models').mkdir(exist_ok=True)
        (self.storage_path / 'temp').mkdir(exist_ok=True)
        (self.storage_path / 'backup').mkdir(exist_ok=True)
        
        # Deduplication cache
        self._checksum_cache: Dict[str, str] = {}
        
        # Storage metrics
        self._storage_metrics = {
            'total_writes': 0,
            'total_reads': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'dedup_savings_bytes': 0
        }
    
    @timer
    async def store_model_artifact(
        self,
        model_name: str,
        version: str,
        artifact_type: str,
        data: bytes,
        metadata: Optional[Dict[str, Any]] = None,
        compress: bool = True
    ) -> ModelArtifact:
        """
        Store a model artifact.
        
        Args:
            model_name: Name of the model
            version: Model version
            artifact_type: Type of artifact
            data: Raw artifact data
            metadata: Optional metadata
            compress: Whether to compress the data
            
        Returns:
            ModelArtifact with storage information
        """
        with self._handle_error("storing model artifact"):
            # Calculate checksum for deduplication
            checksum = hashlib.sha256(data).hexdigest()
            
            # Check if artifact already exists
            existing = await self._find_artifact_by_checksum(checksum)
            
            if existing:
                # Deduplicated - just create new reference
                artifact = ModelArtifact(
                    artifact_id=f"{model_name}_v{version}_{artifact_type}",
                    model_name=model_name,
                    version=version,
                    artifact_type=artifact_type,
                    file_path=existing.file_path,
                    file_size_bytes=existing.file_size_bytes,
                    checksum=checksum,
                    compression=existing.compression,
                    created_at=datetime.utcnow(),
                    metadata=metadata
                )
                
                self._storage_metrics['dedup_savings_bytes'] += len(data)
                
                logger.info(
                    f"Deduplicated artifact for {model_name} v{version} "
                    f"(saved {len(data):,} bytes)"
                )
            else:
                # New artifact - store it
                artifact = await self._store_new_artifact(
                    model_name, version, artifact_type,
                    data, checksum, metadata, compress
                )
            
            # Record in database
            await self._record_artifact(artifact)
            
            # Update metrics
            self._storage_metrics['total_writes'] += 1
            record_metric(
                'registry_storage.artifact_stored',
                1,
                tags={
                    'model': model_name,
                    'version': version,
                    'type': artifact_type,
                    'deduplicated': str(existing is not None)
                }
            )
            
            return artifact
    
    async def _store_new_artifact(
        self,
        model_name: str,
        version: str,
        artifact_type: str,
        data: bytes,
        checksum: str,
        metadata: Optional[Dict[str, Any]],
        compress: bool
    ) -> ModelArtifact:
        """Store a new artifact to disk."""
        # Prepare file path
        artifact_id = f"{model_name}_v{version}_{artifact_type}"
        file_name = f"{artifact_id}_{checksum[:8]}"
        
        if compress:
            file_name += ".gz"
            file_data = await run_in_executor(gzip.compress, data, 9)
            compression = "gzip"
        else:
            file_data = data
            compression = None
        
        # Determine storage location
        model_dir = self.storage_path / 'models' / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        file_path = model_dir / file_name
        
        # Write to disk
        await run_in_executor(file_path.write_bytes, file_data)
        
        # Create artifact record
        artifact = ModelArtifact(
            artifact_id=artifact_id,
            model_name=model_name,
            version=version,
            artifact_type=artifact_type,
            file_path=str(file_path.relative_to(self.storage_path)),
            file_size_bytes=len(file_data),
            checksum=checksum,
            compression=compression,
            created_at=datetime.utcnow(),
            metadata=metadata
        )
        
        # Update checksum cache
        self._checksum_cache[checksum] = artifact.file_path
        
        return artifact
    
    @timer
    async def retrieve_model_artifact(
        self,
        model_name: str,
        version: str,
        artifact_type: str
    ) -> Optional[bytes]:
        """
        Retrieve a model artifact.
        
        Args:
            model_name: Name of the model
            version: Model version
            artifact_type: Type of artifact
            
        Returns:
            Raw artifact data or None if not found
        """
        with self._handle_error("retrieving model artifact"):
            # Get artifact metadata
            artifact = await self._get_artifact_metadata(
                model_name, version, artifact_type
            )
            
            if not artifact:
                logger.warning(
                    f"Artifact not found: {model_name} v{version} {artifact_type}"
                )
                self._storage_metrics['cache_misses'] += 1
                return None
            
            # Read from disk
            file_path = self.storage_path / artifact.file_path
            
            if not file_path.exists():
                logger.error(f"Artifact file missing: {file_path}")
                return None
            
            file_data = await run_in_executor(file_path.read_bytes)
            
            # Decompress if needed
            if artifact.compression == "gzip":
                data = await run_in_executor(gzip.decompress, file_data)
            else:
                data = file_data
            
            # Verify checksum
            actual_checksum = hashlib.sha256(data).hexdigest()
            if actual_checksum != artifact.checksum:
                logger.error(
                    f"Checksum mismatch for {artifact.artifact_id}: "
                    f"expected {artifact.checksum}, got {actual_checksum}"
                )
                return None
            
            # Update metrics
            self._storage_metrics['total_reads'] += 1
            self._storage_metrics['cache_hits'] += 1
            
            record_metric(
                'registry_storage.artifact_retrieved',
                1,
                tags={
                    'model': model_name,
                    'version': version,
                    'type': artifact_type
                }
            )
            
            return data
    
    async def delete_model_artifacts(
        self,
        model_name: str,
        version: Optional[str] = None
    ) -> int:
        """
        Delete model artifacts.
        
        Args:
            model_name: Name of the model
            version: Optional specific version
            
        Returns:
            Number of artifacts deleted
        """
        with self._handle_error("deleting model artifacts"):
            # Get artifacts to delete
            artifacts = await self._list_artifacts(
                model_name, version
            )
            
            deleted_count = 0
            freed_bytes = 0
            
            for artifact in artifacts:
                # Check if file is used by other artifacts
                refs = await self._count_artifact_references(
                    artifact.checksum
                )
                
                if refs <= 1:
                    # Safe to delete physical file
                    file_path = self.storage_path / artifact.file_path
                    if file_path.exists():
                        await run_in_executor(file_path.unlink)
                        freed_bytes += artifact.file_size_bytes
                
                # Delete artifact record
                await self._delete_artifact_record(artifact.artifact_id)
                deleted_count += 1
            
            logger.info(
                f"Deleted {deleted_count} artifacts for {model_name} "
                f"(freed {freed_bytes:,} bytes)"
            )
            
            record_metric(
                'registry_storage.artifacts_deleted',
                deleted_count,
                tags={
                    'model': model_name,
                    'bytes_freed': freed_bytes
                }
            )
            
            return deleted_count
    
    async def get_storage_stats(self) -> StorageStats:
        """Get storage statistics."""
        async with self.db_pool.acquire() as conn:
            # Get aggregate stats
            query = """
                SELECT 
                    COUNT(*) as artifact_count,
                    COUNT(DISTINCT model_name) as model_count,
                    SUM(file_size_bytes) as total_size,
                    MIN(created_at) as oldest,
                    MAX(created_at) as newest
                FROM model_artifacts
            """
            
            row = await conn.fetchrow(query)
            
            # Calculate compression ratio
            compressed_query = """
                SELECT 
                    SUM(file_size_bytes) as compressed_size,
                    SUM(original_size_bytes) as original_size
                FROM model_artifacts
                WHERE compression IS NOT NULL
            """
            
            compression_row = await conn.fetchrow(compressed_query)
            
            if compression_row and compression_row['original_size']:
                compression_ratio = (
                    compression_row['compressed_size'] /
                    compression_row['original_size']
                )
            else:
                compression_ratio = 1.0
            
            # Calculate deduplication ratio
            dedup_ratio = (
                self._storage_metrics['dedup_savings_bytes'] /
                max(1, row['total_size'] + 
                    self._storage_metrics['dedup_savings_bytes'])
            )
            
            return StorageStats(
                total_size_bytes=row['total_size'] or 0,
                artifact_count=row['artifact_count'] or 0,
                model_count=row['model_count'] or 0,
                compression_ratio=compression_ratio,
                deduplication_ratio=dedup_ratio,
                oldest_artifact=row['oldest'],
                newest_artifact=row['newest']
            )
    
    async def cleanup_old_artifacts(
        self,
        retention_days: int = 90,
        keep_latest_versions: int = 3
    ) -> int:
        """
        Clean up old artifacts based on retention policy.
        
        Args:
            retention_days: Days to retain artifacts
            keep_latest_versions: Number of latest versions to keep
            
        Returns:
            Number of artifacts cleaned up
        """
        with self._handle_error("cleaning up old artifacts"):
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
            
            async with self.db_pool.acquire() as conn:
                # Find artifacts to delete
                query = """
                    WITH latest_versions AS (
                        SELECT DISTINCT ON (model_name)
                            model_name, version
                        FROM model_artifacts
                        ORDER BY model_name, created_at DESC
                        LIMIT $1
                    )
                    SELECT artifact_id
                    FROM model_artifacts
                    WHERE created_at < $2
                    AND (model_name, version) NOT IN (
                        SELECT model_name, version FROM latest_versions
                    )
                """
                
                rows = await conn.fetch(
                    query,
                    keep_latest_versions,
                    cutoff_date
                )
                
                # Delete artifacts
                deleted_count = 0
                for row in rows:
                    await self.delete_model_artifacts(
                        row['model_name'],
                        row['version']
                    )
                    deleted_count += 1
            
            logger.info(
                f"Cleaned up {deleted_count} old artifacts "
                f"(retention: {retention_days} days)"
            )
            
            return deleted_count
    
    async def optimize_storage(self) -> Dict[str, Any]:
        """
        Optimize storage by compressing and deduplicating.
        
        Returns:
            Optimization results
        """
        with self._handle_error("optimizing storage"):
            results = {
                'compressed_count': 0,
                'deduplicated_count': 0,
                'bytes_saved': 0,
                'errors': []
            }
            
            # Find uncompressed artifacts
            uncompressed = await self._find_uncompressed_artifacts()
            
            for artifact in uncompressed:
                try:
                    # Retrieve and compress
                    data = await self.retrieve_model_artifact(
                        artifact.model_name,
                        artifact.version,
                        artifact.artifact_type
                    )
                    
                    if data:
                        # Re-store with compression
                        new_artifact = await self.store_model_artifact(
                            artifact.model_name,
                            artifact.version,
                            artifact.artifact_type,
                            data,
                            artifact.metadata,
                            compress=True
                        )
                        
                        bytes_saved = (
                            artifact.file_size_bytes - 
                            new_artifact.file_size_bytes
                        )
                        
                        results['compressed_count'] += 1
                        results['bytes_saved'] += bytes_saved
                        
                except Exception as e:
                    results['errors'].append({
                        'artifact_id': artifact.artifact_id,
                        'error': str(e)
                    })
            
            logger.info(
                f"Storage optimization complete: "
                f"compressed {results['compressed_count']} artifacts, "
                f"saved {results['bytes_saved']:,} bytes"
            )
            
            return results
    
    async def _find_artifact_by_checksum(
        self,
        checksum: str
    ) -> Optional[ModelArtifact]:
        """Find artifact by checksum."""
        if checksum in self._checksum_cache:
            # Get from database
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT * FROM model_artifacts
                    WHERE checksum = $1
                    LIMIT 1
                """
                
                row = await conn.fetchrow(query, checksum)
                
                if row:
                    return ModelArtifact(
                        artifact_id=row['artifact_id'],
                        model_name=row['model_name'],
                        version=row['version'],
                        artifact_type=row['artifact_type'],
                        file_path=row['file_path'],
                        file_size_bytes=row['file_size_bytes'],
                        checksum=row['checksum'],
                        compression=row['compression'],
                        created_at=row['created_at'],
                        metadata=json.loads(row['metadata']) if row['metadata'] else None
                    )
        
        return None
    
    async def _get_artifact_metadata(
        self,
        model_name: str,
        version: str,
        artifact_type: str
    ) -> Optional[ModelArtifact]:
        """Get artifact metadata."""
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT * FROM model_artifacts
                WHERE model_name = $1
                AND version = $2
                AND artifact_type = $3
                ORDER BY created_at DESC
                LIMIT 1
            """
            
            row = await conn.fetchrow(query, model_name, version, artifact_type)
            
            if row:
                return ModelArtifact(
                    artifact_id=row['artifact_id'],
                    model_name=row['model_name'],
                    version=row['version'],
                    artifact_type=row['artifact_type'],
                    file_path=row['file_path'],
                    file_size_bytes=row['file_size_bytes'],
                    checksum=row['checksum'],
                    compression=row['compression'],
                    created_at=row['created_at'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else None
                )
        
        return None
    
    async def _record_artifact(self, artifact: ModelArtifact) -> None:
        """Record artifact in database."""
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO model_artifacts (
                    artifact_id, model_name, version, artifact_type,
                    file_path, file_size_bytes, checksum, compression,
                    created_at, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (artifact_id) DO UPDATE SET
                    file_path = $5,
                    file_size_bytes = $6,
                    checksum = $7,
                    compression = $8,
                    metadata = $10
                """,
                artifact.artifact_id,
                artifact.model_name,
                artifact.version,
                artifact.artifact_type,
                artifact.file_path,
                artifact.file_size_bytes,
                artifact.checksum,
                artifact.compression,
                artifact.created_at,
                json.dumps(artifact.metadata) if artifact.metadata else None
            )
    
    async def _list_artifacts(
        self,
        model_name: str,
        version: Optional[str] = None
    ) -> List[ModelArtifact]:
        """List artifacts for a model."""
        async with self.db_pool.acquire() as conn:
            if version:
                query = """
                    SELECT * FROM model_artifacts
                    WHERE model_name = $1 AND version = $2
                """
                rows = await conn.fetch(query, model_name, version)
            else:
                query = """
                    SELECT * FROM model_artifacts
                    WHERE model_name = $1
                """
                rows = await conn.fetch(query, model_name)
            
            return [
                ModelArtifact(
                    artifact_id=row['artifact_id'],
                    model_name=row['model_name'],
                    version=row['version'],
                    artifact_type=row['artifact_type'],
                    file_path=row['file_path'],
                    file_size_bytes=row['file_size_bytes'],
                    checksum=row['checksum'],
                    compression=row['compression'],
                    created_at=row['created_at'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else None
                )
                for row in rows
            ]
    
    async def _count_artifact_references(
        self,
        checksum: str
    ) -> int:
        """Count references to a checksum."""
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT COUNT(*) FROM model_artifacts
                WHERE checksum = $1
            """
            
            return await conn.fetchval(query, checksum)
    
    async def _delete_artifact_record(
        self,
        artifact_id: str
    ) -> None:
        """Delete artifact record."""
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM model_artifacts WHERE artifact_id = $1",
                artifact_id
            )
    
    async def _find_uncompressed_artifacts(self) -> List[ModelArtifact]:
        """Find uncompressed artifacts."""
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT * FROM model_artifacts
                WHERE compression IS NULL
                ORDER BY file_size_bytes DESC
                LIMIT 100
            """
            
            rows = await conn.fetch(query)
            
            return [
                ModelArtifact(
                    artifact_id=row['artifact_id'],
                    model_name=row['model_name'],
                    version=row['version'],
                    artifact_type=row['artifact_type'],
                    file_path=row['file_path'],
                    file_size_bytes=row['file_size_bytes'],
                    checksum=row['checksum'],
                    compression=row['compression'],
                    created_at=row['created_at'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else None
                )
                for row in rows
            ]