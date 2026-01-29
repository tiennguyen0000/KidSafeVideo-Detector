"""MinIO storage client."""
import os
import io
from minio import Minio
from minio.error import S3Error
from typing import Optional, BinaryIO
import logging

logger = logging.getLogger(__name__)


class StorageClient:
    """Client for interacting with MinIO object storage."""
    
    def __init__(self, host: str = None, port: int = None, 
                 access_key: str = None, secret_key: str = None,
                 bucket: str = None):
        """
        Initialize storage client.
        
        Args:
            host: MinIO host (default from env MINIO_HOST)
            port: MinIO port (default from env MINIO_PORT)
            access_key: MinIO access key (default from env MINIO_ACCESS_KEY)
            secret_key: MinIO secret key (default from env MINIO_SECRET_KEY)
            bucket: Bucket name (default from env MINIO_BUCKET)
        """
        self.host = host or os.environ.get('MINIO_HOST', 'localhost')
        self.port = port or int(os.environ.get('MINIO_PORT', '9000'))
        self.access_key = access_key or os.environ.get('MINIO_ACCESS_KEY', 'minioadmin')
        self.secret_key = secret_key or os.environ.get('MINIO_SECRET_KEY', 'minioadmin123')
        self.bucket = bucket or os.environ.get('MINIO_BUCKET', 'video-storage')
        
        self.client = Minio(
            f"{self.host}:{self.port}",
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=False
        )
        
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Ensure the bucket exists, create if not."""
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info(f"Created bucket: {self.bucket}")
        except S3Error as e:
            logger.error(f"Error ensuring bucket exists: {e}")
    
    def upload_file(self, object_key: str, file_path: str, content_type: str = None) -> bool:
        """Upload a file to MinIO."""
        try:
            self.client.fput_object(
                self.bucket,
                object_key,
                file_path,
                content_type=content_type or 'application/octet-stream'
            )
            logger.info(f"Uploaded file to {object_key}")
            return True
        except S3Error as e:
            logger.error(f"Error uploading file to {object_key}: {e}")
            return False
    
    def upload_data(self, object_key: str, data: bytes, content_type: str = None) -> bool:
        """Upload binary data to MinIO."""
        try:
            data_stream = io.BytesIO(data)
            self.client.put_object(
                self.bucket,
                object_key,
                data_stream,
                length=len(data),
                content_type=content_type or 'application/octet-stream'
            )
            logger.info(f"Uploaded data to {object_key}")
            return True
        except S3Error as e:
            logger.error(f"Error uploading data to {object_key}: {e}")
            return False
    
    def download_file(self, object_key: str, file_path: str) -> bool:
        """Download a file from MinIO."""
        try:
            self.client.fget_object(self.bucket, object_key, file_path)
            logger.info(f"Downloaded {object_key} to {file_path}")
            return True
        except S3Error as e:
            logger.error(f"Error downloading {object_key}: {e}")
            return False
    
    def download_data(self, object_key: str, silent: bool = False) -> Optional[bytes]:
        """Download data from MinIO as bytes."""
        try:
            response = self.client.get_object(self.bucket, object_key)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error as e:
            if not silent:
                logger.error(f"Error downloading data from {object_key}: {e}")
            return None
    
    def download_bytes(self, object_key: str) -> Optional[bytes]:
        """Alias for download_data for Spark compatibility."""
        return self.download_data(object_key, silent=True)
    
    def object_exists(self, object_key: str) -> bool:
        """Check if an object exists in MinIO."""
        try:
            self.client.stat_object(self.bucket, object_key)
            return True
        except S3Error:
            return False
    
    def delete_object(self, object_key: str) -> bool:
        """Delete an object from MinIO."""
        try:
            self.client.remove_object(self.bucket, object_key)
            logger.info(f"Deleted object: {object_key}")
            return True
        except S3Error as e:
            logger.error(f"Error deleting {object_key}: {e}")
            return False
    
    def list_objects(self, prefix: str = None) -> list:
        """List objects in MinIO with optional prefix."""
        try:
            objects = self.client.list_objects(self.bucket, prefix=prefix, recursive=True)
            return [obj.object_name for obj in objects]
        except S3Error as e:
            logger.error(f"Error listing objects with prefix {prefix}: {e}")
            return []
    
    def get_object_url(self, object_key: str, external: bool = True) -> str:
        """
        Get presigned URL for an object.
        
        Args:
            object_key: Path to the object in the bucket
            external: If True, replace internal hostname with external hostname
                     for browser access (default True)
        """
        try:
            url = self.client.presigned_get_object(self.bucket, object_key)
            
            if external:
                # Replace internal hostname with external hostname for browser access
                # Internal: minio:9000 -> External: localhost:9000
                external_host = os.environ.get('MINIO_EXTERNAL_HOST', 'localhost')
                external_port = os.environ.get('MINIO_EXTERNAL_PORT', '9000')
                internal_endpoint = f"{self.host}:{self.port}"
                external_endpoint = f"{external_host}:{external_port}"
                url = url.replace(internal_endpoint, external_endpoint)
            
            return url
        except S3Error as e:
            logger.error(f"Error generating URL for {object_key}: {e}")
            return ""
    
    def get_public_url(self, object_key: str) -> str:
        """Get public URL for an object (if bucket is public)."""
        external_host = os.environ.get('MINIO_EXTERNAL_HOST', 'localhost')
        external_port = os.environ.get('MINIO_EXTERNAL_PORT', '9000')
        return f"http://{external_host}:{external_port}/{self.bucket}/{object_key}"


# Global storage client instance
storage = StorageClient()
