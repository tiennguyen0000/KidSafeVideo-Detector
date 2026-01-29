"""
Spark-based Preprocessing for Video Classification.

Provides parallel processing for:
- Frame extraction from videos
- Transcript chunking and processing
- MinIO upload of preprocessed data

Optimizations:
- mapPartitions for batch processing within partitions
- Broadcast variables for shared configuration
- Coalesce to reduce output file count
"""

import os
import json
import logging
import tempfile
from typing import List, Dict, Iterator, Optional, Tuple
from io import BytesIO

logger = logging.getLogger(__name__)


class SparkPreprocessor:
    """
    Spark-based parallel preprocessor for video data.
    
    Extracts frames and processes transcripts in parallel using Spark,
    then uploads to MinIO storage.
    """
    
    def __init__(
        self,
        spark=None,
        num_frames: int = 16,
        num_chunks: int = 5,
        frame_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize the Spark preprocessor.
        
        Args:
            spark: SparkSession instance (optional, will create if None)
            num_frames: Number of frames to extract per video
            num_chunks: Number of text chunks per transcript
            frame_size: Target frame size (width, height)
        """
        self.num_frames = num_frames
        self.num_chunks = num_chunks
        self.frame_size = frame_size
        self.mode = 'ultra_light'
        self.db_config = {}
        self.storage_config = {}
        
        if spark is None:
            from .session import get_spark
            self.spark = get_spark()
        else:
            self.spark = spark
    
    def set_config(self, db_config: dict, storage_config: dict, mode: str):
        """
        Set configuration for database and storage connections.
        
        Args:
            db_config: Database connection config (host, port, user, password, database)
            storage_config: MinIO storage config (host, port, access_key, secret_key, bucket)
            mode: Model mode ('ultra_light' or 'balanced')
        """
        self.db_config = db_config
        self.storage_config = storage_config
        self.mode = mode
        logger.info(f"SparkPreprocessor configured for mode: {mode}")
    
    def extract_frames_parallel(
        self,
        sample_plan: List[Dict],
        num_partitions: int = 4
    ) -> List[Dict]:
        """
        Extract frames in parallel using Spark.
        
        Args:
            sample_plan: List of sample dicts with video_id, sample_id, label
            num_partitions: Number of Spark partitions
            
        Returns:
            List of frame extraction results
        """
        if not sample_plan:
            return []
        
        logger.info(f"Spark frame extraction: {len(sample_plan)} samples, {num_partitions} partitions")
        
        # Broadcast configuration
        config_bc = self.spark.sparkContext.broadcast({
            'num_frames': self.num_frames,
            'frame_size': self.frame_size,
            'storage': self.storage_config,
            'db': self.db_config,
            'mode': self.mode,
        })
        
        # Group samples by video_id to avoid downloading same video multiple times
        from collections import defaultdict
        samples_by_video = defaultdict(list)
        for s in sample_plan:
            samples_by_video[s['video_id']].append(s)
        
        # Create RDD of video groups
        video_groups = list(samples_by_video.items())
        samples_rdd = self.spark.sparkContext.parallelize(video_groups, num_partitions)
        
        def process_video_group(partition: Iterator) -> Iterator[Dict]:
            """Process a partition of video groups."""
            import cv2
            import subprocess
            import tempfile
            import glob
            from PIL import Image
            from io import BytesIO
            from minio import Minio
            
            cfg = config_bc.value
            num_frames = cfg['num_frames']
            frame_size = cfg['frame_size']
            storage_cfg = cfg['storage']
            
            # Create MinIO client
            minio_client = Minio(
                f"{storage_cfg['host']}:{storage_cfg['port']}",
                access_key=storage_cfg['access_key'],
                secret_key=storage_cfg['secret_key'],
                secure=False
            )
            bucket = storage_cfg.get('bucket', 'video-storage')
            
            for video_id, samples in partition:
                try:
                    # Download video once
                    video_path = None
                    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                        video_path = f.name
                        
                    # Get storage path from sample or construct it
                    storage_path = None
                    for s in samples:
                        if 'storage_path' in s:
                            storage_path = s['storage_path']
                            break
                    
                    if not storage_path:
                        # Try common paths
                        for label in ['Safe', 'Aggressive', 'Sexual', 'Superstition']:
                            test_path = f"videos/{label}/{video_id}.mp4"
                            try:
                                minio_client.fget_object(bucket, test_path, video_path)
                                storage_path = test_path
                                break
                            except:
                                continue
                    else:
                        minio_client.fget_object(bucket, storage_path, video_path)
                    
                    if not storage_path or not os.path.exists(video_path):
                        for s in samples:
                            yield {
                                'sample_id': s['sample_id'],
                                'video_id': video_id,
                                'status': 'failed',
                                'error': 'Video not found'
                            }
                        continue
                    
                    # Extract all frames
                    with tempfile.TemporaryDirectory() as temp_dir:
                        extract_cmd = [
                            'ffmpeg', '-y', '-i', video_path,
                            '-vf', f'fps=1,scale={frame_size[0]}:{frame_size[1]}',
                            '-q:v', '2',
                            f'{temp_dir}/frame_%04d.jpg'
                        ]
                        subprocess.run(extract_cmd, capture_output=True)
                        
                        extracted_frames = sorted(glob.glob(f'{temp_dir}/frame_*.jpg'))
                        num_extracted = len(extracted_frames)
                        
                        # Process each sample (may have different random frame selection)
                        for sample in samples:
                            sample_id = sample['sample_id']
                            
                            try:
                                # Select frames uniformly or randomly
                                if num_extracted >= num_frames:
                                    indices = [int(i * num_extracted / num_frames) for i in range(num_frames)]
                                else:
                                    indices = list(range(num_extracted))
                                
                                # Upload selected frames
                                frame_count = 0
                                for j, idx in enumerate(indices[:num_frames]):
                                    if idx < len(extracted_frames):
                                        with open(extracted_frames[idx], 'rb') as f:
                                            data = f.read()
                                        
                                        path = f"samples/{sample_id}/frame_{j:02d}.jpg"
                                        minio_client.put_object(
                                            bucket, path, 
                                            BytesIO(data), len(data),
                                            content_type='image/jpeg'
                                        )
                                        frame_count += 1
                                
                                # Pad with black frames if needed
                                for j in range(frame_count, num_frames):
                                    black = Image.new('RGB', frame_size, (0, 0, 0))
                                    buffer = BytesIO()
                                    black.save(buffer, format='JPEG')
                                    data = buffer.getvalue()
                                    
                                    path = f"samples/{sample_id}/frame_{j:02d}.jpg"
                                    minio_client.put_object(
                                        bucket, path,
                                        BytesIO(data), len(data),
                                        content_type='image/jpeg'
                                    )
                                
                                yield {
                                    'sample_id': sample_id,
                                    'video_id': video_id,
                                    'num_frames': num_frames,
                                    'status': 'success'
                                }
                                
                            except Exception as e:
                                yield {
                                    'sample_id': sample_id,
                                    'video_id': video_id,
                                    'status': 'failed',
                                    'error': str(e)
                                }
                    
                    # Cleanup
                    if video_path and os.path.exists(video_path):
                        os.unlink(video_path)
                        
                except Exception as e:
                    for s in samples:
                        yield {
                            'sample_id': s['sample_id'],
                            'video_id': video_id,
                            'status': 'failed',
                            'error': str(e)
                        }
        
        results = samples_rdd.mapPartitions(process_video_group).collect()
        
        success = sum(1 for r in results if r.get('status') == 'success')
        failed = sum(1 for r in results if r.get('status') == 'failed')
        logger.info(f"Frame extraction complete: {success} success, {failed} failed")
        
        return results
    
    def extract_transcripts_parallel(
        self,
        sample_plan: List[Dict],
        num_partitions: int = 4
    ) -> List[Dict]:
        """
        Process transcripts in parallel using Spark.
        
        Note: Actual STT (Whisper) runs sequentially first.
        Spark handles chunking and storage.
        
        Args:
            sample_plan: List of samples with transcripts
            num_partitions: Number of Spark partitions
            
        Returns:
            List of transcript processing results
        """
        if not sample_plan:
            return []
        
        logger.info(f"Spark transcript processing: {len(sample_plan)} samples")
        
        config_bc = self.spark.sparkContext.broadcast({
            'num_chunks': self.num_chunks,
            'storage': self.storage_config,
        })
        
        samples_rdd = self.spark.sparkContext.parallelize(sample_plan, num_partitions)
        
        def process_transcript(partition: Iterator) -> Iterator[Dict]:
            """Process transcripts in partition."""
            from minio import Minio
            from io import BytesIO
            import json
            
            cfg = config_bc.value
            num_chunks = cfg['num_chunks']
            storage_cfg = cfg['storage']
            
            minio_client = Minio(
                f"{storage_cfg['host']}:{storage_cfg['port']}",
                access_key=storage_cfg['access_key'],
                secret_key=storage_cfg['secret_key'],
                secure=False
            )
            bucket = storage_cfg.get('bucket', 'video-storage')
            
            for sample in partition:
                sample_id = sample.get('sample_id')
                transcript = sample.get('transcript', '')
                
                try:
                    if transcript:
                        words = transcript.split()
                        chunk_size = max(1, len(words) // num_chunks)
                        chunks = []
                        
                        for i in range(num_chunks):
                            start = i * chunk_size
                            end = start + chunk_size if i < num_chunks - 1 else len(words)
                            chunks.append(' '.join(words[start:end]))
                        
                        while len(chunks) < num_chunks:
                            chunks.append("")
                        
                        transcript_data = {
                            'full_text': transcript,
                            'chunks': chunks[:num_chunks]
                        }
                        
                        data = json.dumps(transcript_data).encode('utf-8')
                        path = f"samples/{sample_id}/transcript.json"
                        minio_client.put_object(
                            bucket, path,
                            BytesIO(data), len(data),
                            content_type='application/json'
                        )
                        
                        yield {
                            'sample_id': sample_id,
                            'num_chunks': len(chunks),
                            'status': 'success'
                        }
                    else:
                        yield {
                            'sample_id': sample_id,
                            'status': 'skipped',
                            'reason': 'no transcript'
                        }
                        
                except Exception as e:
                    yield {
                        'sample_id': sample_id,
                        'status': 'failed',
                        'error': str(e)
                    }
        
        results = samples_rdd.mapPartitions(process_transcript).collect()
        
        success = sum(1 for r in results if r.get('status') == 'success')
        logger.info(f"Transcript processing complete: {success} success")
        
        return results
    
    def preprocess_samples(
        self,
        samples: List[Dict],
        storage_client=None,
        progress_callback=None
    ) -> Dict[str, any]:
        """
        Full preprocessing: frames + transcripts.
        
        Args:
            samples: List of sample dicts
            storage_client: Optional storage client (unused, uses config)
            progress_callback: Optional progress callback
            
        Returns:
            Dict with success/failed counts
        """
        frame_results = self.extract_frames_parallel(samples)
        chunk_results = self.extract_transcripts_parallel(samples)
        
        success = sum(1 for r in frame_results if r.get('status') == 'success')
        failed = sum(1 for r in frame_results if r.get('status') == 'failed')
        
        return {
            'success': success,
            'failed': failed,
            'frame_results': frame_results,
            'chunk_results': chunk_results,
        }
