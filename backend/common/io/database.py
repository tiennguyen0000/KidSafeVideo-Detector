"""Database client for Postgres."""
import os
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from typing import List, Dict, Any, Optional
from contextlib import contextmanager


class DatabaseClient:
    """Client for interacting with Postgres database."""
    
    def __init__(self):
        self.host = os.environ.get('POSTGRES_HOST', 'localhost')
        self.port = int(os.environ.get('POSTGRES_PORT', '5432'))
        self.user = os.environ.get('POSTGRES_USER', 'video_classifier')
        self.password = os.environ.get('POSTGRES_PASSWORD', 'changeme123')
        self.database = os.environ.get('POSTGRES_DB', 'video_classifier')
        
        self._connection = None
    
    def connect(self):
        """Establish database connection."""
        if self._connection is None or self._connection.closed:
            self._connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database
            )
        return self._connection
    
    def close(self):
        """Close database connection."""
        if self._connection and not self._connection.closed:
            self._connection.close()
            self._connection = None
    
    @contextmanager
    def cursor(self, dict_cursor=True):
        """Context manager for database cursor."""
        conn = self.connect()
        cursor_factory = RealDictCursor if dict_cursor else None
        cur = conn.cursor(cursor_factory=cursor_factory)
        try:
            yield cur
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cur.close()
    
    def execute(self, query: str, params: tuple = None, fetch: bool = False) -> Optional[List[Dict]]:
        """Execute a SQL query."""
        with self.cursor() as cur:
            cur.execute(query, params)
            if fetch:
                return cur.fetchall()
            return None
    
    def upsert_video(self, video_id: str, filename: str, label: str, 
                     transcript: Optional[str] = None, storage_path: Optional[str] = None,
                     title: Optional[str] = None) -> None:
        """Insert or update video record."""
        query = """
            INSERT INTO videos (video_id, filename, label, title, transcript, storage_path, status_preprocess)
            VALUES (%s, %s, %s, %s, %s, %s, 'pending')
            ON CONFLICT (video_id) DO UPDATE
            SET filename = EXCLUDED.filename,
                label = EXCLUDED.label,
                title = EXCLUDED.title,
                transcript = EXCLUDED.transcript,
                storage_path = COALESCE(EXCLUDED.storage_path, videos.storage_path),
                status_preprocess = CASE 
                    WHEN EXCLUDED.storage_path IS NOT NULL AND videos.storage_path IS NULL 
                    THEN 'pending'  -- New upload, reset to pending
                    ELSE videos.status_preprocess  -- Keep existing status
                END,
                updated_at = CURRENT_TIMESTAMP
        """
        self.execute(query, (video_id, filename, label, title, transcript, storage_path))
    
    def upsert_video_for_inference(self, video_id: str, filename: str, storage_path: str, title: str = None) -> None:
        """
        Insert video record for inference-only (no label).
        Used for videos from YouTube search that need classification.
        
        These videos will NOT be added to training samples.
        """
        query = """
            INSERT INTO videos (video_id, filename, label, title, storage_path, status_preprocess)
            VALUES (%s, %s, NULL, %s, %s, 'pending_preprocessing')
            ON CONFLICT (video_id) DO UPDATE
            SET filename = COALESCE(EXCLUDED.filename, videos.filename),
                title = COALESCE(EXCLUDED.title, videos.title),
                storage_path = COALESCE(EXCLUDED.storage_path, videos.storage_path),
                status_preprocess = CASE 
                    WHEN videos.label IS NULL THEN 'pending_preprocessing'  -- Keep pending_preprocessing for inference-only
                    ELSE videos.status_preprocess
                END,
                updated_at = CURRENT_TIMESTAMP
        """
        self.execute(query, (video_id, filename, title, storage_path))
    
    def get_video(self, video_id: str) -> Optional[Dict]:
        """Get video by ID."""
        query = "SELECT * FROM videos WHERE video_id = %s"
        results = self.execute(query, (video_id,), fetch=True)
        return results[0] if results else None
    
    def get_videos_by_status(self, status: str, limit: int = 100) -> List[Dict]:
        """Get videos by preprocessing status."""
        query = """
            SELECT * FROM videos 
            WHERE status_preprocess = %s 
            ORDER BY created_at ASC
            LIMIT %s 
        """
        ## can nhac bo limit
        return self.execute(query, (status, limit), fetch=True)
    
    def update_video_status(self, video_id: str, status: str) -> None:
        """Update video preprocessing status."""
        query = "UPDATE videos SET status_preprocess = %s, updated_at = CURRENT_TIMESTAMP WHERE video_id = %s"
        self.execute(query, (status, video_id))
    
    def get_videos_pending_inference(self, limit: int = 100) -> List[Dict]:
        """
        Get videos pending inference with their sample_id.
        
        Returns videos that:
        - Have status_preprocess = 'pending_inference'
        - Have inference samples in training_samples_v2 (split='inference')
        """
        query = """
            SELECT v.*, s.sample_id
            FROM videos v
            INNER JOIN training_samples_v2 s ON v.video_id = s.video_id
            WHERE v.status_preprocess = 'pending_inference'
              AND s.split = 'inference'
            ORDER BY v.created_at ASC
            LIMIT %s
        """
        return self.execute(query, (limit,), fetch=True) or []
    
    def update_video_transcript(self, video_id: str, transcript: str) -> None:
        """Update video transcript."""
        query = "UPDATE videos SET transcript = %s WHERE video_id = %s"
        self.execute(query, (transcript, video_id))
    
    def upsert_video_preprocess(self, video_id: str, **kwargs) -> None:
        """Insert or update video preprocessing record."""
        fields = list(kwargs.keys())
        values = list(kwargs.values())
        
        set_clause = ', '.join([f"{field} = EXCLUDED.{field}" for field in fields])
        
        query = f"""
            INSERT INTO video_preprocess (video_id, {', '.join(fields)})
            VALUES (%s, {', '.join(['%s'] * len(fields))})
            ON CONFLICT (video_id) DO UPDATE
            SET {set_clause}, updated_at = CURRENT_TIMESTAMP
        """
        self.execute(query, tuple([video_id] + values))
    
    def get_video_preprocess(self, video_id: str) -> Optional[Dict]:
        """Get video preprocessing record."""
        query = "SELECT * FROM video_preprocess WHERE video_id = %s"
        results = self.execute(query, (video_id,), fetch=True)
        return results[0] if results else None
    
    def insert_training_sample(self, video_id: str, label: str, frame_indices: List[int],
                               transcript_available: bool, mode_ready: str, split: str = 'train') -> None:
        """Insert training sample record."""
        query = """
            INSERT INTO training_samples (video_id, label, frame_indices, transcript_available, mode_ready, split)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        self.execute(query, (video_id, label, Json(frame_indices), transcript_available, mode_ready, split))
    
    def get_training_samples(self, split: str = None, mode: str = None) -> List[Dict]:
        """Get training samples."""
        conditions = []
        params = []
        
        if split:
            conditions.append("split = %s")
            params.append(split)
        
        if mode:
            conditions.append("mode_ready = %s")
            params.append(mode)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        query = f"SELECT * FROM training_samples {where_clause} ORDER BY id"
        
        return self.execute(query, tuple(params) if params else None, fetch=True)
    
    def get_fusion_config(self, mode: str) -> Dict[str, Dict[str, float]]:
        """Get fusion weights for a mode."""
        query = """
            SELECT class_name, w_text, w_img 
            FROM fusion_config 
            WHERE mode = %s AND is_active = TRUE
        """
        results = self.execute(query, (mode,), fetch=True)
        
        config = {}
        for row in results:
            config[row['class_name']] = {
                'w_text': row['w_text'],
                'w_img': row['w_img']
            }
        
        return config
    
    def upsert_fusion_config(self, mode: str, weights: Dict[str, Dict[str, float]]) -> None:
        """
        Insert or update fusion configuration for a mode.
        
        Args:
            mode: Model mode ('ultra_light' or 'balanced')
            weights: Dict mapping class_name to {'text': w_text, 'image': w_img}
                    or {'w_text': w_text, 'w_img': w_img}
        """
        # Deactivate old configs
        query = "UPDATE fusion_config SET is_active = FALSE WHERE mode = %s"
        self.execute(query, (mode,))
        
        # Insert new configs
        for class_name, weight_dict in weights.items():
            # Support both formats: {'text': x, 'image': y} or {'w_text': x, 'w_img': y}
            w_text = weight_dict.get('text', weight_dict.get('w_text', 0.5))
            w_img = weight_dict.get('image', weight_dict.get('w_img', 0.5))
            
            query = """
                INSERT INTO fusion_config (mode, class_name, w_text, w_img, is_active)
                VALUES (%s, %s, %s, %s, TRUE)
            """
            self.execute(query, (mode, class_name, w_text, w_img))
    
    def get_active_models(self, mode: str) -> Dict[str, Dict]:
        """Get active model artifacts for a mode."""
        query = """
            SELECT model_type, version, artifact_path, metrics
            FROM model_registry
            WHERE mode = %s AND is_active = TRUE
        """
        results = self.execute(query, (mode,), fetch=True)
        
        models = {}
        for row in results:
            models[row['model_type']] = {
                'version': row['version'],
                'artifact_path': row['artifact_path'],
                'metrics': row['metrics']
            }
        
        return models
    
    def register_model(self, mode: str, model_type: str, version: str, 
                      artifact_path: str, metrics: Dict = None, set_active: bool = True,
                      training_config: Dict = None, best_epoch: int = None,
                      total_epochs: int = None, training_time: float = None) -> int:
        """Register a new model version.
        
        Returns:
            model_id: The ID of the registered model
        """
        if set_active:
            # Deactivate previous models
            query = "UPDATE model_registry SET is_active = FALSE WHERE mode = %s AND model_type = %s"
            self.execute(query, (mode, model_type))
        
        # Insert new model
        query = """
            INSERT INTO model_registry (mode, model_type, version, artifact_path, metrics, is_active,
                                      training_config, best_epoch, total_epochs, training_time_seconds)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        result = self.execute(query, (
            mode, model_type, version, artifact_path,
            Json(metrics) if metrics else None,
            set_active,
            Json(training_config) if training_config else None,
            best_epoch, total_epochs, training_time
        ), fetch=True)
        
        return result[0]['id'] if result else None
    
    def insert_prediction(self, video_id: str, mode: str, y_pred: str,
                         p_text: Dict, p_img: Dict, p_final: Dict, confidence: float, metadata: Dict = None) -> None:
        """Insert prediction record.
        
        For two-stage predictions:
        - p_text: binary text probs
        - p_img: binary image probs  
        - p_final: multiclass probs (or binary if Safe)
        - metadata: additional info (stage, binary_probs, multiclass_probs)
        """
        # If metadata provided, merge it into p_final for backward compatibility
        if metadata:
            p_final_extended = dict(p_final)
            p_final_extended['_metadata'] = metadata
            p_final = p_final_extended
        
        query = """
            INSERT INTO predictions (video_id, mode, y_pred, p_text, p_img, p_final, confidence)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        self.execute(query, (video_id, mode, y_pred, Json(p_text), Json(p_img), Json(p_final), confidence))
    
    
    def save_training_metrics(self, model_id: int, epoch: int, split: str,
                                loss: float, accuracy: float, precision: float,
                                recall: float, f1: float, per_class_metrics: Dict,
                                confusion_matrix: list, learning_rate: float = None,
                                batch_size: int = None) -> None:
        """Save training metrics for a specific epoch."""
        query = """
            INSERT INTO training_metrics (
                model_id, epoch, split, loss, accuracy, precision, recall, f1,
                per_class_metrics, confusion_matrix, learning_rate, batch_size
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (model_id, epoch, split) DO UPDATE SET
                loss = EXCLUDED.loss,
                accuracy = EXCLUDED.accuracy,
                precision = EXCLUDED.precision,
                recall = EXCLUDED.recall,
                f1 = EXCLUDED.f1,
                per_class_metrics = EXCLUDED.per_class_metrics,
                confusion_matrix = EXCLUDED.confusion_matrix,
                learning_rate = EXCLUDED.learning_rate,
                batch_size = EXCLUDED.batch_size
        """
        self.execute(query, (
            model_id, epoch, split, loss, accuracy, precision, recall, f1,
            Json(per_class_metrics), Json(confusion_matrix),
            learning_rate, batch_size
        ))
    
    def get_training_metrics(self, model_id: int) -> List[Dict]:
        """Get all training metrics for a model."""
        query = """
            SELECT * FROM training_metrics
            WHERE model_id = %s
            ORDER BY epoch ASC, split ASC
        """
        return self.execute(query, (model_id,), fetch=True) or []
    
    def get_latest_prediction(self, video_id: str, mode: str = None) -> Optional[Dict]:
        """Get latest prediction for a video."""
        if mode:
            query = """
                SELECT * FROM predictions 
                WHERE video_id = %s AND mode = %s 
                ORDER BY created_at DESC 
                LIMIT 1
            """
            params = (video_id, mode)
        else:
            query = """
                SELECT * FROM predictions 
                WHERE video_id = %s 
                ORDER BY created_at DESC 
                LIMIT 1
            """
            params = (video_id,)
        
        results = self.execute(query, params, fetch=True)
        return results[0] if results else None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics."""
        stats = {}
        
        # Count videos by status
        query = "SELECT status_preprocess, COUNT(*) as count FROM videos GROUP BY status_preprocess"
        status_counts = self.execute(query, fetch=True)
        stats['videos_by_status'] = {row['status_preprocess']: row['count'] for row in status_counts}
        
        # Count videos by label
        query = "SELECT label, COUNT(*) as count FROM videos GROUP BY label"
        label_counts = self.execute(query, fetch=True)
        stats['videos_by_label'] = {row['label']: row['count'] for row in label_counts}
        
        # Count predictions
        query = "SELECT COUNT(*) as count FROM predictions"
        pred_count = self.execute(query, fetch=True)
        stats['total_predictions'] = pred_count[0]['count']
        
        return stats

    # ========================================================================
    # TRAINING SAMPLES V2 (Pre-generated with augmentation)
    # ========================================================================
    def upsert_samples(self, sample_id: str, video_id: str, label: str,
                        augment_idx: int, selected_frames: str, selected_chunks: str) -> None:
        """Insert pre-generated training sample."""
        query = """
            INSERT INTO training_samples_v2 
            (sample_id, video_id, label, augment_idx, selected_frames, selected_chunks)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (sample_id) DO UPDATE
            SET label = EXCLUDED.label,
                selected_frames = EXCLUDED.selected_frames,
                selected_chunks = EXCLUDED.selected_chunks,
                updated_at = CURRENT_TIMESTAMP
        """
        self.execute(query, (sample_id, video_id, label, augment_idx, selected_frames, selected_chunks))
    
    
    def get_training_samples_v2(self, split: str) -> List[Dict]:
        """Get v2 training samples by split."""
        query = """
            SELECT * FROM training_samples_v2 WHERE split = %s
        """
        return self.execute(query, (split,), fetch=True) or []
    
    def update_sample_split_v2(self, sample_id: str, split: str) -> None:
        """Update split for a v2 sample."""
        query = """
            UPDATE training_samples_v2 SET split = %s, updated_at = CURRENT_TIMESTAMP
            WHERE sample_id = %s
        """
        self.execute(query, (split, sample_id))

    # ========================================================================
    # SAMPLE PREDICTIONS (Per-sample predictions for error analysis)
    # ========================================================================
    def save_sample_predictions_batch(self, model_id: int, predictions: List[Dict]) -> None:
        """Save batch of sample predictions for error analysis.
        
        Args:
            model_id: ID of the trained model
            predictions: List of dicts with keys:
                - sample_id: Training sample ID
                - true_label: Ground truth label
                - predicted_label: Model's predicted label
                - confidence: Prediction confidence (optional)
                - p_text: Text model probabilities (optional)
                - p_img: Image model probabilities (optional)
                - p_final: Final fused probabilities (optional)
                - split: Data split ('train', 'val', 'test')
        """
        if not predictions:
            return
            
        query = """
            INSERT INTO sample_predictions 
            (model_id, sample_id, true_label, predicted_label, confidence, p_text, p_img, p_final, split)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (model_id, sample_id) DO UPDATE SET
                true_label = EXCLUDED.true_label,
                predicted_label = EXCLUDED.predicted_label,
                confidence = EXCLUDED.confidence,
                p_text = EXCLUDED.p_text,
                p_img = EXCLUDED.p_img,
                p_final = EXCLUDED.p_final,
                split = EXCLUDED.split
        """
        
        # Batch insert
        conn = self.connect()
        with conn.cursor() as cursor:
            for pred in predictions:
                cursor.execute(query, (
                    model_id,
                    pred['sample_id'],
                    pred['true_label'],
                    pred['predicted_label'],
                    pred.get('confidence'),
                    Json(pred.get('p_text')) if pred.get('p_text') else None,
                    Json(pred.get('p_img')) if pred.get('p_img') else None,
                    Json(pred.get('p_final')) if pred.get('p_final') else None,
                    pred.get('split', 'val')
                ))
            conn.commit()

    def get_sample_predictions(self, model_id: int, true_label: str = None, 
                                predicted_label: str = None, split: str = None,
                                limit: int = 100) -> List[Dict]:
        """Get sample predictions for error analysis."""
        conditions = ["model_id = %s"]
        params = [model_id]
        
        if true_label:
            conditions.append("true_label = %s")
            params.append(true_label)
        if predicted_label:
            conditions.append("predicted_label = %s")
            params.append(predicted_label)
        if split:
            conditions.append("split = %s")
            params.append(split)
        
        params.append(limit)
        
        query = f"""
            SELECT * FROM sample_predictions
            WHERE {' AND '.join(conditions)}
            ORDER BY confidence ASC
            LIMIT %s
        """
        return self.execute(query, tuple(params), fetch=True) or []

    # ========================================================================
    # GATE WEIGHTS ANALYSIS
    # ========================================================================
    def save_gate_weights(self, model_id: int, gate_weights: Dict, split: str = 'val') -> None:
        """Save gate weights analysis for a model.
        
        Stores the gate weights in the model's metrics JSON field.
        
        Args:
            model_id: ID of the model
            gate_weights: Dict with per_class and overall gate weights stats
            split: Data split used for analysis ('val', 'train', 'test')
        """
        # Get current metrics
        query = "SELECT metrics FROM model_registry WHERE id = %s"
        result = self.execute(query, (model_id,), fetch=True)
        
        if not result:
            return
        
        current_metrics = result[0].get('metrics') or {}
        
        # Add gate weights to metrics
        current_metrics['gate_weights'] = {
            split: gate_weights,
            'analyzed_at': datetime.now().isoformat()
        }
        
        # Update metrics
        update_query = """
            UPDATE model_registry
            SET metrics = %s
            WHERE id = %s
        """
        self.execute(update_query, (Json(current_metrics), model_id))
    
    def get_gate_weights(self, model_id: int) -> Optional[Dict]:
        """Get gate weights analysis for a model.
        
        Returns:
            Dict with gate weights data or None if not available
        """
        query = "SELECT metrics FROM model_registry WHERE id = %s"
        result = self.execute(query, (model_id,), fetch=True)
        
        if not result:
            return None
        
        metrics = result[0].get('metrics') or {}
        return metrics.get('gate_weights')


# Global database client instance
db = DatabaseClient()
