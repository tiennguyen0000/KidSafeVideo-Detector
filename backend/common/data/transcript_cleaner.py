"""Transcript preprocessing and cleaning utilities."""
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TranscriptCleaner:
    """Clean and preprocess video transcripts."""
    
    def __init__(self):
        # Common channel CTAs (case-insensitive patterns)
        self.cta_patterns = [
            r'\bsubscribe\b.*',  # "Subscribe" + anything after
            r'\blike\b.*?\bsubscribe\b',  # "like ... subscribe"
            r'\bshare\b.*?\bsubscribe\b',  # "share ... subscribe"
            r'subscribe\s+cho\s+kênh.*',
            r'đăng\s+ký\s+kênh.*',
            r'theo\s+dõi\s+kênh.*',
            r'bấm\s+chuông.*',
            r'nhấn\s+chuông.*',
            r'để\s+không\s+bỏ\s+lỡ.*',
            r'video\s+tiếp\s+theo.*',
            r'xem\s+thêm.*',  # "xem thêm" + rest of sentence
        ]
        
        # Channel-specific identifiers (prevent channel bias)
        self.channel_patterns = [
            r'ghiền\s+mì\s+gõ',
            r'kênh\s+\w+',  # Generic "kênh X"
            r'channel\s+\w+',
        ]
        
        # Intro/outro markers
        self.intro_outro_patterns = [
            r'^.*?xin\s+chào.*?(?=\.|!|\?)',  # Intro greetings
            r'cảm\s+ơn\s+.*?đã\s+xem.*$',     # Outro thanks
            r'hẹn\s+gặp\s+lại.*$',             # See you later
        ]
        
        # Compile all patterns
        self.all_patterns = (
            self.cta_patterns + 
            self.channel_patterns + 
            self.intro_outro_patterns
        )
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE | re.UNICODE) 
            for pattern in self.all_patterns
        ]
    
    def clean(self, transcript: str) -> str:
        """
        Clean transcript by removing noise.
        
        Args:
            transcript: Raw transcript text
            
        Returns:
            Cleaned transcript
        """
        if not transcript:
            return ""
        
        cleaned = transcript
        
        # 1. Remove CTAs and channel-specific content
        for pattern in self.compiled_patterns:
            cleaned = pattern.sub(' ', cleaned)  # Replace with space, not empty
        
        # 2. Remove multiple punctuation artifacts (e.g., ". .")
        cleaned = re.sub(r'[.!?]\s*[.!?]+', '.', cleaned)
        
        # 3. Remove standalone punctuation
        cleaned = re.sub(r'\s+[.!?,;:]\s+', ' ', cleaned)
        
        # 4. Remove multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # 5. Fix spacing around punctuation
        cleaned = re.sub(r'\s+([.!?,;:])', r'\1', cleaned)
        
        # 6. Remove leading/trailing punctuation and whitespace
        cleaned = cleaned.strip(' .!?,;:')
        
        # 7. Remove very short transcripts (likely just noise)
        if len(cleaned) < 10:
            logger.warning(f"Transcript too short after cleaning: '{cleaned}'")
            return ""
        
        return cleaned
    
    def clean_with_metadata(self, transcript: str) -> dict:
        """
        Clean transcript and return metadata.
        
        Returns:
            {
                'original': original text,
                'cleaned': cleaned text,
                'removed_chars': number of chars removed,
                'removed_percentage': percentage removed
            }
        """
        if not transcript:
            return {
                'original': '',
                'cleaned': '',
                'removed_chars': 0,
                'removed_percentage': 0.0
            }
        
        cleaned = self.clean(transcript)
        
        original_len = len(transcript)
        cleaned_len = len(cleaned)
        removed_chars = original_len - cleaned_len
        removed_percentage = (removed_chars / original_len * 100) if original_len > 0 else 0.0
        
        return {
            'original': transcript,
            'cleaned': cleaned,
            'removed_chars': removed_chars,
            'removed_percentage': round(removed_percentage, 2)
        }


# Global instance
transcript_cleaner = TranscriptCleaner()


def clean_transcript(transcript: Optional[str]) -> Optional[str]:
    """
    Clean a transcript string.
    
    Args:
        transcript: Raw transcript text or None
        
    Returns:
        Cleaned transcript or None if empty
    """
    if not transcript:
        return None
    
    cleaned = transcript_cleaner.clean(transcript)
    
    # Return None if cleaned transcript is empty
    return cleaned if cleaned else None


def preprocess_transcript_batch(transcripts: list) -> list:
    """
    Clean a batch of transcripts.
    
    Args:
        transcripts: List of transcript strings
        
    Returns:
        List of cleaned transcripts
    """
    return [clean_transcript(t) for t in transcripts]
