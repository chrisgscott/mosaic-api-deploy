"""
Unit tests for DocProcessor
"""

import pytest
from doc_processor import DocProcessor, DocArtifact


def test_process_bytes_simple_text():
    """Test basic text processing."""
    processor = DocProcessor()
    
    content = b"Hello world!\nThis is a test document.\n\n\nWith multiple lines."
    artifact = processor.process_bytes("test.txt", content)
    
    assert artifact.text == "Hello world!\nThis is a test document.\n\nWith multiple lines."
    assert artifact.metadata['filename'] == "test.txt"
    assert artifact.metadata['size_bytes'] == len(content)
    assert artifact.metadata['mime_type'] == "text/plain"
    assert len(artifact.metadata['content_hash']) == 64  # SHA256 hex
    assert artifact.metadata['text_length'] == len(artifact.text)
    assert artifact.warnings == []


def test_process_bytes_unicode_normalization():
    """Test Unicode normalization."""
    processor = DocProcessor()
    
    # Content with Unicode normalization issues
    content = "Café naïve résumé".encode('utf-8')
    artifact = processor.process_bytes("unicode.txt", content)
    
    assert "Café naïve résumé" in artifact.text
    assert artifact.metadata['mime_type'] == "text/plain"
    assert artifact.warnings == []


def test_process_bytes_encoding_errors():
    """Test handling of encoding errors."""
    processor = DocProcessor()
    
    # Invalid UTF-8 bytes
    content = b"Hello \xff\xfe world"
    artifact = processor.process_bytes("bad_encoding.txt", content)
    
    assert "Hello" in artifact.text
    assert "world" in artifact.text
    assert len(artifact.warnings) > 0
    assert "UTF-8 decode errors" in artifact.warnings[0]


def test_process_bytes_whitespace_normalization():
    """Test whitespace normalization."""
    processor = DocProcessor()
    
    content = b"Line 1   with   spaces\n\n\n\nLine 2\t\twith\ttabs\n\n\n"
    artifact = processor.process_bytes("whitespace.txt", content)
    
    # Should normalize multiple spaces and excessive newlines
    assert "Line 1 with spaces" in artifact.text
    assert "Line 2 with tabs" in artifact.text
    assert "\n\n\n\n" not in artifact.text  # No more than 2 consecutive newlines


def test_content_hash_consistency():
    """Test that identical content produces identical hashes."""
    processor = DocProcessor()
    
    content = b"Test content for hash consistency"
    
    artifact1 = processor.process_bytes("file1.txt", content)
    artifact2 = processor.process_bytes("file2.txt", content)  # Different filename
    
    # Same content should produce same hash regardless of filename
    assert artifact1.metadata['content_hash'] == artifact2.metadata['content_hash']
    assert artifact1.text == artifact2.text


if __name__ == "__main__":
    # Run basic tests
    test_process_bytes_simple_text()
    test_process_bytes_unicode_normalization()
    test_process_bytes_encoding_errors()
    test_process_bytes_whitespace_normalization()
    test_content_hash_consistency()
    print("✅ All DocProcessor tests passed!")
