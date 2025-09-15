"""
Document Processing Module

Handles file format detection, text extraction, normalization, and metadata generation
before documents are passed to MindsDB handlers.
"""

import hashlib
import mimetypes
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
# import boto3  # Removed - using Supabase storage instead of AWS S3
import os
import io
import base64
from pypdf import PdfReader
from PIL import Image


@dataclass
class DocArtifact:
    """Processed document artifact with normalized text and metadata."""
    text: str
    metadata: Dict
    warnings: List[str]


class DocProcessor:
    """Document processor for extracting and normalizing text from various sources."""
    
    def __init__(self, s3_client=None, enable_visual_ai=False, openai_client=None):
        self.s3_client = s3_client
        self.enable_visual_ai = enable_visual_ai
        self.openai_client = openai_client
        self.visual_processing_cost = 0.0
        
    def process_bytes(self, filename: str, content: bytes) -> DocArtifact:
        """Process raw bytes into normalized text artifact."""
        start_time = time.time()
        warnings = []
        
        # Detect MIME type
        mime_type, _ = mimetypes.guess_type(filename)
        if not mime_type:
            mime_type = "application/octet-stream"
            
        # Text extraction with PDF support
        metadata_extra = {}
        if mime_type == 'application/pdf':
            text, pdf_metadata = self._extract_pdf_text_and_images(content)
            # Merge PDF-specific metadata
            for key, value in pdf_metadata.items():
                if key not in ['warnings']:  # Don't overwrite warnings
                    metadata_extra[key] = value
            if pdf_metadata.get('warnings'):
                warnings.extend(pdf_metadata['warnings'])
        elif mime_type.startswith('text/') or mime_type == 'application/octet-stream':
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                text = content.decode('utf-8', errors='replace')
                warnings.append("UTF-8 decode errors replaced with placeholder characters")
        else:
            # Future: add DOCX, etc. handlers here
            text = content.decode('utf-8', errors='replace')
            warnings.append(f"Unsupported MIME type {mime_type}, treating as text")
            
        # Text normalization
        text = self._normalize_text(text)
        
        # Generate content hash and structural analysis
        content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        structural_hash = hashlib.sha256(self._extract_structure(text).encode('utf-8')).hexdigest()
        
        # Analyze text characteristics
        analysis = self._analyze_text(text)
        
        # Build comprehensive metadata
        metadata = {
            # Basic file info
            'filename': filename,
            'size_bytes': len(content),
            'mime_type': mime_type,
            'content_hash': content_hash,
            'text_length': len(text),
            
            # Processing info
            'processed_at': datetime.now().isoformat(),
            'processor_version': '0.1.0',
            'processing_time_ms': int((time.time() - start_time) * 1000),
            'original_encoding': 'utf-8' if mime_type != 'application/pdf' else 'pdf',
            'extraction_method': 'pdf' if mime_type == 'application/pdf' else 'text',
            
            # Text analysis
            'language': analysis['language'],
            'paragraph_count': analysis['paragraph_count'],
            'line_count': analysis['line_count'],
            'estimated_reading_time': analysis['estimated_reading_time'],
            'text_quality_score': analysis['text_quality_score'],
            'complexity_score': analysis['complexity_score'],
            
            # Structural analysis
            'has_tables': analysis['has_tables'],
            'has_code_blocks': analysis['has_code_blocks'],
            'structural_hash': structural_hash,
            'similarity_hash': self._generate_similarity_hash(text)
        }
        
        # Add PDF-specific metadata if available
        if metadata_extra:
            metadata.update(metadata_extra)
        
        return DocArtifact(
            text=text,
            metadata=metadata,
            warnings=warnings
        )
    
    def process_s3(self, bucket: str, key: str) -> DocArtifact:
        """Process document from S3 object."""
        if not self.s3_client:
            raise ValueError("S3 client not configured")
            
        # Fetch object from S3
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read()
        except Exception as e:
            raise ValueError(f"Failed to fetch s3://{bucket}/{key}: {e}")
            
        # Extract filename from key
        filename = key.split('/')[-1] if '/' in key else key
        
        # Process the content
        artifact = self.process_bytes(filename, content)
        
        # Add S3-specific metadata
        artifact.metadata['source_uri'] = f's3://{bucket}/{key}'
        artifact.metadata['s3_bucket'] = bucket
        artifact.metadata['s3_key'] = key
        
        return artifact
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent processing."""
        # Unicode normalization (NFKC - canonical decomposition + canonical composition)
        text = unicodedata.normalize('NFKC', text)
        
        # Remove null bytes and other control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        
        # Normalize whitespace (collapse multiple spaces, normalize line endings)
        lines = text.split('\n')
        normalized_lines = []
        for line in lines:
            # Collapse multiple spaces to single space, strip trailing whitespace
            normalized_line = ' '.join(line.split())
            normalized_lines.append(normalized_line)
            
        # Join with single newlines, remove excessive blank lines
        text = '\n'.join(normalized_lines)
        
        # Remove more than 2 consecutive newlines
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')
            
        return text.strip()
    
    def _extract_pdf_text_and_images(self, content: bytes) -> Tuple[str, Dict]:
        """Extract text and images from PDF content with spatial positioning."""
        warnings = []
        pdf_metadata = {}
        
        try:
            # Create PDF reader from bytes
            pdf_file = io.BytesIO(content)
            reader = PdfReader(pdf_file)
            
            # Extract basic PDF metadata
            pdf_metadata['page_count'] = len(reader.pages)
            pdf_metadata['pdf_version'] = reader.metadata.get('/Producer', 'Unknown') if reader.metadata else 'Unknown'
            
            # Extract text and images with positioning
            text_parts = []
            all_images = []
            
            for page_num, page in enumerate(reader.pages):
                try:
                    # Extract text for this page
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_parts.append(page_text)
                    
                    # Extract images with position data
                    page_images = self._extract_images_from_page(page, page_num + 1)
                    all_images.extend(page_images)
                    
                except Exception as e:
                    warnings.append(f"Failed to extract content from page {page_num + 1}: {str(e)}")
            
            # Process images with AI if enabled
            if self.enable_visual_ai and self.openai_client and all_images:
                visual_content = self._process_images_with_ai(all_images, text_parts)
                full_text = self._integrate_visual_content_into_text(text_parts, visual_content)
                pdf_metadata['visual_content'] = visual_content
                pdf_metadata['visual_processing_cost'] = self.visual_processing_cost
            else:
                full_text = '\n\n'.join(text_parts)
                pdf_metadata['visual_content'] = {
                    'image_count': len(all_images),
                    'images': [],
                    'ai_analysis_enabled': self.enable_visual_ai and self.openai_client is not None
                }
            
            # PDF-specific quality indicators
            if not full_text.strip():
                warnings.append("PDF appears to contain no extractable text (may be image-based)")
                pdf_metadata['text_extractable'] = False
            else:
                pdf_metadata['text_extractable'] = True
            
            # Check for potential OCR needs
            if pdf_metadata['page_count'] > 0:
                avg_chars_per_page = len(full_text) / pdf_metadata['page_count']
                if avg_chars_per_page < 50:
                    warnings.append("PDF may contain scanned images requiring OCR")
                    pdf_metadata['likely_scanned'] = True
                else:
                    pdf_metadata['likely_scanned'] = False
            
            pdf_metadata['warnings'] = warnings
            pdf_metadata['visual_processing_cost'] = self.visual_processing_cost
            return full_text, pdf_metadata
            
        except Exception as e:
            error_msg = f"Failed to process PDF: {str(e)}"
            warnings.append(error_msg)
            pdf_metadata['warnings'] = warnings
            pdf_metadata['extraction_error'] = str(e)
            return "", pdf_metadata
    
    def _analyze_text(self, text: str) -> Dict:
        """Analyze text characteristics for metadata."""
        lines = text.split('\n')
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        words = text.split()
        
        # Basic counts
        line_count = len(lines)
        paragraph_count = len(paragraphs)
        word_count = len(words)
        
        # Reading time estimation (average 200 words per minute)
        estimated_reading_time = max(1, round(word_count / 200))
        
        # Language detection (simple heuristic for now)
        language = self._detect_language(text)
        
        # Text quality score (based on character distribution)
        quality_score = self._calculate_quality_score(text)
        
        # Complexity score (based on sentence length and vocabulary)
        complexity_score = self._calculate_complexity_score(text, words)
        
        # Structural detection
        has_tables = self._detect_tables(text)
        has_code_blocks = self._detect_code_blocks(text)
        
        return {
            'language': language,
            'paragraph_count': paragraph_count,
            'line_count': line_count,
            'estimated_reading_time': estimated_reading_time,
            'text_quality_score': quality_score,
            'complexity_score': complexity_score,
            'has_tables': has_tables,
            'has_code_blocks': has_code_blocks
        }
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection (defaulting to English for now)."""
        # Future: integrate langdetect library
        # For now, simple heuristic based on common words
        english_indicators = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that']
        text_lower = text.lower()
        english_count = sum(1 for word in english_indicators if word in text_lower)
        
        return 'en' if english_count >= 3 else 'unknown'
    
    def _calculate_quality_score(self, text: str) -> float:
        """Calculate text quality score (0.0-1.0)."""
        if not text.strip():
            return 0.0
            
        # Factors that indicate good quality text
        total_chars = len(text)
        printable_chars = len([c for c in text if c.isprintable() or c in '\n\r\t'])
        alpha_chars = len([c for c in text if c.isalpha()])
        
        # Quality indicators
        printable_ratio = printable_chars / total_chars if total_chars > 0 else 0
        alpha_ratio = alpha_chars / total_chars if total_chars > 0 else 0
        
        # Penalize excessive repetition
        repetition_penalty = 1.0
        if len(text) > 100:
            # Check for repeated patterns
            sample = text[:500]  # Sample first 500 chars
            unique_chars = len(set(sample))
            char_diversity = unique_chars / len(sample) if sample else 0
            repetition_penalty = min(1.0, char_diversity * 2)
        
        quality_score = (printable_ratio * 0.3 + alpha_ratio * 0.4 + repetition_penalty * 0.3)
        return min(1.0, max(0.0, quality_score))
    
    def _calculate_complexity_score(self, text: str, words: List[str]) -> float:
        """Calculate text complexity score (0.0-1.0)."""
        if not words:
            return 0.0
            
        # Sentence length analysis
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
            
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # Vocabulary diversity
        unique_words = len(set(word.lower() for word in words))
        vocabulary_diversity = unique_words / len(words) if words else 0
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Normalize and combine factors
        sentence_complexity = min(1.0, avg_sentence_length / 20)  # 20 words = high complexity
        vocab_complexity = vocabulary_diversity  # Already 0-1
        word_length_complexity = min(1.0, avg_word_length / 8)  # 8 chars = high complexity
        
        complexity_score = (sentence_complexity * 0.4 + vocab_complexity * 0.3 + word_length_complexity * 0.3)
        return min(1.0, max(0.0, complexity_score))
    
    def _detect_tables(self, text: str) -> bool:
        """Detect if text contains table-like structures."""
        lines = text.split('\n')
        
        # Look for common table indicators
        table_indicators = [
            lambda line: '|' in line and line.count('|') >= 2,  # Markdown tables
            lambda line: '\t' in line and line.count('\t') >= 2,  # Tab-separated
            lambda line: '  ' in line and len(line.split('  ')) >= 3,  # Space-separated columns
        ]
        
        table_lines = 0
        for line in lines:
            if any(indicator(line.strip()) for indicator in table_indicators):
                table_lines += 1
                
        # If more than 10% of lines look like tables, consider it has tables
        return table_lines > len(lines) * 0.1 and table_lines >= 2
    
    def _detect_code_blocks(self, text: str) -> bool:
        """Detect if text contains code blocks."""
        # Common code indicators
        code_indicators = [
            '```',  # Markdown code blocks
            '    ',  # Indented code (4+ spaces at line start)
            'def ',  # Python functions
            'function ',  # JavaScript functions
            'class ',  # Class definitions
            'import ',  # Import statements
            '#!/',  # Shebang lines
            '{',  # Curly braces (common in many languages)
            '}',
            '();',  # Function calls with semicolons
        ]
        
        code_score = 0
        lines = text.split('\n')
        
        for line in lines:
            stripped = line.strip()
            if any(indicator in line for indicator in code_indicators):
                code_score += 1
            # Check for indented blocks (common in Python, YAML, etc.)
            if line.startswith('    ') and stripped:
                code_score += 0.5
                
        # If more than 15% of content suggests code, mark as having code blocks
        return code_score > len(lines) * 0.15
    
    def _extract_structure(self, text: str) -> str:
        """Extract structural elements for structural hashing."""
        lines = text.split('\n')
        structure_elements = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
                
            # Detect headings (markdown style)
            if stripped.startswith('#'):
                level = len(stripped) - len(stripped.lstrip('#'))
                structure_elements.append(f"H{level}")
            # Detect list items
            elif stripped.startswith(('- ', '* ', '+ ')) or re.match(r'^\d+\.', stripped):
                structure_elements.append("LIST")
            # Detect potential table rows
            elif '|' in stripped or '\t' in stripped:
                structure_elements.append("TABLE")
            # Regular paragraph
            else:
                structure_elements.append("P")
                
        return ''.join(structure_elements)
    
    def _generate_similarity_hash(self, text: str) -> str:
        """Generate a fuzzy hash for near-duplicate detection."""
        # Simple similarity hash based on word frequency
        words = re.findall(r'\w+', text.lower())
        if not words:
            return hashlib.sha256(b'').hexdigest()[:16]
            
        # Get most common words (ignoring very common stop words)
        stop_words = {'the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with', 'for', 'as', 'was', 'on', 'are', 'you'}
        filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Create frequency signature
        word_freq = {}
        for word in filtered_words[:100]:  # Limit to first 100 significant words
            word_freq[word] = word_freq.get(word, 0) + 1
            
        # Sort by frequency and create signature
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        signature = ''.join([word[:3] for word, _ in top_words])
        
        return hashlib.sha256(signature.encode()).hexdigest()[:16]
    
    def _extract_images_from_page(self, page, page_num: int) -> List[Dict]:
        """Extract images from a PDF page with spatial positioning."""
        images = []
        
        if '/XObject' in page.get('/Resources', {}):
            xobjects = page['/Resources']['/XObject'].get_object()
            
            for obj_name in xobjects:
                obj = xobjects[obj_name]
                if obj.get('/Subtype') == '/Image':
                    try:
                        # Let pypdf handle decompression automatically
                        image_data = obj.get_data()
                        width = obj.get('/Width', 0)
                        height = obj.get('/Height', 0)
                        
                        # Skip very small images (likely decorative)
                        if width < 50 or height < 50 or len(image_data) < 100:
                            continue
                        
                        # Store filter info for debugging
                        filter_info = obj.get('/Filter', 'None')
                        
                        images.append({
                            'data': image_data,
                            'page_number': page_num,
                            'width': width,
                            'height': height,
                            'x': width // 2,  # Simplified positioning
                            'y': height // 2,
                            'name': obj_name,
                            'filter': filter_info,
                            'colorspace': obj.get('/ColorSpace'),
                            'bits_per_component': obj.get('/BitsPerComponent', 8)
                        })
                    except Exception as e:
                        # Skip problematic images
                        continue
        
        return images
    
    def _estimate_image_position(self, page, obj_name: str) -> Dict:
        """Estimate image position on page (simplified approach)."""
        # This is a simplified estimation - full position extraction requires
        # parsing the page's content stream which is complex
        page_height = float(page.mediabox.height)
        page_width = float(page.mediabox.width)
        
        # Return estimated center position for now
        # In a full implementation, we'd parse the content stream
        return {
            'x': page_width / 2,
            'y': page_height / 2,
            'estimated': True
        }
    
    def _process_images_with_ai(self, images: List[Dict], text_parts: List[str]) -> Dict:
        """Process extracted images with GPT-4o-mini vision API."""
        processed_images = []
        total_cost = 0.0
        
        for img in images:
            try:
                # Get surrounding text context
                context = self._get_image_context(img, text_parts)
                context['image_info'] = img  # Pass image metadata for FlateDecode handling
                
                # Analyze image with AI
                analysis = self._analyze_image_with_ai(img['data'], context)
                
                processed_images.append({
                    'page_number': img['page_number'],
                    'width': img['width'],
                    'height': img['height'],
                    'description': analysis.get('description', ''),
                    'type': analysis.get('type', 'unknown'),
                    'confidence': analysis.get('confidence', 'none')
                })
                
                # Track cost (approximate: $0.00015 per image for GPT-4o-mini)
                total_cost += 0.00015
                
            except Exception as e:
                print(f"Warning: Failed to analyze image on page {img['page_number']}: {e}")
        
        self.visual_processing_cost += total_cost
        
        return {
            'images': processed_images,
            'total_cost': total_cost,
            'ai_analysis_enabled': True
        }
        
        return {
            'image_count': len(images),
            'processed_count': len(processed_images),
            'images': processed_images,
            'ai_analysis_enabled': True,
            'processing_cost': total_cost
        }
    
    def _get_image_context(self, image: Dict, text_parts: List[str]) -> Dict:
        """Get surrounding text context for an image."""
        page_num = image['page_number'] - 1  # 0-indexed
        
        if page_num < len(text_parts):
            page_text = text_parts[page_num]
            
            # Split into sentences/paragraphs for context
            sentences = page_text.split('. ')
            
            # For now, use first and last parts as context
            # In full implementation, we'd use position to find nearby text
            preceding_text = '. '.join(sentences[:2]) if len(sentences) > 1 else sentences[0][:200]
            following_text = '. '.join(sentences[-2:]) if len(sentences) > 1 else sentences[-1][:200]
            
            # Extract headings (lines that are short and might be titles)
            lines = page_text.split('\n')
            headings = [line.strip() for line in lines if len(line.strip()) < 100 and line.strip().isupper()]
            
            return {
                'page_number': image['page_number'],
                'preceding_text': preceding_text,
                'following_text': following_text,
                'nearby_headings': headings[:3]  # Top 3 headings
            }
        
        return {
            'page_number': image['page_number'],
            'preceding_text': '',
            'following_text': '',
            'nearby_headings': []
        }
    
    def _analyze_image_with_ai(self, image_data: bytes, context: Dict) -> Dict:
        """Analyze image using GPT-4o-mini vision with context."""
        try:
            # Convert image to supported format (PNG/JPEG)
            converted_image_data = self._convert_image_to_supported_format(image_data, context.get('image_info'))
            if not converted_image_data:
                return {
                    'description': "Failed to convert image to supported format",
                    'type': 'unknown',
                    'confidence': 'none'
                }
            
            # Convert image to base64
            image_b64 = base64.b64encode(converted_image_data).decode('utf-8')
            
            # Create context-aware prompt
            prompt = f"""Analyze this image from page {context['page_number']} of a document.

Context before image: "{context['preceding_text'][:200]}..."
Context after image: "{context['following_text'][:200]}..."
Nearby headings: {context['nearby_headings']}

Based on this context, describe the image and extract any data or insights. 
If it's a chart/graph, extract key data points. If it's a diagram, describe components.
Focus on how it relates to the surrounding content. Be concise but comprehensive."""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                    ]
                }],
                max_tokens=300
            )
            
            description = response.choices[0].message.content
            
            return {
                'description': description,
                'type': self._classify_visual_content(description),
                'confidence': 'high'  # GPT-4o-mini is generally reliable
            }
            
        except Exception as e:
            return {
                'description': f"Failed to analyze image: {str(e)}",
                'type': 'unknown',
                'confidence': 'none'
            }
    
    def _convert_image_to_supported_format(self, image_data: bytes, image_info: Dict = None) -> bytes:
        """Convert image to PNG format for GPT-4o-mini compatibility."""
        try:
            # Skip very small images (likely icons or artifacts)
            if len(image_data) < 100:
                return None
            
            # Try to open the image with PIL
            image_file = io.BytesIO(image_data)
            image = Image.open(image_file)
            
            # Skip very small images (likely icons or decorative elements)
            if image.width < 50 or image.height < 50:
                return None
            
            # Convert to RGB if necessary (for PNG compatibility)
            if image.mode in ('RGBA', 'LA', 'P'):
                # Keep transparency for RGBA, convert others to RGB
                if image.mode != 'RGBA':
                    image = image.convert('RGB')
            elif image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            
            # Save as PNG
            output_buffer = io.BytesIO()
            image.save(output_buffer, format='PNG', optimize=True)
            return output_buffer.getvalue()
            
        except Exception as e:
            # For FlateDecode and other compressed formats, try alternative approach
            if image_info:
                return self._handle_flatdecode_image(image_data, image_info)
            return None
    
    def _handle_flatdecode_image(self, image_data: bytes, image_info: Dict) -> bytes:
        """Handle FlateDecode images using PDF metadata for reconstruction."""
        try:
            width = image_info.get('width', 0)
            height = image_info.get('height', 0)
            colorspace = image_info.get('colorspace')
            bits_per_component = image_info.get('bits_per_component', 8)
            
            if width <= 0 or height <= 0:
                return None
            
            # Calculate expected data size based on colorspace
            if colorspace == '/DeviceGray':
                expected_size = width * height * (bits_per_component // 8)
                mode = 'L'
            elif colorspace == '/DeviceRGB':
                expected_size = width * height * 3 * (bits_per_component // 8)
                mode = 'RGB'
            elif colorspace == '/DeviceCMYK':
                expected_size = width * height * 4 * (bits_per_component // 8)
                mode = 'CMYK'
            else:
                # Try RGB as default
                expected_size = width * height * 3 * (bits_per_component // 8)
                mode = 'RGB'
            
            # Check if data size matches expectations
            if len(image_data) == expected_size:
                try:
                    image = Image.frombytes(mode, (width, height), image_data)
                    
                    # Convert CMYK to RGB for compatibility
                    if mode == 'CMYK':
                        image = image.convert('RGB')
                    
                    output_buffer = io.BytesIO()
                    image.save(output_buffer, format='PNG')
                    return output_buffer.getvalue()
                    
                except Exception:
                    pass
            
            return None
            
        except Exception:
            return None

    def _handle_special_image_formats(self, image_data: bytes) -> bytes:
        """Handle special image formats that PIL can't directly process."""
        try:
            # For FlateDecode images, pypdf returns raw pixel data
            # We need to reconstruct the image using the dimensions from the PDF object
            data_size = len(image_data)
            
            # Common bit depths and color spaces for PDF images
            # Most FlateDecode images are 8-bit RGB or RGBA
            bytes_per_pixel_options = [1, 3, 4]  # Grayscale, RGB, RGBA
            
            # Try to find matching dimensions based on data size
            for bytes_per_pixel in bytes_per_pixel_options:
                if data_size % bytes_per_pixel == 0:
                    total_pixels = data_size // bytes_per_pixel
                    
                    # Try common aspect ratios and sizes
                    for width in range(50, 3000, 10):
                        if total_pixels % width == 0:
                            height = total_pixels // width
                            
                            if 50 <= height <= 3000 and width * height * bytes_per_pixel == data_size:
                                try:
                                    if bytes_per_pixel == 1:
                                        image = Image.frombytes('L', (width, height), image_data)
                                    elif bytes_per_pixel == 3:
                                        image = Image.frombytes('RGB', (width, height), image_data)
                                    elif bytes_per_pixel == 4:
                                        image = Image.frombytes('RGBA', (width, height), image_data)
                                    
                                    # Verify the image makes sense (not just noise)
                                    if self._validate_reconstructed_image(image):
                                        output_buffer = io.BytesIO()
                                        image.save(output_buffer, format='PNG')
                                        return output_buffer.getvalue()
                                        
                                except Exception:
                                    continue
            
            return None
            
        except Exception:
            return None
    
    def _validate_reconstructed_image(self, image: Image.Image) -> bool:
        """Validate that a reconstructed image contains meaningful content."""
        try:
            # Basic validation - check if image has reasonable variance
            # (not just solid color or pure noise)
            import numpy as np
            
            # Convert to array for analysis
            img_array = np.array(image)
            
            # Check if image has reasonable variance (not solid color)
            if img_array.std() < 5:
                return False
                
            # Check if it's not pure noise (too much variance)
            if img_array.std() > 100:
                return False
                
            return True
            
        except Exception:
            # If numpy isn't available or validation fails, assume valid
            return True
    
    def _classify_visual_content(self, description: str) -> str:
        """Classify visual content type based on AI description."""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['chart', 'graph', 'plot', 'bar', 'line', 'pie']):
            return 'chart'
        elif any(word in description_lower for word in ['diagram', 'flowchart', 'schematic']):
            return 'diagram'
        elif any(word in description_lower for word in ['table', 'grid', 'rows', 'columns']):
            return 'table'
        elif any(word in description_lower for word in ['photo', 'image', 'picture']):
            return 'photo'
        else:
            return 'other'
    
    def _integrate_visual_content_into_text(self, text_parts: List[str], visual_content: Dict) -> str:
        """Integrate AI visual analysis into text at appropriate positions."""
        if not visual_content.get('images'):
            return '\n\n'.join(text_parts)
        
        # Group images by page
        images_by_page = {}
        for img in visual_content['images']:
            page_num = img['page_number'] - 1  # 0-indexed
            if page_num not in images_by_page:
                images_by_page[page_num] = []
            images_by_page[page_num].append(img)
        
        # Insert visual analysis into each page's text
        enhanced_text_parts = []
        for page_num, page_text in enumerate(text_parts):
            if page_num in images_by_page:
                # Add visual content analysis to this page
                visual_analyses = []
                for img in images_by_page[page_num]:
                    analysis_text = f"\n\n[VISUAL CONTENT: {img['ai_analysis']['description']}]\n\n"
                    visual_analyses.append(analysis_text)
                
                # Insert visual content (simplified: add at end of page)
                enhanced_page = page_text + ''.join(visual_analyses)
                enhanced_text_parts.append(enhanced_page)
            else:
                enhanced_text_parts.append(page_text)
        
        return '\n\n'.join(enhanced_text_parts)


def create_doc_processor(mock_s3_mode: bool = False, enable_visual_ai: bool = False) -> DocProcessor:
    """Factory function to create DocProcessor with OpenAI client. S3 functionality removed."""
    openai_client = None
    
    # Initialize OpenAI client if API key is available
    try:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if openai_api_key:
            import openai
            openai_client = openai.OpenAI(api_key=openai_api_key)
        else:
            print("Warning: OPENAI_API_KEY not found in environment variables")
    except Exception as e:
        if enable_visual_ai:
            print(f"Warning: Failed to initialize OpenAI client: {e}")
    
    # Always return DocProcessor without S3 client since we use Supabase storage
    return DocProcessor(s3_client=None, enable_visual_ai=enable_visual_ai, openai_client=openai_client)
