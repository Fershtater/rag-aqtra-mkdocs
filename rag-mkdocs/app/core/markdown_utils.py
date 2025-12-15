"""
Utilities for working with Markdown documents.

Functions for extracting sections, anchors and improved chunking.
"""

import re
from typing import List, Tuple, Optional


def extract_sections(text: str) -> List[Tuple[int, str, str]]:
    """
    Extracts sections from Markdown text.
    
    Args:
        text: Markdown text
        
    Returns:
        List of tuples (level, title, section_content)
    """
    sections = []
    lines = text.split('\n')
    current_section_level = 0
    current_section_title = ""
    current_section_content = []
    
    for line in lines:
        # Check if line is a header
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if header_match:
            # Save previous section if it exists
            if current_section_content:
                sections.append((
                    current_section_level,
                    current_section_title,
                    '\n'.join(current_section_content)
                ))
            
            # Start new section
            current_section_level = len(header_match.group(1))
            current_section_title = header_match.group(2).strip()
            current_section_content = []
        else:
            current_section_content.append(line)
    
    # Add last section
    if current_section_content:
        sections.append((
            current_section_level,
            current_section_title,
            '\n'.join(current_section_content)
        ))
    
    return sections


def slugify(text: str) -> str:
    """
    Converts text to slug for anchors.
    
    Args:
        text: Header text
        
    Returns:
        Slug (e.g., "app-development" from "App Development")
    """
    # Convert to lowercase
    text = text.lower()
    # Replace spaces and special characters with hyphens
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    # Remove hyphens at start and end
    return text.strip('-')


def find_section_for_text(text: str, position: int) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Finds section that text at specified position belongs to.
    
    Args:
        text: Full document text
        position: Position in text
        
    Returns:
        Tuple (title, level, anchor) or (None, None, None)
    """
    sections = extract_sections(text)
    current_pos = 0
    last_title = None
    last_level = None
    last_anchor = None
    
    for level, title, content in sections:
        section_start = current_pos
        section_end = current_pos + len(content)
        
        if section_start <= position <= section_end:
            return title, level, slugify(title)
        
        # Update last encountered section
        if level <= (last_level or 999):  # Higher header level
            last_title = title
            last_level = level
            last_anchor = slugify(title)
        
        current_pos = section_end + len(title) + level + 2  # +2 for "# " and "\n"
    
    return last_title, last_level, last_anchor


def build_doc_url(base_url: str, source: str, section_anchor: Optional[str] = None) -> str:
    """
    Builds full document URL with optional section anchor.
    
    Args:
        base_url: Base documentation URL (e.g., "https://docs.aqtra.io/")
        source: File path relative to docs/ (e.g., "docs/app-development/button.md")
        section_anchor: Section anchor (e.g., "primary-button") or None
        
    Returns:
        Full URL (e.g., "https://docs.aqtra.io/app-development/button.html#primary-button")
    """
    # Remove docs/ prefix if present
    if source.startswith("docs/"):
        path = source[5:]  # Remove "docs/"
    else:
        path = source
    
    # Remove .md
    if path.endswith(".md"):
        path = path[:-3]
    
    # Replace separators with /
    path = path.replace("\\", "/")
    
    # Build URL
    url = f"{base_url.rstrip('/')}/{path}.html"
    
    # Add anchor if present
    if section_anchor:
        url += f"#{section_anchor}"
    
    return url

