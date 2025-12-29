"""
Optional script for updating vector index.

Usage:
    python update_index.py

This script:
1. Loads all .md files from data/mkdocs_docs
2. Splits them into chunks
3. Recreates FAISS vector index
4. Saves index to vectorstore/faiss_index

Alternative: use /update_index endpoint via API.
"""

import logging
import sys
from pathlib import Path

# Add project root directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from app.core.rag_chain import build_or_load_vectorstore, chunk_documents, load_mkdocs_documents

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function for updating index."""
    # Load environment variables
    load_dotenv()
    
    logger.info("=" * 60)
    logger.info("UPDATING VECTOR INDEX")
    logger.info("=" * 60)
    
    try:
        # Load documents
        logger.info("Loading documents from data/mkdocs_docs...")
        documents = load_mkdocs_documents()
        
        if not documents:
            logger.error("No documents found for indexing")
            logger.error("Make sure data/mkdocs_docs contains .md files")
            sys.exit(1)
        
        logger.info(f"Loaded {len(documents)} documents")
        
        # Split into chunks
        logger.info("Splitting documents into chunks...")
        chunks = chunk_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Recreate index
        logger.info("Creating vector index...")
        vectorstore = build_or_load_vectorstore(
            chunks=chunks,
            force_rebuild=True
        )
        
        logger.info("=" * 60)
        logger.info("Index successfully updated!")
        logger.info(f"Number of documents in index: {vectorstore.index.ntotal}")
        logger.info(f"Vector dimension: {vectorstore.index.d}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error updating index: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

