"""
Script for updating vector index.

Usage:
    python scripts/update_index.py [--docs-path PATH] [--vectorstore-dir PATH]

This script:
1. Loads all .md files from docs path
2. Splits them into chunks
3. Recreates FAISS vector index
4. Saves index to vectorstore directory

Alternative: use /update_index endpoint via API.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from app.rag.indexing import build_or_load_vectorstore, load_mkdocs_documents
from app.settings import Settings

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
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Update FAISS vector index from documentation")
    parser.add_argument(
        "--docs-path",
        type=str,
        help="Path to documentation directory (default: from Settings.DOCS_PATH)"
    )
    parser.add_argument(
        "--vectorstore-dir",
        type=str,
        help="Path to vectorstore directory (default: from Settings.VECTORSTORE_DIR)"
    )
    args = parser.parse_args()
    
    # Get settings
    settings = Settings()
    
    # Use provided paths or fall back to settings
    docs_path = args.docs_path or settings.DOCS_PATH
    vectorstore_dir = args.vectorstore_dir or settings.VECTORSTORE_DIR
    
    logger.info("=" * 60)
    logger.info("UPDATING VECTOR INDEX")
    logger.info("=" * 60)
    logger.info(f"Docs path: {docs_path}")
    logger.info(f"Vectorstore dir: {vectorstore_dir}")
    
    try:
        # Load documents
        logger.info(f"Loading documents from {docs_path}...")
        documents = load_mkdocs_documents(docs_path=docs_path)
        
        if not documents:
            logger.error(f"No documents found for indexing in {docs_path}")
            logger.error("Make sure the docs path contains .md files")
            sys.exit(1)
        
        logger.info(f"Loaded {len(documents)} documents")
        
        # Build index (force rebuild)
        logger.info(f"Creating vector index in {vectorstore_dir}...")
        vectorstore = build_or_load_vectorstore(
            chunks=None,  # Will auto-load from docs_path
            index_path=vectorstore_dir,
            docs_path=docs_path,
            force_rebuild=True
        )
        
        logger.info("=" * 60)
        logger.info("Index successfully updated!")
        if hasattr(vectorstore, 'index') and hasattr(vectorstore.index, 'ntotal'):
            logger.info(f"Number of documents in index: {vectorstore.index.ntotal}")
            if hasattr(vectorstore.index, 'd'):
                logger.info(f"Vector dimension: {vectorstore.index.d}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error updating index: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

