# Refactoring Status

## Phase 0: Repo Hygiene ✅ COMPLETED
- Updated .gitignore to ignore var/, app/vectorstore/, **/faiss_index/
- Created var/ directory structure (var/vectorstore/faiss_index, var/db, var/logs)
- Updated get_vectorstore_dir() to use VECTORSTORE_DIR env var (default: var/vectorstore/faiss_index)
- Updated build_or_load_vectorstore() to use configurable index_path

## Phase 1: Split API into Routers - IN PROGRESS
- ✅ Created app/api/routes/ directory
- ✅ Created health.py router
- ✅ Created metrics.py router
- ✅ Created admin_index.py router
- ✅ Created app/api/schemas.py (moved Query, QueryResponse, EscalateRequest, ErrorResponse)
- ⏳ TODO: Create query_v1.py router
- ⏳ TODO: Create answer_v2.py router
- ⏳ TODO: Create stream.py router
- ⏳ TODO: Create prompt_debug.py router
- ⏳ TODO: Create escalate.py router (if needed)
- ⏳ TODO: Update main.py to use routers

## Phase 2: Create Services Layer - PENDING
- ⏳ TODO: Create app/services/ directory
- ⏳ TODO: Create AnswerService
- ⏳ TODO: Create ConversationService
- ⏳ TODO: Create PromptService

## Phase 3: Split rag_chain - PENDING
- ⏳ TODO: Create app/rag/ directory
- ⏳ TODO: Create indexing.py
- ⏳ TODO: Create retrieval.py
- ⏳ TODO: Create chain.py
- ⏳ TODO: Create not_found.py
- ⏳ TODO: Update app/core/rag_chain.py as compatibility facade

## Phase 4: Schemas Cleanup - PENDING
- ⏳ TODO: Move app/api/v2_models.py to app/api/schemas/v2.py
- ⏳ TODO: Update imports

## Next Steps
1. Complete Phase 1 by creating remaining routers
2. Update main.py to use routers via include_router()
3. Test backward compatibility
4. Proceed with Phase 2-4

