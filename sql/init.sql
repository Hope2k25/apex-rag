-- Apex RAG System - PostgreSQL Schema Initialization
-- This file is automatically run when the Docker container starts
-- Based on Plan B: Data Infrastructure

-- ============================================
-- EXTENSIONS
-- ============================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
-- CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ============================================
-- TABLE: semantic_chunks
-- Stores document chunks with embeddings for vector search
-- ============================================

CREATE TABLE IF NOT EXISTS semantic_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_file TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    header_path TEXT,  -- Breadcrumb path (e.g., "Doc > Section > Subsection")
    content TEXT NOT NULL,
    content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    -- embedding float8[],  -- gte-modernbert-base produces 768-dim vectors
    embedding float8[], -- FALLBACK: Use float array for Windows without pgvector
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    content_hash TEXT,
    
    -- ========================================
    -- LLM ENRICHMENT FIELDS (Future expansion)
    -- These fields are populated by LLM analysis after initial deterministic ingestion
    -- ========================================
    llm_summary TEXT,                    -- LLM-generated summary of the chunk
    llm_questions TEXT[],                -- Questions this chunk can answer
    llm_keywords TEXT[],                 -- LLM-extracted keywords
    llm_domain_tags TEXT[],              -- Domain classification (e.g., "web-api", "database")
    llm_use_case_tags TEXT[],            -- Use case tags (e.g., "implementation", "debugging")
    llm_quality_score FLOAT,             -- LLM assessment of content quality (0-1)
    llm_enriched_at TIMESTAMPTZ,         -- When LLM enrichment was performed
    llm_enrichment_model TEXT,           -- Which model was used for enrichment
    llm_enrichment_version INTEGER DEFAULT 0,  -- Version of enrichment schema
    
    UNIQUE(source_file, chunk_index)
);

-- Indexes for semantic_chunks
-- CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON semantic_chunks 
--     -- USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON semantic_chunks 
    USING GIN (content_tsv);

CREATE INDEX IF NOT EXISTS idx_chunks_metadata ON semantic_chunks 
    USING GIN (metadata jsonb_path_ops);

CREATE INDEX IF NOT EXISTS idx_chunks_source ON semantic_chunks (source_file);

CREATE INDEX IF NOT EXISTS idx_chunks_llm_keywords ON semantic_chunks 
    USING GIN (llm_keywords);

CREATE INDEX IF NOT EXISTS idx_chunks_llm_domain ON semantic_chunks 
    USING GIN (llm_domain_tags);

-- ============================================
-- TABLE: code_entities
-- Stores code entities linked to the DKB graph
-- ============================================

CREATE TABLE IF NOT EXISTS code_entities (
    id TEXT PRIMARY KEY,  -- DKB entity ID (e.g., "src/user.py:UserService.get_user")
    entity_type TEXT NOT NULL,  -- "class", "function", "method", "module"
    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    start_line INTEGER,
    end_line INTEGER,
    signature TEXT,
    docstring TEXT,
    docstring_embedding float8[],  -- Embedding of docstring + signature
    metadata JSONB DEFAULT '{}',
    content_hash TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- ========================================
    -- LLM ENRICHMENT FIELDS
    -- ========================================
    llm_purpose TEXT,                    -- LLM-generated purpose description
    llm_usage_examples TEXT[],           -- LLM-generated usage examples
    llm_related_concepts TEXT[],         -- Related concepts/patterns
    llm_complexity_rating INTEGER,       -- Complexity rating 1-10
    llm_enriched_at TIMESTAMPTZ,
    llm_enrichment_model TEXT,
    llm_enrichment_version INTEGER DEFAULT 0
);

-- Indexes for code_entities
-- CREATE INDEX IF NOT EXISTS idx_code_embedding ON code_entities 
--     -- USING hnsw (docstring_embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_code_type ON code_entities (entity_type);

CREATE INDEX IF NOT EXISTS idx_code_file ON code_entities (file_path);

-- ============================================
-- TABLE: memory_notes
-- Stores agent memories (episodic, semantic, procedural)
-- ============================================

CREATE TABLE IF NOT EXISTS memory_notes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT DEFAULT 'default',
    memory_type TEXT NOT NULL CHECK (memory_type IN ('episodic', 'semantic', 'procedural')),
    content TEXT NOT NULL,
    context TEXT,
    keywords TEXT[],
    embedding float8[],
    source_ref TEXT,  -- Link to source document/code entity
    usage_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    deleted_reason TEXT,
    restored_from UUID  -- Reference to checkpoint if restored
);

-- Indexes for memory_notes
-- CREATE INDEX IF NOT EXISTS idx_memory_embedding ON memory_notes 
--     -- USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_memory_active ON memory_notes (is_active);

CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_notes (memory_type);

CREATE INDEX IF NOT EXISTS idx_memory_keywords ON memory_notes 
    USING GIN (keywords);

-- ============================================
-- TABLE: memory_checkpoints
-- Stores snapshots of memory state for rollback
-- ============================================

CREATE TABLE IF NOT EXISTS memory_checkpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    checkpoint_name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by TEXT DEFAULT 'SYSTEM',
    reason TEXT,
    memory_snapshot JSONB NOT NULL,  -- Serialized state of all active memories
    is_current BOOLEAN DEFAULT FALSE
);

-- Index for checkpoints
CREATE INDEX IF NOT EXISTS idx_checkpoint_current ON memory_checkpoints (is_current);

CREATE INDEX IF NOT EXISTS idx_checkpoint_created ON memory_checkpoints (created_at DESC);

-- ============================================
-- TABLE: ingestion_manifest
-- Tracks ingested documents and their status
-- ============================================

CREATE TABLE IF NOT EXISTS ingestion_manifest (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_file TEXT NOT NULL UNIQUE,
    output_file TEXT,
    file_type TEXT NOT NULL,  -- "pdf", "html", "markdown", "rst", "code"
    knowledge_type TEXT,
    chunk_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    error_message TEXT,
    content_hash TEXT,
    ingested_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- ========================================
    -- LLM ENRICHMENT STATUS
    -- ========================================
    llm_enrichment_status TEXT DEFAULT 'pending' 
        CHECK (llm_enrichment_status IN ('pending', 'processing', 'completed', 'failed', 'skipped')),
    llm_enriched_at TIMESTAMPTZ,
    llm_enrichment_error TEXT
);

-- Index for manifest
CREATE INDEX IF NOT EXISTS idx_manifest_status ON ingestion_manifest (status);

CREATE INDEX IF NOT EXISTS idx_manifest_type ON ingestion_manifest (file_type);

CREATE INDEX IF NOT EXISTS idx_manifest_llm_status ON ingestion_manifest (llm_enrichment_status);

-- ============================================
-- TABLE: libraries
-- Stores installed library/package information for auto-API docs
-- ============================================

CREATE TABLE IF NOT EXISTS libraries (
    id TEXT PRIMARY KEY,  -- e.g., "pypi:fastapi:0.115.6"
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    language TEXT NOT NULL,  -- "python", "javascript", "go", etc.
    ecosystem TEXT NOT NULL,  -- "pypi", "npm", "crates", etc.
    repository_url TEXT,
    license TEXT,
    homepage_url TEXT,
    documentation_url TEXT,
    metadata JSONB DEFAULT '{}',
    documented_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_libraries_lang ON libraries (language, name);

-- ============================================
-- TABLE: api_elements
-- Stores API elements from libraries (classes, functions, etc.)
-- ============================================

CREATE TABLE IF NOT EXISTS api_elements (
    id TEXT PRIMARY KEY,  -- e.g., "fastapi.routing.APIRouter"
    library_id TEXT REFERENCES libraries(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    element_type TEXT NOT NULL,  -- "class", "function", "method", "interface", etc.
    module TEXT,
    parent TEXT,  -- Parent class for methods
    visibility TEXT DEFAULT 'public',
    signature TEXT,
    docstring TEXT,
    docstring_embedding float8[],
    parameters JSONB DEFAULT '[]',  -- Array of ParameterInfo
    returns JSONB,  -- ReturnInfo
    type_parameters TEXT[],  -- For generics
    examples JSONB DEFAULT '[]',  -- Array of ExampleInfo
    see_also TEXT[],
    deprecated BOOLEAN DEFAULT FALSE,
    deprecation_message TEXT,
    since_version TEXT,
    source_file TEXT,
    start_line INTEGER,
    end_line INTEGER,
    source_hash TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- LLM enrichment for API elements
    llm_usage_tips TEXT,
    llm_common_patterns TEXT[],
    llm_enriched_at TIMESTAMPTZ,
    llm_enrichment_model TEXT
);

-- CREATE INDEX IF NOT EXISTS idx_api_embedding ON api_elements 
--     -- USING hnsw (docstring_embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_api_library ON api_elements (library_id);

CREATE INDEX IF NOT EXISTS idx_api_type ON api_elements (element_type);

-- ============================================
-- TABLE: error_patterns
-- Stores error patterns extracted from library source code
-- ============================================

CREATE TABLE IF NOT EXISTS error_patterns (
    id TEXT PRIMARY KEY,  -- e.g., "err:fastapi:ValueError:prefix-slash"
    library_id TEXT REFERENCES libraries(id) ON DELETE CASCADE,
    api_element_id TEXT REFERENCES api_elements(id) ON DELETE SET NULL,
    exception_type TEXT NOT NULL,
    message_pattern TEXT NOT NULL,
    message_regex TEXT,  -- For fuzzy matching
    message_embedding float8[],  -- For semantic error matching
    condition TEXT,  -- When this error is raised
    source_line INTEGER,
    language TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    times_encountered INTEGER DEFAULT 0,
    last_seen TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- LLM enrichment for errors
    llm_explanation TEXT,  -- LLM explanation of what causes this error
    llm_fix_suggestions TEXT[],  -- LLM-generated fix suggestions
    llm_enriched_at TIMESTAMPTZ
);

-- CREATE INDEX IF NOT EXISTS idx_error_embedding ON error_patterns 
--     -- USING hnsw (message_embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_error_type ON error_patterns (language, exception_type);

CREATE INDEX IF NOT EXISTS idx_error_library ON error_patterns (library_id);

-- Full-text search on error messages
CREATE INDEX IF NOT EXISTS idx_error_message_trgm ON error_patterns 
    USING GIN (message_pattern gin_trgm_ops);

-- ============================================
-- TABLE: error_fixes
-- Stores known fixes for error patterns (can be user-submitted or LLM-generated)
-- ============================================

CREATE TABLE IF NOT EXISTS error_fixes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    error_pattern_id TEXT REFERENCES error_patterns(id) ON DELETE CASCADE,
    fix_description TEXT NOT NULL,
    fix_code TEXT,  -- Code example of the fix
    source TEXT DEFAULT 'manual',  -- "manual", "llm", "stackoverflow", etc.
    upvotes INTEGER DEFAULT 0,
    downvotes INTEGER DEFAULT 0,
    verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_fix_error ON error_fixes (error_pattern_id);

-- ============================================
-- FUNCTIONS
-- ============================================

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
DROP TRIGGER IF EXISTS update_chunks_updated_at ON semantic_chunks;
CREATE TRIGGER update_chunks_updated_at
    BEFORE UPDATE ON semantic_chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_code_updated_at ON code_entities;
CREATE TRIGGER update_code_updated_at
    BEFORE UPDATE ON code_entities
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_manifest_updated_at ON ingestion_manifest;
CREATE TRIGGER update_manifest_updated_at
    BEFORE UPDATE ON ingestion_manifest
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_libraries_updated_at ON libraries;
CREATE TRIGGER update_libraries_updated_at
    BEFORE UPDATE ON libraries
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_api_elements_updated_at ON api_elements;
CREATE TRIGGER update_api_elements_updated_at
    BEFORE UPDATE ON api_elements
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_error_patterns_updated_at ON error_patterns;
CREATE TRIGGER update_error_patterns_updated_at
    BEFORE UPDATE ON error_patterns
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- HELPER FUNCTIONS FOR HYBRID SEARCH
-- ============================================

-- Hybrid search function combining vector and BM25
CREATE OR REPLACE FUNCTION hybrid_search(
    query_text TEXT,
    query_embedding float8[],
    alpha FLOAT DEFAULT 0.7,
    result_limit INTEGER DEFAULT 20
)
RETURNS TABLE(
    id UUID,
    content TEXT,
    source_file TEXT,
    header_path TEXT,
    dense_score FLOAT,
    sparse_score FLOAT,
    combined_score FLOAT
) AS $$
BEGIN
    -- STUBBED HYBRID SEARCH (Sparse only due to missing pgvector)
    RETURN QUERY
    SELECT 
        sc.id,
        sc.content,
        sc.source_file,
        sc.header_path,
        0.0::FLOAT AS dense_score,
        ts_rank(sc.content_tsv, websearch_to_tsquery('english', query_text))::FLOAT AS sparse_score,
        ts_rank(sc.content_tsv, websearch_to_tsquery('english', query_text))::FLOAT AS combined_score
    FROM semantic_chunks sc
    WHERE sc.content_tsv @@ websearch_to_tsquery('english', query_text)
    ORDER BY sparse_score DESC
    LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- FUNCTION: Error lookup with fuzzy matching
-- ============================================

CREATE OR REPLACE FUNCTION lookup_error(
    p_error_message TEXT,
    p_exception_type TEXT,
    p_language TEXT,
    result_limit INTEGER DEFAULT 5
)
RETURNS TABLE(
    error_id TEXT,
    pattern TEXT,
    condition TEXT,
    api_element_id TEXT,
    library_id TEXT,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ep.id,
        ep.message_pattern,
        ep.condition,
        ep.api_element_id,
        ep.library_id,
        similarity(ep.message_pattern, p_error_message) AS sim
    FROM error_patterns ep
    WHERE ep.language = p_language
      AND (p_exception_type IS NULL OR ep.exception_type = p_exception_type)
      AND similarity(ep.message_pattern, p_error_message) > 0.3
    ORDER BY sim DESC
    LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- INITIAL DATA
-- ============================================

-- Insert a system checkpoint as baseline
INSERT INTO memory_checkpoints (checkpoint_name, reason, memory_snapshot, is_current, created_by)
VALUES (
    'system-init',
    'Initial system checkpoint at database creation',
    '{"memories": [], "memory_links": [], "memory_count": 0}'::jsonb,
    TRUE,
    'SYSTEM'
) ON CONFLICT DO NOTHING;

-- ============================================
-- GRANT PERMISSIONS (for non-superuser access)
-- ============================================

-- These would be configured based on your user setup
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO apex_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO apex_user;

-- Indexes for semantic_chunks
-- CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON semantic_chunks 
--     -- USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON semantic_chunks 
    USING GIN (content_tsv);

CREATE INDEX IF NOT EXISTS idx_chunks_metadata ON semantic_chunks 
    USING GIN (metadata jsonb_path_ops);

CREATE INDEX IF NOT EXISTS idx_chunks_source ON semantic_chunks (source_file);

-- ============================================
-- TABLE: code_entities
-- Stores code entities linked to the DKB graph
-- ============================================

CREATE TABLE IF NOT EXISTS code_entities (
    id TEXT PRIMARY KEY,  -- DKB entity ID (e.g., "src/user.py:UserService.get_user")
    entity_type TEXT NOT NULL,  -- "class", "function", "method", "module"
    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    start_line INTEGER,
    end_line INTEGER,
    signature TEXT,
    docstring TEXT,
    docstring_embedding float8[],  -- Embedding of docstring + signature
    metadata JSONB DEFAULT '{}',
    content_hash TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for code_entities
-- CREATE INDEX IF NOT EXISTS idx_code_embedding ON code_entities 
--     -- USING hnsw (docstring_embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_code_type ON code_entities (entity_type);

CREATE INDEX IF NOT EXISTS idx_code_file ON code_entities (file_path);

-- ============================================
-- TABLE: memory_notes
-- Stores agent memories (episodic, semantic, procedural)
-- ============================================

CREATE TABLE IF NOT EXISTS memory_notes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT DEFAULT 'default',
    memory_type TEXT NOT NULL CHECK (memory_type IN ('episodic', 'semantic', 'procedural')),
    content TEXT NOT NULL,
    context TEXT,
    keywords TEXT[],
    embedding float8[],
    source_ref TEXT,  -- Link to source document/code entity
    usage_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    deleted_reason TEXT,
    restored_from UUID  -- Reference to checkpoint if restored
);

-- Indexes for memory_notes
-- CREATE INDEX IF NOT EXISTS idx_memory_embedding ON memory_notes 
--     -- USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_memory_active ON memory_notes (is_active);

CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_notes (memory_type);

CREATE INDEX IF NOT EXISTS idx_memory_keywords ON memory_notes 
    USING GIN (keywords);

-- ============================================
-- TABLE: memory_checkpoints
-- Stores snapshots of memory state for rollback
-- ============================================

CREATE TABLE IF NOT EXISTS memory_checkpoints (
    checkpoint_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    checkpoint_name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by TEXT DEFAULT 'SYSTEM',
    reason TEXT,
    memory_snapshot JSONB NOT NULL,  -- Serialized state of all active memories
    is_current BOOLEAN DEFAULT FALSE
);

-- Index for checkpoints
CREATE INDEX IF NOT EXISTS idx_checkpoint_current ON memory_checkpoints (is_current);

CREATE INDEX IF NOT EXISTS idx_checkpoint_created ON memory_checkpoints (created_at DESC);

-- ============================================
-- TABLE: ingestion_manifest
-- Tracks ingested documents and their status
-- ============================================

CREATE TABLE IF NOT EXISTS ingestion_manifest (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_file TEXT NOT NULL UNIQUE,
    output_file TEXT,
    file_type TEXT NOT NULL,  -- "pdf", "html", "markdown", "rst", "code"
    knowledge_type TEXT,
    chunk_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    error_message TEXT,
    content_hash TEXT,
    ingested_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for manifest
CREATE INDEX IF NOT EXISTS idx_manifest_status ON ingestion_manifest (status);

CREATE INDEX IF NOT EXISTS idx_manifest_type ON ingestion_manifest (file_type);

-- ============================================
-- Removed duplicate triggers and function definition that were here.


-- ============================================
-- HELPER FUNCTIONS FOR HYBRID SEARCH
-- ============================================

-- Hybrid search function combining vector and BM25
CREATE OR REPLACE FUNCTION hybrid_search(
    query_text TEXT,
    query_embedding float8[],
    alpha FLOAT DEFAULT 0.7,
    result_limit INTEGER DEFAULT 20
)
RETURNS TABLE(
    id UUID,
    content TEXT,
    source_file TEXT,
    header_path TEXT,
    dense_score FLOAT,
    sparse_score FLOAT,
    combined_score FLOAT
) AS $$
BEGIN
    -- STUBBED HYBRID SEARCH (Sparse only due to missing pgvector)
    RETURN QUERY
    SELECT 
        sc.id,
        sc.content,
        sc.source_file,
        sc.header_path,
        0.0::FLOAT AS dense_score,
        ts_rank(sc.content_tsv, websearch_to_tsquery('english', query_text))::FLOAT AS sparse_score,
        ts_rank(sc.content_tsv, websearch_to_tsquery('english', query_text))::FLOAT AS combined_score
    FROM semantic_chunks sc
    WHERE sc.content_tsv @@ websearch_to_tsquery('english', query_text)
    ORDER BY sparse_score DESC
    LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- INITIAL DATA
-- ============================================

-- Insert a system checkpoint as baseline
INSERT INTO memory_checkpoints (checkpoint_name, reason, memory_snapshot, is_current, created_by)
VALUES (
    'system-init',
    'Initial system checkpoint at database creation',
    '{"memories": [], "memory_links": [], "memory_count": 0}'::jsonb,
    TRUE,
    'SYSTEM'
) ON CONFLICT DO NOTHING;

-- ============================================
-- GRANT PERMISSIONS (for non-superuser access)
-- ============================================

-- These would be configured based on your user setup
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO apex_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO apex_user;
