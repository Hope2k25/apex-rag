-- Apex RAG System - PostgreSQL Schema Initialization
-- This file is automatically run when the Docker container starts
-- Based on Plan B: Data Infrastructure

-- ============================================
-- EXTENSIONS
-- ============================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
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
    embedding VECTOR(768),  -- gte-modernbert-base produces 768-dim vectors
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    content_hash TEXT,
    UNIQUE(source_file, chunk_index)
);

-- Indexes for semantic_chunks
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON semantic_chunks 
    USING hnsw (embedding vector_cosine_ops);

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
    docstring_embedding VECTOR(768),  -- Embedding of docstring + signature
    metadata JSONB DEFAULT '{}',
    content_hash TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for code_entities
CREATE INDEX IF NOT EXISTS idx_code_embedding ON code_entities 
    USING hnsw (docstring_embedding vector_cosine_ops);

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
    embedding VECTOR(768),
    source_ref TEXT,  -- Link to source document/code entity
    usage_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    deleted_reason TEXT,
    restored_from UUID  -- Reference to checkpoint if restored
);

-- Indexes for memory_notes
CREATE INDEX IF NOT EXISTS idx_memory_embedding ON memory_notes 
    USING hnsw (embedding vector_cosine_ops);

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
CREATE TRIGGER update_chunks_updated_at
    BEFORE UPDATE ON semantic_chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_code_updated_at
    BEFORE UPDATE ON code_entities
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_manifest_updated_at
    BEFORE UPDATE ON ingestion_manifest
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- HELPER FUNCTIONS FOR HYBRID SEARCH
-- ============================================

-- Hybrid search function combining vector and BM25
CREATE OR REPLACE FUNCTION hybrid_search(
    query_text TEXT,
    query_embedding VECTOR(768),
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
    RETURN QUERY
    WITH dense_results AS (
        SELECT 
            sc.id,
            sc.content,
            sc.source_file,
            sc.header_path,
            1 - (sc.embedding <=> query_embedding) AS score
        FROM semantic_chunks sc
        WHERE sc.embedding IS NOT NULL
        ORDER BY sc.embedding <=> query_embedding
        LIMIT 100
    ),
    sparse_results AS (
        SELECT 
            sc.id,
            ts_rank(sc.content_tsv, websearch_to_tsquery('english', query_text)) AS score
        FROM semantic_chunks sc
        WHERE sc.content_tsv @@ websearch_to_tsquery('english', query_text)
        ORDER BY score DESC
        LIMIT 100
    ),
    combined AS (
        SELECT 
            COALESCE(d.id, s_lookup.id) AS id,
            COALESCE(d.content, s_lookup.content) AS content,
            COALESCE(d.source_file, s_lookup.source_file) AS source_file,
            COALESCE(d.header_path, s_lookup.header_path) AS header_path,
            COALESCE(d.score, 0.0) AS dense_score,
            COALESCE(s.score, 0.0) AS sparse_score,
            (alpha * COALESCE(d.score, 0.0) + (1 - alpha) * COALESCE(s.score, 0.0)) AS combined_score
        FROM dense_results d
        FULL OUTER JOIN sparse_results s ON d.id = s.id
        LEFT JOIN semantic_chunks s_lookup ON s.id = s_lookup.id
    )
    SELECT 
        c.id,
        c.content,
        c.source_file,
        c.header_path,
        c.dense_score::FLOAT,
        c.sparse_score::FLOAT,
        c.combined_score::FLOAT
    FROM combined c
    ORDER BY c.combined_score DESC
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
