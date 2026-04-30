-- ProjectCerebro — pgvector initialization
-- Runs automatically when PostgreSQL container starts

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Trial embeddings table
-- Stores 128-dimensional EEGNet encoder embeddings
-- for retrieval-augmented explainability
CREATE TABLE IF NOT EXISTS trial_embeddings (
    trial_id        TEXT PRIMARY KEY,
    subject_id      TEXT NOT NULL,
    dataset         TEXT NOT NULL,  -- 'physionet', 'bci_iv_2a', 'private'
    run_id          TEXT,
    label           TEXT NOT NULL,  -- 'left', 'right', 'feet', 'tongue'
    label_code      INTEGER,        -- T1=1, T2=2
    embedding       vector(128),    -- EEGNet encoder output
    model_version   TEXT,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Index for fast cosine similarity search
CREATE INDEX IF NOT EXISTS trial_embeddings_vector_idx
    ON trial_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Index for filtering by subject or dataset
CREATE INDEX IF NOT EXISTS trial_embeddings_subject_idx
    ON trial_embeddings (subject_id);

CREATE INDEX IF NOT EXISTS trial_embeddings_dataset_idx
    ON trial_embeddings (dataset);

CREATE INDEX IF NOT EXISTS trial_embeddings_label_idx
    ON trial_embeddings (label);