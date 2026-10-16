-- supabase/setup.sql  —  run ONCE in Supabase: SQL Editor -> New query -> paste -> Run
-- Creates everything Legal Document Navigator needs in the cloud.

-- 1. pgvector: lets Postgres store embedding vectors
create extension if not exists vector;

-- 2. documents: one row per uploaded PDF (replaces data/uploads/names.json)
create table if not exists documents (
    doc_id      text primary key,                 -- e.g. 'legal-document-sample-818b29'
    name        text not null,                    -- original file name, shown in the sidebar
    owner       text,                             -- browser ID: each browser sees only its own rows
    created_at  timestamptz not null default now()
);
create index if not exists documents_owner_idx on documents (owner);

-- 3. chunks: the clauses + their vectors (replaces chroma_db/)
--    384 = size of BAAI/bge-small-en-v1.5 vectors. A different embedding model needs a new number.
create table if not exists chunks (
    id         text primary key,
    doc_id     text not null references documents(doc_id) on delete cascade,
    text       text not null,
    embedding  vector(384) not null,
    metadata   jsonb not null default '{}'::jsonb  -- section, pages, embed_text, ...
);
create index if not exists chunks_doc_id_idx on chunks (doc_id);

-- 4. chats: the chat history of each document (replaces data/chats/)
create table if not exists chats (
    doc_id      text primary key references documents(doc_id) on delete cascade,
    messages    jsonb not null default '[]'::jsonb,
    updated_at  timestamptz not null default now()
);
-- "on delete cascade": deleting a document also deletes its chunks and its chat

-- 5. Security: block the public (anon) key. The app uses the secret key, which is allowed.
alter table documents enable row level security;
alter table chunks    enable row level security;
alter table chats     enable row level security;

-- 6. A private storage bucket for the PDF files (replaces data/uploads/*.pdf)
insert into storage.buckets (id, name, public)
values ('uploads', 'uploads', false)
on conflict (id) do nothing;