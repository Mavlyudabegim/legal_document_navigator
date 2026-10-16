-- supabase/add_owner.sql  —  run ONCE if you created the tables before "private per browser"
-- SQL Editor -> New query -> paste -> Run

-- 1. every document remembers which browser uploaded it
alter table documents add column if not exists owner text;
create index if not exists documents_owner_idx on documents (owner);

-- 2. documents uploaded before this change have no owner, so nobody sees them any more.
--    This removes them (their chunks and chats are removed too: "on delete cascade").
--    Their PDF files stay in Storage -> uploads; delete them there by hand if you like.
delete from documents where owner is null;