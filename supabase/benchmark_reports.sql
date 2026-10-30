-- Run this once in the Supabase SQL Editor.
-- The backend stores the complete benchmark summary and per-sample dataset in
-- JSONB so the existing report shape is preserved.

create table if not exists public.benchmark_reports (
  id uuid primary key default gen_random_uuid(),
  benchmark_scope text,
  mode text not null,
  run_timestamp text,
  summary jsonb not null,
  dataset jsonb not null,
  created_at timestamptz not null default now()
);

create index if not exists benchmark_reports_created_at_idx
  on public.benchmark_reports (created_at desc);

create index if not exists benchmark_reports_scope_idx
  on public.benchmark_reports (benchmark_scope);

-- The backend uses the service-role key. Keep the table inaccessible to the
-- public anon role until explicit frontend read policies are needed.
alter table public.benchmark_reports enable row level security;

create table if not exists public.documents (
  id uuid primary key default gen_random_uuid(),
  source_name text not null unique,
  full_text text not null,
  character_count integer not null default 0,
  chunks_count integer not null default 0,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists documents_updated_at_idx on public.documents (updated_at desc);
alter table public.documents enable row level security;

create table if not exists public.detector_metrics (
  id uuid primary key default gen_random_uuid(),
  metrics jsonb not null,
  created_at timestamptz not null default now()
);

create index if not exists detector_metrics_created_at_idx on public.detector_metrics (created_at desc);
alter table public.detector_metrics enable row level security;
