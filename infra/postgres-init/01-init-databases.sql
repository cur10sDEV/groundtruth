-- Runs once on first boot of an empty postgres data volume.
-- Flagsmith (main + edge engine) and Langfuse expect their databases to exist.
-- CREATE DATABASE has no IF NOT EXISTS, so \gexec runs the generated
-- "CREATE DATABASE ..." statement only when the database is missing.
-- The postgres image executes init scripts with psql, which supports \gexec.

SELECT 'CREATE DATABASE flagsmith'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'flagsmith')\gexec

SELECT 'CREATE DATABASE flag_engine'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'flag_engine')\gexec

SELECT 'CREATE DATABASE langfuse'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'langfuse')\gexec
