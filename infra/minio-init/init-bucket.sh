#!/bin/sh
until mc alias set local http://minio:9000 minioadmin minioadmin; do sleep 2; done
mc mb --ignore-existing local/documents
mc version enable local/documents
mc anonymous set none local/documents
# lifecycle: expire non-current versions after 30 days
mc ilm rule add local/documents --noncurrent-expire-days 30 --noncurrent-expire-newer 1
