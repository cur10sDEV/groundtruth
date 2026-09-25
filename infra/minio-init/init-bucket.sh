#!/bin/sh
until mc alias set local http://minio:9000 minioadmin minioadmin; do sleep 2; done
mc mb --ignore-existing local/documents
mc version enable local/documents
mc anonymous set none local/documents
# lifecycle: expire non-current versions after 30 days
mc ilm rule add local/documents --noncurrent-expire-days 30 --noncurrent-expire-newer 1
# bucket events -> RabbitMQ: the sole ingest trigger. MinIO publishes
# s3:ObjectCreated:* under documents/ to the minio.events exchange; the
# worker's translator is the only path into the ingestion queue. The
# notify_amqp target itself is configured via MINIO_NOTIFY_AMQP_*_PRIMARY
# environment variables on the minio service.
if [ -z "$(mc event ls local/documents arn:minio:sqs::primary:amqp 2>/dev/null)" ]; then
  mc event add local/documents arn:minio:sqs::primary:amqp --event put --prefix documents/ --ignore-existing
fi
