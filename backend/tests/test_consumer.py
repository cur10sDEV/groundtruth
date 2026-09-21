from app.ingestion.consumer import process_message


def test_process_message_signature():
    # unit-safety: no queue required for signature; integration behavior needs RabbitMQ
    assert callable(process_message)
