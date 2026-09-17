> Optimizations and Techniques from [[01. Fact checker]] can be applied here also

## Funcitonal Requirements

- Zero Hallucinations, 100% Correctness
- Guarantee: System must not assert any claim that cannot be traced to retrieved chunk / passage
- User interacting via web interface
- We need to serve data from just-updated document (~15 mints sweet time for ingestion and creating indexes)


## Capacity Estimation

- 10M docs, 10KB each, 200 QPS, ~1K docs updated / day
- 100GB in S3 (raw storage only) - there could be multiple versions, etc
- PG will also be in some gbs or might be some tb - depending on what you are storing
- The largest storage needs is for vector db
	- 10<sup>7</sup> (docs) x 25 (avg no of chunks per document) x 1536 (dimensions) x 4B (each dimension size for `float32`) = 1.5TB (*this is of only dense vectors*)
	- 10<sup>7</sup> (docs) x 25 (no of chunks) x 500B (size of each chunk) = ~500GB to 1TB depending on the db's internal storage formats and requirements (*this is of only sparse vectors*)
	- It also stores various metadata fields that will be someting to add to the total


## Storage

### S3

- Prefix should be like this = **`s3://<bucket_name>/<org_id>/<user_id>/<doc_id>/<uuid>.ext`**
- Never store the file with the original filename - always UUID - the original filename should be in the metadata
- Versioning in s3 as failover if something in the ingestion pipeline fails
- Lifecycle for non-current versions of about n days (30) so that they gets deleted and you wont be charged for the redundant data

### Postgres

- For citations and having a consistent store about what chunks belogs to what documents and what documents belongs to what user
- And the other usual tables like users, organizations (if you have), and logs table, and whatnot

```sql
CREATE TABLE documents (
	id UUID PRIMARY KEY
	user_id UUID
	org_id UUID
	original_filename TEXT
	status ENUM(PENDING,PROCESSING,EMBEDDED,FAILED)
	content_hash TEXT -- for deduplication and also unchanged re-uploads
	current_version INTEGER DEFAULT 1-- this will be used for query
	
	created_at TIMESTAMP
	updated_at TIMESTAMP
	
	FOREIGN KEY (user_id) REFERENCES users(id)
	FOREIGN KEY (org_id) REFERENCES organizations(id)
	
	-- either store the s3 key explicitly or code level construction
)

CREATE TABLE chunks (
	id UUID PRIMARY KEY
	doc_id UUID
	user_id UUID -- this denormalization so that no joins and faster query
	chunk_text TEXT
	page_number INTEGER
	start_offset INTEGER -- in the ui/ux we can highlight inside the original doc
	end_offset INTEGER
	version INTEGER DEFAULT 1 -- this will be used for query
	
	created_at TIMESTAMP
	updated_at TIMESTAMP
	
	FOREIGN KEY (doc_id) REFERENCES documents(id)
	FOREIGN KEY (user_id) REFERENCES users(id)
	FOREIGN KEY (org_id) REFERENCES organizations(id)
	
	-- either store the s3 key explicitly or code level construction
)
```

### Vector DB (Qdrant)

- Store the chunks / embeddings in the vector db alongwith the whole metadata enrichment
- Only store the embeddings / vectors - for both dense vectors (cosine similarity) and sparse vectors (bm25) but not the actual chunk text (if you really want you can - its just that we are storing them in pg so...)

***Okay so the thing is you can either use ElasticSearch for keyword searching (BM25) or you can use your vector db - totally depends on your choice and operations***

> You can only store vectors without the chunk text in the likes of qdrant with parameter `with_payload=False` when creating / defining collection and vectors.


## Caching

### What to store

- the final answer text
- the cited chunk IDs (so citation links on a cache hit still resolve and are clickable, exactly like a fresh generation - pg stored)
- the cited doc IDs (can invalidate particular cache entries when a doc gets updated / deleted)
- the faithfulness check result (so you're not recomputing it on every cache hit)

- Have a moderate TTL max about a day - but fully depends on the query pattern
- This should also have information isolation based on user / org so that one org's / user's cached query does not get serve to other purely based on similarity
- For invalidation: 
	- Rely on TTL (simple) - regardless TTL will be present no matter what you choose
	- **SET** - Have a reverse index lookup for `doc_id -> cache entries` so that can easily get the cache keys and delete them all - fast lookup
		- You can also look at RedisVL for this type of thing


## Ingestion Service (Write Path)

- Use s3 events for object creation, updation, deletion
- S3 events will be feeded into the queue like SQS or Kafka, and an ingestor service can then take care of:
	- Parsing (reading) - file and layout dependent
	- Preprocessing (cleanup) - ready to be chunked
	- Chunking with metadata enrichment - doc_id, user_id, org_id, page_number, character_offsets, chunk_index, word_count, char_count, urls, tags, type and other domain specific info like year, topic, etc - this helps us in future with metadata filtering
	- Each chunk will have a UUID - so that we can actually trace back to it - when CRUD happens or we implement RAG with Sources - so that an endpoint like getChunkById - will help the user see the acutal chunk used in the generation process - double checking whether the claim is grounded in the source material or not
	- Embeddings - storing in vector database alognwith that previous metadata
	- Updating the status at the db layer = PENDING | PROCESSING | FAILED , etc


## Query and Retrieval Pipeline (Read Path)

- We will get the user query in a POST request with body and headers and whatnot
- Runs the standard validation using pydantic, security checkpoints (rate-limit, auth, etc)
- Runs the LLM validation checks - guardrails, PII, profanity, etc
- Perform semantic cahing lookup for either the user query and serve the result to the user if present (*end here if cached*)
- You tell your LLM to generate a more verbose and accurate query by removing typos, grammatical mistakes, identifying Entities, rephrasing the text
- Also you can generate K queries for a given query in the same pattern for higher recall
- Now you perform the hybrid search on both dense and sparse vectors (parallely)
- Retrived chunks go through RRF
- Optionally can also go through re-ranker if wants to
- LLM generation - Pass the retrieved chunks to the LLM as the context
- Caching with the original user query as key

> **Your LLM / Agent needs to know what metadata (and the structure) it can filter query upon**


## UI/UX

- We have to design cancellation of operations with proper cleanup if the user wants to cancel a upload or processing of documents
- Stream events / status / tool-use to the user (thinking, processing) using SSE
- You can either send the whole response at once after every check - guardrails, PII, profanity, etc - but should show progress updates so that user is not in the dead air
- Or you can incrementally check the response streamed from the LLM using something like sliding window - you take sentences, custom token sizes (100) - accumulate them - run your validation pipeline and sends the output to the user - this way you are only tens of milliseconds slower than the actual stream response - but is much more better UX


## Security / Validations

- System should have boundaries like RBAC wherever necessary - so that the agent can only be able to retrieve docs and chunks owned by the user / organization and has permissions to do so. NO INFO LEAK
- The above information isolation also applies to semantic cache
- Prevent Abuse - Rate Limiting the service, Standard Validation of inputs and outputs like query / answer / file size
- Standard Guardrails, Prompt Injection Safety, PII detection, etc
- Max file size validation in s3 and also on frontend


## Failure Modes

- Timeouts in case of any of the storage system
- For factual correctness and zero hallucination - we cascade the errors to the user explicitly telling that unable to process the query at the moment
- Timeouts / Rate limits / failures in LLM generation
	- retry with max no of retries and exponential backoff
	- you can have another fallback model
	- otherwise same behaviour of error cascading to the user
- If cache is down - then ofcourse the full retrieval pipeline
- Write path failure 
	- s3 file access failed - retry and after DQL
	- Partial updates or writes to any storage layer - then have to take care of that - we have events in the queue so we can reprocess that - have to figure out a transactional sort of way for doing things - might need to persist data before we are 100% sure that everything has been executed successfully


## Updation

- When updating a document - it will by default has version 1 as the *current_version* and all its chunks will use this and store it in the postgres as well as qdrant vectors - this way when we are upadting a document we dont have to delete the old before successfully ingesting the new version across systems - Also this way the query path will work as it is using the old data until new ones become available
- We just have to flip the current_version to the latest version and now that will be used across the read path whenever fetching vectors, chunks from different storage layers
- Invalidate the cache for the particular document


## Cleanup for old files

- A periodic cleanup job will run and find out all the chunks, vectors with versions lower than the current_version of the documents and deletes them in batches with retries and put failed operations in a DLQ for further troubleshooting and manual intervention


## Things to look out for

- Threshold for matching in both vector db and semantic cache
- Tool calls if any - might stream these events / status

