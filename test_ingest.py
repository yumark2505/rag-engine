from rag.pipeline import RAGIngestionPipeline, Pipeline

ingestion = RAGIngestionPipeline(data_dir="./data/Document")
result = ingestion.ingest()
print(result)