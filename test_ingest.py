from rag.pipeline import RAGIngestionPipeline

ingestion = RAGIngestionPipeline(data_dir="./data/Document")
result = ingestion.ingest()
print(result)