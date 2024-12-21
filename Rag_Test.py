# Import the RAG system
from RAG import RAGSystem

# Initialize the system (you can specify different models if needed)
rag = RAGSystem(model_name="llama2", embed_model_name="llama2")

# Load documents from your ./documents directory
documents = rag.load_documents()

# Process the documents if any were loaded successfully
if documents:
    num_chunks = rag.process_documents(documents)
    print(f"Processed {num_chunks} chunks from Strata reports")

# Now you can query your documents
# Example:
response = rag.query("What are the major maintenance issues identified in the building?")
print(response)

# Or get a comprehensive summary if you have property IDs set up
# summary = rag.summarize_property("property_id")
# print(summary)