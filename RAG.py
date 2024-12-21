import chromadb
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
import os

class RAGSystem:
    def __init__(self, model_name="llama2", embed_model_name="llama2"):
        """Initialize the RAG system with specified models."""
        self.llm = Ollama(model=model_name, base_url='http://host.docker.internal:11434')
        self.embeddings = OllamaEmbeddings(model=embed_model_name, base_url='http://host.docker.internal:11434')
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name="document_collection",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

    def load_documents(self, directory_path="./documents"):
        """Load PDF documents from a directory."""
        loader = DirectoryLoader(
            directory_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        try:
            documents = loader.load()
            print(f"Successfully loaded {len(documents)} documents")
            return documents
        except Exception as e:
            print(f"Error loading documents: {str(e)}")
            return []

    def process_documents(self, documents):
        """Process documents and store them in the vector database."""
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Prepare documents for ChromaDB
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [str(i) for i in range(len(chunks))]
        
        # Generate embeddings and add to ChromaDB
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        return len(chunks)

    def query(self, question, k=3):
        """Query the RAG system about Strata reports."""
        # Search for relevant documents
        results = self.collection.query(
            query_texts=[question],
            n_results=k
        )
        
        # Prepare context from retrieved documents
        context_docs = [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(
                results['documents'][0],
                results['metadatas'][0]
            )
        ]
        
        # Construct prompt with context and specific instructions for Strata analysis
        context_text = "\n\n".join([doc.page_content for doc in context_docs])
        prompt = f"""You are a professional Strata report analyzer helping Aidan evaluate potential property purchases. 
Use the following context from Strata reports to answer the question. Focus on:
- Identifying any major issues or red flags
- Financial implications and costs
- Building maintenance and repair history
- Future planned works
- Overall building condition
- Compliance issues

If you cannot find the specific information in the context, say so clearly.

Context:
{context_text}

Question: {question}

Answer:"""
        
        # Generate response using Ollama
        response = self.llm.invoke(prompt)
        return response

    def add_single_document(self, content, metadata=None):
        """Add a single document to the vector database."""
        if metadata is None:
            metadata = {}
        
        doc = Document(page_content=content, metadata=metadata)
        chunks = self.text_splitter.split_documents([doc])
        
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [str(i + self.collection.count()) for i in range(len(chunks))]
        
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        return len(chunks)

    def summarize_property(self, property_id):
        """Generate a comprehensive summary of a specific property."""
        summary_prompt = f"""Please provide a comprehensive summary of this property's Strata report, including:

1. Overall Building Condition
2. Major Issues or Concerns
3. Recent Repairs and Maintenance
4. Upcoming Required Works
5. Financial Health of the Strata
6. Compliance Status
7. Recommendations

Context:
{{context}}

Summary:"""
        
        # Get all chunks related to this property
        results = self.collection.query(
            query_texts=["building condition maintenance repairs financial"],
            n_results=10,  # Get more chunks for a comprehensive summary
            where={"property_id": property_id} if property_id else None
        )
        
        context_text = "\n\n".join(results['documents'][0])
        prompt = summary_prompt.replace("{{context}}", context_text)
        
        return self.llm.invoke(prompt)

# Example usage
if __name__ == "__main__":
    # Initialize the RAG system
    rag = RAGSystem()
    
    # Load and process Strata reports
    documents = rag.load_documents("./documents")
    if documents:
        num_chunks = rag.process_documents(documents)
        print(f"Processed {num_chunks} chunks from Strata reports")
    
    # Example queries specific to Strata reports
    questions = [
        "What are the major maintenance issues identified in the building?",
        "Are there any upcoming special levies or significant expenses?",
        "What is the overall condition of the building's common areas?",
        "Are there any compliance issues I should be concerned about?",
        "What major repairs have been completed in the last 5 years?",
        "Does anything in the report suggest that the building is in a poor state?",
        "Is there an indication that the building is in a poor state?",
        "Is there any indication that the builder has become insolvent",
        "Are there any legal proceedings against the builder?",
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        response = rag.query(question)
        print(f"Answer: {response}")
