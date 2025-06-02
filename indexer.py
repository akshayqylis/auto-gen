import os
import pickle
from pathlib import Path
from typing import List, Dict, Any
import hashlib

# PDF processing
import PyPDF2
from sentence_transformers import SentenceTransformer

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class PDFVectorStore:
    def __init__(self, model_name: str = 'BAAI/bge-base-en-v1.5'):
        """Initialize the PDF Vector Store with Qdrant in-memory client"""
        self.client = QdrantClient(":memory:")  # In-memory Qdrant instance
        self.model = SentenceTransformer(model_name)
        self.collection_name = "pdf_documents"
        self.chunk_size = 500  # Characters per chunk
        self.chunk_overlap = 50  # Overlap between chunks
        
        # Create collection
        self._create_collection()
    
    def _create_collection(self):
        """Create a collection in Qdrant"""
        # Get embedding dimension from the model
        sample_embedding = self.model.encode("sample text")
        vector_size = len(sample_embedding)
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"Created collection '{self.collection_name}' with vector size {vector_size}")
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {str(e)}")
            return ""
        return text
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary if possible
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + self.chunk_size // 2:
                    chunk = text[start:break_point + 1]
                    start = break_point + 1 - self.chunk_overlap
                else:
                    start = end - self.chunk_overlap
            else:
                start = end
            
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def index_pdf(self, pdf_path: str) -> bool:
        """Index a PDF file into the vector store"""
        if not os.path.exists(pdf_path):
            print(f"PDF file not found: {pdf_path}")
            return False
        
        print(f"Processing PDF: {pdf_path}")
        
        # Extract text from PDF
        text = self._extract_text_from_pdf(pdf_path)
        if not text.strip():
            print(f"No text extracted from {pdf_path}")
            return False
        
        # Split into chunks
        chunks = self._chunk_text(text)
        print(f"Created {len(chunks)} chunks from PDF")
        
        # Create embeddings and index
        points = []
        for i, chunk in enumerate(chunks):
            # Generate embedding
            embedding = self.model.encode(chunk).tolist()
            
            # Create unique ID for each chunk
            chunk_id = hashlib.md5(f"{pdf_path}_{i}_{chunk[:50]}".encode()).hexdigest()
            
            # Create point with metadata
            point = PointStruct(
                id=chunk_id,
                vector=embedding,
                payload={
                    "text": chunk,
                    "source_file": os.path.basename(pdf_path),
                    "chunk_index": i,
                    "file_path": pdf_path
                }
            )
            points.append(point)
        
        # Upload to Qdrant
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"Successfully indexed {len(points)} chunks from {pdf_path}")
            return True
        except Exception as e:
            print(f"Error indexing chunks: {str(e)}")
            return False
    
    def search(self, query: str, limit: int = 5, score_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Search for relevant chunks based on query"""
        try:
            query_embedding = self.model.encode(query).tolist()
            
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
                # Removed score_threshold as it might be too restrictive
            )
            
            results = []
            for result in search_results:
                # Apply score threshold manually
                if result.score >= score_threshold:
                    results.append({
                        "text": result.payload["text"],
                        "score": result.score,
                        "source_file": result.payload["source_file"],
                        "chunk_index": result.payload["chunk_index"]
                    })
            
            return results
        except Exception as e:
            print(f"Error during search: {str(e)}")
            return []
    
    def debug_collection(self):
        """Debug method to see what's in the collection"""
        try:
            # Check collection info
            collection_info = self.client.get_collection(self.collection_name)
            print(f"Collection info: {collection_info}")
            
            # Try to get points count
            try:
                count_result = self.client.count(collection_name=self.collection_name)
                print(f"Points count: {count_result}")
            except:
                print("Could not get points count")
            
            # Try scroll with different parameters
            try:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=5,
                    with_payload=True,
                    with_vectors=True
                )
                print(f"Scroll result type: {type(scroll_result)}")
                print(f"Scroll result: {scroll_result}")
            except Exception as e:
                print(f"Scroll error: {e}")
                
        except Exception as e:
            print(f"Debug error: {e}")
    
    def save_store(self, save_path: str):
        """Save the vector store to disk for persistence"""
        try:
            # Debug first
            print("Debugging collection before save...")
            self.debug_collection()
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Get all points from the collection - handle different return formats
            try:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=10000,  # Adjust based on your needs
                    with_payload=True,
                    with_vectors=True
                )
                
                print(f"Scroll result type: {type(scroll_result)}")
                
                # Handle different possible return formats
                if isinstance(scroll_result, tuple) and len(scroll_result) >= 2:
                    points = scroll_result[0]  # (points, next_page_offset)
                elif isinstance(scroll_result, list):
                    points = scroll_result
                else:
                    points = scroll_result
                
                print(f"Points type: {type(points)}")
                print(f"Points length: {len(points) if points else 'None'}")
                
                if points is None:
                    print("No points found in collection")
                    return
                    
            except Exception as e:
                print(f"Error retrieving points from collection: {e}")
                return
            
            # Convert Record objects to a more pickle-friendly format
            serializable_points = []
            for i, point in enumerate(points):
                try:
                    print(f"Processing point {i}: {type(point)}")
                    serializable_point = {
                        "id": point.id,
                        "vector": point.vector.tolist() if hasattr(point.vector, 'tolist') else list(point.vector),
                        "payload": point.payload
                    }
                    serializable_points.append(serializable_point)
                except Exception as e:
                    print(f"Error processing point {i} (id: {getattr(point, 'id', 'unknown')}): {e}")
                    continue
            
            if not serializable_points:
                print("No valid points to save")
                return
            
            store_data = {
                "points": serializable_points,
                "model_name": 'BAAI/bge-base-en-v1.5',
                "collection_name": self.collection_name,
                "vector_size": len(serializable_points[0]["vector"]) if serializable_points else 384
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(store_data, f)
            
            print(f"Vector store saved to {save_path} with {len(serializable_points)} points")
            
        except Exception as e:
            print(f"Error saving vector store: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def load_store(self, load_path: str):
        """Load a previously saved vector store"""
        try:
            with open(load_path, 'rb') as f:
                store_data = pickle.load(f)
            
            # Recreate collection and load points
            points = store_data["points"]
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
            
            print(f"Vector store loaded from {load_path} with {len(points)} points")
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")


def main():
    """Main function to process PDFs from a folder"""
    # Configuration
    PDF_FOLDER = "pdfs"  # Folder containing PDF files
    VECTOR_STORE_PATH = "vector_store/pdf_store.pkl"  # Path to save the vector store
    
    # Initialize vector store
    vector_store = PDFVectorStore()
    
    # Process all PDFs in the folder
    if not os.path.exists(PDF_FOLDER):
        print(f"Creating PDF folder: {PDF_FOLDER}")
        os.makedirs(PDF_FOLDER)
        print("Please add PDF files to the 'pdfs' folder and run the script again.")
        return
    
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {PDF_FOLDER}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file}")
    
    # Index each PDF
    successful_indexing = 0
    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        if vector_store.index_pdf(pdf_path):
            successful_indexing += 1
        print("-" * 50)
    
    print(f"\nIndexing complete! Successfully processed {successful_indexing}/{len(pdf_files)} PDFs")
    
    # Save the vector store for later use
    vector_store.save_store(VECTOR_STORE_PATH)
    
    # Test search functionality
    print("\n" + "="*50)
    print("Testing search functionality...")
    test_query = "What is the main topic of the document?"
    results = vector_store.search(test_query, limit=3)
    
    if results:
        print(f"\nSearch results for: '{test_query}'")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.3f}")
            print(f"   Source: {result['source_file']}")
            print(f"   Text: {result['text'][:200]}...")
    else:
        print("No relevant results found for the test query.")


if __name__ == "__main__":
    main()
