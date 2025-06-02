def search_knowledge_base(query: str, max_results: int = 3, relevance_threshold: float = 0.3) -> str:
    """
    Search the indexed PDF knowledge base for relevant information.
    
    This tool searches through indexed PDF documents to find relevant context
    for answering user questions. If no relevant information is found above
    the threshold, it returns a message indicating general knowledge should be used.
    
    Args:
        query (str): The search query or question to find relevant information for
        max_results (int): Maximum number of relevant chunks to return (default: 3)
        relevance_threshold (float): Minimum similarity score for relevance (default: 0.3)
    
    Returns:
        str: Either relevant context from indexed documents or indication to use general knowledge
    """
    
    try:
        # Configuration - use absolute path to ensure it works from any directory
        import os
        # Get the directory where your indexing script is located
        BASE_DIR = os.path.expanduser(".")  # Update this to your actual path
        VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store", "pdf_store.pkl")
        MODEL_NAME = 'BAAI/bge-base-en-v1.5'
        
        # Check if vector store exists
        if not os.path.exists(VECTOR_STORE_PATH):
            return "Knowledge base not found. Please ensure the PDF indexing has been completed first. I'll answer using my general knowledge."
        
        # Initialize components
        model = SentenceTransformer(MODEL_NAME)
        client = QdrantClient(":memory:")
        collection_name = "pdf_documents"
        
        # Load the saved vector store
        try:
            with open(VECTOR_STORE_PATH, 'rb') as f:
                store_data = pickle.load(f)
            
            # Debug: Check what was loaded
            if store_data is None:
                return "Knowledge base file exists but contains no data. Please re-run the indexing script."
            
            if not isinstance(store_data, dict):
                return f"Knowledge base file format is incorrect. Expected dict, got {type(store_data)}. Please re-run the indexing script."
            
            points = store_data.get("points", [])
            if not points:
                return f"Knowledge base contains no indexed points. Available keys: {list(store_data.keys())}. Please re-run the indexing script."
            
            # Recreate collection
            sample_embedding = model.encode("sample text")
            vector_size = len(sample_embedding)
            
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            
            # Convert serialized points back to PointStruct for uploading
            upload_points = []
            for point_data in points:
                try:
                    # Handle the new serializable format
                    if isinstance(point_data, dict):
                        upload_point = PointStruct(
                            id=point_data.get("id"),
                            vector=point_data.get("vector"),
                            payload=point_data.get("payload", {})
                        )
                        upload_points.append(upload_point)
                    # Handle old Record objects (fallback)
                    elif hasattr(point_data, 'id') and hasattr(point_data, 'vector') and hasattr(point_data, 'payload'):
                        upload_point = PointStruct(
                            id=point_data.id,
                            vector=point_data.vector if isinstance(point_data.vector, list) else point_data.vector.tolist(),
                            payload=point_data.payload
                        )
                        upload_points.append(upload_point)
                except Exception as e:
                    print(f"Error processing point: {e}")
                    continue
            
            if not upload_points:
                return "No valid points found in the knowledge base. Please re-run the indexing script."
            
            # Load points into collection
            client.upsert(
                collection_name=collection_name,
                points=upload_points
            )
            
        except Exception as e:
            return f"Error loading knowledge base: {str(e)}. I'll answer using my general knowledge."
        
        # Search for relevant information
        try:
            query_embedding = model.encode(query).tolist()
            
            search_results = client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=max_results * 2  # Get more results to filter by threshold
            )
            
            # Filter results by relevance threshold
            relevant_results = []
            for result in search_results:
                if result.score >= relevance_threshold:
                    relevant_results.append({
                        "text": result.payload["text"],
                        "score": result.score,
                        "source_file": result.payload["source_file"],
                        "chunk_index": result.payload["chunk_index"]
                    })
            
            # Limit to max_results
            relevant_results = relevant_results[:max_results]
            
            if not relevant_results:
                return f"No relevant information found in the knowledge base for this query (searched {len(search_results)} chunks, none above threshold {relevance_threshold}). I'll answer using my general knowledge."
            
            # Format the response with relevant context
            context_parts = []
            context_parts.append(f"Found {len(relevant_results)} relevant chunks from the knowledge base:")
            context_parts.append("")
            
            for i, result in enumerate(relevant_results, 1):
                context_parts.append(f"**Source {i}** (from {result['source_file']}, relevance: {result['score']:.3f}):")
                context_parts.append(result['text'])
                context_parts.append("")
            
            context_parts.append("---")
            context_parts.append("Please use the above context from the indexed documents to answer the user's question. If the context doesn't fully address the question, you can supplement with your general knowledge but prioritize the indexed information.")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            return f"Error during search: {str(e)}. I'll answer using my general knowledge."
            
    except Exception as e:
        return f"Unexpected error in knowledge base search: {str(e)}. I'll answer using my general knowledge."