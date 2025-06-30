import os
import json
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from tqdm import tqdm
import uuid # For generating unique IDs for Qdrant points
import numpy as np
import ollama # Import the ollama library

# --- Configuration ---
# Directory where your PDF files are located. Create 'hwmanuals/pdf' and place your PDFs here.
PDF_DIR = "hwmanuals/pdf"
# File to keep track of which PDF files have been indexed and their modification times.
INDEXED_FILES_RECORD = "hwmanuals/qdrant_indexed_files_record.json" # Stored locally

# Qdrant client configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333 # Default REST API port
QDRANT_GRPC_PORT = 6334 # Default gRPC port (more efficient for batch operations)
QDRANT_COLLECTION_NAME = "PdfManualChunks" # Qdrant collection name

CHUNK_SIZE = 500  # Number of characters per text chunk
CHUNK_OVERLAP = 50  # Number of characters that overlap between consecutive chunks

# --- Embedding Configuration ---
# Choose your embedding source: "ollama" or "sentence-transformer"
EMBEDDING_SOURCE = "sentence-transformer" # Set default to ollama

# Sentence Transformer specific configuration
SENTENCE_TRANSFORMER_MODEL_NAME = "BAAI/bge-base-en-v1.5"
SENTENCE_TRANSFORMER_EMBEDDING_DIMENSION = 768

# Ollama specific configuration
OLLAMA_API_BASE_URL = "http://localhost:11434" # Default Ollama API base URL
OLLAMA_MODEL_NAME = "mxbai-embed-large" # Model to use with Ollama for embeddings
OLLAMA_EMBEDDING_DIMENSION = 768 # Dimension for nomic-embed-text embeddings

# Dynamic Embedding Dimension (will be set during initialization)
EMBEDDING_DIMENSION = None 

# --- Global Variables for Model and Qdrant Client (initialized once) ---
embedding_model = None        # Will hold the loaded SentenceTransformer model
ollama_client = None          # Will hold the Ollama client instance
qdrant_client = None          # Will hold the Qdrant client instance
indexed_files = {}            # Stores {filepath: last_mod_time} for idempotency checking

def initialize_components():
    """
    Initializes the embedding model (SentenceTransformer or Ollama) and Qdrant client,
    ensuring the collection exists.
    This function should be called before any indexing or searching operations.
    """
    global embedding_model, ollama_client, qdrant_client, EMBEDDING_DIMENSION

    # 1. Initialize Embedding Model based on EMBEDDING_SOURCE
    if EMBEDDING_SOURCE == "sentence-transformer":
        if embedding_model is None:
            print(f"Loading embedding model: {SENTENCE_TRANSFORMER_MODEL_NAME}...")
            try:
                embedding_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_NAME)
                EMBEDDING_DIMENSION = SENTENCE_TRANSFORMER_EMBEDDING_DIMENSION
                print("Sentence Transformer model loaded successfully.")
            except Exception as e:
                print(f"Error loading Sentence Transformer model: {e}")
                print("Please ensure you have an active internet connection or the model is cached locally.")
                print("You might need to install 'sentence-transformers' if you haven't: pip install sentence-transformers")
                exit(1) # Exit if model cannot be loaded, as it's critical

    # 2. Create directories if they don't exist
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(INDEXED_FILES_RECORD), exist_ok=True)
    print(f"Ensured '{PDF_DIR}' directory exists.")

    # 3. Initialize Qdrant Client
    if qdrant_client is None:
        print(f"Connecting to Qdrant at: {QDRANT_HOST}:{QDRANT_GRPC_PORT}")
        try:
            qdrant_client = QdrantClient(host=QDRANT_HOST, grpc_port=QDRANT_GRPC_PORT, prefer_grpc=True)
            # Check if Qdrant is accessible by trying to list collections
            qdrant_client.get_collections() 
            print("Qdrant client connected.")

            # 4. Create Qdrant Collection if it doesn't exist
            if not qdrant_client.collection_exists(collection_name=QDRANT_COLLECTION_NAME):
                print(f"Creating Qdrant collection '{QDRANT_COLLECTION_NAME}' with dimension {EMBEDDING_DIMENSION}...")
                qdrant_client.recreate_collection(
                    collection_name=QDRANT_COLLECTION_NAME,
                    vectors_config=models.VectorParams(size=EMBEDDING_DIMENSION, distance=models.Distance.COSINE),
                )
                print(f"Collection '{QDRANT_COLLECTION_NAME}' created.")
            else:
                # Check if the existing collection has the correct dimension.
                # If not, you might need to recreate it or handle the mismatch.
                collection_info = qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
            
        except Exception as e:
            print(f"Error initializing Qdrant client or collection: {e}")
            print("Please ensure your Qdrant Docker container is running and accessible.")
            exit(1)

def load_indexed_record():
    """
    Loads the record of already indexed files from INDEXED_FILES_RECORD.
    This record is used to check if a file needs re-indexing.
    """
    global indexed_files
    if os.path.exists(INDEXED_FILES_RECORD):
        try:
            with open(INDEXED_FILES_RECORD, 'r') as f:
                indexed_files = json.load(f)
            print(f"Loaded indexed files record from {INDEXED_FILES_RECORD}.")
        except json.JSONDecodeError as e:
            print(f"Error decoding {INDEXED_FILES_RECORD}: {e}. Starting with an empty record.")
            indexed_files = {}
    else:
        indexed_files = {}
        print("No existing indexed files record found. Starting fresh.")

def save_indexed_record():
    """Saves the current record of indexed files to INDEXED_FILES_RECORD."""
    try:
        with open(INDEXED_FILES_RECORD, 'w') as f:
            json.dump(indexed_files, f, indent=4)
    except IOError as e:
        print(f"Error saving indexed files record to {INDEXED_FILES_RECORD}: {e}")

def get_file_modification_time(filepath):
    """
    Returns the last modification timestamp of a given file.
    Used for the idempotency checker.
    """
    try:
        return os.path.getmtime(filepath)
    except OSError as e:
        print(f"Error getting modification time for {filepath}: {e}")
        return None

def extract_text_from_pdf(filepath):
    """
    Extracts text from a PDF file page by page.
    Returns a list of dictionaries, where each dictionary contains
    the page number and the text extracted from that page.
    """
    text_content = []
    try:
        with open(filepath, 'rb') as file:
            reader = PdfReader(file)
            num_pages = len(reader.pages)
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text = page.extract_text()
                if text:
                    text_content.append({"page_num": page_num + 1, "text": text})
        return text_content
    except Exception as e:
        print(f"Error extracting text from PDF '{filepath}': {e}")
        return None

def chunk_text(text_segments, chunk_size, chunk_overlap):
    """
    Chunks text segments (from individual pages) into smaller pieces.
    Each chunk retains its original page number and source file information.
    """
    chunks = []
    for segment in text_segments:
        full_text = segment['text']
        page_num = segment['page_num']
        if not full_text:
            continue

        start_idx = 0
        while start_idx < len(full_text):
            end_idx = min(start_idx + chunk_size, len(full_text))
            chunk_content = full_text[start_idx:end_idx].strip()
            
            if chunk_content: # Only add non-empty chunks
                chunks.append({
                    'text': chunk_content,
                    'page': page_num,
                    'start_char': start_idx,
                    'end_char': end_idx
                })
            
            # Move start index for the next chunk
            if end_idx == len(full_text):
                break 
            start_idx += (chunk_size - chunk_overlap)
            
            if start_idx >= len(full_text):
                 start_idx = len(full_text) - chunk_size if len(full_text) > chunk_size else 0
                 if start_idx < 0: start_idx = 0

    return chunks

def embed_chunks_sentence_transformer(chunks):
    """
    Generates embeddings for a list of text chunks using the Sentence Transformer model.
    """
    if embedding_model is None:
        raise RuntimeError("Sentence Transformer embedding model not initialized. Call initialize_components() first.")
    
    texts_to_embed = [chunk['text'] for chunk in chunks]
    if not texts_to_embed:
        return np.array([]) # Return empty numpy array if no texts to embed
    
    # Encode the texts to get their embeddings
    embeddings = embedding_model.encode(texts_to_embed, show_progress_bar=False, convert_to_numpy=True)
    return embeddings

def embed_chunks_ollama(chunks):
    """
    Generates embeddings for a list of text chunks using the Ollama client.
    """

    texts_to_embed = [chunk['text'] for chunk in chunks]
    if not texts_to_embed:
        return np.array([]) # Return empty numpy array if no texts to embed
    
    embeddings = []
    for text in tqdm(texts_to_embed, desc="Generating Ollama Embeddings"):
        try:
            # Ollama's embed function expects a single prompt for embedding
            response = ollama.embed(model="mxbai-embed-large", input=text)
            embeddings.append(response['embeddings'])
        except Exception as e:
            print(f"Error generating Ollama embedding for a chunk: {e}")
            # Append a zero vector or handle error as appropriate
            embeddings.append(np.zeros(EMBEDDING_DIMENSION).tolist()) 
            
    return np.array(embeddings)


def embed_chunks(chunks):
    """
    Generates embeddings for a list of text chunks using the chosen embedding source.
    """
    if EMBEDDING_SOURCE == "sentence-transformer":
        return embed_chunks_sentence_transformer(chunks)
    elif EMBEDDING_SOURCE == "ollama":
        return embed_chunks_ollama(chunks)
    else:
        raise ValueError(f"Unknown EMBEDDING_SOURCE: {EMBEDDING_SOURCE}")

def index_pdfs():
    """
    Main function to process, chunk, embed, and index PDF files using Qdrant.
    It checks for existing index files and updates only new or modified PDFs.
    """
    global qdrant_client, indexed_files

    initialize_components() # Ensure Qdrant client and collection are ready
    load_indexed_record()   # Load previously indexed files record

    # Find all PDF files in the specified directory and its subdirectories
    pdf_files = []
    for root, _, files in os.walk(PDF_DIR):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))

    if not pdf_files:
        print(f"No PDF files found in '{PDF_DIR}'. Please place your PDF manuals there.")
        return

    print(f"Found {len(pdf_files)} PDF files in '{PDF_DIR}'.")

    # List to hold Qdrant PointStructs for batch upsert
    points_to_upsert = []
    
    # Iterate through each found PDF file
    for filepath in tqdm(pdf_files, desc="Processing PDFs"):
        file_mod_time = get_file_modification_time(filepath)
        
        needs_processing = False
        if filepath not in indexed_files:
            needs_processing = True
            print(f"Indexing new file: '{filepath}'...")
        elif indexed_files[filepath] != file_mod_time:
            needs_processing = True
            print(f"Re-indexing modified file: '{filepath}'...")
            
            # If the file was modified, delete its existing points from Qdrant
            try:
                qdrant_client.delete_points(
                    collection_name=QDRANT_COLLECTION_NAME,
                    points_selector=models.PointSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="source_file",
                                    match=models.MatchValue(value=filepath),
                                ),
                            ],
                        )
                    ),
                )
                print(f"Successfully deleted old chunks for '{filepath}' from Qdrant.")
            except Exception as e:
                print(f"Error deleting old chunks for '{filepath}': {e}. Proceeding, but duplicates might occur if deletion failed.")
        else:
            print(f"Skipping '{filepath}' as it is already indexed and unchanged.")
            continue # Skip to the next file if no processing is needed

        if needs_processing:
            # 1. Extract Text from PDF
            text_segments = extract_text_from_pdf(filepath)
            if not text_segments:
                print(f"Could not extract text from '{filepath}'. Skipping.")
                continue

            # 2. Chunk Extracted Text
            file_chunks = []
            for segment in text_segments:
                chunks_from_page = chunk_text([segment], CHUNK_SIZE, CHUNK_OVERLAP)
                file_chunks.extend(chunks_from_page)

            if not file_chunks:
                print(f"No text chunks generated for '{filepath}'. Skipping.")
                continue
            
            # 3. Embed Chunks
            chunk_embeddings = embed_chunks(file_chunks)
            if chunk_embeddings.size == 0:
                print(f"No embeddings generated for '{filepath}'. Skipping.")
                continue

            # Prepare data for Qdrant batch upsert
            for i, chunk_data in enumerate(file_chunks):
                # Qdrant payload (metadata)
                payload = {
                    "text": chunk_data['text'],
                    "source_file": filepath,
                    "page": chunk_data['page'],
                    "start_char": chunk_data['start_char'],
                    "end_char": chunk_data['end_char']
                }
                
                # Create a PointStruct for Qdrant
                try:
                    points_to_upsert.append(
                        models.PointStruct(
                            id=str(uuid.uuid4()), # Unique ID for each point
                            vector=chunk_embeddings[i].tolist(), # Convert numpy array to list
                            payload=payload
                        )
                    )
                except:
                    continue
            
            # Update the indexed files record with the new modification time
            indexed_files[filepath] = file_mod_time
            save_indexed_record() # Save after each file to prevent data loss on crash

    if not points_to_upsert:
        print("No new or updated PDF files found to index. Qdrant is up to date.")
        return # No new points to add

    print(f"Upserting {len(points_to_upsert)} new or updated chunks to Qdrant...")
    try:
        # Batch upsert points to Qdrant
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            wait=True, # Wait for the operation to complete
            points=points_to_upsert
        )
        
        print(f"Successfully upserted {len(points_to_upsert)} chunks to Qdrant.")
        # Get current collection count
        collection_info = qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_name)
        print(f"Total items in Qdrant collection '{QDRANT_COLLECTION_NAME}': {collection_info.points_count}")

    except Exception as e:
        print(f"Error upserting chunks to Qdrant: {e}")
    
    print("\nPDF indexing process completed!")


def get_qdrant_embeddings_plain_text():
    """
    Retrieves all points (including their embeddings) from the specified Qdrant collection
    and prints them in a plain text format.
    """
    global qdrant_client
    if qdrant_client is None: # Check if client initialization failed
        print("Qdrant client not initialized. Cannot retrieve embeddings.")
        return

    print(f"\nRetrieving embeddings from collection: '{QDRANT_COLLECTION_NAME}'...")

    try:
        all_points = []
        offset = None # Start with no offset
        has_more = True

        while has_more:
            points, current_offset = qdrant_client.scroll(
                collection_name=QDRANT_COLLECTION_NAME,
                limit=100, # Number of points to retrieve per scroll batch
                offset=offset,
                with_vectors=True, # IMPORTANT: Request the actual vector embeddings
                with_payload=True  # Request the associated metadata
            )
            all_points.extend(points)
            offset = current_offset # Update offset for next iteration
            has_more = (current_offset is not None)

        if not all_points:
            print(f"No points found in collection '{QDRANT_COLLECTION_NAME}'.")
            return

        print(f"Successfully retrieved {len(all_points)} points.")
        print("\n--- Qdrant Embeddings and Metadata ---")

        for i, point in enumerate(all_points):
            print(f"\n--- Point {i+1} (ID: {point.id}) ---")
            
            if point.vector is not None:
                vector_str = "[" + ", ".join(f"{x:.4f}" for x in point.vector[:5]) + "..., " + ", ".join(f"{x:.4f}" for x in point.vector[-5:]) + "]"
                if len(point.vector) > 10:
                    print(f"Vector (partial): {vector_str} (Dimension: {len(point.vector)})")
                else:
                    print(f"Vector: {point.vector} (Dimension: {len(point.vector)})")
            else:
                print("Vector: Not available")

            if point.payload:
                print("Payload (Metadata):")
                for key, value in point.payload.items():
                    if isinstance(value, str) and len(value) > 150:
                        print(f"  {key}: {value[:147]}...")
                    else:
                        print(f"  {key}: {value}")
            else:
                print("Payload: None")
        
        print("\n--- End of Embeddings List ---")

    except Exception as e:
        print(f"Error retrieving points from Qdrant: {e}")

def search_indexed_pdfs(query_text, k=5):
    """
    Searches the indexed PDF chunks for the most similar chunks to the query using Qdrant.
    This function will initialize the Qdrant client if it's not already in memory.

    Args:
        query_text (str): The text query (e.g., a question) to search for.
        k (int): The number of top similar results to return.

    Returns:
        list: A list of dictionaries, each containing 'score' (similarity score, higher is better for cosine),
              'text' (the chunk text), and 'chunk_metadata' (source file, page, etc.).
    """
    global qdrant_client, embedding_model, ollama_client

    # Ensure components are initialized and client is ready
    initialize_components()

    # Check if collection exists and has points
    try:
        collection_info = qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        if collection_info.points_count == 0:
            print(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' is empty. Please ensure 'index_pdfs()' was run successfully and found files.")
            return []
    except Exception as e:
        print(f"Could not retrieve Qdrant collection info: {e}. Ensure collection exists and Qdrant is running.")
        return []

    print(f"Searching for: '{query_text}' in Qdrant using {EMBEDDING_SOURCE} embeddings...")
    
    try:
        # Embed the query text based on the chosen source
        if EMBEDDING_SOURCE == "sentence-transformer":
            if embedding_model is None:
                raise RuntimeError("Sentence Transformer model not initialized for search.")
            query_embedding = embedding_model.encode([query_text]).tolist()
        elif EMBEDDING_SOURCE == "ollama":
            if ollama_client is None:
                raise RuntimeError("Ollama client not initialized for search.")
            response = ollama.embed(model=OLLAMA_MODEL_NAME, input=query_text)
            query_embedding = response['embedding']
        else:
            raise ValueError(f"Unknown EMBEDDING_SOURCE: {EMBEDDING_SOURCE}")

        # Perform similarity search in Qdrant
        search_result = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_embedding[0],
            limit=k,
            with_payload=True,  # Retrieve the associated metadata
            with_vectors=True  # No need to retrieve the vectors themselves for display
        )
        print('serarch res',search_result)
        formatted_results = []
        for hit in search_result:
            formatted_results.append({
                'score': hit.score,
                'text': hit.payload.get('text', 'N/A'),
                'chunk_metadata': {
                    "source_file": hit.payload.get('source_file', 'N/A'),
                    "page": hit.payload.get('page', 'N/A'),
                    "start_char": hit.payload.get('start_char', 'N/A'),
                    "end_char": hit.payload.get('end_char', 'N/A')
                }
            })
            
        return formatted_results

    except Exception as e:
        print(f"Error during Qdrant search: {e}")
        return []

# --- Example Usage ---
if __name__ == "__main__":
    print("--- Starting PDF Indexing Process with Qdrant ---")
    # Step 1: Run the indexing process.
    # This will create 'hwmanuals/pdf' directory.
    # Place your PDF files inside 'hwmanuals/pdf' before running.
    # Ensure your Qdrant Docker container and Ollama (if EMBEDDING_SOURCE is "ollama") are running before executing this.
    index_pdfs()

    print("\n--- Demonstrating Search ---")
    # Step 2: After indexing, you can search the documents.
    search_query = "NVIDIA Ampere Architecture"
    search_results = search_indexed_pdfs(search_query, k=10) # Get top 3 relevant chunks

    if search_results:
        print(f"\nTop {len(search_results)} relevant chunks for query: '{search_query}'")
        for i, result in enumerate(search_results):
            chunk_meta = result['chunk_metadata']
            print(f"\n--- Result {i+1} (Score: {result['score']:.4f}) ---")
            print(f"Source File: {chunk_meta.get('source_file', 'N/A')}")
            print(f"Page: {chunk_meta.get('page', 'N/A')}")
            print("Chunk Text:")
            print("---")
            print(result['text'])
            print("---")
    else:
        print("\nNo search results found. Ensure PDFs are present and indexing was successful.")

    # Uncomment the line below to retrieve and print all embeddings and metadata from Qdrant
    # get_qdrant_embeddings_plain_text()

    print("\n--- Process Finished ---")
