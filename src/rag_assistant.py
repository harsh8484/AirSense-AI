import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class LocalRAGAssistant:
    def __init__(self, doc_dir="data/raw/documents"):
        self.doc_dir = doc_dir
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.chunks = []
        self.chunk_sources = []
        self.is_indexed = False
        
    def load_and_index_documents(self):
        """
        Reads document files from the documents folder, breaks them into paragraphs (chunks),
        and builds a TF-IDF index.
        """
        txt_files = glob.glob(os.path.join(self.doc_dir, "*.txt"))
        if not txt_files:
            print(f"[WARNING] No documents found in {self.doc_dir} for RAG indexing.")
            return False
            
        self.chunks = []
        self.chunk_sources = []
        
        for file_path in txt_files:
            source_name = os.path.basename(file_path).replace(".txt", "").replace("_", " ").upper()
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                # Split content into paragraphs/blocks (separated by double newlines or single newlines with content)
                paragraphs = [p.strip() for p in content.split("\n") if len(p.strip()) > 30]
                
                for p in paragraphs:
                    self.chunks.append(p)
                    self.chunk_sources.append(source_name)
            except Exception as e:
                print(f"[ERROR] Failed reading document {file_path}: {e}")
                
        if self.chunks:
            # Fit TF-IDF matrix
            self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
            self.is_indexed = True
            print(f"[SUCCESS] Indexed {len(self.chunks)} text chunks from RAG documents.")
            return True
        return False
        
    def query(self, user_query, top_n=2):
        """
        Queries the vector index for similar passages, then synthesizes a response.
        """
        if not self.is_indexed:
            success = self.load_and_index_documents()
            if not success:
                return {
                    "answer": "Hello! I am your AirSense AI Assistant. It looks like the knowledge base documents haven't been generated yet. Please generate mock data or upload reference text files to start searching CPCB and WHO guidelines.",
                    "citations": []
                }
                
        # Transform user query
        query_vec = self.vectorizer.transform([user_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top matching chunks
        top_indices = similarities.argsort()[::-1][:top_n]
        
        retrieved_chunks = []
        retrieved_sources = []
        max_scores = []
        
        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0.05:  # Relevance threshold
                retrieved_chunks.append(self.chunks[idx])
                retrieved_sources.append(self.chunk_sources[idx])
                max_scores.append(round(score, 3))
                
        # Generate Answer based on retrieved chunks (Simulated GenAI LLM Response Generator)
        if not retrieved_chunks:
            answer = (
                "I couldn't find specific details regarding that in my current knowledge base. "
                "However, generally, PM2.5 levels are highly influenced by Aerosol Optical Depth (AOD) "
                "along with weather factors like Planetary Boundary Layer Height (PBLH) which traps "
                "emissions, Relative Humidity which changes particle size, and Wind Speed which scatters pollutants. "
                "Please let me know if you would like me to explain standard CPCB or WHO air quality thresholds!"
            )
            citations = []
        else:
            # We construct a smart summarized response combining retrieved data
            citations = list(set(retrieved_sources))
            
            # Simple offline context synthesis
            joined_context = " ".join(retrieved_chunks)
            
            # Smart custom response templates based on query keywords
            q = user_query.lower()
            if "delhi" in q or "aqi" in q or "high" in q or "reduce" in q or "pm2.5" in q:
                answer = f"Based on the **{', '.join(citations)}** documents, here is the context on air quality:\n\n"
                
                # Check what content we have
                if "NAAQS" in joined_context or "cpcb" in q or "standards" in q:
                    answer += (
                        "- **CPCB Air Quality Standards**: The National Ambient Air Quality Standard (NAAQS) "
                        "for PM2.5 in India is **60 µg/m³** over 24 hours, and **40 µg/m³** annually.\n"
                        "- **AQI Categories**: CPCB defines AQI bands. Good (0-50), Satisfactory (51-100), "
                        "Moderately Polluted (101-200), Poor (201-300), Very Poor (301-400), and Severe (401-500). "
                        "PM2.5 above 250 µg/m³ triggers the Severe category, affecting healthy individuals.\n"
                    )
                if "who" in q or "guidelines" in q or "world health" in q:
                    answer += (
                        "- **WHO Guidelines**: The World Health Organization is significantly stricter, recommending "
                        "an annual average PM2.5 of **5 µg/m³** and a 24-hour limit of **15 µg/m³** to reduce health risks.\n"
                    )
                if "stubble" in q or "burning" in q or "policy" in q or "reduce" in q or "mitigation" in q:
                    answer += (
                        "- **Mitigation & Policies**: Primary interventions include stubble burning mitigation "
                        "( Happy Seeder machines, Pusa bio-decomposers), transitioning public transit to Electric Vehicles (EV), "
                        "and implementing dust controls (road watering, mechanical sweeping, and green belt borders).\n"
                    )
                if "aod" in q or "satellite" in q or "reanalysis" in q:
                    answer += (
                        "- **Scientific Estimation**: Satellite Aerosol Optical Depth (AOD) measures total atmospheric column particles. "
                        "Surface PM2.5 is calculated by adjusting AOD with Planetary Boundary Layer Height (PBLH) "
                        "and Relative Humidity (RH). A low PBLH compresses pollution near the ground, while high humidity causes aerosol hygroscopic growth.\n"
                    )
                # Fallback if no template matched
                if len(answer) < 80:
                    answer += f"Retrieved Context:\n" + "\n".join([f"- {c}" for c in retrieved_chunks])
            else:
                answer = "Here is what I found in the RAG Knowledge Base:\n\n"
                for i, chunk in enumerate(retrieved_chunks):
                    answer += f"**From {retrieved_sources[i]}** (relevance score: {max_scores[i]}):\n{chunk}\n\n"
                    
        return {
            "answer": answer,
            "citations": citations,
            "chunks_retrieved": len(retrieved_chunks)
        }

if __name__ == "__main__":
    rag = LocalRAGAssistant()
    rag.load_and_index_documents()
    res = rag.query("What is the CPCB standard for PM2.5?")
    print("Answer:\n", res["answer"])
    print("Citations:", res["citations"])
