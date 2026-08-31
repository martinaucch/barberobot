import os
# Fix for Mac Apple Silicon (MPS) Out of Memory errors when loading large embedding models
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import chainlit as cl
from llama_index.core import Settings
from llama_index.llms.groq import Groq 
from llama_index.core.query_engine import RetrieverQueryEngine
from dotenv import load_dotenv 
import time

# Import our custom hybrid retriever setup
from hybrid_retriever import setup_hybrid_retriever

# Load environment variables from your .env file
load_dotenv()

# Define the persona and instructions for BarberoBot
prompt_text = """You are the digital teaching assistant to Professor Alessandro Barbero, dedicated to helping users explore the "Barberotheca" lecture archive.

Your role is to act as a scholarly recommendation and guidance system. When answering a user's question, strictly follow these instructions:

1. TONE & PERSONA:
   - Polite, academic yet accessible, enthusiastic about history, and faithful to Professor Barbero's narrative style.
   - Speak on behalf of the archive (e.g., "Il Professor Barbero ha trattato questo argomento in...", "Nelle sue lezioni emerge che...").

2. GROUNDING & CONSTRAINTS:
   - Rely ONLY on the provided context (transcript excerpts and Knowledge Graph metadata).
   - If the Professor has NOT mentioned the topic in the provided transcripts, state honestly that the archive does not contain relevant lectures on this topic. Do not invent or pull facts from outside the provided context.

3. RESPONSE STRUCTURE:
   Format your output clearly using markdown:

   - **Il tema nell'archivio**: State immediately whether the topic is discussed and in which context/lesson(s).
   - **Cosa dice il Professore**: A structured, engaging summary of Barbero's arguments and storytelling based on the transcript chunks.
   - **Fonti e Approfondimenti**:
     - **Lezione**: [Lesson Title / Date]
     - **Guarda su YouTube**: [Link from dcterms:source metadata, if available]
     - **Trascrizione completa**: [Link from schema:mainEntityOfPage, if available]
   - **Consigli correlati** (if series metadata is present):
     - Mention the macro-theme or series (e.g., "Questa lezione fa parte della serie '[Series Title]' ([Year])").
     - Suggest related lessons in the same series to further the user's exploration.

4. INLINE CITATIONS:
   - Each context chunk provided below is numbered (e.g. Fonte 1). 
   - Quando usi un'informazione da una fonte, DEVI inserire il numero della fonte tra parentesi quadre (es. [1], [2]) direttamente nel testo di "**Cosa dice il Professore**".
   - CRITICAL: Do NOT output a bullet point for "Citazioni in linea" and do NOT append a list of sources at the end of the response. The citations must ONLY be inline brackets.

Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer: """

# ==========================================
# 1. SETUP THE LLM (Global Scope)
# ==========================================
global_llm = Groq(
    model="qwen/qwen3.8-27b", 
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0.1
)
Settings.llm = global_llm

# ==========================================
# 2. INITIALIZE HYBRID PIPELINE (Global Scope)
# ==========================================
print("Loading BarberoBot models into memory (this will take ~30-60s)...")
global_hybrid_retriever = setup_hybrid_retriever()
global_query_engine = RetrieverQueryEngine.from_args(
    retriever=global_hybrid_retriever,
)
print("Models loaded! Chainlit is ready.")

@cl.on_chat_start
async def start():
    """
    Runs once when the user opens the web browser.
    """
    time.sleep(1)
    msg = cl.Message(content="BarberoBot is ready! What would you like to know?")
    await msg.send()

    try:
        # ==========================================
        # 3. SET SYSTEM PROMPT & SAVE TO SESSION
        # ==========================================
        from llama_index.core import PromptTemplate
        global_query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": PromptTemplate(prompt_text)}
        )
        
        cl.user_session.set("query_engine", global_query_engine)


        msg.content = "BarberoBot è pronto! Chiedimi ciò che desideri riguardo alle lezioni del professor Barbero."
        await msg.update()
        
    except Exception as e:
        msg.content = f"Error initializing system: {str(e)}\n\nCheck your GROQ_API_KEY and make sure you ran vector_store.py!"
        await msg.update()

@cl.on_message
async def main(message: cl.Message):
    """
    Runs every time the user types a message in the chat.
    """
    query_engine = cl.user_session.get("query_engine")

    # Create an empty message for streaming the response
    msg = cl.Message(content="", author="BarberoBot")
    await msg.send()

    try:
        # ==========================================
        # EXECUTE QUERY & SOURCE ATTRIBUTION
        # ==========================================
        # 1. Retrieve nodes manually so we can number them for inline citations
        retriever = query_engine.retriever
        retrieved_nodes = await cl.make_async(retriever.retrieve)(message.content)
        
        # 2. Separate graph nodes from vector nodes
        graph_nodes = []
        vector_nodes = []
        for node in retrieved_nodes:
            source_type = node.node.metadata.get("source", "ChromaDB (Transcript)")
            if source_type == "rdflib_graph_metadata":
                graph_nodes.append(node)
            else:
                vector_nodes.append(node)
        
        # 3. Create Chainlit elements ONLY for the vector nodes (transcripts)
        #    Each snippet gets its own unique citation so clicking opens exactly that passage.
        import re
        from collections import defaultdict
        
        # First pass: assign lesson IDs and count occurrences per lesson
        node_info = []
        lesson_counts = defaultdict(int)
        
        for i, node in enumerate(vector_nodes, 1):
            original_text = node.node.get_content()
            file_id = node.node.metadata.get("file_id", "Unknown File")
            
            metadata_str = retriever.graph_retriever.get_lesson_metadata(file_id)
            id_match = re.search(r'id=(\d+)', metadata_str)
            citation_num = id_match.group(1) if id_match else str(i + 100)
            
            title_match = re.search(r'- Titolo Lezione: (.*)', metadata_str)
            lesson_name = title_match.group(1).strip() if title_match else file_id
            
            lesson_counts[citation_num] += 1
            node_info.append((node, original_text, citation_num, lesson_name))
        
        # Second pass: build citation labels and elements
        elements = []
        lesson_seen = defaultdict(int)
        
        for node, original_text, citation_num, lesson_name in node_info:
            lesson_seen[citation_num] += 1
            
            # If a lesson has multiple snippets, add sub-index (e.g. [36.1], [36.2])
            if lesson_counts[citation_num] > 1:
                citation_label = f"{citation_num}.{lesson_seen[citation_num]}"
            else:
                citation_label = citation_num
            
            node.node.set_content(f"Fonte {citation_label}:\n{original_text}")
            
            elements.append(
                cl.Text(
                    name=f"[{citation_label}]",
                    content=f"### {lesson_name}\n\n**Passaggio citato:**\n\n{original_text}",
                    display="side"
                )
            )

        # Recombine nodes: unnumbered graph metadata first, then numbered transcripts
        final_nodes = graph_nodes + vector_nodes

        # 4. Synthesize the response with the modified nodes
        response = await cl.make_async(query_engine.synthesize)(
            message.content, nodes=final_nodes
        )
        
        # 5. Send the main text with elements attached (this enables Chainlit's inline citations)
        msg.content = str(response)
        msg.elements = elements
        await msg.update()
        
        # Close the side panel so it doesn't auto-open; user must click a citation to open it
        await cl.ElementSidebar.set_elements([])
        
    except Exception as e:
        msg.content = f"Sorry, I encountered an error while processing that: {str(e)}"
        await msg.update()