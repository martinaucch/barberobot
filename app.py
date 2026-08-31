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
    time.sleep(3)
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
        # Execute the Hybrid GraphRAG query!
        # CRITICAL FIX: Only pass the user's short message to the retriever!
        response = await cl.make_async(query_engine.query)(message.content)
        
        # Send the main text
        msg.content = str(response)
        await msg.update()

        # ==========================================
        # SOURCE ATTRIBUTION
        # ==========================================
        if response.source_nodes:
            sources_message = "\n\n**Sources Retrieved:**\n"
            
            for i, node in enumerate(response.source_nodes):
                source_type = node.node.metadata.get("source", "ChromaDB (Transcript)")
                
                if source_type == "rdflib_graph_metadata":
                    fact = node.node.get_text().replace('\n', ' | ')
                    sources_message += f"* **Lesson Metadata:** {fact}\n"
                else:
                    file_id = node.node.metadata.get("file_id", "Unknown File")
                    snippet = node.node.get_text()[:100].replace("\n", " ") + "..."
                    sources_message += f"* **Transcript Chunk** (`{file_id}`): {snippet}\n"

            # Send the sources as a secondary message
            await cl.Message(content=sources_message, author="System").send()
        
    except Exception as e:
        msg.content = f"Sorry, I encountered an error while processing that: {str(e)}"
        await msg.update()