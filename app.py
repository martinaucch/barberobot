import os
# Fix for Mac Apple Silicon (MPS) Out of Memory errors when loading large embedding models
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import chainlit as cl
from llama_index.core import Settings
from llama_index.llms.groq import Groq 
from llama_index.core.query_engine import RetrieverQueryEngine
from dotenv import load_dotenv 

# Import our custom hybrid retriever setup
from hybrid_retriever import setup_hybrid_retriever

# Load environment variables from your .env file
load_dotenv()

# Define the persona and instructions for BarberoBot
prompt_text = """You are BarberoBot, an AI assistant specialized in the lectures and historical knowledge of Professor Alessandro Barbero.

MISSION
Your mission is to provide precise and detailed information to users based on the transcripts of Alessandro Barbero's lectures and the associated Knowledge Graph facts. 

Your KNOWLEDGE BASE consists of:
1. Semantic text chunks from the original lecture transcripts (Vector Search).
2. Explicit historical facts and metadata extracted from the Barberotheca Knowledge Graph (Graph Traversal).

INSTRUCTIONS
- Always prioritize the context provided to you by the retriever. 
- Break down complex historical concepts, making them understandable while maintaining Barbero's engaging style.
- IF the retrieved context does not contain the required information, you may use your general knowledge, but you MUST preface it by saying: "I am not entirely sure about this based on the lecture transcripts, but..."
- If you use a Knowledge Graph Fact from the context, try to mention the explicit relationship (e.g., "According to the graph, X is connected to Y").

Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer: """

@cl.on_chat_start
async def start():
    """
    Runs once when the user opens the web browser.
    Initializes the Groq LLM and the Hybrid GraphRAG pipeline.
    """
    msg = cl.Message(content="Loading BarberoBot (Connecting to Groq, ChromaDB, and RDFlib)...")
    await msg.send()

    try:
        # ==========================================
        # 1. SETUP THE LLM
        # ==========================================
        llm = Groq(
            model="qwen/qwen3.8-27b", 
            api_key=os.environ.get("GROQ_API_KEY"),
            temperature=0.1
        )
        Settings.llm = llm
        
        # ==========================================
        # 2. INITIALIZE HYBRID PIPELINE
        # (This also sets Settings.embed_model)
        # ==========================================
        hybrid_retriever = setup_hybrid_retriever()
        
        # Connect the hybrid retriever to the Query Engine
        query_engine = RetrieverQueryEngine.from_args(
            retriever=hybrid_retriever,
        )

        # Inject our custom BarberoBot prompt so it only affects the LLM, NOT the retriever!
        from llama_index.core import PromptTemplate
        qa_prompt = PromptTemplate(prompt_text)
        query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt})

        # Store the query engine in the user's session state
        cl.user_session.set("query_engine", query_engine)

        msg.content = "BarberoBot is ready! Ask me anything about Professor Barbero's lectures, the Battle of Lepanto, the Crusades, or any other historical topic he covers."
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
                
                if source_type == "rdflib_graph_traversal":
                    fact = node.node.get_text()
                    sources_message += f"* **Graph Fact:** {fact}\n"
                else:
                    file_id = node.node.metadata.get("file_id", "Unknown File")
                    snippet = node.node.get_text()[:100].replace("\n", " ") + "..."
                    sources_message += f"* **Transcript Chunk** (`{file_id}`): {snippet}\n"

            # Send the sources as a secondary message
            await cl.Message(content=sources_message, author="System").send()
        
    except Exception as e:
        msg.content = f"Sorry, I encountered an error while processing that: {str(e)}"
        await msg.update()