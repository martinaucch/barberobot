import os
import re
import time
from collections import defaultdict
import chainlit as cl
from dotenv import load_dotenv 

# LlamaIndex Imports
from llama_index.core import Settings
from llama_index.llms.groq import Groq 
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole

# Import our custom hybrid retriever setup
from hybrid_retriever import setup_hybrid_retriever



# Load environment variables
load_dotenv()

# Define the persona and instructions for BarberoBot
prompt_text = """You are the digital teaching assistant to Professor Alessandro Barbero, dedicated to helping users explore the "Barberotheca" lecture archive.

Your role is to act as a scholarly recommendation and guidance system. When answering a user's question, strictly follow these instructions:

1. TONE & PERSONA:
   - Polite, academic yet accessible, enthusiastic about history, and faithful to Professor Barbero's narrative style.
   - Speak on behalf of the archive (e.g., "Il Professor Barbero ha trattato questo argomento in...", "Nelle sue lezioni emerge che...").
   - Do NOT use greetings like "Ciao!" or "Benvenuto!" if there is already a conversation history. For follow-up questions, jump straight into the answer.

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
     - Suggest related lessons in the same series to further the user's exploration (e.g. "Le altre lezioni di questa serie sono '[sibling_lessons]').
     - Se nei metadati compare "Altre lezioni con temi/entità in comune", consiglia anche queste ultime dicendo qualcosa come: "Potrebbero interessarti anche queste altre lezioni su temi simili: ...".
    
4. CONVERSATIONAL CONTEXT:
   - Below is the recent conversation history. Use it to understand follow-up questions (e.g. if the user says "dimmi di più", refer to the last topic discussed).

Context information is below.
---------------------
{context_str}
---------------------
Recent Conversation History:
---------------------
{chat_history}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer: """

@cl.on_chat_start
async def start():
    """
    Runs once when the user opens the web browser. This initializes the isolated user session.
    """
    # 1. Setup the LLM specific to this session
    llm = Groq(
        model="qwen/qwen3.8-27b", 
        api_key=os.environ.get("GROQ_API_KEY"),  
        temperature=0.1,
        context_window=32768,
    )
    Settings.llm = llm
    Settings.context_window = 32768

    # Show the logo for 2 seconds before loading
    import asyncio
    await asyncio.sleep(2)

    # 2. Let the user know the system is booting up
    
    msg = cl.Message(content="Caricamento dei modelli della Barberotheca in memoria...")
    await msg.send()

    try:
        # 3. Initialize the Hybrid Retriever and Query Engine FOR THIS USER ONLY
        hybrid_retriever = setup_hybrid_retriever()
        
        query_engine = RetrieverQueryEngine.from_args(
            retriever=hybrid_retriever,
        )
        
        # Apply the custom prompt
        query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": PromptTemplate(prompt_text)}
        )
        
        # 4. Store the engine and memory in the isolated user session dictionary
        memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
        cl.user_session.set("memory", memory)
        cl.user_session.set("query_engine", query_engine)

        msg.content = "BarberoBot è pronto! Chiedimi ciò che desideri riguardo alle lezioni del professor Barbero."
        await msg.update()
        
    except Exception as e:
        msg.content = f"Errore durante l'inizializzazione del sistema: {str(e)}"
        await msg.update()

@cl.on_message
async def main(message: cl.Message):
    """
    Runs every time the user types a message in the chat.
    """
    # 1. Retrieve the isolated query engine and memory for THIS specific user
    query_engine = cl.user_session.get("query_engine")
    memory = cl.user_session.get("memory")
    
    if not query_engine or not memory:
        await cl.Message(content="Il sistema non è stato inizializzato correttamente. Ricarica la pagina.").send()
        return

    # Create an empty message for streaming the response
    msg = cl.Message(content="", author="BarberoBot")
    await msg.send()

    try:
        # ==========================================
        # FORMAT HISTORY & UPDATE PROMPT
        # ==========================================
        chat_history_list = memory.get()
        history_str = ""
        for m in chat_history_list:
            role = "Utente" if m.role == MessageRole.USER else "Assistente"
            history_str += f"{role}: {m.content}\n\n"
            
        if not history_str:
            history_str = "Nessuna conversazione precedente."

        dynamic_prompt = prompt_text.replace("{chat_history}", history_str)
        query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": PromptTemplate(dynamic_prompt)}
        )

        # ==========================================
        # CONDENSE QUERY & EXECUTE RETRIEVAL
        # ==========================================
        # If we have history, rewrite the query to include context (e.g. "di che nazionalità era?" -> "di che nazionalità era Giovanna d'Arco?")
        actual_query = message.content
        if chat_history_list:
            condense_prompt = f"""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question that includes any relevant context from the conversation (like names or subjects).
If the follow up question is already standalone, just return it as is. Do not answer the question, ONLY return the rephrased question.

Conversation:
{history_str}

Follow Up Question: {message.content}
Standalone Question:"""
            # Use the LLM to rewrite the query
            condense_resp = await cl.make_async(Settings.llm.complete)(condense_prompt)
            actual_query = str(condense_resp).strip()
            print(f"Original query: '{message.content}' -> Rewritten: '{actual_query}'")

        # 2. Retrieve nodes manually using the contextualized query
        retriever = query_engine.retriever
        retrieved_nodes = await cl.make_async(retriever.retrieve)(actual_query)
        print(f"DEBUG: retrieved_nodes count = {len(retrieved_nodes)}")
        
        # 3. Separate graph nodes from vector nodes
        graph_nodes = []
        vector_nodes = []
        for node in retrieved_nodes:
            source_type = node.node.metadata.get("source", "ChromaDB (Transcript)")
            if source_type == "rdflib_graph_metadata":
                graph_nodes.append(node)
            else:
                vector_nodes.append(node)
                
        # 4. Synthesize the response FIRST (without citations in the text)
        final_nodes = graph_nodes + vector_nodes
        response = await cl.make_async(query_engine.synthesize)(
            message.content, nodes=final_nodes
        )
        
        final_text = str(response)
        
        # 5. Programmatically build the sources list and cl.Text elements
        elements = []
        lessons_dict = defaultdict(list)
        
        for i, node in enumerate(vector_nodes, 1):
            original_text = node.node.get_content()
            file_id = node.node.metadata.get("file_id", "Unknown File")
            
            # Extract metadata
            metadata_str = retriever.graph_retriever.get_lesson_metadata(file_id)
            id_match = re.search(r'id=(\d+)', metadata_str)
            citation_num = id_match.group(1) if id_match else str(i + 100)
            
            title_match = re.search(r'- Titolo Lezione: (.*)', metadata_str)
            lesson_name = title_match.group(1).strip() if title_match else file_id
            
            # Create a simple element name (e.g. "Estratto 1")
            element_name = f"Estratto {i}"
            
            # Append the element
            elements.append(
                cl.Text(
                    name=element_name,
                    content=f"### {lesson_name}\n\n**Testo originale:**\n\n{original_text}",
                    display="side"
                )
            )
            
            # Group by lesson for the final summary
            lessons_dict[citation_num].append({
                "title": lesson_name,
                "element_name": element_name
            })
            
        # 6. Append the Sources section to the final text
        if lessons_dict:
            final_text += "\n\n---\n**Fonti usate per questa risposta:**\n"
            for lesson_id, chunks in lessons_dict.items():
                title = chunks[0]["title"]
                element_names = [c["element_name"] for c in chunks]
                elements_str = ", ".join(element_names)
                
                # Link to Barberotheca
                lesson_url = f"https://metamuses.github.io/barberotheca/lesson.html?id={lesson_id}"
                
                final_text += f"- **[{title}]({lesson_url})** ({elements_str})\n"
                
        # 7. Save the new exchange to the memory buffer
        memory.put(ChatMessage(role=MessageRole.USER, content=message.content))
        memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=final_text))

        msg.content = final_text
        msg.elements = elements
        await msg.update()
        
    except Exception as e:
        msg.content = f"Mi dispiace, si è verificato un errore tecnico: {str(e)}"
        await msg.update()