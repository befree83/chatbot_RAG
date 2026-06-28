from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from core.rag_system import RAGSystem

class Chatbot:
    """
    Handles conversational flow execution, context injections,
    strict instruction enforcement, and multi-turn session tracking.
    """
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        # Enforce exact model assignment specifications matching instructions
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        self.chat_history: List[Any] = []
        
        # System instructions guaranteeing strict alignment with corporate text context
        self.system_prompt = SystemMessage(
            content=(
                "You are an advanced, internal corporate AI chatbot assistant. "
                "Your objective is to answer questions using strictly the context fragments provided below.\n\n"
                "CRITICAL CONSTRAINTS:\n"
                "1. Base your answers ONLY on the provided context fragments. Do not invent facts, extrapolate, or use external knowledge.\n"
                "2. If the context fragments do not contain enough details to fully answer a question, state clearly: "
                "'I am sorry, but I do not possess that information within the corporate knowledge base records.'\n"
                "3. Maintain a professional, crisp, corporate tone at all times.\n"
                "4. All replies must be written clearly in English."
            )
        )

    def process_message(self, user_query: str) -> Dict[str, Any]:
        """
        Executes the query pipeline: context extraction, historical evaluation, 
        response generation, and state persistence.
        """
        # 1. Retrieve context fragments using the RAG Engine
        relevant_chunks = self.rag_system.retrieve_context(user_query, k=3)
        context_str = "\n---\n".join([chunk.page_content for chunk in relevant_chunks])
        
        # 2. Structure context injection block
        context_injection = SystemMessage(
            content=f"--- CONTEXT FRAGMENTS STARTED ---\n{context_str}\n--- CONTEXT FRAGMENTS ENDED ---"
        )
        
        # 3. Assemble dynamic processing sequence payloads
        execution_messages = [self.system_prompt, context_injection]
        
        # Inject conversational memory window
        execution_messages.extend(self.chat_history[-6:])
        
        # Append target current query statement
        execution_messages.append(HumanMessage(content=user_query))
        
        try:
            # 4. Invoke LLM execution call
            response = self.llm.invoke(execution_messages)
            output_text = response.content
            
            # 5. Persist messaging records to memory sequence
            self.chat_history.append(HumanMessage(content=user_query))
            self.chat_history.append(AIMessage(content=output_text))
            
            # Extract tracking targets for diagnostic visualization interfaces
            sources = list(set([os.path.basename(chunk.metadata.get('source', 'unknown')) for chunk in relevant_chunks]))
            
            return {
                "response": output_text,
                "sources": sources
            }
        except Exception as e:
            return {
                "response": f"An operational framework processing anomaly occurred: {str(e)}",
                "sources": []
            }

    def clear_session_memory(self):
        """Resets the active structural conversation state tracking vector."""
        self.chat_history.clear()