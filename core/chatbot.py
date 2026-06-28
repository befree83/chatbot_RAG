from typing import Any, List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from core.rag_system import RAGSystem

class Chatbot:
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        self.chat_history: List[Any] = []
        self.system_prompt = SystemMessage(
            content="You are a professional corporate AI assistant. Answer based ONLY on the provided context."
        )

    def process_conversation_turn(self, user_query: str) -> Dict[str, Any]:
        relevant_chunks = self.rag_system.retrieve_context(user_query)
        context_str = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
        
        messages = [self.system_prompt, HumanMessage(content=f"Context: {context_str}")]
        messages.extend(self.chat_history[-6:])
        messages.append(HumanMessage(content=user_query))
        
        response = self.llm.invoke(messages)
        self.chat_history.append(HumanMessage(content=user_query))
        self.chat_history.append(AIMessage(content=response.content))
        
        return {"response": response.content, "sources": []}
