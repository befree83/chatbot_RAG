import sys
from dotenv import load_dotenv
from core.rag_system import RAGSystem
from core.chatbot import Chatbot

def bootstrap_application_runtime() -> None:
    load_dotenv()
    try:
        rag_service = RAGSystem(documents_dir="documents")
        rag_service.load_and_index_documents()
        
        enterprise_bot = Chatbot(rag_system=rag_service)
        
        print("System initialized. Ask your questions.")
        while True:
            user_input = input("\nUser > ").strip()
            if user_input.lower() in ["/exit", "quit", "/salir"]:
                break
            
            result = enterprise_bot.process_conversation_turn(user_input)
            print(f"\nBot > {result['response']}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    bootstrap_application_runtime()
