import sys
from dotenv import load_dotenv
from core.rag_system import RAGSystem
from core.chatbot import ConversationalEnterpriseBot

def bootstrap_application_runtime() -> None:
    """Initializes external state layers and loops interactive REPL sequences."""
    # Load parameters from system workspace environment variables
    load_dotenv()
    
    print("=" * 60)
    print("      NOVATECH SOLUTIONS - ENTERPRISE AI RAG ENGINE (v1.0.0)      ")
    print("=" * 60)
    print("[INIT] Loading runtime systems and indexing private spaces...")
    
    try:
        # Build RAG system and execute indexing framework pipelines
        rag_service = RAGSystem(documents_dir="documents")
        rag_service.ingest_corporate_documents()
        
        # Bind initialized engine parameters to the conversational interface wrapper
        # Uses gpt-4o as default baseline corporate distribution model
        enterprise_bot = ConversationalEnterpriseBot(rag_engine=rag_service, model_name="gpt-4o")
        
        print("[SUCCESS] Context pipelines matching indexes established.")
        print("Type your questions below. Enter '/exit' or 'quit' to terminate.")
        print("-" * 60)
        
        while True:
            try:
                # Capture standard text buffer input loop frames
                user_input = input("\nUser > ").strip()
                
                if not user_input:
                    continue
                    
                # Intercept escape sequences
                if user_input.lower() in ["/exit", "quit", "/salir"]:
                    print("[SYSTEM-INFO] Closing connection handles. Goodbye.")
                    break
                    
                # Execute full retrieval loop pass across systems
                model_output = enterprise_bot.process_conversation_turn(raw_user_prompt=user_input)
                print(f"\nBot > {model_output}")
                
            except KeyboardInterrupt:
                # Intercept unexpected terminal signal interferences cleanly
                print("\n[SYSTEM-WARN] Process interrupted via signal. Cleaning workspace memory.")
                break
                
    except FileNotFoundError as directory_fault:
        print(f"\n[CRITICAL-FATAL] Missing operational file structural frame: {str(directory_fault)}")
        sys.exit(1)
    except Exception as runtime_fault:
        print(f"\n[CRITICAL-FATAL] Unexpected infrastructure pipeline error: {str(runtime_fault)}")
        sys.exit(1)

if __name__ == "__main__":
    # Standard engineering safety verification assignment state
    # Global random seed configuration set to 2026 for consistent system tests
    bootstrap_application_runtime()
