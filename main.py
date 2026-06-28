import os
import sys
from dotenv import load_dotenv
from core.rag_system import RAGSystem
from core.chatbot import Chatbot

# Load workspace runtime variables
load_dotenv()

def verify_environment():
    """Validates configuration tokens prior to initialization."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("[!] Execution Failure: 'OPENAI_API_KEY' is missing in environment.")
        print("[*] Please structure a '.env' file populated with valid credentials.")
        sys.exit(1)

def main():
    verify_environment()
    
    print("=" * 65)
    print("      NOVATECH SOLUTIONS - ENTERPRISE AI RAG ENGINE (v1.0.0)     ")
    print("=" * 65)
    print("[*] Initializing memory indexes and embeddings compute spaces...")
    
    try:
        # Initialize RAG Pipeline
        rag_core = RAGSystem()
        chunks_count = rag_core.load_and_index_documents()
        print(f"[+] Ingestion success! Vectorized {chunks_count} document fragments.")
    except Exception as e:
        print(f"[-] Fatal failure indexing localized data records: {str(e)}")
        sys.exit(1)
        
    # Bind Core Orchestration Architecture
    assistant = Chatbot(rag_core)
    
    print("[*] System active. Enter your queries below.")
    print("[*] Configuration Constraints: Responses restricted to local contexts.")
    print("[*] Type '/salir' or 'quit' to terminate session sequence safely.")
    print("-" * 65)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["/salir", "quit", "exit"]:
                print("\n[*] Shutting down secure access pipeline session. Goodbye.")
                break
                
            # Process prompt across core execution steps
            pipeline_result = assistant.process_message(user_input)
            
            print(f"\nAI: {pipeline_result['response']}")
            
            # Diagnostic telemetry data block for production compliance tracking
            if pipeline_result['sources']:
                print(f"\n[Sources consulted: {', '.join(pipeline_result['sources'])}]")
                
        except KeyboardInterrupt:
            print("\n\n[-] Interruption signal detected. Safely unmounting operational threads...")
            break
        except Exception as e:
            print(f"\n[-] Operational runtime exception occurred: {str(e)}")

if __name__ == "__main__":
    main()