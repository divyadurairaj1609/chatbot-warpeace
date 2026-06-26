import anthropic
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()

print("=" * 50)
print("   Claude Chatbot with Vector Database")
print("=" * 50)

# Load embedding model
print("\nLoading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./vector_db")

# Connect to Claude
claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Ask user which knowledge base to use
print("\nWhich knowledge base do you want to use?")
print("  1. Divya's AI Learning Journey")
print("  2. War and Peace")
print("  3. Both (search across everything!)")

while True:
    choice = input("\nEnter 1, 2 or 3: ").strip()
    if choice in ["1", "2", "3"]:
        break
    print("Please enter 1, 2 or 3!")

# Set collection based on choice
if choice == "1":
    collections = [client.get_collection("divya_journal")]
    kb_name = "Divya's AI Learning Journey"
elif choice == "2":
    collections = [client.get_collection("war_and_peace")]
    kb_name = "War and Peace"
else:
    collections = [
        client.get_collection("divya_journal"),
        client.get_collection("war_and_peace")
    ]
    kb_name = "Both Knowledge Bases"

print(f"\nUsing: {kb_name}")
print("-" * 50)
print("Type your message and press Enter to chat!")
print("Type 'switch' to change knowledge base")
print("Type 'quit' to exit\n")

# Conversation history
history = []

def search_vector_db(question, n_results=3):
    """Search across selected collections"""
    question_embedding = model.encode(question).tolist()
    all_results = []

    for collection in collections:
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=n_results
        )
        all_results.extend(results['documents'][0])

    return all_results

# Main chat loop
while True:
    user_input = input("\nYou: ").strip()

    if user_input.lower() == "quit":
        print("Goodbye!")
        break

    if user_input.lower() == "switch":
        print("\nWhich knowledge base do you want to use?")
        print("  1. Divya's AI Learning Journey")
        print("  2. War and Peace")
        print("  3. Both")
        while True:
            choice = input("Enter 1, 2 or 3: ").strip()
            if choice in ["1", "2", "3"]:
                break
        if choice == "1":
            collections = [client.get_collection("divya_journal")]
            kb_name = "Divya's AI Learning Journey"
        elif choice == "2":
            collections = [client.get_collection("war_and_peace")]
            kb_name = "War and Peace"
        else:
            collections = [
                client.get_collection("divya_journal"),
                client.get_collection("war_and_peace")
            ]
            kb_name = "Both Knowledge Bases"
        print(f"Switched to: {kb_name}")
        history = []  # Clear history on switch
        continue

    if not user_input:
        continue

    # Search vector DB
    print(f"\nSearching {kb_name}...")
    relevant_docs = search_vector_db(user_input)
    context = "\n".join([f"- {doc}" for doc in relevant_docs])

    # Build message with context
    message_with_context = f"""Use the following context from the knowledge base to answer the question. If the context is not relevant, answer from your own knowledge.

KNOWLEDGE BASE: {kb_name}

CONTEXT:
{context}

USER QUESTION: {user_input}"""

    # Add to history
    history.append({
        "role": "user",
        "content": message_with_context
    })

    # Send to Claude
    response = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system="You are a helpful friendly assistant. When context is provided from a knowledge base, use it to give accurate answers.",
        messages=history
    )

    reply = response.content[0].text

    # Save reply to history
    history.append({
        "role": "assistant",
        "content": reply
    })

    print(f"\nClaude: {reply}")
    print("-" * 50)