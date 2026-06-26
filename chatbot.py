import anthropic
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()

print("=" * 50)
print("   War and Peace Chatbot")
print("=" * 50)

print("\nLoading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.PersistentClient(path="./vector_db")
collection = client.get_collection("war_and_peace")
claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

print(f"Connected! ({collection.count()} pages loaded)")
print("-" * 50)
print("Ask me anything about War and Peace!")
print("Type 'quit' to exit\n")

history = []

def search_vector_db(question, n_results=3):
    question_embedding = model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results
    )
    return results['documents'][0]

while True:
    user_input = input("\nYou: ").strip()

    if user_input.lower() == "quit":
        print("Goodbye!")
        break

    if not user_input:
        continue

    print("\nSearching War and Peace...")
    relevant_docs = search_vector_db(user_input)
    context = "\n".join([f"- {doc}" for doc in relevant_docs])

    message_with_context = f"""Use the following context from War and Peace to answer the question.

CONTEXT:
{context}

USER QUESTION: {user_input}"""

    history.append({"role": "user", "content": message_with_context})

    response = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system="You are a helpful assistant answering questions about War and Peace by Leo Tolstoy.",
        messages=history
    )

    reply = response.content[0].text
    history.append({"role": "assistant", "content": reply})

    print(f"\nClaude: {reply}")
    print("-" * 50)