# clean_chroma.py
import chromadb
from chromadb.config import Settings

# Σύνδεση στη ChromaDB
client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

# Διαγραφή της συλλογής
try:
    client.delete_collection("general")
    print("✅ Deleted 'general' collection")
except:
    print("ℹ️ 'general' collection didn't exist")

try:
    client.delete_collection("invoices")
    print("✅ Deleted 'invoices' collection")
except:
    print("ℹ️ 'invoices' collection didn't exist")

print("\nChromaDB cleaned. Now re-upload your files.")