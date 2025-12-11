# test_rag.py
import sys
sys.path.append('.')  # Προσθήκη του current directory

from core.integrations.rag_adapter import rag_search

try:
    result = rag_search('test query', 'general', 3)
    print('RAG Search Result:', result)
    print('\nType:', type(result))
    print('Keys:', result.keys() if isinstance(result, dict) else 'Not a dict')
except Exception as e:
    print('Error:', e)
    import traceback
    traceback.print_exc()