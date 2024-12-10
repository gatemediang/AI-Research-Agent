from typing import Tuple, Dict, Any
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.gemini import Gemini
from llama_index.core.tools import BaseTool, FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.prompts import PromptTemplate
from llama_parse import LlamaParse
from llama_index.core.agent import ReActAgent
from sklearn.metrics.pairwise import cosine_similarity
from llama_index.core.agent import FnAgentWorker
from constants import *
from prompt1 import *
from llama_index.readers.wikipedia import WikipediaReader
import wikipedia


parser = LlamaParse(api_key=parse_key, result_type="markdown")

extractor = {".pdf": parser}
# set the default tim as gemini
Settings.llm = Gemini(model="models/gemini-1.5-flash", api_key=API_KEY)
gemini_embedding = GeminiEmbedding(model="models/gemini-1.5-flash", api_key=API_KEY)
Settings.embed_model = gemini_embedding


docs = SimpleDirectoryReader("docs", file_extractor=extractor).load_data()


# First verify the exact page titles
def get_wikipedia_pages(search_terms):
    pages = []
    for term in search_terms:
        page = wikipedia.page(term, auto_suggest=False)
        print(f"online search connected")
        pages.append(page.title)
    return pages


# Get verified page titles
search_terms = [
    "Artificial Intelligence",
    "Large Language Model",
    "Natural Language Processing",
]

verified_pages = get_wikipedia_pages(search_terms)
# print("\nVerified pages:", verified_pages)

# Then use the verified titles with WikipediaReader
wiki_docs = WikipediaReader().load_data(pages=verified_pages)

# Combine documents and create index
all_docs = docs + wiki_docs
index = VectorStoreIndex.from_documents(all_docs)
query_engine = index.as_query_engine()


index = VectorStoreIndex.from_documents(docs)

agent_tools = [
    QueryEngineTool(
        query_engine,
        metadata=ToolMetadata(
            name="knowledge_base",
            description="Use this tool to get information about Generative AI, Machine Learning and research topics on AI from Wikipedia and docs folder",
        ),
    )
]
agent = ReActAgent.from_tools(tools=agent_tools, verbose=True)

result = agent.query("Explain in one sentence what is Large Language Model?")
print(result)
