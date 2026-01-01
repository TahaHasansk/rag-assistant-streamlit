from langchain_community.llms import Ollama

def get_llm():
    return Ollama(
        model="llama3",
        base_url="http://ollama:11434"
    )
