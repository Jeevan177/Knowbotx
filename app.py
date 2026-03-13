import streamlit as st
import os
from dotenv import load_dotenv

# Text splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Document loaders
from langchain_community.document_loaders import PDFPlumberLoader

# Tools
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Vector DB
from langchain_community.vectorstores import FAISS

# Prompt templates
from langchain_core.prompts import ChatPromptTemplate

# Schema
from langchain_core.documents import Document

# LangGraph
from langgraph.graph import END, START, StateGraph

# LLM
from langchain_groq import ChatGroq

# Typing
from typing import Literal, List
from typing_extensions import TypedDict

# Pydantic
from pydantic import BaseModel, Field
load_dotenv()

groq_api_key = os.environ.get("GROQ_API_KEY")

if groq_api_key is None:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.3-70b-versatile")


class RouteQuery(BaseModel):
  """Route a user query to the most relevant datasources."""

  datasources: Literal["vectorstores", "wiki_search", "llm_fallback"] = Field(
      ...,
      description="Given a user question choose it to wikipedia, a vectorstore, or llm_fallback."
  )

structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt
system = """
You are an AI that retrieves and answers user questions with high accuracy.

- If the question is related to the uploaded PDF, return the answer from the document without summarization or modifications.
- If the question is outside the PDF’s content, search Wikipedia and provide a **precise summary**.
- If the answer is not found in either the PDF or Wikipedia, respond with the answer from the llm itself.

Follow these rules strictly to ensure precision.
"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
question_router = route_prompt | structured_llm_router

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]

def pdf_loader(uploaded_file):
    if uploaded_file is not None:
      # Step 2: Save the uploaded file temporarily
      temp_path = f"./{uploaded_file.name}"  
      with open(temp_path, "wb") as f:
          f.write(uploaded_file.read())

      loader = PDFPlumberLoader(temp_path)

      docs = loader.load()
      text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=1000,
          chunk_overlap=100)
      all_text = "".join([doc.page_content for doc in docs])
      docs_split = text_splitter.split_text(all_text)
    return docs_split

def get_vector_store(docs_split):
   
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

    docs_split = [Document(page_content=text_chunk) for text_chunk in docs_split]

    vector_store = FAISS.from_documents(docs_split, embedding = embeddings)
    return vector_store

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    # print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = st.session_state["vector_store"].similarity_search(question, k=5)
    
    retrieved_texts = "\n\n".join([doc.page_content for doc in documents])

    if not retrieved_texts.strip():
        return {"documents": "I couldn't find relevant information in the uploaded PDF.", "question": question}

    llm_response = llm.invoke(f"Based on the following PDF content, answer the question:\n\n{retrieved_texts}\n\nQuestion: {question}")
    return {"documents": llm_response.content, "question": question}

def wiki_search(state):
    wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=1)
    wiki = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

    # print("---WIKIPEDIA SEARCH STARTED---")
    question = state["question"]

    docs = wiki.invoke({"query": question})

    if isinstance(docs, dict):
        wikipedia_content = docs.get("summary", "").strip()
    elif isinstance(docs, str):
        wikipedia_content = docs #or handle it in a different way.
    else:
        return {"documents": "Error: Unexpected Wikipedia response.", "question": question}

    if not wikipedia_content:
        return {"documents": "I couldn't find any relevant information on Wikipedia.", "question": question}

    
    wikipedia_content = docs.replace("summary", "").strip()  # Ensure we handle empty responses

    # If Wikipedia doesn't return useful info, avoid sending an empty prompt to LLM
    if not wikipedia_content:
        return {"documents": "I couldn't find any relevant information.", "question": question}

    # Improved prompt for the LLM
    prompt = f"""
    You are an AI assistant that provides answers based on reliable information.
    
    - The user's question: "{question}"
    - The following information was retrieved from Wikipedia:
    
    "{wikipedia_content}"
    
    Based on this, provide a clear and concise answer.
    """

    # Generate response using LLM
    llm_response = llm.invoke(prompt)

    # Ensure the correct return format for LangGraph
    return {"documents": llm_response.content, "question": question}

def llm_fallback(state):
    question = state["question"]
    llm_response = llm.invoke(question)
    return {"documents": llm_response.content, "question": question}


def route_question(state):
    """
    Route question to wiki search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    # print("---ROUTE QUESTION---")
    question = state["question"]

    source = question_router.invoke({"question": question})

    if source.datasources =='wiki_search':
        # print("---ROUTE QUESTION TO Wiki SEARCH---")
        return "wiki_search"
    elif source.datasources == "vectorstores":
        # print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    else:
        return "llm_fallback"

def graph():
    workflow = StateGraph(GraphState)
    
    # Define the nodes
    workflow.add_node("wiki_search", wiki_search)  # web search
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("llm_fallback", llm_fallback)  # LLM Fallback ✅

    # Build graph
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "wiki_search": "wiki_search",
            "vectorstore": "retrieve",
            "llm_fallback": "llm_fallback",  # ✅ Added this to avoid KeyError
        },
    )
    
    workflow.add_edge("retrieve", END)
    workflow.add_edge("wiki_search", END)
    workflow.add_edge("llm_fallback", END)  # ✅ Add this to properly route LLM fallback

    # Compile
    app = workflow.compile()
    return app


def user_input(user_question, app):

    response = app.invoke({"question": "user_question"})
    # print(response['documents'])
    st.write("Reply: ", response["documents"])

def main():
    st.set_page_config("Chat PDF")
    st.header("KnowBotX") #KnowBotX – (Knowledge Bot + AI + Docs)

    # Initialize chat history if not present
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []  

    user_question = st.chat_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = []
                for pdf in pdf_docs:
                    raw_text.extend(pdf_loader(pdf))  # Process each uploaded PDF
                
                store = get_vector_store(raw_text)  # ✅ Get FAISS in memory
                st.session_state["vector_store"] = store  # Store in session state
                st.success("Processing Complete!")

    # Display previous chat messages
    for chat in st.session_state["chat_history"]:
        with st.chat_message(chat["role"]):
            st.write(chat["message"])

    if user_question:
        if "vector_store" in st.session_state:
            app = graph()  # Initialize the graph before calling `app.invoke()`
            response = app.invoke({"question": user_question})
            answer = response["documents"].replace('<think>', '').replace('</think>', '')

            # Save user query
            st.session_state["chat_history"].append({"role": "user", "message": user_question})
            # Save AI response
            st.session_state["chat_history"].append({"role": "assistant", "message": answer})

            # Display response in chat format
            with st.chat_message("user"):
                st.write(user_question)

            with st.chat_message("assistant"):
                st.write(answer)

        # else:
        #     st.warning("Please upload and process a PDF first.")

if __name__ == "__main__":
    main()
