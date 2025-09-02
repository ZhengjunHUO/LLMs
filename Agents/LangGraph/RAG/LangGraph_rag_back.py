from fastapi import FastAPI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field
from typing import List
from typing_extensions import TypedDict

# Build vectorDB & retriever
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
raw_docs = [WebBaseLoader(url).load() for url in urls]
docs = [doc for raw_doc in raw_docs for doc in raw_doc]

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=200, chunk_overlap=0)
doc_chunks = splitter.split_documents(docs)

vec_db = Chroma.from_documents(
    documents=doc_chunks,
    collection_name="rag",
    embedding=OllamaEmbeddings(model="qwen2.5-coder:14b")
)
retriever = vec_db.as_retriever()


# Prepare LLM 
llm = ChatOllama(model="qwen2.5-coder:14b", temperature=0)

class CheckDocs(BaseModel):
    is_relevant: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
llm_structured = llm.with_structured_output(CheckDocs)

# Prepare chains
system_instruction = """Return only a 'yes' or 'no' depends on whether the document is relevant to the question"""
eval_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_instruction),
        ("human", "Document: \n {doc} \n\nQuestion: {question}"),
    ]
)
retriv_eval_chain = eval_prompt | llm_structured

gen_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "You are an assistant for question-answering tasks. If the context is not provided try to figure out the answer yourself, otherwise use the following pieces of retrieved context to answer the question but don't mention the context in the answer. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question}\nContext: {context}\nAnswer:"),
    ]
)
gen_resp_chain = gen_prompt | llm | StrOutputParser()

reform_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a question re-writer that converts an input question to a better version that is optimized for web search. Look at the input and try to reason about the underlying semantic meaning."""),
        ("human", "This is the original question: {question} \n Try to rewrite a better one"),
    ]
)
reform_question_chain = reform_prompt | llm | StrOutputParser()


# Build workflow
class State(TypedDict):
    docs: List[str]
    question: str
    retry: int
    gen_without_context: str
    resp: str

def init_state(state):
    print("*** INIT STATE ***")

    return {"retry": 3}

def retrv_docs(state):
    print("*** RETRIEVE DOCS ***")
    print("  current state: ", state)

    docs = retriever.invoke(state["question"])
    return {"docs": docs}

def eval_docs(state):
    print("*** CHECK DOCS RELEVANCE ***")
    related_docs = []
    gen_without_context = "FALSE"

    for doc in state["docs"]:
        eval_rslt = retriv_eval_chain.invoke({"doc": doc.page_content, "question": state["question"]})
        print("  ==> ", doc.metadata['source'], doc.page_content[:80], " is relevant: ", eval_rslt.is_relevant)
        if eval_rslt.is_relevant == "yes":
            related_docs.append(doc)
    if len(related_docs) == 0:
        print("  !!! NO RELATED DOCS FOUND !!!")
        gen_without_context = "TRUE"

    return {"docs": related_docs, "gen_without_context": gen_without_context}

def reform_question(state):
    print("*** REFORM QUESTION ***")
    print("  ORIGINAL QUESTION: ", state["question"])

    reformed_question = reform_question_chain.invoke({"question": state["question"]})
    print("  REFORMED QUESTION: ", reformed_question)
    return {"question": reformed_question, "retry": state["retry"] - 1}

def gen_resp(state):
    print("*** GENERATE RESPONSE ***")

    resp = gen_resp_chain.invoke({"context": state["docs"], "question": state["question"]})
    return {"resp": resp}

def retry_or_gen(state):
    print("*** MAKING DECISION ***")
    if state["gen_without_context"] == "FALSE":
        print("  WILL GENERATE RESPONSE")
        return "do_gen_resp"

    if state["retry"] == 0:
        print("  UNABLE TO RETRIEVE DOC, WILL GENERATE RESPONSE")
        return "do_gen_resp"

    print("  TRY TO REFORM THE QUESTION AND RETRIEVE AGAIN")
    return "do_reform_question"

graph = StateGraph(State)
graph.add_node("init_state", init_state)
graph.add_node("retrv_docs", retrv_docs)
graph.add_node("eval_docs", eval_docs)
graph.add_node("reform_question", reform_question)
graph.add_node("gen_resp", gen_resp)

graph.add_edge(START, "init_state")
graph.add_edge("init_state", "retrv_docs")
graph.add_edge("retrv_docs", "eval_docs")
graph.add_conditional_edges(
    "eval_docs",
    retry_or_gen,
    {
        "do_gen_resp": "gen_resp",
        "do_reform_question": "reform_question",
    }
)
graph.add_edge("reform_question", "retrv_docs")
graph.add_edge("gen_resp", END)

workflow = graph.compile()

# Launch API
app = FastAPI()

class Req(BaseModel):
    question: str

@app.post("/answer")
async def answer_question(request: Req):
    user_input = {"question": request.question}
    for stdout in workflow.stream(user_input):
        for key, value in stdout.items():
            print("Node ", key)

    return {"resp": value["resp"]}