import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document



load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

llm=ChatOpenAI(model="gpt-4o")
# print(llm)

loader = WebBaseLoader("https://docs.smith.langchain.com/administration/tutorials/manage_spend");
docs = loader.load();
# print(docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()

vectorstoredb = FAISS.from_documents(documents, embeddings)

query="LangSmith has two usage limits: total traces and extended"
result = vectorstoredb.similarity_search(query, k=1);
# print("Som >>>> " + result[0].page_content)

prompt = ChatPromptTemplate.from_template(
    """

Answer the following question  based only on the provided context : 
<context>
{context}
</context>
"""
)
document_chain = create_stuff_documents_chain(llm, prompt=prompt)

# print(document_chain)

# p = document_chain.invoke({
#                        "input" : "LangSmith has two usage limits: total traces and extended",
#                        "context" : [Document(page_content="LangSmith has two usage limits: total traces and extended retention traces. These correspond to the two metrics we've been tracking on our usage graph.")]

#                        })

# print(p)

retriever = vectorstoredb.as_retriever()
retrieval_chain = create_retrieval_chain(retriever,document_chain)
response = retrieval_chain.invoke({"input" : "LangSmith has two usage limits: total traces and extended"})
print("ANSWER >>>> ")
print(response['answer'])

print("RESPONSE >>>> ")
print((response['context']))
# # print("RESPONSE >>>> ", "\n\n".join([doc.page_content for doc in response['context']]))
