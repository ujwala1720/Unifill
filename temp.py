
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = "paste api key here"
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

urls =[""]
loaders = UnstructuredURLLoader(urls)
data = loaders.load()
text_splitter = CharacterTextSplitter(separator='\n',chunk_size=1000,chunk_overlap=200)
docs = text_splitter.split_documents(data)
embeddings = OpenAIEmbeddings()
vectorstore_OpenAI = Pinecone.from_documents(docs,embeddings)
#vectorstore_OpenAI.save_local("Pinecone")
#Assign this to a variable
PROMPT=PromptTemplate(template=prompt.prompt_template, input_variables=["context", "question"])
X =Pinecone.load_local("Pinecone", OpenAIEmbeddings())
llm=OpenAI(temperature=0, model_name='text-davinci-003')
chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=kal.as_retriever())
ans=chain({"question": "your question"}, return_only_outputs=True)
print(ans["ans"])









