#imports
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = "E:/LLMS/vectorstores/db_faiss"


cstm_prmt_template = """ Use the following info to answer the user's question. If you dont know the answer, just say that you dont know, dont make anything up.

Context: {context}
Question: {question}

Only returns the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector stores
    """
    prompt = PromptTemplate(template=cstm_prmt_template, input_variables=['context','question'])
    return prompt

def load_llm():
    llm = CTransformers(
        model = 'E:/LLMS/llama-2-7b-chat.Q8_0.gguf',
        model_type = "llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= "stuff",
        retriever = db.as_retriever(search_kwargs={'k':2}),
        return_source_documents = True,
        chain_type_kwargs= {'prompt': prompt}
    )
    return qa_chain

def bot_mcbots():
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs = {'device':'cpu'} )

    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm,qa_prompt,db )
    
    return qa

def final_answer(query):
    qa_result = bot_mcbots()
    response = qa_result({'query':query})
    return response

##chainlit stuffs

@cl.on_chat_start
async def start():
    chain = bot_mcbots()
    msg = cl.Message(content="Booting Baymax....")
    await msg.send()
    msg.content = "Hello, I am Baymax, your personal healthcare companion.  Do you have a question regarding your health?"
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=False, answer_prefix_tokens = ["FINAL", "ANSWER"] 
    )
    cb.answer_reached=True
    res = await chain.acall(message, callbacks=[cb])
    answer = res.get("result")
    sources = res.get("sources_documents")

    if sources:
        answer += f"\nSources:"+ str(sources)
    else:
        answer += f"\nNo Sources Found"
    await cl.Message(content=answer).send()