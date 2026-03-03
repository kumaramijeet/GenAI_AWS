#1 import the required modules
from langchain_aws import ChatBedrockConverse
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationSummaryBufferMemory

#2a write a function for invoking model-client connection with bedrock
def demo_chatbot():
    demo_llm = ChatBedrockConverse(
        credentials_profile_name='default',
        model="amazon.nova-pro-v1:0",
        temperature=0.1,
        max_tokens=1000
    )
    return demo_llm

#3 create a function for conversation buffer memory (llm and max token limit)
def demo_memory():
    llm_data = demo_chatbot()
    memory = ConversationSummaryBufferMemory(llm=llm_data, max_token_limit=2000)
    return memory


#4 create a function for conversation chain - input text + memory
def demo_conversation(input_text, memory):
    llm_chain_data = demo_chatbot()
    llm_conversation = ConversationChain(
        llm = llm_chain_data,
        memory = memory,
        verbose = True
    )
    #5 chat response using invoke
    chat_reply = llm_conversation.invoke(input_text)
    return chat_reply['response']
    
