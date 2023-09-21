import os
import openai
import streamlit as st
import tiktoken
from itertools import islice
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient


# Set the service endpoint and API key from the enivironment

load_dotenv()

service_name = os.getenv("MY_SEARCH_SERVICE")
admin_key = os.getenv("SEARCH_SERVICE_ADMIN_KEY")
engine_name = "gpt-35-turbo"

index_name = os.getenv("INDEX_NAME")
openai.api_type = "azure"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_version = "2023-07-01-preview"
openai.api_base = os.getenv("OPENAI_API_BASE")

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def remove_system_messages(context, num_to_remove=3):
    count_removed = 0
    first_hit_skipped = False

    # Create a new list to store the messages we want to keep
    new_context = []

    for message in context:
        if message['role'] == 'system':
            if not first_hit_skipped:
                # Skip the first 'system' message
                first_hit_skipped = True
                new_context.append(message)
                continue
            if count_removed < num_to_remove:
                # Remove this 'system' message
                count_removed += 1
                continue
        # Add the message to the new context
        new_context.append(message)
        
    return new_context

# Create an SDK client
endpoint = f"https://{service_name}.search.windows.net/"
search_client = SearchClient(endpoint=endpoint,
                             index_name=index_name,
                             credential=AzureKeyCredential(admin_key))

# First
st.title("💬 AILA - A.I. Assistant") 
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role":"system","content":"You are an AI assistant that helps people find information."},{"role": "assistant", "content": "Do you have a question about any of the transcripts in the database?"}]

for msg in st.session_state.messages:
    if msg["role"] != "system":
        st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)

    if num_tokens_from_messages(st.session_state.messages) > 10000:
        print("Warning: num tokens from messages is greater than 10000. This may cause issues with the GPT-3 model.")
        st.session_state.messages = remove_system_messages(st.session_state.messages, num_to_remove=4)
    print(num_tokens_from_messages(st.session_state.messages))

    query_results = search_client.search(query_type='semantic',
                               query_language='en-us',
                               semantic_configuration_name='semantictestconfig',
                               search_text=prompt,
                               select='content,people,locations,metadata_storage_name',
                               query_caption='extractive',
                               query_answer='extractive',
                               include_total_count=True,)
    
    top_three = islice(query_results, 3)

    for doc in top_three:
        result_list = doc["content"]
        result_list = result_list.replace("\n", " ")
        st.session_state.messages.append({"role": "system", "content": f'{doc["metadata_storage_name"]}\n{result_list}'})


    st.session_state.messages.append({"role": "system", "content": "Only answer questions based on the transcripts above. You may infer answers from the transcript but be clear that there is no direct answer. If the answer is not found in the transcript at all, answer 'I don't know'. If you need to reference a transcript refer to it by its filename"})
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = openai.ChatCompletion.create(engine=engine_name, messages=st.session_state.messages)
    msg = response.choices[0].message
    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg.content)




def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
