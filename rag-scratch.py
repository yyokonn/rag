# Ref: https://github.com/nishanthegde/rag-from-scratch/tree/main
import chromadb
import openai
import os

from openai import OpenAI

openai.api_key = os.environ["OPENAI_API_KEY"]

DATA_PATH = "data/"
# Global variable to maintain mapping
ID_TO_CHUNK_MAPPING = {}


# Pre-process text to store
def get_text(fname: str) -> str:
    """
    Read fname into a vectorstore
    """
    with open(fname, "r") as f:
        text = f.read()
    return text


def chunk_text(text: str, chunk_size: int, chunk_overlap=50) -> list:
    """
    Split text into n chunks of length chunk-size number of characters
    """
    text = text.replace("\n", " ")
    return [
        text[i : min(len(text), i + chunk_size)]
        for i in range(0, len(text), chunk_size - chunk_overlap)
    ]


def get_embeddings(text_list: list, model="text-embedding-ada-002"):
    """
    Embed each text in text_list using model from openai API.
    """
    # client = OpenAI()
    Embedding_list = openai.embeddings.create(input=text_list[:2], model=model).data
    embeddings = [Embedding_list[i].embedding for i in range(len(Embedding_list))]
    return embeddings


def update_ID_TO_CHUNK_MAPPING(ids: list, text_list: list) -> None:
    """
    Update ID_TO_CHUNK_MAPPING with new ids and corresponding texts in text_list for context retrieval.
    """
    ID_TO_CHUNK_MAPPING.update({id: chunk for id, chunk in list(zip(ids, text_list))})


def get_context(
    knowledge, query: str, n_results: int, filter={"source": "input.txt"}
) -> list:
    """
    Retrieve n_results number of relevant chunks of text from knowledge using query. 
    Filter the text using the filter condition.
    """
    results = knowledge.query(
        query_embeddings=get_embeddings(query), n_results=n_results, where=filter
    )

    context = [ID_TO_CHUNK_MAPPING[id] for id in results["ids"][0]]
    return context


def store_embeddings(text_list: list):
    """
    Create embeddings for text_list using get_embeddings().
    Store the embeddings in a Chroma database with unique IDs.
    Update ID_TO_CHUNK_MAPPING with IDs and corresponding text chunks.
    """
    embeddings = get_embeddings(text_list)

    client = chromadb.Client()
    # get a collection or create if it doesn't exist already
    collection = client.get_or_create_collection("knowledge")

    num_records = len(embeddings)
    ids = [f"id{i}" for i in range(num_records)]
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=[{"source": "input.txt"}] * num_records,
    )

    update_ID_TO_CHUNK_MAPPING(ids, text_list)

    return collection


def construct_prompt(question: str, context: list[str]) -> str:
    """
    Format the prompt with more instructions.
    Insert the question and context into the final prompt.
    """
    formatted_context = "\n".join(context)

    prompt = f"Use the following context to answer the question.\n\
               If you don't know the answer, just say that you don't know.\n\
               Context: {formatted_context}\n\
               Question: {question}\n\
               Answer: "
    return prompt


def ask_gpt(prompt: str):
    """
    Ask GPT 3.5-turbo the prompt. 
    Returns the model's response.
    """
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant for question-answering tasks.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    assistant_msg = response.choices[0].message.content

    return assistant_msg


def main():
    fname = DATA_PATH + "input.txt"
    text = get_text(fname)
    text_list = chunk_text(text, chunk_size=500, chunk_overlap=50)

    knowledge = store_embeddings(text_list)

    # Chatbot that can answer multiple questions, until the user quits.
    print(
        "Ask a question, e.g. Why didn't the writer want to work on AI in the mid 1980's. \
        If you want to quit the conversation, say 'quit.'"
    )
    while True:
        query = input("User: ")
        if query == "quit":
            break
        context = get_context(knowledge, query, n_results=5)
        prompt = construct_prompt(query, context)
        response = ask_gpt(prompt)
        print(f"Response: {response}")


if __name__ == "__main__":
    main()
