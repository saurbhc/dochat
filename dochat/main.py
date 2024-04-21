import argparse


import os
from glob import iglob

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.base.base_query_engine import BaseQueryEngine


def get_documents(files: str, verbose: bool = False) -> list[dict]:
    if verbose:
        print(f"Loading data from {files}...")

    documents = []
    for _file in iglob(files, recursive=True):
        _file = os.path.abspath(_file)
        if os.path.isdir(_file):
            reader = SimpleDirectoryReader(
                input_dir=_file,
                filename_as_id=True,
            )
        else:
            reader = SimpleDirectoryReader(
                input_files=[_file],
                filename_as_id=True,
            )

        documents.extend(reader.load_data(show_progress=verbose))

    return documents


def get_embed_model(verbose: bool = False) -> HuggingFaceEmbedding:
    if verbose:
        print("Loading HuggingFace embedding model...")
    return HuggingFaceEmbedding(
        model_name="BAAI/bge-large-en-v1.5",
        trust_remote_code=True,
    )


def get_index(
    documents: list[dict],
    embed_model: HuggingFaceEmbedding,
    verbose: bool = False,
) -> VectorStoreIndex:
    Settings.embed_model = embed_model
    return VectorStoreIndex.from_documents(documents, show_progress=verbose)


def get_llm(
    model: str = "llama3:8b",
    request_timeout: float = 120.0,
    verbose: bool = False,
) -> Ollama:
    if verbose:
        print("Loading Ollama LLM model...")
    return Ollama(model=model, request_timeout=request_timeout)


def get_prompt_template(verbose: bool = False) -> PromptTemplate:
    if verbose:
        print("Creating prompt template...")
    qa_prompt_tmpl_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    return PromptTemplate(qa_prompt_tmpl_str)


def get_query_engine(
    llm: Ollama,
    index: VectorStoreIndex,
    prompt_template: PromptTemplate,
    streaming: bool = True,
    verbose: bool = False,
) -> BaseQueryEngine:
    if verbose:
        print("Creating query engine...")

    # Create the query engine, where we use a cohere reranker on the fetched nodes
    Settings.llm = llm
    query_engine = index.as_query_engine(streaming=streaming)
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": prompt_template},
    )
    return query_engine


def main() -> int:
    """
    Main entry point for dochat CLI

    Usage:
        dochat <command> [<args>...]
        dochat -h | --help
        dochat <files>
    """
    parser = argparse.ArgumentParser(description="dochat CLI")
    parser.add_argument("files", type=str, help="Path to the file/folder to index")
    parser.add_argument("prompt", type=str, help="Prompt to use for querying")
    parser.add_argument("-v", "--verbose", action="store_true", default=True)
    args = parser.parse_args()

    print(f"Indexing files from {args.files}...")
    documents = get_documents(files=args.files, verbose=args.verbose)
    embed_model = get_embed_model(verbose=args.verbose)
    index = get_index(
        documents=documents,
        embed_model=embed_model,
        verbose=args.verbose,
    )
    llm = get_llm(verbose=args.verbose)
    prompt_template = get_prompt_template(verbose=args.verbose)
    query_engine = get_query_engine(
        llm=llm,
        index=index,
        prompt_template=prompt_template,
        verbose=args.verbose,
    )

    print(f"Indexing complete! Querying with '{args.prompt}'")
    streaming_response = query_engine.query(args.prompt)
    streaming_response.print_response_stream()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
