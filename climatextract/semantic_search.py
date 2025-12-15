"""Extract raw text from PDF files, store it in DuckDB,
and provide basic search functionality (vector search, full text search)."""

import asyncio
from dataclasses import dataclass, field
import datetime
from hashlib import sha256
import os.path
import json
from typing import Any, Dict, List, Optional
import tiktoken

import pandas as pd
import numpy as np

from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
import duckdb

from climatextract.config import EmbeddingModel


class Page:
    """
    Class for storing a (page number, page content) from a pdf page, 
    along with information how the page was found.

    Most important fields:
    - page content: text of the page
    - page label: string page number (usually "1", "2", "3", ..., but this is not guaranteed)
    - page index: numeric page number (0, 1, 2, ...)
    - vector_similarity: cosine similarity from semantic search

    - complete_file_path: path to the PDF file
    - short_file_name: name of the PDF file

    Other fields are related to how a page has been found.
    """

    def __init__(self, data: Dict[str, Any]):

        self.page_index = data["page_index"]
        self.page_label = data["page_label"]
        self.page_content = data["page_content"]
        # cosine similarity from vector search
        self.vector_similarity = data["vector_similarity"]

        self.complete_file_path = data["complete_file_path"]
        self.short_file_name = data["short_file_name"]

        # "full_text_search" or "vector_search"
        self.search_method = data["search_method"]
        # bm25 similarity from full text search
        self.bm25_similarity = data["bm25_similarity"]
        # boolean, whether the page is adjacent (i.e., found via the context_window)
        # to one of the original search results
        self.is_adjacent = data["is_adjacent"]
        # timestamp when page embedding was stored in database
        self.page_embedding_created_at = data["page_embedding_created_at"]
        # timestamp when query embedding was stored in database
        self.query_embedding_created_at = data["query_embedding_created_at"]


class EmbeddingsRepository:
    """Class to manage the DuckDB database for storing embeddings and related data."""

    def __init__(self, database_name: str):
        self.database_name = database_name
        self.database_existent = False

        if self.database_exists():
            print(f"""Embedding database with name '{self.database_name}' already exists.
                  It is your resposibility to ensure newly added embeddings are created
                  in the same way as the existing ones.""")

    def database_exists(self) -> bool:
        """Checks if a file named 'database_name' already exists."""

        if self.database_existent:
            return self.database_existent

        self.database_existent = os.path.isfile(self.database_name)
        return self.database_existent

    def get_database_name(self) -> str:
        """Returns the file path/name of the corresponding .duckDb database 
        (which may or may not exist)."""
        return self.database_name

    def create_database(self, embed_dim: int) -> None:
        """Create database schema (tables, indexes) if needed."""

        database_path = "./data/processed/embeddings"
        if not os.path.exists(database_path):
            os.makedirs(database_path)

        with duckdb.connect(self.database_name) as con:
            # Create the tables only if they don't exist
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS pages(
                    id STRING,
                    short_file_name STRING,
                    page_index INTEGER,
                    page_label STRING,
                    page_content STRING,
                    embedding FLOAT[{embed_dim}],
                    embed_created_by STRING,
                    embed_created_at STRING
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS pdf_files(
                    short_file_name STRING,
                    complete_file_path STRING,
                    number_of_pages INTEGER
                )
            """)
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS search_query_embeddings(
                    search_query STRING,
                    embedding FLOAT[{embed_dim}],
                    embed_created_by STRING,
                    embed_created_at STRING
                )
                """)
            # FTS extension
            con.execute("INSTALL fts")
            con.execute("LOAD fts")

    def insert_search_query_record(self,
                                   search_query: str,
                                   embedding: List[float],
                                   embed_created_by: str,
                                   embed_created_at: str) -> None:
        """Insert a single record into search_embeddings table."""

        if not self.search_query_exists(search_query):
            with duckdb.connect(self.database_name) as con:
                con.execute(
                    "INSERT INTO search_query_embeddings VALUES (?, ?, ?, ?)",
                    [search_query, embedding, embed_created_by, embed_created_at]
                )

    def remove_search_query_record(self, search_query: str, embed_created_at: str) -> None:
        """Remove search_query from database."""
        with duckdb.connect(self.database_name) as con:
            con.execute("""DELETE FROM search_query_embeddings
                            WHERE search_query = ?
                            AND embed_created_at = ?""", [search_query, embed_created_at])

    def search_query_exists(self, search_query: str) -> bool:
        """Check if search_query already exists in database."""
        with duckdb.connect(self.database_name) as con:
            con.execute(
                "SELECT COUNT(*) FROM search_query_embeddings WHERE search_query = ?",
                [search_query]
            )
            search_query_in_database = bool(con.fetchall()[0][0])
            return search_query_in_database

    def get_search_query_embedding(self, search_query: str) -> List[float]:
        """Retrieve the embedding for a given search_query."""
        with duckdb.connect(self.database_name) as con:
            res = con.execute(
                "SELECT embedding FROM search_query_embeddings WHERE search_query = ?",
                [search_query]
            ).fetch_df()
            return res["embedding"].tolist()

    def get_search_query_embedding_creation_date(self, search_query: str) -> str:
        """Retrieve the creation date for a given search_query embedding."""
        with duckdb.connect(self.database_name) as con:
            res = con.execute(
                "SELECT embed_created_at FROM search_query_embeddings WHERE search_query = ?",
                [search_query]
            ).fetch_df()
            return res["embed_created_at"].tolist()

    def pdf_exists(self, short_file_name: str) -> bool:
        """Check if a PDF file (by short_file_name) already exists in database."""
        with duckdb.connect(self.database_name) as con:
            con.execute(
                "SELECT COUNT(*) FROM pdf_files WHERE short_file_name = ?",
                [short_file_name]
            )
            pdf_in_database = bool(con.fetchall()[0][0])
            return pdf_in_database

    def persist_pdf_with_pages_in_database(self, pages_nodes, embeddings,
                                           embed_created_by: str, embed_created_at: str) -> None:
        """Inserts records for all pages of a PDF file and summary information 
        about the file into the database."""

        file_path = pages_nodes[0].metadata["file_name"]
        _, tail = os.path.split(file_path)
        number_of_pages = len(pages_nodes)

        page_embeddings = [(page, embedding)
                           for page, embedding in zip(pages_nodes, embeddings)]

        self._insert_pdf_record(tail, file_path, number_of_pages)

        for index, (document, embedding) in enumerate(page_embeddings):

            page_label = document.metadata["page_label"]
            page_content = document.get_text()
            doc_identity = str(page_content) + str(document.metadata)
            doc_id = str(sha256(doc_identity.encode(
                "utf-8", "surrogatepass")).hexdigest())

            self._insert_pdf_page_record(
                doc_id, tail, index, page_label, page_content,
                embedding, embed_created_by, embed_created_at)

        self._update_fts_index()

    def remove_pdf(self, short_file_name: str) -> None:
        """Remove short_file_name from database. Update index for full text search."""
        with duckdb.connect(self.database_name) as con:
            con.execute("DELETE FROM pdf_files WHERE short_file_name = ?", [
                        short_file_name])
            con.execute("DELETE FROM pages WHERE short_file_name = ?", [
                        short_file_name])
            con.execute(
                """PRAGMA create_fts_index('pages', 'id', 'page_content', overwrite=1);""")

    def _insert_pdf_record(self, short_file_name: str,
                           complete_file_path: str, number_of_pages: int) -> None:
        """Insert a single record into pdf_files table."""
        with duckdb.connect(self.database_name) as con:
            con.execute(
                "INSERT INTO pdf_files VALUES (?, ?, ?)",
                [short_file_name, complete_file_path, number_of_pages]
            )

    def _insert_pdf_page_record(self,
                                page_id: str,
                                short_file_name: str,
                                page_index: int,
                                page_label: str,
                                page_content: str,
                                embedding: List[float],
                                embed_created_by: str,
                                embed_created_at: str
                                ) -> None:
        """Insert a single record into the pages table."""
        with duckdb.connect(self.database_name) as con:
            con.execute(
                "INSERT INTO pages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [page_id, short_file_name, page_index, page_label,
                    page_content, embedding, embed_created_by, embed_created_at]
            )

    def _update_fts_index(self) -> None:
        """Update Full Text Search index after each insert if you want it always fresh"""
        with duckdb.connect(self.database_name) as con:
            con.execute(
                """PRAGMA create_fts_index('pages', 'id', 'page_content', overwrite=1);""")

    def pdf_vector_search(self,
                          short_file_name: str,
                          search_vector: List[float],
                          limit: Optional[int],
                          similarity: str,
                          embed_dim: int
                          ) -> pd.DataFrame:
        """
        Perform a vector-based search inside "short_file_name", 
        using either array_cosine_similarity (descending)
        or array_distance (ascending).
        """
        with duckdb.connect(self.database_name) as con:
            if similarity == "cosine":
                base_query = f"""
                    FROM pages
                    LEFT JOIN pdf_files
                        ON (pdf_files.short_file_name = pages.short_file_name)
                    SELECT
                        pdf_files.complete_file_path,
                        page_index,
                        page_label,
                        page_content,
                        array_cosine_similarity(embedding, $searchVector::FLOAT[{embed_dim}]) AS similarity
                    WHERE pages.short_file_name = ($file_name)
                    ORDER BY similarity DESC
                """
                if limit:
                    base_query += " LIMIT ($limit)"
                params = {
                    "searchVector": search_vector,
                    "file_name": short_file_name
                }
                if limit:
                    params["limit"] = limit
                return con.execute(base_query, params).fetch_df()
            elif similarity == "euclidean":
                base_query = f"""
                    FROM pages
                    LEFT JOIN pdf_files
                        ON (pdf_files.short_file_name = pages.short_file_name)
                    SELECT
                        pdf_files.complete_file_path,
                        page_index,
                        page_label,
                        page_content,
                        array_distance(embedding, $searchVector::FLOAT[{embed_dim}]) AS similarity
                    WHERE pages.short_file_name = ($file_name)
                    ORDER BY similarity
                """
                if limit:
                    base_query += " LIMIT ($limit)"
                params = {
                    "searchVector": search_vector,
                    "file_name": short_file_name
                }
                if limit:
                    params["limit"] = limit
                return con.execute(base_query, params).fetch_df()
            else:
                raise ValueError(
                    "Unknown similarity measure: use 'cosine' or 'euclidean'.")

    def pdf_full_text_search(self,
                             short_file_name: str,
                             search_text: str,
                             limit: int
                             ) -> pd.DataFrame:
        """
        Perform a full-text search  inside "short_file_name" using the FTS extension in DuckDB.
        """
        with duckdb.connect(self.database_name) as con:
            return con.execute("""
                FROM pages
                LEFT JOIN pdf_files
                    ON (pdf_files.short_file_name = pages.short_file_name)
                SELECT
                    pdf_files.complete_file_path,
                    page_index,
                    page_label,
                    page_content,
                    fts_main_pages.match_bm25(id, $searchTerm, fields := 'page_content') AS similarity
                WHERE pages.short_file_name = ($file_name)
                    AND similarity IS NOT NULL
                ORDER BY similarity DESC
                LIMIT ($limit)
            """, {
                "searchTerm": search_text,
                "limit": limit,
                "file_name": short_file_name
            }).fetch_df()

    def get_pdf_pages_by_indices(self,
                                 short_file_name: str,
                                 page_index_list: List[int],
                                 page_index_list_subset: List[int],
                                 search_method: str,
                                 search_query: str,
                                 search_vector: List[float],
                                 embed_dim: int,
                                 search_embed_created_at: str) -> pd.DataFrame:
        """
        Retrieve subset of pages (page_index) from short_file_name;
        add search-related info.
        """
        page_indices_str = ', '.join(str(index) for index in page_index_list)
        page_indices_str_subset = ', '.join(
            str(index) for index in page_index_list_subset)

        if page_indices_str == "":
            print("Warning: No pages found. Returning empty dataframe.")
            return pd.DataFrame(columns=["complete_file_path",
                                         "short_file_name",
                                         "page_index",
                                         "page_label",
                                         "page_content",
                                         "search_method",
                                         "vector_similarity",
                                         "bm25_similarity",
                                         "is_adjacent"])
        else:
            with duckdb.connect(self.database_name) as con:
                # Eucledian distance not implmented here
                return con.execute(f"""
                    FROM pages
                    LEFT JOIN pdf_files ON (pdf_files.short_file_name = pages.short_file_name)
                    SELECT
                        pdf_files.complete_file_path,
                        pages.short_file_name,
                        page_index,
                        page_label,
                        page_content,
                        $search_method AS search_method,
                        array_cosine_similarity(embedding, $searchVector::FLOAT[{embed_dim}]) AS vector_similarity,
                        fts_main_pages.match_bm25(id, $searchTerm, fields := 'page_content') as bm25_similarity,
                        pages.page_index NOT IN ({page_indices_str_subset}) AS is_adjacent,
                        pages.embed_created_at AS page_embedding_created_at,
                        $query_embed_created_at AS query_embedding_created_at
                    WHERE pages.short_file_name = ($file_name)
                        AND pages.page_index IN ({page_indices_str})
                """, {
                    "search_method": search_method,
                    "searchVector": search_vector,
                    "searchTerm": search_query,
                    "query_embed_created_at": search_embed_created_at,
                    "file_name": short_file_name
                }).fetch_df()


class SearchQuery:
    """Embed search query with an embedding model and store them in a database

    To initialize a new search query, run 'embed_search_query_and_save_to_database'.
    To retrieve the search query embedding, run 'get_query_embedding'."""

    def __init__(self, search_query: str, repository: EmbeddingsRepository):
        self._search_query = search_query
        self._repository = repository
        self._embedding = []
        self._embed_model_repr = ""
        self._embed_created_at = ""

    def embed_search_query_and_save_to_database(self, embed_model: EmbeddingModel):
        """Embed the search query and save it to the database."""

        if not self._repository.database_exists():
            print(
                f"Database '{self._repository.database_name}' not found. Creating new database.")
            self._repository.create_database(embed_model.get_embed_dimension())

        if self._repository.search_query_exists(self._search_query):
            print(
                (
                    f"Search query '{self._search_query}'"
                    f" already exists in the database. Nothing added.\n"
                ))
            return True
        else:
            print(
                (
                    f"Embed search query '{self._search_query}'"
                    f" with {embed_model}. Persist it in the database."
                ))
            self._embedding = embed_model.get_embeddings(
                [self._search_query])[0]
            self._embed_model_repr = repr(embed_model)
            self._embed_created_at = datetime.datetime.now()
            self._repository.insert_search_query_record(
                self._search_query, self._embedding, self._embed_model_repr, self._embed_created_at)
            return True

    def remove_search_query_from_database(self):
        """Remove search query from database."""
        if self._repository.search_query_exists(self._search_query):
            self._repository.remove_search_query_record(
                self._search_query, self.get_query_embedding_creation_date())
        else:
            print(
                f"Search query '{self._search_query}' not found in database. Nothing removed.\n")

    def get_query_text(self):
        """Return the search query text."""
        return self._search_query

    def get_query_embedding(self):
        """Return the search query embedding."""
        if len(self._embedding) == 0:
            self._embedding = self._repository.get_search_query_embedding(self._search_query)[
                0]

        if len(self._embedding) == 0:
            raise KeyError(
                (
                    f"Search query '{self._search_query}' not found in database."
                    f"Please run 'embed_search_query_and_save_to_database' first."
                ))

        return self._embedding

    def get_query_embedding_creation_date(self):
        """Return the creation date of the search query embedding."""
        if not self._embed_created_at:
            self._embed_created_at = self._repository.get_search_query_embedding_creation_date(
                self._search_query)[0]

        if not self._embed_created_at:
            raise KeyError(
                (
                    f"Search query '{self._search_query}' not found in database."
                    f"Please run 'embed_search_query_and_save_to_database' first."
                ))

        return self._embed_created_at

    def get_embed_dim(self):
        """Return the embedding dimension."""
        return len(self.get_query_embedding())


class Pdfdoc:
    """Class that holds a single pdf file. 
    Raw text and embeddings from each page are actually stored in a database.

    It has three public methods:
    - load_pdf_and_embed_and_save_to_database(embed_model: EmbeddingModel) - 
        Stores the raw text and embeddings in a database, making it available for retrieval.
    - remove_pdf_from_database() - Removes the pdf file from the database.
    - retrieve_relevant_pages() - Searches for the most relevant pages inside this pdf file."""

    def __init__(self, filename: str, repository: EmbeddingsRepository):
        self.filename = filename
        _, self.short_filename = os.path.split(self.filename)
        self._repository = repository

    def retrieve_relevant_pages(self,
                                search_query: SearchQuery,
                                params,
                                return_df: bool = False
                                ) -> pd.DataFrame:
        """Retrieve the most relevant pages from this PDF file.

        Retrieval strategy:
        1. Define search query and (embed model, search method)
        2. Search for the k most similar pages (Pa, ..., Pz)
        3. Add additional pages:
           (Pa-context_window, ..., Pa, ... Pa+context_window,
              ...,
           Pz-context_window, Pz, Pz+context_window)
        4. Return a list with page objects

        Args:
        - search_query: str - The search query to use.
        - params: SemanticSearchParams - Parameters for the search.
        - return_df: bool - If True, return a DataFrame. If False, return a list of Page objects.

        Returns:
        - List[Page] - A list of Page objects (if return_df = False)."""

        if params is None:
            params = SemanticSearchParams()

        similarity_top_k = params.similarity_top_k
        similarity_min_k = getattr(params, "similarity_min_k", None)
        context_window = params.context_window
        search_method = params.search_method

        if not self._repository.pdf_exists(self.short_filename):
            raise KeyError(
                f"PDF file '{self.short_filename}' not found in database. "
                "Please run 'load_pdf_and_embed_and_save_to_database' first."
            )

        query_text = search_query.get_query_text()
        query_embedding = search_query.get_query_embedding()
        embed_dim = search_query.get_embed_dim()
        search_embed_created_at = search_query.get_query_embedding_creation_date()

        if search_method == "vector_search":
            # Fetch all pages, then apply percentile-based filtering with safeguards
            base_df = self._repository.pdf_vector_search(
                short_file_name=self.short_filename,
                search_vector=query_embedding,
                limit=None,  # no limit; we'll cap after thresholding
                similarity="cosine",
                embed_dim=embed_dim
            )
            if base_df.empty:
                res_df = base_df
            else:
                # Compute percentile threshold across all pages in this PDF
                percentile = getattr(params, "percentile_threshold", 95) or 95
                percentile_cutoff = np.percentile(base_df["similarity"], percentile)
                filtered = base_df[base_df["similarity"] >= percentile_cutoff]

                max_pages = similarity_top_k if similarity_top_k else 7
                min_pages = similarity_min_k if similarity_min_k else 0
                # Respect the cap: a floor above the cap is not possible
                if max_pages and min_pages:
                    min_pages = min(min_pages, max_pages)

                filtered = filtered.sort_values("similarity", ascending=False)
                res_df = filtered.head(max_pages)

                # Floor: ensure at least min_pages are processed by falling back to top-N overall
                if min_pages and len(res_df) < min_pages:
                    fallback_df = base_df.sort_values(
                        "similarity", ascending=False)
                    res_df = fallback_df.head(min_pages)
        elif search_method == "full_text_search":
            res_df = self._repository.pdf_full_text_search(
                short_file_name=self.short_filename,
                search_text=query_text,
                limit=similarity_top_k
            )
        else:
            raise ValueError(
                "search_method must be 'vector_search' or 'full_text_search'.")

        page_indices_with_adjacents = self._find_adjacent_pages(
            res_df[["page_index"]], context_window
        )

        res_df2 = self._repository.get_pdf_pages_by_indices(
            short_file_name=self.short_filename,
            page_index_list=page_indices_with_adjacents,
            page_index_list_subset=res_df.page_index.to_list(),
            search_method=search_method,
            search_query=query_text,
            search_vector=query_embedding,
            embed_dim=embed_dim,
            search_embed_created_at=search_embed_created_at
        )

        if return_df:
            return res_df2
        else:
            return self._transform_df2pages(res_df2)

    async def load_pdf_and_embed_and_save_to_database(self,
                                                      embed_model: EmbeddingModel,
                                                      embed_only: bool = False):
        """Load raw text from PDF file (each page is one document), 
        convert to "nodes", create embeddings, and persist results in database"""

        if not self._repository.database_exists():
            # Create the DB schema if this is the first PDF
            print(
                f"Database '{self._repository.database_name}' not found. Creating new database.")
            self._repository.create_database(embed_model.get_embed_dimension())

        if self._repository.pdf_exists(self.short_filename):
            print(
                f"PDF {self.filename} already exists in the database. Nothing added.\n")
            return True

            # Typically, we don't create embeddings for the pdfs everytime
            # Current code already counts embedding tokens when they are created
            # (in config -> EmbeddingModel.get_embeddings())
            # If we would still like to count embedding tokens every time, uncomment this code
            # Currently commented because counting embedding tokens everytime,
            # just like creating embeddings everytime, is time costly

            # if hasattr(embed_model, 'token_counter'):
            #     print(f"Counting tokens for existing PDF content...")
            #     # Load the same way we would for new content to count tokens
            #     pages_nodes = self._load_raw_text_pagewise_from_pdf()

            #     total_tokens = 0
            #     for node in pages_nodes:
            #         tokens = embed_model.token_counter.count_tokens(node.get_text())
            #         total_tokens += tokens

            #     embed_model.token_counter.add_embedding_tokens(total_tokens)

            # return True
        else:
            # check if the pdf is marked as "encrypted_pdf" and if yes, skip embeding
            with open('./data/docs/pdf_info.json', "r", encoding="utf-8") as f:
                pdf_info = json.load(f)
                if (self.filename in pdf_info
                    and "in_sample" in pdf_info[self.filename]
                        and "encrypted_pdf" in pdf_info[self.filename]["in_sample"]):
                    print(
                        f"PDF {self.filename} marked as 'encrypted_pdf' and hence will be skipped.")
                    return False
            print(f"Load PDF {self.filename}")
            try:
                pages_nodes = self._load_raw_text_pagewise_from_pdf()
            except Exception as e:
                print(f"Fehler beim Laden von {self.filename}: {str(e)}")
                self._handle_problematic_pdf(str(e))
                return False

            if not pages_nodes:
                # No pages extracted (image-only/blank/unreadable PDF)
                msg = "No pages extracted from PDF (empty reader output)"
                print(f"PDF {self.filename}: {msg}. Skipping.")
                self._handle_problematic_pdf(msg)
                return False

            # Tokenization setup
            encoding = tiktoken.get_encoding("cl100k_base")
            max_tokens = 8192
            processed_texts = []

            truncated_count = 0  # Initialize counter

            for page in pages_nodes:
                original_text = page.get_text()
                tokens = encoding.encode(original_text)

                if len(tokens) > max_tokens:
                    truncated_text = encoding.decode(tokens[:max_tokens])
                    processed_texts.append(truncated_text)
                    truncated_count += 1  # Increment counter
                    # Write to log file
                    with open("problematic_pdfs_log.txt", encoding="utf-8", mode="a") as f:
                        f.write(
                            f"PDF: {self.filename}, Page: {page.metadata['page_label']}\n")
                else:
                    processed_texts.append(original_text)

            # Modified print statement
            print(
                (
                    f"Embedding {len(processed_texts)}"
                    f" pages with {embed_model} ({truncated_count} page(s) truncated)"
                ))
            embeddings = await embed_model.aget_embeddings(processed_texts)

            # Save results
            embed_model_repr = repr(embed_model)
            embed_created_at = datetime.datetime.now()
            self._repository.persist_pdf_with_pages_in_database(
                pages_nodes, embeddings, embed_model_repr, embed_created_at)
            if embed_only:
                print(
                    f"PDF {self.filename} was embedded successfully, no further processing.")

            return True

    def _handle_problematic_pdf(self, error_message):
        """Loggt problematische PDFs und aktualisiert die JSON-Datei mit os.path"""
        # Add log entry
        with open("problematic_pdfs_log.txt", "a", encoding="utf-8") as log_file:
            log_file.write(f"PDF: {self.filename} - {error_message} \n")

        with open('./data/docs/pdf_info.json', "r", encoding="utf-8") as f:
            pdf_info = json.load(f)

        # add entry in json file
        if self.filename in pdf_info:
            if "in_sample" not in pdf_info[self.filename]:
                pdf_info[self.filename]["in_sample"] = []
            if "encrypted_pdf" not in pdf_info[self.filename]["in_sample"]:
                pdf_info[self.filename]["in_sample"].append("encrypted_pdf")

        # save json
        with open('./data/docs/pdf_info.json', "w", encoding="utf-8") as f:
            json.dump(pdf_info, f, indent=4, ensure_ascii=False)

    def remove_pdf_from_database(self):
        """Remove this PDF file from the database."""
        self._repository.remove_pdf(self.short_filename)

    def _load_raw_text_pagewise_from_pdf(self):
        """
        Load PDF file (each page is one document) and convert documents to nodes.
        (pure llama.index functionality)
        """
        # every page is loaded as a separate document
        reader = PDFReader(return_full_document=False)

        # Each node should represent a page. Chunk size should be large enough,
        # so that we map every document
        # (in our case this is a page) to its own node.
        page_is_node_ingest_pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=100000),
                # extractors.TitleExtractor(),
                # Extract `document_title` metadata field using our llm?
            ]
        )

        pages_from_pdf = reader.load_data(self.filename)
        pages_nodes = page_is_node_ingest_pipeline.run(
            documents=pages_from_pdf)

        return pages_nodes

    def _find_adjacent_pages(self, selected_pages: pd.DataFrame, context_window: int) -> str:
        """
        For each page in selected_pages, add the pages around it (± context_window).
        Return a comma-separated string of indices to retrieve.
        """
        if selected_pages.empty:
            return []

        indices = []
        page_index_col = "page_index"
        for i in range(-context_window, context_window + 1):
            indices.append(selected_pages + i)

        page_indices = pd.concat(indices)[page_index_col].unique().tolist()
        return page_indices

    def _transform_df2pages(self, df: pd.DataFrame) -> List[Page]:
        """Map query results in a DataFrame to a list of Page objects."""
        pages = []
        for row in df.itertuples():
            data = {"complete_file_path": row.complete_file_path,
                    "short_file_name": row.short_file_name,
                    "page_index": row.page_index,
                    "page_label": row.page_label,
                    "page_content": row.page_content,
                    "search_method": row.search_method,
                    "vector_similarity": row.vector_similarity,
                    "bm25_similarity": row.bm25_similarity,
                    "is_adjacent": row.is_adjacent,
                    "page_embedding_created_at": row.page_embedding_created_at,
                    "query_embedding_created_at": row.query_embedding_created_at}
            page = Page(data)
            pages.append(page)
        return pages


if __name__ == "__main__":

    @dataclass
    class SemanticSearchParams:
        """Parameters for semantic search."""
        path_to_embedding_repository: str = field(
            default="embeddings_from_2025_02_22.duckdb")
    #    path_to_embedding_repository: str = field(
    #       default="data/processed/embeddings/text-embedding-ada-002_from_2025_03_05.duckdb")
        # field(default="text-embedding-ada-002")
        emb_model: str = field(default="text-embedding-3-large")
        search_query: str = field(default="""What are the total CO2 emissions in different years?
                                Include Scope 1, Scope 2, and Scope 3 emissions if available.""")
        similarity_top_k: int = field(default=7)
        similarity_min_k: int = field(default=4)
        percentile_threshold: int = field(default=95)
        context_window: int = field(default=0)
        search_method: str = field(default="vector_search")

    semantic_search_params = SemanticSearchParams()

    embeddings_repo = EmbeddingsRepository(
        database_name=semantic_search_params.path_to_embedding_repository)
    embed_model = EmbeddingModel(model_name=semantic_search_params.emb_model)

    search_query = SearchQuery(
        search_query=semantic_search_params.search_query, repository=embeddings_repo)

    # search_query.remove_search_query_from_database()
    # pdfdoc = Pdfdoc("./data/pdfs/puma_2018_en.pdf", repository = embeddings_repo)
    # pdfdoc.remove_pdf_from_database()

    pdfdoc = Pdfdoc("./data/pdfs/apple_2021_en.pdf",
                    repository=embeddings_repo)
    pdfdoc = Pdfdoc("./data/pdfs/puma_2018_en.pdf", repository=embeddings_repo)

    if asyncio.run(pdfdoc.load_pdf_and_embed_and_save_to_database(embed_model)) and \
            search_query.embed_search_query_and_save_to_database(embed_model):

        # vector search with cosine similarity
        # (results look very similar to what llama.index returned in the past)
        # & return DataFrame
        res2 = pdfdoc.retrieve_relevant_pages(
            search_query=search_query, params=semantic_search_params, return_df=True)

        print(res2[["short_file_name", "page_index", "vector_similarity"]])
        print(res2)

        # full text search & return list of Pages
        res2 = pdfdoc.retrieve_relevant_pages(
            search_query=search_query, params=semantic_search_params, return_df=True)

        print(res2)
