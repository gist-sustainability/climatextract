"""Data Lake Manager for handling Azure storage operations."""

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import urlparse

# Optional Azure imports for data lake downloads
try:
    from azure.storage.blob import BlobServiceClient
except Exception:
    BlobServiceClient = None

try:
    from azure_authentication import customized_azure_login
    credential = customized_azure_login.CredentialFactory().select_credential()
except Exception:
    credential = None


class DataLakeManager:
    """Manages data lake operations for downloading and preparing required files.
    
    This class breaks the workflow into four separate, focused methods:
    1. handle_embedding_database()
    2. check_files_needing_embedding()
    3. check_files_needing_download()
    4. download_missing_pdfs()
    """
    
    def __init__(self, blob_path_pdfs: str = "pdfs",
                 blob_path_embeddings: str = "embeddings",
                 storage_account_url: Optional[str] = None,
                 storage_credential: Optional[Any] = None):
        """Initialize the DataLakeManager.

        Args:

            blob_path_pdfs: Blob container/path for PDF files.
            blob_path_embeddings: Blob container/path for embedding databases.
            storage_account_url: Azure storage account URL for data lake access.
                Falls back to the AZURE_STORAGE_ACCOUNT_URL environment variable.
            storage_credential: Shared Access Signature (SAS) token or Azure credential object. Falls
                back to the AZURE_STORAGE_SAS_TOKEN environment variable, then to
                the Entra ID credential.
        """
        self.storage_account_url = (storage_account_url
                                    or os.environ.get("AZURE_STORAGE_ACCOUNT_URL"))
        self.storage_credential = (storage_credential
                                   or os.environ.get("AZURE_STORAGE_SAS_TOKEN")
                                   or credential)
        self.blob_path_pdfs = blob_path_pdfs
        self.blob_path_embeddings = blob_path_embeddings
        self._blob_service = None

    def _get_blob_service(self):
        """Get or create the blob service client."""
        if self._blob_service is None:
            if not self.storage_account_url:
                logging.getLogger(__name__).debug("No storage account URL configured")
                return None
            if BlobServiceClient is None:
                logging.getLogger(__name__).debug("Azure storage library 'azure-storage-blob' not installed.")
                return None
            if self.storage_credential is None:
                logging.getLogger(__name__).debug("Cannot authenticate against storage account: Neither a SAS token nor Azure credential available")
                return None

            self._blob_service = BlobServiceClient(
                account_url=self.storage_account_url,
                credential=self.storage_credential
            )
        return self._blob_service
        
    def execute_complete_workflow(self, 
                                filename_list: List[str],
                                embeddings_repo,
                                input_mode: str = "text") -> bool:
        """Execute the complete data lake workflow using all four steps.
        
        This method orchestrates all four steps:
        1. Handle embedding database
        2. Check files needing embedding  
        3. Check files needing download
        4. Download missing PDFs
        
        Args:
            filename_list: List of PDF file paths to process.
            embeddings_repo: Repository object for embeddings.
            input_mode: Processing mode ('text' or 'text+table').
            
        Returns:
            bool: True if workflow completed successfully, False otherwise.
        """
        # Step 1: Handle embedding database
        if not self.handle_embedding_database(embeddings_repo):
            return False
        
        # Step 2: Check which files need embedding
        missing_files = self.check_files_needing_embedding(filename_list, embeddings_repo)
        
        if not missing_files and input_mode == "text":
            return True

        if input_mode == "text+table":
            logging.getLogger(__name__).info("text+table mode requires PDF files for table extraction")

        # Step 3: Check which files need download
        files_to_download = self.check_files_needing_download(missing_files, filename_list, input_mode)
        
        # Step 4: Download missing PDFs
        return self.download_missing_pdfs(files_to_download)
    
    def handle_embedding_database(self, embeddings_repo) -> bool:
        """Step 1: Handle embedding database download if needed.

        Args:
            embeddings_repo: Repository object to check if database exists.

        Returns:
            bool: True if database is available or successfully downloaded, False on failure.
        """
        blob_service = self._get_blob_service()
        if not blob_service:
            logging.getLogger(__name__).info("Data lake not configured — database will be created locally if needed")
            return True

        # Handle embedding database
        if not embeddings_repo.database_exists():
            logging.getLogger(__name__).info("Embedding database not found locally")
            response = input("Download database from data lake? [y/N] ").strip().lower()
            embeddings_db_path = embeddings_repo.get_database_name()

            if response == "y":
                try:
                    container, prefix = self._split_blob_path(self.blob_path_embeddings)
                    container_client = blob_service.get_container_client(container)
                    blob_name = prefix + os.path.basename(embeddings_db_path)
                    blob_client = container_client.get_blob_client(blob_name)

                    dir_path = os.path.dirname(embeddings_db_path)
                    if dir_path:
                        os.makedirs(dir_path, exist_ok=True)
                    with open(embeddings_db_path, "wb") as f:
                        blob_client.download_blob().readinto(f)
                except Exception as e:
                    logging.getLogger(__name__).warning(
                        "Embedding database download failed (%s). "
                        "A new database will be created locally at %s",
                        e, embeddings_db_path)
                    # Remove the partial download so an empty/corrupt file is
                    # not later mistaken for a valid database.
                    try:
                        if os.path.exists(embeddings_db_path):
                            os.remove(embeddings_db_path)
                    except OSError:
                        pass

        return True
    
    def check_files_needing_embedding(self, filename_list: List[str], embeddings_repo) -> List[str]:
        """Step 2: Check which files need embedding.
        
        Args:
            filename_list: List of all PDF file paths to process.
            embeddings_repo: Repository object to check if PDFs are already embedded.
            
        Returns:
            List[str]: List of file paths that need embedding.
        """
        missing_files = []
        if embeddings_repo.database_exists():
            for filepath in filename_list:
                short_filename = os.path.basename(filepath)
                if not embeddings_repo.pdf_exists(short_filename):
                    missing_files.append(filepath)
        else:
            missing_files = filename_list.copy()

        return missing_files
    
    def check_files_needing_download(self, missing_files: List[str], all_files: List[str], input_mode: str) -> List[str]:
        """Step 3: Check which files need to be downloaded.
        
        Args:
            missing_files: List of files that need embedding.
            all_files: List of all PDF files in the workflow.
            input_mode: Processing mode ('text' or 'text+table').
            
        Returns:
            List[str]: List of file paths that need to be downloaded.
        """
        files_to_download = []
        
        if input_mode == "text+table":
            # For text+table mode: download ANY missing PDF file (regardless of embedding status)
            for filepath in all_files:
                if not os.path.exists(filepath):
                    files_to_download.append(filepath)
        else:
            # For text mode: only download files that need embedding
            for filepath in missing_files:
                if not os.path.exists(filepath):
                    files_to_download.append(filepath)
        
        return files_to_download
    
    def download_missing_pdfs(self, files_to_download: List[str]) -> bool:
        """Step 4: Download missing PDF files from data lake.
        
        Args:
            files_to_download: List of PDF file paths to download.
            
        Returns:
            bool: True if all files downloaded successfully, False on failure or user decline.
        """
        if not files_to_download:
            return True

        blob_service = self._get_blob_service()
        if not blob_service:
            logging.getLogger(__name__).warning("PDF files needed but data lake is not available")
            return False

        container, prefix = self._split_blob_path(self.blob_path_pdfs)

        # Calculate total size
        total_size_bytes = 0
        try:
            container_client = blob_service.get_container_client(container)
            for filepath in files_to_download:
                blob_name = prefix + os.path.basename(filepath)
                blob_client = container_client.get_blob_client(blob_name)
                props = blob_client.get_blob_properties()
                total_size_bytes += int(props.size or 0)
        except Exception:
            total_size_bytes = 0  # If we can't get sizes, proceed anyway

        # Convert to readable format
        total_size_mb = total_size_bytes / (1024 * 1024)

        if total_size_mb > 0:
            logging.getLogger(__name__).info(
                "Files to download: %d (%.1f MB)", len(files_to_download), total_size_mb)
        else:
            logging.getLogger(__name__).info(
                "Files to download: %d", len(files_to_download))

        response = input("Download PDFs from data lake? [y/N] ").strip().lower()

        if response != "y":
            return False

        try:
            container_client = blob_service.get_container_client(container)
            for filepath in files_to_download:
                blob_name = prefix + os.path.basename(filepath)
                blob_client = container_client.get_blob_client(blob_name)

                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, "wb") as f:
                    blob_client.download_blob().readinto(f)
        except Exception as e:
            logging.getLogger(__name__).warning("PDF download failed: %s", e)
            return False

        return True
    
    @staticmethod
    def _split_blob_path(blob_path: str) -> tuple:
        """Split a blob path into container name and optional prefix.
        
        Examples:
            "pdfs"                -> ("pdfs", "")
            "mycontainer/folder"  -> ("mycontainer", "folder/")
            "mycontainer/a/b"     -> ("mycontainer", "a/b/")
        """
        parts = blob_path.strip("/").split("/", 1)
        container = parts[0]
        prefix = (parts[1].rstrip("/") + "/") if len(parts) > 1 else ""
        return container, prefix
    
    def download_directory_from_blob(self, local_dir: str) -> bool:
        """Download all PDFs from a matching blob prefix into a local directory.

        Uses the directory name as a blob prefix to select which files to download.
        E.g. local_dir="data/pdfs/sample_160" looks for blobs under "sample_160/" in
        the configured blob_path_pdfs container.

        Args:
            local_dir: Local directory path to download PDFs into.

        Returns:
            bool: True if download succeeded, False otherwise.
        """
        blob_service = self._get_blob_service()
        if not blob_service:
            logging.getLogger(__name__).warning("Data lake not available — cannot download directory contents")
            return False

        container, base_prefix = self._split_blob_path(self.blob_path_pdfs)
        dir_name = os.path.basename(os.path.normpath(local_dir))
        prefix = base_prefix + dir_name + "/"

        try:
            container_client = blob_service.get_container_client(container)
            blobs = [b for b in container_client.list_blobs(name_starts_with=prefix)
                     if b.name.endswith(".pdf")]
        except Exception:
            logging.getLogger(__name__).warning("Failed to list blobs under prefix '%s'", prefix)
            return False

        if not blobs:
            logging.getLogger(__name__).info("No PDFs found in data lake under '%s'", prefix)
            return False

        total_size_mb = sum(b.size or 0 for b in blobs) / (1024 * 1024)
        logging.getLogger(__name__).info(
            "Found %d PDFs (%.1f MB) in data lake under '%s'", len(blobs), total_size_mb, prefix)

        response = input("Download from data lake? [y/N] ").strip().lower()
        if response != "y":
            return False

        try:
            os.makedirs(local_dir, exist_ok=True)
            for blob in blobs:
                blob_client = container_client.get_blob_client(blob.name)
                local_path = os.path.join(local_dir, os.path.basename(blob.name))
                with open(local_path, "wb") as f:
                    blob_client.download_blob().readinto(f)
        except Exception as e:
            logging.getLogger(__name__).warning("Failed to download PDFs from data lake: %s", e)
            return False

        return True

    def download_pdfs_if_not_locally_available(self, pdf_input: str | List[str]) -> bool:
        """Check all input PDFs for local availability and offer to download missing ones.

        This is an upfront check before the pipeline runs. Files not downloaded here
        may still be prompted for later if they are also missing from the embedding database.

        Args:
            pdf_input: A single PDF path or list of PDF paths.

        Returns:
            bool: True if all files are available or download succeeded, False if user declined.
        """
        if isinstance(pdf_input, str):
            pdf_input = [pdf_input]

        files_to_download = [f for f in pdf_input if f.endswith('.pdf') and not Path(f).is_file()]

        if not files_to_download:
            return True

        total = len(pdf_input)
        missing = len(files_to_download)
        logging.getLogger(__name__).info(
            "%d of %d PDF files are not available locally. "
            "Files missing locally and not in the embedding database will not be processed.",
            missing, total)

        return self.download_missing_pdfs(files_to_download)


class DataLakeAdministrator:
    """Administrative operations on the data lake storage account,
        which require a storage account key. Not needed for normal users.
    """

    def __init__(self, storage_account_url: Optional[str] = None,
                 account_key: Optional[str] = None):
        """Initialize the DataLakeAdministrator.

        Args:
            storage_account_url: Azure storage account URL for data lake access.
                Falls back to the AZURE_STORAGE_ACCOUNT_URL environment variable.
            account_key: Storage account key. Falls back to the
                AZURE_STORAGE_ACCOUNT_KEY environment variable. Only an 
                administrator with the account key can generate SAS tokens.
        """
        self.storage_account_url = (storage_account_url
                                    or os.environ.get("AZURE_STORAGE_ACCOUNT_URL"))
        self.account_key = account_key or os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")

    def generate_account_sas_for_blob_readonly(self, expiry_days: int = 270) -> str:
        """Generate a read-only Shared Access Signature (SAS) token for the blob service.

        Hand the token to users who should read the data lake without an Entra ID
        login; they set it as AZURE_STORAGE_SAS_TOKEN. It covers every container of
        the storage account, permits no writes or deletes, and can only be revoked by
        rotating the account key.

        Args:
            expiry_days: How long the token stays valid.

        Returns:
            str: The SAS token as a query string, without a leading '?'.

        Raises:
            ValueError: If the storage account URL or the account key is missing.
        """
        from azure.storage.blob import (
            AccountSasPermissions,
            ResourceTypes,
            generate_account_sas,
        )

        if not self.storage_account_url:
            raise ValueError("No storage account URL configured")

        if not self.account_key:
            raise ValueError(
                "No account key available — pass account_key or set AZURE_STORAGE_ACCOUNT_KEY")

        now = datetime.now(timezone.utc)
        return generate_account_sas(
            account_name=self._account_name_from_url(self.storage_account_url),
            account_key=self.account_key,
            # service: list containers; container: list blobs; object: read them
            resource_types=ResourceTypes(service=True, container=True, object=True),
            permission=AccountSasPermissions(read=True, list=True),
            start=now - timedelta(minutes=15),
            expiry=now + timedelta(days=expiry_days),
            protocol="https"
        )

    @staticmethod
    def _account_name_from_url(storage_account_url: str) -> str:
        """Extract the storage account name from its endpoint URL.

        Examples:
            "https://myaccount.blob.core.windows.net" -> "myaccount"
            "http://127.0.0.1:10000/devstoreaccount1" -> "devstoreaccount1"
        """
        parsed = urlparse(storage_account_url)
        host = parsed.hostname or ""

        # Every Azure storage endpoint — public, sovereign clouds, private endpoints,
        # blob and dfs — names the account first and carries ".core." in the host.
        if ".core." in host:
            return host.split(".")[0]

        # Storage emulators (Azurite) address the account by path rather than hostname
        path_parts = [p for p in parsed.path.split("/") if p]
        if parsed.scheme and path_parts:
            return path_parts[0]

        raise ValueError(f"Cannot determine account name from URL: {storage_account_url}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    admin = DataLakeAdministrator()
    try:
        if True:
            sas_token = admin.generate_account_sas_for_blob_readonly(expiry_days=270)
            print(f"Generated SAS token: {sas_token}")
            print(f"Set in your environment: AZURE_STORAGE_SAS_TOKEN={sas_token}")
    except ValueError as e:
        logging.getLogger(__name__).error("Failed to generate SAS token: %s", e)
