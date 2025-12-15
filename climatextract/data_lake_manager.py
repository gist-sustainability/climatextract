"""Data Lake Manager for handling Azure storage operations."""

import os
from typing import List, Optional

# Optional Azure imports for data lake downloads
try:
    from azure.storage.blob import BlobServiceClient
except Exception:
    BlobServiceClient = None

try:
    from azure_authentication import customized_azure_login
    credential = customized_azure_login.CredentialFactory().select_credential()
except Exception as e:
    print(f"Error applying azure authentication, data lake not available")
    credential = None


class DataLakeManager:
    """Manages data lake operations for downloading and preparing required files.
    
    This class breaks the workflow into four separate, focused methods:
    1. handle_embedding_database()
    2. check_files_needing_embedding()
    3. check_files_needing_download()
    4. download_missing_pdfs()
    """
    
    def __init__(self, storage_account_url: Optional[str]):
        """Initialize the DataLakeManager.
        
        Args:
            storage_account_url: Azure storage account URL for data lake access.
        """
        self.storage_account_url = storage_account_url
        self._blob_service = None
        
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
        
        # Add user messaging for text+table mode
        if input_mode == "text+table":
            print("text+table mode requires PDF files for table extraction.")
            print("Checking for missing PDF files...")
        
        # Step 2: Check which files need embedding
        missing_files = self.check_files_needing_embedding(filename_list, embeddings_repo)
        
        if not missing_files and input_mode == "text":
            return True
        
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
            print("Cannot access data lake.")
            return True
        
        # Handle embedding database
        if not embeddings_repo.database_exists():
            print("Embedding database not found locally.")
            response = input("Download database from data lake? [y/N] ").strip().lower()
            embeddings_db_path = embeddings_repo.get_database_name()

            if response == "y":
                try:
                    container_client = blob_service.get_container_client("embeddings")
                    blob_name = os.path.basename(embeddings_db_path)
                    blob_client = container_client.get_blob_client(blob_name)
                    
                    os.makedirs(os.path.dirname(embeddings_db_path), exist_ok=True)
                    with open(embeddings_db_path, "wb") as f:
                        blob_client.download_blob().readinto(f)
                    print("Database downloaded.")
                except Exception:
                    print("Database download failed.")
                    return False
            else:
                print("Database will be created when needed.")
        
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
        
        if missing_files:
            print(f"Number of files that need embedding: {len(missing_files)}")
        
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
            print("Cannot access data lake.")
            return False
        
        # Calculate total size
        total_size_bytes = 0
        try:
            container_client = blob_service.get_container_client("pdfs")
            for filepath in files_to_download:
                blob_name = os.path.basename(filepath)
                blob_client = container_client.get_blob_client(blob_name)
                props = blob_client.get_blob_properties()
                total_size_bytes += int(props.size or 0)
        except Exception:
            total_size_bytes = 0  # If we can't get sizes, proceed anyway
        
        # Convert to readable format
        total_size_mb = total_size_bytes / (1024 * 1024)
        
        if total_size_mb > 0:
            print(f"Files to download: {len(files_to_download)} ({total_size_mb:.1f} MB)")
        else:
            print(f"Files to download: {len(files_to_download)}")
        
        response = input("Download PDFs from data lake? [y/N] ").strip().lower()
        
        if response != "y":
            print("Download declined.")
            return False
            
        try:
            container_client = blob_service.get_container_client("pdfs")
            for filepath in files_to_download:
                blob_name = os.path.basename(filepath)
                blob_client = container_client.get_blob_client(blob_name)
                
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, "wb") as f:
                    blob_client.download_blob().readinto(f)
            print("PDFs downloaded.")
        except Exception:
            print("PDF download failed.")
            return False
        
        return True
    
    def _get_blob_service(self):
        """Get or create the blob service client."""
        if self._blob_service is None:
            if not self.storage_account_url:
                print("No storage account configured.")
                return None
            if BlobServiceClient is None or credential is None:
                print("Azure libraries not available.")
                return None
            self._blob_service = BlobServiceClient(
                account_url=self.storage_account_url, 
                credential=credential
            )
        return self._blob_service