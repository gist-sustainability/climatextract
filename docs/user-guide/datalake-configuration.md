# Datalake configuration

Use storage in the cloud to share large files with team members.

## Teamwork: Automatically populate data folders

If you work in a team where everyone uses the same PDF files, it can be tedious to ensure everone has access to the same PDF files and additional data. Our solution:

- One superuser stores all the PDF files (and embedding databases) online in an Azure datalake.
- Everyone who has access can download the files from the datalake on demand as needed. No need to copy and paste PDF files by hand, and everyone uses the same database.

Add to your `.env` file:

```bash
AZURE_STORAGE_ACCOUNT_URL=https://<your-datalake-name>.blob.core.windows.net/
AZURE_STORAGE_AUTODOWNLOAD_PDFS=True # This will download PDFs if not available locally. This can fill up your local storage. Not needed if embeddings are already generated and stored in the embeddings database.
```

## More documentation needed

mayby Maaz can describe the file in the data lake, and what it does?