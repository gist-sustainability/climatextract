"""Handles extraction of text and tables from PDF pages"""
import html
import os
import csv
import re
from typing import Dict, List

import torch
from pdf2image import convert_from_path
from transformers import DetrImageProcessor
from transformers import TableTransformerForObjectDetection
from docling.document_converter import DocumentConverter

from climatextract.semantic_search import Page

class PageTextAndTableExtractor:
    """Handles extraction of text and tables from PDF pages
    Consolidates all table extraction functionality (also methods previously in table_helper.py)
    """
    
    def __init__(self):
        """Initialize the extractor"""
        pass
    

    async def extract_text_and_tables_from_pages(self, relevant_pages: List[Page], filename: str) -> List[str]:
        """Extract both text and tables from relevant pages."""
        relevant_tables = self.get_tables_for_relevant_pages(relevant_pages, filename)
        # If there are relevant pages with tables, pass them to LLM
        if any(inner_list for inner_list in relevant_tables):
            return [''.join(str(x) for x in relevant_tables[idx]) + ' ' +  page.page_content for idx, page in enumerate(relevant_pages)]
        else:
            return [page.page_content for page in relevant_pages]
        
    def get_tables_for_relevant_pages(self, relevant_pages, filename):
        """Extract tables from relevant pages and return them with the filtered relevant pages."""
        relevant_page_numbers = [
            page.page_index + 1 for page in relevant_pages]

        extracted_tables = self.extract_tables_from_pages(
            filename, relevant_page_numbers)

        tables = self.match_relevant_pages_and_tables(
            relevant_pages, extracted_tables)

        return tables
    
    def match_relevant_pages_and_tables(self, relevant_pages, extracted_tables):
        """Match relevant pages with extracted tables."""
        tables_padded = []
        for page in relevant_pages:
            page_number = page.page_index + 1
            if page_number in extracted_tables.keys():
                tables_padded.append(extracted_tables[page_number])
            else:
                tables_padded.append([])
        return tables_padded
    
    

    def extract_tables_from_pages(self, file_name: str, rel_page_numbers: List[int]) -> Dict[int, list]:
        """
        Extract tables from the specified pages of the PDF file,
        either by reading in already extracted tables or by extracting them from the PDF file, 
        checking if the page contains a table first.
        """
        tables = {}
        for page in rel_page_numbers:
            report_name = file_name.split('/')[-1].replace('.pdf', '')
            path_to_table_cells = f'./data/processed/tables/{report_name}_{page}_table_cells.csv'
            # In case table has already been extracted from the page
            if os.path.exists(path_to_table_cells):
                with open(path_to_table_cells, 'r', encoding='utf-8') as f:
                    table = list(csv.reader(f, delimiter=","))
                tables[page] = table
            # In case table has not been extracted from the page
            else:
                # Check whether the page contains a table
                if self.check_if_table_on_page(file_name, page):
                    table_extracted = self.extract_table_from_page(file_name, page)
                    table_cleaned = self.clean_table(table_extracted)
                    self.save_table(path_to_table_cells, table_cleaned)
                    tables[page] = table_cleaned

        return tables


    def check_if_table_on_page(self, file_name: str, relevant_page: int) -> bool:
        '''Check if the page contains a table'''
        processor = DetrImageProcessor.from_pretrained(
            "microsoft/table-transformer-detection")
        model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-detection")

        image = convert_from_path(
            file_name, first_page=relevant_page, last_page=relevant_page, fmt='jpeg')[0]
        image = image.convert("RGB")
        width, height = image.size
        image.resize((int(width*0.5), int(height*0.5)))

        inputs = processor(images=image, return_tensors="pt")

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Identify the tables in the image of pdf page
        results = processor.post_process_object_detection(
            outputs, threshold=0.7, target_sizes=[(height, width)])[0]
        if len(results['scores']) != 0 and any(t > 0.98 for t in results['scores']):
            return True
        else:
            return False


    def extract_table_from_page(self, file_name: str, page: int) -> list:
        """Extracts the table from the filename on the specified page"""
        converter = DocumentConverter()
        doc = converter.convert(source=file_name, page_range=(
            page, page))
        data = doc.document.export_to_markdown()
        data = html.unescape(data)
        data = data.replace('-', '')
        data = data.replace('  ', '')
        data_reformatted = [i.split('|')
                            for i in data.split('\n') if len(i.strip()) > 0]
        return data_reformatted


    def clean_table(self, table: list) -> list:
        """Performs string manipulations to clean markdown table"""
        result = []
        counter_row = 0
        counter_col = 0
        filter_data = []
        max_length = max(map(len, table))
        for i in table:

            if len(i) >= max_length/2:
                transform_i = []
                for x in i:
                    if re.match(r"^\ ?[2][0][0-9]{2}\ *$", x):
                        transform_i.append(x)
                    else:
                        x = x.replace(",", "")
                        x = x.replace("n/a", "")
                        x = x.replace("N/A", "")
                        x = x.replace("\n", " ")
                        x = x.replace("<", "")
                        x = x.replace(">", "")
                        x = x.replace("=", "")

                        if x[-1:].isspace():
                            x = x[:-1]

                        if re.match(r"\ *[+-]?(?:\d+(?:\.\d+)?|\.\d+)$", x):
                            x = float(x)
                        elif re.match(r"^[A-Za-z0-9\)\('\"\\s]+[0-9a-z\)][\d+]$", x):
                            x = x[:-1]
                            if re.match(r"^[A-Za-z0-9\)\('\"\\s]+[0-9a-z\)][\d+]$", x):
                                x = x[:-1]
                        elif re.match(r"^[0-9.\s]+[0-9.]$", x):
                            x = ""
                        transform_i.append(x)
                filter_data.append(transform_i)

        for idx_i, i in enumerate(filter_data):
            for idx_j, j in enumerate(i):
                if isinstance(j, float):
                    for a in range(idx_j, -1, -1):
                        if isinstance(i[a], str) and len(i[a]) > 3:
                            row_name = i[a]
                            counter_row = 1
                            break
                        else:
                            pass

                    for b in range(idx_i, -1, -1):
                        if len(filter_data[b]) <= idx_j:
                            continue
                        col_candidate = filter_data[b][idx_j]
                        if isinstance(col_candidate, str) and len(col_candidate) > 3:
                            col_name = col_candidate
                            counter_col = 1
                            break

                    if counter_row == 1 and counter_col == 1:
                        result.append((row_name, col_name, j))
                        counter_row = 0
                        counter_col = 0

        return result


    def save_table(self, path_to_table_cells: str, table: list):
        """Save the table to a CSV file"""
        with open(path_to_table_cells, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(table)
        return
