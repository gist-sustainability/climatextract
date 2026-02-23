from climatextract import extract
from climatextract import extract_and_evaluate

# Current interface
# result_path = extract("data/pdfs/sato holdings_2022_report.pdf")
# print(f"Results saved to: {result_path}")

result_path = extract_and_evaluate(pdf_input="./data/pdfs/sato oyj_2022_report.pdf", gold_standard_path="./data/evaluation_dataset/gist_2025.csv")
print(f"Results saved to: {result_path}")