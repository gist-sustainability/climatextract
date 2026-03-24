# GIST Testing

### Prerequisites

Ensure you have the `pytest` library installed. If not, run the following command 

`pip install pytest~=8.3.5`

Before the tests execute successfully, you will need to download the embeddings and extract table information from the PDF files. Either configure `climatextract.toml` with `filename_list = ["./data/pdfs/sato holdings_2022_report.pdf"]` and `input_mode = "text+table"`, or use the public API:

```python
from climatextract import extract
extract(["./data/pdfs/sato holdings_2022_report.pdf"])
```

### Running Tests

To run the tests, follow these steps:

1. Open a terminal and navigate to your project's root directory.

2. Run the following command to execute all tests:

`python -m pytest` 

For verbose output including names of individual test functions, use 

`python -m pytest -v`

For additionally disabled output capturing, use 

`python -m pytest -v -s`

To show a dataframe with mismatches between goldstandard and extracted values: 

`python -m pytest --log-cli-level=INFO`

3. To run specific tests, you can use the following commands:

- To run only the functionality tests:

  `python -m pytest tests/test_acceptance.py::test_functionality`

- To run only the quality tests:

  `python -m pytest tests/test_acceptance.py::test_quality`

4. To save the test output in a file:

  `pytest -v -s > tests/output_tests.txt 2>&1`

### Output of running the tests

1. When all tests pass successfully, the output will display 8 passed tests along with several warnings (triggered by the main function's runtime). These warnings do not affect code quality or test validity. The warnings are informational and do not indicate test failures. The final output will appear as: 

==================== 8 passed, 49 warnings in 133.41s (0:02:13) ====================

2. If one or more tests fail, the output will explicitly highlight the failed result alongside passed tests and warnings. For example: 

==================== short test summary info ====================

FAILED tests/test_acceptance.py::test_quality[default-text] - AssertionError: Expected 32 True values in 'value_match', got 11

FAILED tests/test_acceptance.py::test_quality[structured_json-text] - AssertionError: Expected 32 True values in 'value_match', got 11

FAILED tests/test_acceptance.py::test_quality[default-text+table] - AssertionError: Expected 32 True values in 'value_match', got 11

FAILED tests/test_acceptance.py::test_quality[structured_json-text+table] - AssertionError: Expected 32 True values in 'value_match', got 11

==================== 4 failed, 4 passed, 45 warnings in 127.73s (0:02:07) ====================


### Test Descriptions

#### Functionality Test (`test_functionality`)

This test verifies the main functionality with different prompt types and input modes. It checks that:

1. The main function runs successfully and returns a valid run_id.
2. The MLflow experiment status is 'FINISHED'.
3. Artifacts are generated and contain at least one row of data.

#### Quality Test (`test_quality`)

This test checks the quality of results for different prompt types and input modes. It verifies that:

1. The required artifact '04a_results_available_in_report.csv' is present.
2. The 'value_match' column contains the expected number of True and NA values.
3. Any mismatches in 'value_match' are reported.
4. Optionally, the number of True values in 'unit_match' and mismatches are reported.
