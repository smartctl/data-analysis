# Data Analysis

Data analysis tool.

## Usage

- Setup a Python Virtual Environment
```bash
# Create a Python Virtual Environment
python3 -m venv venv

# Activate the virtual environment
source ./venv/bin/activate

# Upgrade pip to the latest within the virtual environment
pip install --upgrade pip

# Install Python libraries in virtual environment
pip install -r requirements.txt
```

- Update `mapping.yaml` to match your environment.

- Execute the collector `python main.py` and collect for the desired time.
  - Press Ctrl-C to break or stop collection
  - A parquet file will be written to data/collection 
    ```
    data/
    └── collection
        └── metrics_combined_20230806-135903.parquet
    ```
- Load the collection file for data analysis using the Pandas library
  ```python
  import pandas as pd
  fname="data/collection/metrics_combined_20230806-135903.parquet"
  df=pd.read_parquet(fname)
  df.head()
  ```
