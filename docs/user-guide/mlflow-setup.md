# MLflow setup

Climatextract supports [MLflow](https://mlflow.org/docs/latest/ml/getting-started/) for experiment tracking. Basically, it provides a database to store the results from your experiments.

With Azure Databricks, all results can be stored and shared online.


## Setup

Add the following to your ``.env`` file:

```bash
# Tracking URI: "databricks", "./mlruns", "sqlite:///mlflow.db", or server URL
# Use "databricks" if you are using Databricks on Azure.
MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# If using databricks on Azure, add the following:
DATABRICKS_HOST=https://<your-databricks-instance>.azuredatabricks.net/
DATABRICKS_TOKEN=personal-access-token
```

To set up access to an remote Mlflow Tracking Server on Azure Databricks, you need to create a personal access token. Follow these steps: 

1. Set `MLFLOW_TRACKING_URI=databricks`.
2. Log into [Azure](https://portal.azure.com). 
3. Create a Databricks instance or find an existing instance you want to use.
4. Copy the URL which contains azuredatabricks.net and save it in the `.env` file as `DATABRICKS_HOST` variable. 
5. Launch the workspace and click on your initial in the upper right corner. 
6. Navigate to `Settings > User > Developer > Access tokens`and click on `Manage`. Generate a new access token and save it in the `.env` file as `DATABRICKS_TOKEN` variable. Be aware that it takes some time for the token to get activated, so you might get 401 authentication errors in the beginning when running the code. This should be resolved after some time.

In your ``climatextract.toml`` configuration file you can specify the experiment name:

```toml
[mlflow]
# Experiment name
experiment_name = "/Shared/Experiments/my_experiment"
```

---

## Next Steps

- [Sharing Large Files](datalake-configuration.md) – Share PDFs and embeddings with your team
- [Architecture](../concepts/architecture.md) – Understand how the pipeline works