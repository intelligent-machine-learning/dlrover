# Database Table Design

```mysql
create table job_metrics(
    job_uuid varchar(255)
    job_name varchar(255)
    create_at timestamp
    finished_at timestamp
    hyper_params_feature mediumtext
    job_feature mediumtext
    dataset_feature mediumtext
    model_feature mediumtext
    job_runtime mediumtext
    exit_reason varchar(255)
    optimization mediumtext
    resource mediumtext
    customized_data mediumtext
);
```