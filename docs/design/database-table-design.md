# Database Table Design

```mysql
create table job_metrics(
    job_uuid varchar(255) // job uuid, the unique id         
    job_name varchar(255) // job name
    create_at timestamp // job create timestamp
    finished_at timestamp // job finish timestamp
    hyper_params_feature mediumtext // the feature of hyper parameters
    job_feature mediumtext // the feature of the job
    dataset_feature mediumtext // the feature of the training dataset
    model_feature mediumtext // the feature of the training model
    job_runtime mediumtext // job runtime information
    exit_reason varchar(255) // the exit reason of the job
    optimization mediumtext // the optimization information of the job
    resource mediumtext // the resources of the job
    customized_data mediumtext // custimized data
);
```