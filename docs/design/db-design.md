# Database Table Design

```mysql
create table job_metrics(
    uuid varchar(255), // job uuid, the unique id         
    hyper_params_feature, mediumtext // the feature of hyper parameters
    job_feature mediumtext, // the feature of the job
    dataset_feature mediumtext, // the feature of the training dataset
    model_feature mediumtext, // the feature of the training model
    job_runtime mediumtext, // job runtime information
    exit_reason varchar(255), // the exit reason of the job
    optimization mediumtext, // the optimization information of the job
    resource mediumtext, // the resources of the job
    customized_data mediumtext, // custimized data
    type varchar(255) // indicate the data type
);

create table job(
    uuid varchar(255), // job uuid, the unique id
    name varchar(255), // job name
    create_at timestamp, // job create timestamp
    finished_at timestamp, // job finish timestamp
    exit_reason varchar(255), // the exit reason of the job
    status mediumtext // the status of the job, e.g., error information
);

create table cluster(
    cid varchar(255), // unique cluster id
    name varchar(255), // cluster name
    resources mediumtext, // cluster resource information
    status mediumtext, // cluster status
    customized_data mediumtext // cluster customized data
)
```