# Database Table Design

```mysql
create table job_metrics(
    uid varchar(255) NOT NULL, // job unique id         
    hyper_params_feature mediumtext, // the feature of hyper parameters
    job_feature mediumtext, // the feature of the job
    dataset_feature mediumtext, // the feature of the training dataset
    model_feature mediumtext, // the feature of the training model
    job_runtime mediumtext, // job runtime information
    exit_reason varchar(255), // the exit reason of the job
    optimization mediumtext, // the optimization information of the job
    resource mediumtext, // the resources of the job
    customized_data mediumtext, // job metrics custimized data
    type varchar(255), // indicate the data type
    PRIMARY KEY (uid)
);

create table job(
    uid varchar(255) NOT NULL, // job unique id
    name varchar(255), // job name
    scenario varchar(255), // job scenario
    create_at timestamp, // job create timestamp
    started_at timestamp, // job start timestamp
    finished_at timestamp, // job finish timestamp
    status mediumtext, // the status of the job, e.g., error information
    PRIMARY KEY (uid)
);

create table job_node(
    uid varchar(255) NOT NULL, // job node unique id
    name varchar(255), // job node name
    job_uid varchar(255), // the job uid
    type varchar(255), // the type of the job node, e.g., ps, worker
    create_at timestamp, // job node create timestamp
    started_at timestamp, // job node start timestamp
    finished_at timestamp, // job node finish timestamp
    resource mediumtext, // the resources of the job node
    status mediumtext, // the status of the job node
    customized_data mediumtext, // job node custimized data
    PRIMARY KEY (uid)
);

create table cluster(
    uid varchar(255) NOT NULL, // unique cluster id
    name varchar(255), // cluster name
    resources mediumtext, // cluster resource information
    status mediumtext, // cluster status
    customized_data mediumtext, // cluster customized data
    PRIMARY KEY (uid)
)
```
