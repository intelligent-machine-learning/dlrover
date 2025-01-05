create database dlrover;

use dlrover;

create table job_metrics(
    uid varchar(255) NOT NULL,        
    hyper_params_feature mediumtext,
    job_feature mediumtext,
    dataset_feature mediumtext, 
    model_feature mediumtext,
    job_runtime mediumtext,
    exit_reason varchar(255),
    optimization mediumtext,
    resource mediumtext,
    customized_data mediumtext,
    type varchar(255),
    PRIMARY KEY (uid)
);

create table job(
    uid varchar(255) NOT NULL,
    name varchar(255),
    scenario varchar(255),
    created_at timestamp,
    started_at timestamp,
    finished_at timestamp,
    status mediumtext,
    PRIMARY KEY (uid)
);

create table job_node(
    uid varchar(255) NOT NULL,
    name varchar(255),
    job_uid varchar(255),
    type varchar(255),
    created_at timestamp,
    started_at timestamp,
    finished_at timestamp,
    resource mediumtext,
    status mediumtext,
    customized_data mediumtext,
    PRIMARY KEY (uid)
);

create table cluster(
    uid varchar(255) NOT NULL,
    name varchar(255),
    resources mediumtext,
    status mediumtext,
    customized_data mediumtext,
    PRIMARY KEY (uid)
)

