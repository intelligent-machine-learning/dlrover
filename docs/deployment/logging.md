# Logging Configuration Reference

This article primarily introduces logging configuration when using DLRover.

## Usage

### Logging Level

Support: Both Kubernetes Job and Ray job

User can control the log output level of the DLRover framework through the 
environment variable:
`
DLROVER_LOG_LEVEL
`
with the default level: INFO. 

Note: DLRover logs are output to stderr by default.

### Redirect File

Support: Both Kubernetes Job and Ray job

User can control the default log redirection file of the DLRover framework 
through the environment variable:
`
DLROVER_LOG_ROOT_DIR
`.
The default value is empty, meaning logs will not be redirected to any log file.

Use the following env to control the max size of each rotate file
(minimum: 1MB, default: 200MB):
`
DLROVER_LOG_ROTATE_MAX_BYTES
`

Use the following env to control the backup file number:
(minimum: 1, default: 5):
`
DLROVER_LOG_ROTATE_BACKUP_COUNT
`.

### Training Log

Support: Only Kubernetes Job(torchrun mode)

User can control the default log redirection directory for the training side(training process) 
through the environment variable:
`
DLROVER_LOG_AGENT_DIR
`.
The default value is '/tmp', but there will be no redirect logging because the 
'redirects' and 'tee' is configured as Std.None by default.

When the log-directory is specified, the 'redirects' and 'tee' will be set to 
Std.ALL and the log will be redirect to the specified log-directory.

For example:
```text
# specified base-log-directory: /home/admin/logs/dlrover
# specified agent-log-directory: /home/admin/logs/dlrover/agent
/home/admin/logs/dlrover/
  dlrover.log  # dlrover framework logging
  \ agent  
    \ attempt_0 
      \ 0
        \ std.log
        \ err.log
        \ error.json
      \ 1
      ...
      \ 7 
    \ attempt_1
    ...
```

In addition, users can still use the parameters natively supported by dlrover-run 
(inherited from torchrun), such as --log-dir, --redirects, and --tee, to configure 
the logging settings mentioned above. Please refer to the 
[torch documentation](https://docs.pytorch.org/docs/stable/elastic/agent.html#torch.distributed.elastic.agent.server.WorkerSpec) for details.
