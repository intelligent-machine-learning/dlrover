# Enhanced Training Events Exporter Design Document

## Overview

This document describes the design of a training events exporter mechanism integrated with both the training process and the DLRover agent. The primary goal is to capture, trace, and analyze detailed training event data to aid in debugging, performance analysis, and system monitoring.

## Motivation

Capturing training events enhances observability and traceability of the training workflow. With detailed event logs, developers can:

- Analyze training dynamics and identify performance bottlenecks.
- Quickly pinpoint issues during debugging.
- Optimize resource allocation based on insights from the event data.
- Seamlessly integrate with monitoring tools to obtain real-time system health information.

Events are exported in a human-readable text format (with optional JSON support), ensuring that the data is easily collected and analyzed.

## Design Overview

The implementation is modular, promoting extensibility and maintainability. The system comprises the following key components:

### 1. Configuration Management (`config.py`)

- **Purpose:**
  - Parse configuration from JSON files and environment variables.
  - Set up logging and configure default parameters such as asynchronous export, maximum queue size, file output directory, and text formatting options.

### 2. Event Definition (`event.py`)

- **Purpose:**

  - Define an `Event` class encapsulating event properties like timestamp, event name, type, target, and additional content.
  - Provide helper methods to create events of different types.

- **Event Types:**

  - **BEGIN:** Marks the start of a duration-based event.
  - **END:** Marks the end of the duration event.
  - **INSTANT:** Represents a one-off or instantaneous event.

The BEGIN and END events form a pair, they are used to indicate a span of time. the event id is the same for the pair.

- **Core Properties:**
  - `event_id`: A unique identifier for instant and duration events.
  - `event_time`: ISO8601 formatted timestamp.
  - `target`: The system or process the event is associated with.
  - `name`: A descriptive event name.
  - `event_type`: The type indicator (BEGIN, END, or INSTANT).
  - `content`: A dictionary holding event-specific details.

### 3. Event Emission (`emitter.py`)

- **Purpose:**
  - Encapsulate the logic for emitting events using the `EventEmitter` class.
  - Provide convenient methods such as `instant`, `begin`, and `end` to emit events.
  - Implement a `DurationSpan` context manager to manage duration events effortlessly.

### 4. Exporter Implementation (`exporter.py`)

- **Purpose:**

  - Define a generic `EventExporter` interface to support various export targets.

- **Concrete Exporters:**

  - **TextFileExporter:** Exports events to a file in a one-line log format (or in JSON format).
  - **ConsoleExporter:** Outputs events directly to the console.

- **Async Exporting:**

  - **AsyncExporter:** Wraps another exporter to enable asynchronous event exporting using a dedicated worker thread.

- **Formatting:**

  - The default log format is:

    ```
    [event_time] [event_id] [target] [name] [event_type] content
    ```

  - JSON formatting is also supported via the `JsonFormatter`.

### 5. Error Handling (`error_handler.py`)

- **Purpose:**
  - Handle exceptions and system signals.
  - Capture uncaught exceptions and system signal events, logging these incidents as error events to assist in debugging.

### 6. Predefined Events (`predefined/`)

- **Purpose:**

  - Standardize event generation across various components of the DLRover ecosystem.
  - Provide predefined schemas for common events in modules such as `DLRoverAgentEvent`, `DLRoverMasterEvent`, and `TrainerProcess`.

- **Predefined Event Examples:**

| Target             | Event Name         | Description                                                     |
| ------------------ | ------------------ | --------------------------------------------------------------- |
| TrainerProcess     | `#start`           | Marks the start of the training process.                        |
| TrainerProcess     | `#init`            | Indicates the initialization phase of training.                 |
| TrainerProcess     | `#load_dataset`    | Event for the dataset loading step.                             |
| TrainerProcess     | `#load_ckpt`       | Represents the checkpoint loading phase.                        |
| TrainerProcess     | `#persist_ckpt`    | Logs the persistence of checkpoints in an asynchronous process. |
| TrainerProcess     | `#finish`          | Denotes the end of the training process.                        |
| TrainerProcess     | `#init_end`        | Signifies the conclusion of the initialization phase.           |
| TrainerProcess     | `#train`           | Represents the training phase.                                  |
| TrainerProcess     | `#epoch`           | Records an epoch event within the training process.             |
| TrainerProcess     | `#step`            | Indicates a step event in the training loop.                    |
| TrainerProcess     | `#substep`         | Denotes a substep within a larger training step.                |
| TrainerProcess     | `#evaluate`        | Marks the evaluation phase of training.                         |
| TrainerProcess     | `#predict`         | Denotes the prediction phase.                                   |
| TrainerProcess     | `#predict_step`    | Represents an individual prediction step.                       |
| TrainerProcess     | `#save`            | Logs checkpoint saving operations.                              |
| TrainerProcess     | `#log`             | Used for general information logging.                           |
| DLRoverAgentEvent  | `#start`           | Marks the start event of the DLRover agent.                     |
| DLRoverAgentEvent  | `#node_join`       | Logs when a node joins the agent ecosystem.                     |
| DLRoverAgentEvent  | `#rendezvous`      | Denotes the rendezvous process within the agent.                |
| DLRoverAgentEvent  | `#network_check`   | Checks network connectivity within the agent environment.       |
| DLRoverAgentEvent  | `#elastic_train`   | Initiates elastic training procedures.                          |
| DLRoverAgentEvent  | `#process_restart` | Indicates a process restart event.                              |
| DLRoverAgentEvent  | `#exit`            | Marks the termination event of the agent.                       |
| DLRoverMasterEvent | `#pod_create`      | Indicates a new pod creation by the master.                     |
| DLRoverMasterEvent | `#pod_change`      | Denotes a change in the pod configuration.                      |
| DLRoverMasterEvent | `#pod_relaunch`    | Logs events for pod relaunch operations.                        |
| DLRoverMasterEvent | `#fault_detect`    | Represents fault detection events by the master.                |
| DLRoverMasterEvent | `#repair`          | Logs repair actions executed by the master.                     |

## Event Emission and Usage

Any process that needs to emit events should inherit from the `Process` class, which provides methods such as:

- `instant()`: Immediately emit an instantaneous event.
- `duration()`: Returns a context manager to generate a duration event.
- `custom_duration()`: Allows subclasses to define custom duration events.
- Additionally, helper methods like `error()` and `info()` are available for logging error or informational messages.

**Usage Examples:**

```python
# Using a context manager for duration event
with process.duration("some duration"):
    # Perform training operations

# Manually controlling the duration event
mydur = process.duration("custom duration")
mydur.begin()
# ... perform operations ...
mydur.end()
# Or for conditional status:
mydur.success()
mydur.fail("Error encountered during processing")

# Including extra arguments in the event content
with process.duration("detailed duration") as dur:
    dur.extra_args(key1="value1", key2="value2")
```

## Summary

The enhanced design provides a robust and flexible framework for exporting training events. It significantly improves observability, debugging capabilities, and seamless integration with monitoring tools, all while maintaining a modular and extendable architecture.
