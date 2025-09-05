from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class TaskType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
    NONE: _ClassVar[TaskType]
    TRAINING: _ClassVar[TaskType]
    EVALUATION: _ClassVar[TaskType]
    PREDICTION: _ClassVar[TaskType]
    WAIT: _ClassVar[TaskType]
    TRAIN_END_CALLBACK: _ClassVar[TaskType]

NONE: TaskType
TRAINING: TaskType
EVALUATION: TaskType
PREDICTION: TaskType
WAIT: TaskType
TRAIN_END_CALLBACK: TaskType

class Response(_message.Message):
    __slots__ = ["success", "reason"]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    success: bool
    reason: str
    def __init__(
        self, success: bool = ..., reason: _Optional[str] = ...
    ) -> None: ...

class Message(_message.Message):
    __slots__ = ["node_id", "node_type", "data", "job_uid"]
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    NODE_TYPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    JOB_UID_FIELD_NUMBER: _ClassVar[int]
    node_id: int
    node_type: str
    data: bytes
    job_uid: str
    def __init__(
        self,
        node_id: _Optional[int] = ...,
        node_type: _Optional[str] = ...,
        data: _Optional[bytes] = ...,
        job_uid: _Optional[str] = ...,
    ) -> None: ...
