# Failover Diagnosis 使用文档

## 概述

Failover Diagnosis 功能允许在 DLRover 的 Ray 架构中，当检测到 actor 失败时，自动收集目标 actor 的日志，并通过自定义的诊断扩展类分析日志，从而决定如何执行容错逻辑。

## 功能特性

1. **API 参数控制**: 通过 `JobConfig` 启用和配置诊断功能
2. **自动日志收集**: 使用 Ray State API 通过 actor name 获取日志
3. **可扩展诊断逻辑**: 支持自定义诊断扩展类
4. **灵活的容错决策**: 根据诊断结果决定继续容错、停止任务或跳过特定 actor

## 配置参数

在 `JobConfig` 中添加以下参数：

```python
from dlrover.python.unified.common.config import JobConfig

config = JobConfig(
    job_name="my_job",
    dl_config={...},
    
    # 启用诊断功能
    enable_failover_diagnosis=True,
    
    # 指定自定义诊断扩展类（可选）
    diagnostic_extension_class="my_package.my_module.MyDiagnosticExtension",
    
    # 获取日志的行数（默认 1000）
    diagnostic_log_lines=2000,
)
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_failover_diagnosis` | bool | False | 是否启用诊断功能 |
| `diagnostic_extension_class` | str | "" | 自定义诊断扩展类的完整路径，格式为 `module.submodule.ClassName` |
| `diagnostic_log_lines` | int | 1000 | 从每个失败 actor 收集的日志行数 |

## 诊断动作类型

诊断扩展类可以返回以下动作类型：

```python
from dlrover.python.unified.api import DiagnosticAction

# 继续执行正常容错逻辑（默认）
DiagnosticAction.CONTINUE_FAILOVER

# 停止任务，不执行容错
DiagnosticAction.STOP_JOB

# 跳过当前 actor 的容错
DiagnosticAction.SKIP_ACTOR

# 自定义动作（由扩展类处理）
DiagnosticAction.CUSTOM
```

## 自定义诊断扩展

### 基础示例

```python
from dlrover.python.unified.api import (
    DiagnosticExtension,
    DiagnosticResult,
    ActorLogContext,
    DiagnosticAction,
)

class MyDiagnosticExtension(DiagnosticExtension):
    """自定义诊断扩展示例"""
    
    async def diagnose(
        self,
        actor_logs: List[ActorLogContext],
        failure_reason: str = "",
    ) -> DiagnosticResult:
        """
        执行诊断分析
        
        Args:
            actor_logs: 包含失败 actor 日志信息的列表
            failure_reason: 失败原因描述
            
        Returns:
            DiagnosticResult: 诊断结果，包含建议的动作
        """
        for ctx in actor_logs:
            # 分析日志内容
            if "OutOfMemoryError" in ctx.log_content:
                return DiagnosticResult(
                    action=DiagnosticAction.STOP_JOB,
                    reason=f"检测到 OOM 错误: {ctx.actor_name}",
                    metadata={"error_type": "OOM", "actor": ctx.actor_name}
                )
            
            if "CUDA out of memory" in ctx.log_content:
                return DiagnosticResult(
                    action=DiagnosticAction.STOP_JOB,
                    reason="检测到 CUDA OOM 错误",
                    metadata={"error_type": "CUDA_OOM"}
                )
        
        # 默认继续执行容错
        return DiagnosticResult(
            action=DiagnosticAction.CONTINUE_FAILOVER,
            reason="未发现致命错误，继续容错"
        )
```

### ActorLogContext 结构

```python
@dataclass
class ActorLogContext:
    actor_name: str          # Actor 名称
    log_content: str         # 日志内容
    log_lines: int           # 日志行数
    actor_metadata: dict     # 额外的元数据（如 actor_id, state, pid 等）
    failure_reason: str      # 失败原因
```

## 注册诊断扩展

有两种方式注册自定义诊断扩展：

### 方式一：通过配置参数（推荐）

```python
config = JobConfig(
    ...
    diagnostic_extension_class="my_package.diagnostic.MyDiagnosticExtension",
)
```

### 方式二：通过 Entry Point

在 `setup.py` 中添加 entry point：

```python
setup(
    ...
    entry_points={
        'dlrover.unified.diagnostic_extension': [
            'my_ext = my_package.diagnostic:MyDiagnosticExtension',
        ],
    },
)
```

## 完整使用示例

```python
from dlrover.python.unified.common.config import JobConfig
from dlrover.python.unified.api import (
    DiagnosticExtension,
    DiagnosticResult,
    ActorLogContext,
    DiagnosticAction,
)

# 1. 定义自定义诊断扩展
class OOMDiagnosticExtension(DiagnosticExtension):
    """检测 OOM 错误的诊断扩展"""
    
    async def diagnose(self, actor_logs, failure_reason=""):
        for ctx in actor_logs:
            log = ctx.log_content.lower()
            if "oom" in log or "out of memory" in log or "cudaerror":
                return DiagnosticResult(
                    action=DiagnosticAction.STOP_JOB,
                    reason=f"检测到内存错误: {ctx.actor_name}",
                    metadata={
                        "error_type": "OOM",
                        "actor_name": ctx.actor_name,
                        "log_snippet": ctx.log_content[:500]
                    }
                )
        
        return DiagnosticResult(
            action=DiagnosticAction.CONTINUE_FAILOVER,
            reason="未检测到 OOM 错误，继续容错"
        )

# 2. 配置任务
config = JobConfig(
    job_name="training_job",
    dl_config={
        "workloads": {
            "trainer": {
                "total": 4,
                "resource": {"cpu": 4, "gpu": 1},
                "entry_point": "train.main",
            }
        }
    },
    enable_failover_diagnosis=True,
    diagnostic_extension_class="__main__.OOMDiagnosticExtension",  # 如果在主模块中定义
    diagnostic_log_lines=1500,
)

# 3. 提交任务
from dlrover.python.unified.driver.main import submit_job
submit_job(config)
```

## 工作原理

当诊断功能启用时，以下流程会在 actor 失败时自动执行：

1. **检测失败**: Master 检测到 actor restart 或 failure report
2. **收集日志**: 使用 Ray State API (`list_actors` + `get_log`) 通过 actor name 获取日志
3. **执行诊断**: 调用诊断扩展类的 `diagnose` 方法分析日志
4. **决策执行**: 根据诊断结果执行相应动作：
   - `CONTINUE_FAILOVER`: 继续正常容错流程
   - `STOP_JOB`: 停止任务
   - `SKIP_ACTOR`: 跳过当前 actor
   - `CUSTOM`: 执行自定义逻辑

## 注意事项

1. **日志获取限制**: Ray State API 只能从存活的节点获取日志，如果 actor 所在节点已下线，可能无法获取完整日志
2. **性能影响**: 诊断过程会增加 failover 的延迟，建议保持诊断逻辑轻量
3. **错误处理**: 诊断过程中的异常会被捕获，默认会继续执行容错

## 调试

启用 debug 日志可以查看诊断过程的详细信息：

```python
import logging
logging.getLogger("dlrover").setLevel(logging.DEBUG)
```

日志中会包含：
- 诊断启动信息
- 日志收集状态
- 诊断结果和决策原因
