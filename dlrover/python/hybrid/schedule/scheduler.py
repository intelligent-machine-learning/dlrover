from dataclasses import dataclass
from typing import Any, Collection, Dict, Literal, Optional

import ray
from ray.actor import ActorClass, ActorHandle
from ray.util.scheduling_strategies import SchedulingStrategyT
from ray.exceptions import RayActorError

from dlrover.python.hybrid.config import ResourceDesc
from dlrover.python.unified.common.constant import DLWorkloadEnv
from dlrover.python.unified.common.enums import SchedulingStrategyType
from dlrover.python.common.log import default_logger as logger


@dataclass
class Node:
    name: str
    resource: ResourceDesc
    envs: Dict[str, str]
    cls: ActorClass
    options: Dict[str, Any]  # as kwargs for actor
    scheduling_strategy: SchedulingStrategyT = None
    kind: Literal["worker", "master"] = "worker"


class Scheduler:
    def __init__(self, strategy: SchedulingStrategyType) -> None:
        self.strategy = strategy
        self.__pg = None  # Placement group for actors
        self.__actors_cache = {}  # name -> actor mapping

    def create_pgs(self, pgs):
        pass

    def create_nodes(self, nodes: Collection[Node]):
        """Create/Get actors for all nodes."""
        # 0. create placement group if not exists
        """TODO: Create placement group if not exists."""
        # 1. ray create_or_exists actors
        for node in nodes:
            if node.name in self.__actors_cache:
                continue
            runtime_env: dict = {
                "env_vars": node.envs,
            }
            # setup working dir
            if DLWorkloadEnv.WORKING_DIR in node.envs:
                runtime_env["working_dir"] = node.envs[DLWorkloadEnv.WORKING_DIR]

            actor = node.cls.options(
                name=node.name,
                lifetime="detached",
                max_restarts=-1,  # Allow unlimited restarts
                get_if_exists=True,
                num_cpus=node.resource.cpu,
                memory=node.resource.memory,
                num_gpus=node.resource.gpu,
                resources=node.resource.user_defined,
                runtime_env=runtime_env,
                scheduling_strategy=node.scheduling_strategy,
            ).remote(**node.options)
            self.__actors_cache[node.name] = actor
            logger.info(f"Created {node}")

        # 2. Check actors with ping
        result = self.execute(nodes, "status")
        logger.info(f"Actors status: {result}")

    def get_actor(self, name: str, refresh: bool = False):
        if refresh or name not in self.__actors_cache:
            self.__actors_cache[name] = ray.get_actor(name)
        return self.__actors_cache[name]

    @staticmethod
    def _invoke_actor(
        actor: ActorHandle, method_name: str, *args, **kwargs
    ) -> ray.ObjectRef:
        return getattr(actor, method_name).remote(*args, **kwargs)

    def execute(self, nodes: Collection[Node], method_name: str, *args, **kwargs):
        """Execute a method on all nodes."""
        actors = [self.get_actor(node.name) for node in nodes]
        tasks = [
            self._invoke_actor(actor, method_name, *args, **kwargs) for actor in actors
        ]

        results = {}
        waiting = {task: actor.name for task, actor in zip(tasks, nodes)}
        while len(waiting) > 0:
            ready, not_ready = ray.wait(
                list(waiting.keys()), num_returns=len(waiting), timeout=10
            )
            next_waiting = {
                task: waiting[task] for task in not_ready
            }  # Update waiting tasks
            for task in ready:
                actor_name = waiting.pop(task)
                try:
                    result = ray.get(task)
                    results[actor_name] = result
                except RayActorError as e:
                    print(f"Error executing {method_name} on {actor_name}: {e}")
                    actor = self.get_actor(actor_name, refresh=True)
                    task = self._invoke_actor(actor, method_name, *args, **kwargs)
                    next_waiting[task] = actor
                except Exception as e:
                    print(
                        f"Unexpected error executing {method_name} on {actor_name}: {e}"
                    )
                    results[actor_name] = e
            waiting = next_waiting
            if len(waiting) > 0:
                print(f"Waiting for {len(waiting)} tasks to complete {method_name} ...")
        return results

    def execute_one(self, node: Node, method_name: str, *args, **kwargs):
        result = self.execute([node], method_name, *args, **kwargs)
        result = result[node.name]
        if isinstance(result, Exception):
            raise result
        return result

    def cleanup(self, nodes: Collection[Node]):
        """Cleanup resources for all nodes."""
        toKill = [node.name for node in nodes]
        while len(toKill) > 0:
            name = toKill.pop()
            try:
                actor = self.get_actor(name)
                ray.kill(actor, no_restart=True)
            except ValueError:
                # Actor not found, continue
                continue
            except RayActorError:
                self.get_actor(name, refresh=True)
                toKill.append(name)

        for node in nodes:
            self.__actors_cache.pop(node.name, None)
