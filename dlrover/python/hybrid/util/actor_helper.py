import asyncio
from functools import partial
from typing import Dict, List, Optional, TypeVar

import ray
from ray.actor import ActorHandle
from ray.exceptions import RayActorError

from dlrover.python.common.log import default_logger as logger

__actors_cache: Dict[str, ActorHandle] = {}


def get_actor_with_cache(name: str, refresh: bool = False):
    """Get a Ray actor by its name, optionally refreshing the cache."""
    # It's safe without a lock, as actors are identified by their names,
    if refresh or name not in __actors_cache:
        try:
            __actors_cache[name] = ray.get_actor(name)
        except Exception as e:
            logger.error(f"Error getting actor {name}: {e}")
            raise
    return __actors_cache[name]


def __invoke_actor(
    actor: ActorHandle, method_name: str, *args, **kwargs
) -> ray.ObjectRef:
    """Invoke a method on a Ray actor."""
    try:
        return getattr(actor, method_name).remote(*args, **kwargs)
    except AttributeError as e:
        logger.error(f"Method {method_name} not found on actor {actor}: {e}")
        return ray.put(None)


def invoke_actors(actors: List[str], method_name: str, *args, **kwargs):
    """Execute a method on all nodes."""
    tasks = [
        __invoke_actor(
            get_actor_with_cache(name), method_name, *args, **kwargs
        )
        for name in actors
    ]

    results: list = [None] * len(actors)
    waiting = {task: task_id for task_id, task in enumerate(tasks)}
    while len(waiting) > 0:
        ready, not_ready = ray.wait(
            list(waiting.keys()), num_returns=len(waiting), timeout=10
        )
        next_waiting = {task: waiting[task] for task in not_ready}
        for task in ready:
            task_id = waiting.pop(task)
            try:
                result = ray.get(task)
                results[task_id] = result
            except RayActorError as e:
                logger.error(
                    f"Error executing {method_name} on {actors[task_id]}: {e}"
                )
                actor = get_actor_with_cache(actors[task_id], refresh=True)
                task = __invoke_actor(actor, method_name, *args, **kwargs)
                next_waiting[task] = task_id
            except Exception as e:
                logger.error(
                    f"Unexpected error executing {method_name} on {actors[task_id]}: {e}"
                )
                results[task_id] = e
        waiting = next_waiting
        if len(waiting) > 0:
            logger.info(
                f"Waiting for {len(waiting)} tasks to complete {method_name} ..."
            )
    return results


def invoke_actor(actor_name: str, method_name: str, *args, **kwargs):
    """Call a method on a Ray actor by its name."""
    return invoke_actors([actor_name], method_name, *args, **kwargs)[0]


def kill_actors(actors: List[str]):
    """Kill Ray actors by their names."""
    logger.info(f"Killing actors: {actors}")
    toKill = actors.copy()
    while len(toKill) > 0:
        name = toKill.pop()
        try:
            actor = get_actor_with_cache(name)
            ray.kill(actor, no_restart=True)
        except ValueError:
            # Actor not found, continue
            continue
        except RayActorError:
            get_actor_with_cache(name, refresh=True)
            toKill.append(name)

    for node in actors:
        __actors_cache.pop(node, None)


T = TypeVar("T")


class ActorProxy:
    def __init__(self, actor: str, cls: Optional[type]):
        self.actor = actor
        get_actor_with_cache(actor)  # warmup actor
        # Optionally filter methods if a class is provided
        if cls is not None:
            self._methods = {
                method for method in dir(cls) if not method.startswith("__")
            }
        else:
            self._methods = None

    def __getattr__(self, name):
        if self._methods is not None and name not in self._methods:
            raise AttributeError(
                f"Method {name} not found in actor {self.actor}."
            )
        return partial(invoke_actor, self.actor, name)

    @staticmethod
    def wrap(actor_name: str, cls: Optional[type[T]] = None) -> "T":
        """Wraps the actor proxy to return an instance of the class."""
        return ActorProxy(actor_name, cls)  # type: ignore


async def invoke_actor_async(
    actor_name: str, method_name: str, *args, **kwargs
):
    """Call a method on a Ray actor by its name."""
    actor = get_actor_with_cache(actor_name)
    while True:
        try:
            res: ray.ObjectRef = getattr(actor, method_name).remote(
                *args, **kwargs
            )
            return await res
        except RayActorError as e:
            print(f"Error executing {method_name} on {actor_name}: {e}")
            actor = get_actor_with_cache(actor_name, refresh=True)
            continue  # Retry with the refreshed actor
        except Exception as e:
            print(
                f"Unexpected error executing {method_name} on {actor_name}: {e}"
            )
            return e


async def invoke_actors_async(
    actors: List[str], method_name: str, *args, **kwargs
):
    return asyncio.gather(
        *[
            invoke_actor_async(actor, method_name, *args, **kwargs)
            for actor in actors
        ]
    )
