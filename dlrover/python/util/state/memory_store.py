
class MemoryStore:
    """
        actor_names = []
    """
    def __init__(self, state_manager, jobname, namespace):
        self.__data_map = {}

    def get(self, key, default_value=None):
        return self.__data_map.get(key, default_value)

    def put(self, key, value):
        self.__data_map[key] = value

    def delete(self, key):
        del self.__data_map[key]

    def add_actor_name(self, actor_name):
        actor_names = self.get("actor_names",[])
        actor_names.append(actor_name)
        self.put("actor_names",actor_names)
        return True 

    def remove_actor_name(self, actor_name):
        actor_names = self.get("actor_names",[])
        if actor_name in actor_names:
            actor_names.remove(actor_name)
 
        self.put("actor_names",actor_names)
        return True 

    def do_checkpoint(self):
        pass

    def load_checkpoint(self):
        pass 



# actor_names = ["PsActor_0","PsActor_1","Worker-0|4","Worker-1|4","Worker-2|4","Worker-3|4"]
# RayClient
"""
负责查询Actor的状态
申请资源规格
增加删除Actor
"""
