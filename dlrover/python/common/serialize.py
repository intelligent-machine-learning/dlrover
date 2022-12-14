import json


class JsonSerializable(object):
    def toJSON(self, indent=None):
        return json.dumps(
            self,
            default=lambda o: o.__dict__,
            sort_keys=True,
            indent=indent,
        )
