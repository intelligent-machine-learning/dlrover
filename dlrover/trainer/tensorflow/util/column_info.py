from dlrover.trainer.tensorflow.util.common_util import add_prop

class Property(object):
    """Property descriptor"""

    def __init__(self, name, property_name, property_doc, default=None):
        self._name = name
        self._names = property_name
        self._default = default
        self._doc = property_doc

    def __get__(self, obj, objtype):
        return obj.__dict__.get(self._name, self._default)

    def __set__(self, obj, value):
        if obj is None:
            return self
        for name in self._names:
            obj.__dict__[name] = value

    @property
    def __doc__(self):
        return self._doc

@add_prop(
    ("dtype", "data type"),
    ("name", "feature_name"),
    ("is_label", "whether it is a label or a feature"),
)
class Column(object):
    @property
    def keys(self):
        """Get keys of a `Column`"""
        return self.added_prop

    def __str__(self):
        result = [k + "=" + str(getattr(self, k)) for k in self.keys]
        return "{" + ";".join(result) + "}"

    def set_default(self):
        """Set default value of column fields"""
        if self.is_sparse:
            self.separator = self.separator or "\\u0001"
            self.group_separator = self.group_separator or "\\u0002"
        else:
            self.separator = self.separator or ","

        self.group = common_util.get_group_num(self.group)
        self.shape = common_util.get_shape(
            self.name, self.is_sparse, self.shape, self.group
        )

    __repr__ = __str__