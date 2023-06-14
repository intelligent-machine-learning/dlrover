def find_modules(model, m_list):
    if isinstance(model, tuple(m_list)):
        return [model]
    res = []
    for mo in model.modules():
        if isinstance(mo, tuple(m_list)):
            res.append(mo)
    return res


# Convert module name in module_list into module type.
# Also ignore any module names that not exist in model.
def to_module_class_by_name(model, module_list):
    module_classes = {}
    for m in module_list:
        if type(m) == str:
            module_classes[m] = None
    unassigned_num = len(module_classes)
    if unassigned_num > 0:
        for m in model.modules():
            if type(m).__name__ in module_classes.keys() and module_classes[type(m).__name__] is None:
                module_classes[type(m).__name__] = type(m)
                unassigned_num -= 1
                if unassigned_num == 0:
                    break
    result = []
    for m in module_list:
        if type(m) == str:
            if module_classes[m] is not None:
                result.append(module_classes[m])
        else:
            result.append(m)
    return type(module_list)(result)
