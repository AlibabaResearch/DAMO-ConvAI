def ignore_nodes(node_names):
    node_names = [node_name.strip().lower() for node_name in node_names]

    def decorator(func):
        def wrapper(*args, **kwargs):
            new_res = ()
            res = func(*args, **kwargs)
            assert isinstance(res, tuple)
            assert isinstance(res[0], list)
            assert isinstance(node_names, list)
            for element_list in res:
                new_element_list = []
                for element in element_list:
                    save_flag = True
                    for node_name in node_names:
                        if node_name in element:
                            save_flag = False
                            break
                    if save_flag:
                        new_element_list.append(element)
                new_res += (list(set(new_element_list)),)
            return new_res

        return wrapper

    return decorator
