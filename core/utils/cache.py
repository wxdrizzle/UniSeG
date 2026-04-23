def cache_res(*names):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # if the result is already cached, directly return it
            result = tuple(self.cfg.var.obj_model.recorder.get(name, None) for name in names)
            exists = (value is not None for value in result)
            if all(exists):
                if len(result) == 1:
                    return result[0]
                return result
            else:
                assert not any(exists), f'Value caching status: {exists}'

            # execute the original function and cache the result
            real_result = func(self, *args, **kwargs)
            if not isinstance(real_result, tuple):
                real_result = (real_result, )
            for name, value in zip(names, real_result):
                self.cfg.var.obj_model.recorder[name] = value
            if len(real_result) == 1:
                return real_result[0]
            return real_result

        return wrapper

    return decorator


def cache_res_by_domain(*names):
    def decorator(func):
        def wrapper(self, domain, *args, **kwargs):
            names_ = [f'{name}_{domain}' for name in names]
            # if the result is already cached, directly return it
            result = tuple(self.cfg.var.obj_model.recorder.get(name, None) for name in names_)
            exists = (value is not None for value in result)
            if all(exists):
                if len(result) == 1:
                    return result[0]
                return result
            else:
                assert not any(exists), f'Value caching status: {exists}'

            # execute the original function and cache the result
            real_result = func(self, domain, *args, **kwargs)
            if not isinstance(real_result, tuple):
                real_result = (real_result, )
            for name, value in zip(names_, real_result):
                self.cfg.var.obj_model.recorder[name] = value
            if len(real_result) == 1:
                return real_result[0]
            return real_result

        return wrapper

    return decorator


def cache_res_by_domain_lv(*names):
    def decorator(func):
        def wrapper(self, domain, lv, *args, **kwargs):
            names_ = [f'{name}_{domain}_{lv}' for name in names]
            # if the result is already cached, directly return it
            result = tuple(self.cfg.var.obj_model.recorder.get(name, None) for name in names_)
            exists = (value is not None for value in result)
            if all(exists):
                if len(result) == 1:
                    return result[0]
                return result
            else:
                assert not any(exists), f'Value caching status: {exists}'

            # execute the original function and cache the result
            real_result = func(self, domain, lv, *args, **kwargs)
            if not isinstance(real_result, tuple):
                real_result = (real_result, )
            for name, value in zip(names_, real_result):
                self.cfg.var.obj_model.recorder[name] = value
            if len(real_result) == 1:
                return real_result[0]
            return real_result

        return wrapper

    return decorator
