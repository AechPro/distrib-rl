def build_component_factory(
    component_name, builders, arg_transformers, require_list=False, optional=False
):
    def register(key, builder, args_transformer=None):
        builders[key] = builder
        if args_transformer:
            arg_transformers[key] = args_transformer

    def build(config):
        if not optional and ((not config) or len(config) == 0):
            raise AttributeError(f"{component_name} is not optional")

        if require_list and type(config) != list:
            raise AttributeError(f"{component_name} must be list of {component_name}s")

        if type(config) == str:
            return build_individual({config: {}})

        if type(config) == list or len(config) > 1:
            if type(config) == dict:
                return [build_individual({key: config[key]}) for key in config]

            return [build(c) for c in config]

        elif len(config) == 1:
            return build_individual(config)

        return None

    def build_individual(config):
        key = list(config.keys())[0]

        Builder = builders.get(key, None)

        if not Builder:
            raise AttributeError(f"No {component_name} found for key '{key}'")

        kwargs = config[key]

        if key in arg_transformers:
            kwargs = arg_transformers[key](**kwargs)

        return Builder(**kwargs)

    return register, build
