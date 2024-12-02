import yaml
from argparse import Namespace


class DictionaryDot:
    """
    字典点方法
    """

    def __init__(self, args_obj=None):
        self.args_obj = args_obj

    def to_dot(self):
        """
        把字典方法转为点方法
        """
        if isinstance(self.args_obj, dict):
            for key1, value1 in self.args_obj.items():
                if isinstance(value1, dict):
                    for key2, value2 in value1.items():
                        if isinstance(value2, dict):
                            ns2 = Namespace(**value2)
                            value1[key2] = ns2
                    ns1 = Namespace(**value1)
                    self.args_obj[key1] = ns1
            ns = Namespace(**self.args_obj)
            return ns
        else:
            raise ValueError('DictionaryDot.to_dot Error')

    def to_dict(self):
        """
        把点方法转为字典方法
        """
        args_dict = vars(self.args_obj)
        for item in args_dict.items():
            if isinstance(item[1], Namespace):
                args_dict[item[0]] = vars(item[1])
        return args_dict

    @staticmethod
    def update(args, yaml_dir, data_style):
        args_dct = vars(args)
        with open(yaml_dir, 'r', encoding='utf-8') as f:
            yaml_args = yaml.safe_load(f)
        args_dct.update({'train': Namespace(**yaml_args['train'])})
        args_dct.update({'model': Namespace(**yaml_args['model'])})
        # if args is not None:
        #     for _, arg in args.items():
        #         for key, value in arg.items():
        #             args_dct.update({key: value})
        args_dct['loader'].data_style = data_style
        args_dct['loader'].data_label = data_style
        configs = Namespace(**args_dct)
        return configs
