from utils.dictionary_dot import DictionaryDot

class Constants:
    def __init__(self, dataset):
        self._dataset = dataset

    def get_learning_rate(self):
        learning_rate = {
            'UCR': 0.006,
            'SWaT': 0.008,
            'NAB': 0.009,
            'MBA': 0.001,
            'SMAP': 0.0008,
            'MSL': 0.002,
            'SMD': 0.001,
            'WADI': 0.0001,
            'synthetic': 0.0001,
            'PSM': 0.001,
            'MSDS': 0.001,
            'NIPS_TS_GECCO': 0.008,
            'NIPS_TS_CCard': 0.008,
            'NIPS_TS_Swan': 0.008,
            'NIPS_TS_Syn_Mulvar': 0.008,
            'AIOps': 0.008,
        }
        lr = learning_rate[self._dataset]
        return lr

    @staticmethod
    def get_dataset_info(dataset):
        if dataset == 'MSL':
            dataset_info = {
                'datasets': ['MSL'],
                # 'files': [
                #     'C-1_', 'C-2_', 'D-14_', 'D-15_', 'D-16_', 'F-4_', 'F-5_', 'F-7_', 'F-8_',
                #     'M-1_', 'M-2_', 'M-3_', 'M-4_', 'M-5_', 'M-6_', 'M-7_', 'P-10_', 'P-11_',
                #     'P-14_', 'P-15_', 'S-2_', 'T-4_', 'T-5_', 'T-8_', 'T-9_', 'T-12_', 'T-13_'
                # ]
                'files':['C-1_']
            }
        elif dataset == 'SMAP':
            dataset_info = {
                'datasets': ['SMAP'],
                # 'files': [
                #     'A-1_', 'A-2_', 'A-3_', 'A-4_', 'A-5_', 'A-6_', 'A-7_', 'A-8_', 'A-9_',
                #     'B-1_', 'D-1_', 'D-2_', 'D-3_', 'D-4_', 'D-5_', 'D-6_', 'D-7_', 'D-8_',
                #     'D-9_', 'D-11_', 'D-12_', 'D-13_', 'E-1_', 'E-2_', 'E-3_', 'E-4_', 'E-5_',
                #     'E-6_', 'E-7_', 'E-8_', 'E-9_', 'E-10_', 'E-11_', 'E-12_', 'E-13_', 'F-1_',
                #     'F-2_', 'F-3_', 'G-1_', 'G-2_', 'G-3_', 'G-4_', 'G-6_', 'G-7_', 'P-1_',
                #     'P-2_', 'P-3_', 'P-4_', 'P-7_', 'R-1_', 'S-1_', 'T-1_', 'T-2_', 'T-3_'
                # ]
                'files':['A-1_']
            }
        elif dataset == 'NIPS':
            dataset_info = {
                'datasets': ['NIPS_TS_CCard', 'NIPS_TS_Swan', 'NIPS_TS_Syn_Mulvar', 'NIPS_TS_GECCO'],
                'files': ['']
            }
        elif dataset == 'PSM':
            dataset_info = {
                'datasets': ['PSM'],
                'files': ['']
            }
        elif dataset == 'WADI':
            dataset_info = {
                'datasets': ['WADI'],
                'files': ['']
            }
        elif dataset == 'SWaT':
            dataset_info = {
                'datasets': ['SWaT'],
                'files': ['']
            }
        elif dataset == 'SMD':
            dataset_info = {
                'datasets': ['SMD'],
                # 'files': [
                #     'machine-1-1_', 'machine-1-2_', 'machine-1-3_', 'machine-1-4_', 'machine-1-5_', 'machine-1-6_',
                #     'machine-1-7_', 'machine-1-8_',
                #     'machine-2-1_', 'machine-2-2_', 'machine-2-3_', 'machine-2-4_', 'machine-2-5_', 'machine-2-6_',
                #     'machine-2-7_', 'machine-2-8_', 'machine-2-9_',
                #     'machine-3-1_', 'machine-3-2_', 'machine-3-3_', 'machine-3-4_', 'machine-3-5_', 'machine-3-6_',
                #     'machine-3-7_', 'machine-3-8_', 'machine-3-9_', 'machine-3-10_', 'machine-3-11_'
                # ]
                'files': ['machine-1-1_']
            }
        elif dataset == 'AIOps':
            dataset_info = {
                'datasets': ['AIOps'],
                'files': ['instance14_', 'instance15_', 'instance23_', 'instance38_', 'instance39_', 'instance44_']
            }
        else:
            raise ValueError('Unknown dataset: {}'.format(dataset))
        dataset_info = DictionaryDot(args_obj=dataset_info).to_dot()
        return dataset_info
