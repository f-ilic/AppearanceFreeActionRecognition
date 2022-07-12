import pprint
import termplotlib as tpl

def print_learnable_params(model):
    print("-----------------------------------------------------------")
    print("Printing parameters of model which have requires_grad==True")
    print("-----------------------------------------------------------")


    learnable_names = []
    learnable_num_param = []

    frozen_names = []
    frozen_num_params = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            learnable_names.append(name)
            learnable_num_param.append(param.numel())
        if param.requires_grad == False:
            frozen_names.append(name)
            frozen_num_params.append(param.numel())

    num_learnable_params = sum(learnable_num_param)
    num_frozen_params = sum(frozen_num_params)
    fig = tpl.figure()
    fig.barh(learnable_num_param, learnable_names)
    fig.show()
    print("-----------------------------------------------------------")
    print(f"Total Number parameters: {(num_learnable_params+num_frozen_params):.2E}")
    print(f"Total Number of learnable parameters: {num_learnable_params:.2E}")
    print(f"Total Number of frozen parameters: {num_frozen_params:.2E}")
    print("-----------------------------------------------------------\n\n\n")


def print_config_options(cfg):
    print("-----------------------------------------------------------")
    print("Printing configuration options")
    print("-----------------------------------------------------------")
    pprint.pprint(cfg)
    print("-----------------------------------------------------------\n\n\n")

def print_data_augmentation_transform(transform):
    print("-----------------------------------------------------------")
    print("Printing Data Augmentation Transform")
    print("-----------------------------------------------------------")
    pprint.pprint(transform.__dict__)
    print("-----------------------------------------------------------\n\n\n")