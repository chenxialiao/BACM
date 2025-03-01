import torch
import torch.nn as nn
from model import ResNets_Adapt, VGGNets_Adapt
import random


def fetch(device, config, encoder_type, finetune_layers, train_output=False):
    adapt = True if finetune_layers == "PA" else False
    if encoder_type == "VGG19":
        encoder = VGGNets_Adapt.VGG("VGG19", adapt=adapt)
        chkpt = torch.load(config[encoder_type], map_location=device)

    elif encoder_type == "Res50":
        encoder = ResNets_Adapt.Resnet(50, drop_ratio=0.5, feat_dim=512, out_h=7, out_w=7, adapt=adapt)
        chkpt = torch.load(config[encoder_type], map_location=device)
    else:
        raise ValueError(f"implementation for {encoder_type} is not ready yet")

    # load state dict
    if adapt:
        load_adapter_state_dict(encoder, chkpt["encoder_state_dict"])
    else:
        encoder.load_state_dict(chkpt["encoder_state_dict"])

    # prepare encoder
    if finetune_layers != "Full":  # if full fine-tuning, no additional setting is needed
        encoder.eval()
        encoder.requires_grad_(False)  # freeze all parameters & set to eval mode
        if finetune_layers == "Partial":
            if encoder_type == "VGG19":
                for name, module in encoder.named_modules():
                    tokens = name.split(".")
                    if len(tokens) > 1 and int(tokens[1]) >= 33:  # train only last 4 conv. layers
                    #if len(tokens) > 1 and int(tokens[1]) >= 46:  # train only last 2 conv. layers
                        module.train()
                        module.requires_grad_(True)
            elif encoder_type == "Res50":
                for name, module in encoder.named_modules():
                    tokens = name.split(".")
                    if ("23" in tokens)  :  # train only last 4 conv. layers
                    # if "23" in tokens:  # train only last 2 conv. layers
                        module.train()
                        module.requires_grad_(True)
            #     fine_tune_layers = [str(i) for i in range(1, 24,  1)]
            #     for name, module in encoder.named_modules():
            #         tokens = name.split(".")
            #     # 检查tokens中的元素是否在我们的微调层数列表中
            #         if any(token in fine_tune_layers for token in tokens) :
            #             module.train()
            #             module.requires_grad_(True)
        else:
            if finetune_layers == "PA":
                for name, param in encoder.named_parameters():
                    tokens = name.split(".")
                    if "conv1x1" in tokens:
                        param.requires_grad = True

            elif finetune_layers == "BN":
                for name, module in encoder.named_modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.requires_grad_(True)
                        module.train()
            elif finetune_layers == "FC":
                for name, module in encoder.named_modules():
                    # 检查模块是否为全连接层（Linear）
                    if isinstance(module, nn.Linear):
                            module.requires_grad_(True)
                            module.train()
    # if finetune_layers == "Random":
    #     encoder.eval()
    #     for param in encoder.parameters():
    #         param.requires_grad_(False)
    #
    #     # 获取模型的所有子模块列表
    #     all_modules = [module for name, module in encoder.named_modules()]
    #
    #     # 随机选择n_random_layers个模块，并设置它们为可训练
    #     random.shuffle(all_modules)
    #     random_modules_to_finetune = all_modules[:n_random_layers]
    #
    #     # 对于每个被选中的模块，启用其参数的梯度计算
    #     for module in random_modules_to_finetune:
    #         module.train()
    #         for param in module.parameters():
    #             param.requires_grad_(True)
    #     if train_output:
    #         encoder.output_layer.requires_grad_(True)
    #         encoder.output_layer.train()

    encoder.to(device)
    return encoder

def load_adapter_state_dict(encoder, state_dict):
    """
    function for loading state_dict for adapter models
    """
    missing_keys, unexpected_keys = encoder.load_state_dict(state_dict, strict=False)
    newdict = {}
    for idx, key in enumerate(encoder.state_dict().keys()):
        if key in missing_keys:
            tokens = key.split(".")
            if "conv3x3" in tokens:
                tokens.remove("conv3x3")
                org_key = ".".join(tokens)
                newdict[key] = state_dict[org_key]
        else:
            newdict[key] = state_dict[key]
    missing_keys, unexpected_keys = encoder.load_state_dict(newdict, strict=False)
    assert unexpected_keys == [], f"there exists left-over keys: {unexpected_keys}"
