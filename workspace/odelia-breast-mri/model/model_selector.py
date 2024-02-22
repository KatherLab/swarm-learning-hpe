from models import ResNet, VisionTransformer, EfficientNet, EfficientNet3D, EfficientNet3Db7, DenseNet121, UNet3D


def select_model(model_name):
    print('Using model:', model_name)

    # Define ResNet layer configurations
    resnet_layers = {
        'ResNet18': [2, 2, 2, 2],
        'ResNet34': [3, 4, 6, 3],
        'ResNet50': [3, 4, 6, 3],
        'ResNet101': [3, 4, 23, 3],
        'ResNet152': [3, 8, 36, 3],
    }

    if model_name in resnet_layers:
        layers = resnet_layers[model_name]
        model = ResNet(in_ch=1, out_ch=1, spatial_dims=3, layers=layers)
    elif model_name in ['efficientnet_l1', 'efficientnet_l2', 'efficientnet_b4', 'efficientnet_b7']:
        model = EfficientNet(model_name=model_name, in_ch=1, out_ch=1, spatial_dims=3)
    elif model_name.startswith('EfficientNet3D'):
        # Define EfficientNet3D configurations based on model_name
        blocks_args_str = {
            'EfficientNet3Db0': [
            "r1_k3_s11_e1_i32_o16_se0.25",
            "r2_k3_s22_e6_i16_o24_se0.25",
            "r2_k5_s22_e6_i24_o40_se0.25",
            "r3_k3_s22_e6_i40_o80_se0.25",
            "r3_k5_s11_e6_i80_o112_se0.25",
            "r4_k5_s22_e6_i112_o192_se0.25",
            "r1_k3_s11_e6_i192_o320_se0.25"],
            'EfficientNet3Db4': [
            "r1_k3_s11_e1_i48_o24_se0.25",
            "r3_k3_s22_e6_i24_o32_se0.25",
            "r3_k5_s22_e6_i32_o56_se0.25",
            "r4_k3_s22_e6_i56_o112_se0.25",
            "r4_k5_s11_e6_i112_o160_se0.25",
            "r5_k5_s22_e6_i160_o272_se0.25",
            "r2_k3_s11_e6_i272_o448_se0.25"],
            'EfficientNet3Db7': [
            "r1_k3_s11_e1_i32_o32_se0.25",
            "r4_k3_s22_e6_i32_o48_se0.25",
            "r4_k5_s22_e6_i48_o80_se0.25",
            "r4_k3_s22_e6_i80_o160_se0.25",
            "r6_k5_s11_e6_i160_o256_se0.25",
            "r6_k5_s22_e6_i256_o384_se0.25",
            "r3_k3_s11_e6_i384_o640_se0.25"],
        }[model_name[-2:]]  # Extract b0, b4, b7 from model_name
        model = EfficientNet3D(in_ch=1, out_ch=1, spatial_dims=3, blocks_args_str=blocks_args_str)
    elif model_name == 'DenseNet121':
        model = DenseNet121(in_ch=1, out_ch=1, spatial_dims=3)
    elif model_name == 'UNet3D':
        model = UNet3D(in_ch=1, out_ch=1, spatial_dims=3)
    else:
        raise Exception("Invalid network model specified")

    return model