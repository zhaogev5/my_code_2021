from .classification_model import Model


def build_model(model_config, datasets_config):
	model_name = model_config['Model_Name']
	pretrained = model_config['Pretrained']
	fusion_mode = model_config['Fusion_Mode']
	# fusion_mode = save_dir
	in_channels = model_config['In_Channels']
	out_channels = model_config['Out_Channels']
	fusion_kernal_size = model_config['Fusion_kernal_size']
	frozen_stage = model_config['Frozen_Stage']
	sensors = datasets_config['Sensors']
	obj_list = datasets_config['Obj_List']
	model = Model(model_name, fusion_mode, sensors, pretrained, obj_list,
	              in_channels, out_channels, fusion_kernal_size, frozen_stage)
	return model
