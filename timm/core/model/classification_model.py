import timm
import torch
import torch.nn as nn


class Model(nn.Module):
	"""docstring for Model"""

	def __init__(self,
	             model_name,
	             fusion_mode,
	             sensors,
	             pretrained,
	             obj_list,
	             in_channels,
	             out_channels,
	             fusion_kernal_size,
	             frozen_stage):

		super(Model, self).__init__()
		self.obj_list = obj_list
		self.sensors = sensors
		self.pretrained = pretrained
		self.model_name = model_name
		self.fusion_mode = fusion_mode
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.fusion_kernal_size = fusion_kernal_size
		self.frozen_stage = frozen_stage

		if self.fusion_mode == 'No_Fusion':
			self.model = timm.create_model(model_name=self.model_name,
			                               pretrained=self.pretrained,
			                               num_classes=len(self.obj_list))
			if self.frozen_stage:
				for key, stage in self.model.named_modules():
					if key != 'fc':
						for i in stage.parameters():
							i.requires_grad = False
					else:
						for i in stage.parameters():
							i.requires_grad = True

		if self.fusion_mode == 'Input_Fusion':
			self.fusion_layer = nn.Sequential(
				nn.Conv2d(len(self.sensors) * self.in_channels, self.in_channels, self.fusion_kernal_size),
				nn.ReLU(),
				nn.BatchNorm2d(self.in_channels))
			self.model = timm.create_model(model_name=self.model_name,
			                               pretrained=self.pretrained,
			                               num_classes=len(self.obj_list))
			if self.frozen_stage:
				for key, stage in self.model.named_modules():
					if key != 'fc':
						for i in stage.parameters():
							i.requires_grad = False
					else:
						for i in stage.parameters():
							i.requires_grad = True

		if self.fusion_mode == 'Feature_Fusion':
			model = timm.create_model(model_name=self.model_name, pretrained=self.pretrained,
			                          num_classes=len(self.obj_list))
			layers_name = []
			for i in model.state_dict():
				layer_name = i.split('.')[0]
				if layer_name not in layers_name:
					layers_name.append(layer_name)
			print(layers_name)

			self.backbone = nn.ModuleDict()
			for sensor in self.sensors:
				self.model = timm.create_model(model_name=self.model_name,
				                               pretrained=self.pretrained,
				                               num_classes=len(self.obj_list))
				self.model = nn.Sequential(self.model.conv1, self.model.bn1, self.model.layer1, self.model.layer2,
				                           self.model.layer3, self.model.layer4)

				self.backbone.update({sensor: self.model})

			self.fusion_layer = nn.Sequential(
				nn.Conv2d(len(self.sensors) * self.out_channels, self.out_channels, self.fusion_kernal_size),
				nn.ReLU())

			self.classifier = nn.Sequential(model.global_pool,
			                                model.fc)

			if self.frozen_stage:
				for i in self.backbone.parameters():
					i.requires_grad = False

		if self.fusion_mode == 'Decision_Fusion':
			self.model = nn.ModuleDict(
				{sensor: timm.create_model(model_name=self.model_name,
				                           pretrained=self.pretrained,
				                           num_classes=len(self.obj_list))
				 for sensor in self.sensors})

			if self.frozen_stage:
				for key, stage in self.model.named_modules():
					if key[-2:] != 'fc':
						for i in stage.parameters():
							i.requires_grad = False
					else:
						for i in stage.parameters():
							i.requires_grad = True

	def forward(self, x):

		if self.fusion_mode == 'No_Fusion':
			inputs = x[self.sensors[0]]
			results = self.model(inputs)

		if self.fusion_mode == 'Input_Fusion':
			inputs = torch.cat([x[i] for i in x], dim=1)
			x = self.fusion_layer(inputs)
			results = self.model(x)

		if self.fusion_mode == 'Feature_Fusion':
			feats = {}
			for sensor in self.sensors:
				feats.update({sensor: self.backbone[sensor](x[sensor])})
			feat = torch.cat([feats[i] for i in feats], dim=1)
			x = self.fusion_layer(feat)
			results = self.classifier(x)

		if self.fusion_mode == 'Decision_Fusion':
			outs = {}
			for sensor in self.sensors:
				outs.update({sensor: self.model[sensor](x[sensor])})
			results = torch.zeros_like(outs[self.sensors[0]])
			for out in outs:
				results = results + outs[out] / len(self.sensors)

		return results
