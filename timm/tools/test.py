import torch
from tqdm import tqdm
from torch.utils.data import DataLoader


def test(args, model, test_datasets, test_config, datasets_config):
    model.eval()

    gpus = test_config['Gpus']
    batch_size = test_config['Batch_Size']
    num_workers = datasets_config['Num_Workers']
    obj_list = datasets_config['Obj_List']
    con_thresh = test_config['Con_Thresh']

    if gpus is not None:
        is_use_gpu = True
        model = model.cuda()
    else:
        is_use_gpu = False

    test_dataloader = DataLoader(test_datasets, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_num_iter = len(test_dataloader)
    high_con_img = []
    # high_con_img_label = []
    low_con_img = []
    with tqdm(total=test_num_iter) as test_bar:
        for iter, data in enumerate(test_dataloader):
            model_input = data['img']
            model_input_path = data['img_path']
            if is_use_gpu:
                for img in model_input:
                    model_input[img] = model_input[img].cuda()

            model_output = model(model_input)
            model_output = torch.nn.functional.softmax(model_output, dim=1)
            maxs, argmaxs = torch.max(model_output, dim=1)

            for i in range(maxs.shape[0]):
                if maxs[i] > con_thresh:
                    high_con_img.append({sensor: {'img_path': model_input_path[sensor]['img_path'][i],
                                                 'img_name': model_input_path[sensor]['img_name'][i],
                                                 'label': argmaxs[i].item()}
                                        for sensor in model_input_path})
                    # high_con_img_label.append(argmaxs[i].item())
                else:
                    low_con_img.append({sensor: {'img_path': model_input_path[sensor]['img_path'][i],
                                                 'img_name': model_input_path[sensor]['img_name'][i]}
                                        for sensor in model_input_path})
            # print(high_con_img, high_con_img_label)
            test_bar.set_description(
                'TEST. Iteration: {}/{}.'.format(iter + 1, test_num_iter))
            test_bar.update(1)
    # high_con_dict = []
    # low_con_dict = []
    # for i in range(len(high_con_img)):
    # 	high_con_dict.append({'img_path': high_con_img[i], 'label': high_con_img_label[i]})
    # for i in range(len(low_con_img)):
    # 	low_con_dict.append({'img_path': low_con_img[i]})
    return high_con_img, low_con_img
