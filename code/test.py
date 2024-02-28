import os
import os.path
from collections import deque
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import DataLoader

from util.option import Options
from util.misc import valid_tensor, get_color_mapped_images, save_individual_images, save_tensor_grid
from util.augmentation import reverse_modification, get_image_modifications
from util.evaluation import get_print, iou
from model.models import get_model
from dataset.val_dataset import ValDataset

import time

def get_average_visuals(net, image, mask, visuals=None, subtract_min_depth=True, conv=True, test_time_aug=False):
    if visuals is None:
        visuals = OrderedDict()

    if test_time_aug:
        conv = False
        images, transforms = get_image_modifications(image)
        masks, _ = get_image_modifications(mask)
    else:
        images = [image]
        masks = [mask]
        transforms = ["original"]

    albedos = torch.empty((0, 3, image.shape[2], image.shape[3])).to(image.device)
    depths = torch.empty((0, 1, image.shape[2], image.shape[3])).to(image.device)

    print('start inference ...')
    with torch.no_grad():
        for curr_image, cur_mask, transform in zip(images, masks, transforms):
            cur_outputs, cur_features = net(curr_image, mask=None if conv else cur_mask)
            albedo, depth, normal, light_env, light_id = cur_outputs

            if valid_tensor(albedo):
                cur_albedo = reverse_modification(albedo, transform, label='albedo', original_shape=image.shape[2:])
            cur_depth = reverse_modification(depth, transform, label='depth', original_shape=image.shape[2:])
            cur_depth[~mask] = 1

            if valid_tensor(albedo):
                albedos = torch.cat((albedos, cur_albedo))
            depths = torch.cat((depths, cur_depth))

    if valid_tensor(albedo):
        albedo_std, albedo_mean = torch.std_mean(albedos, dim=0, keepdim=True)
        visuals['albedo pred'] = albedo_mean

    depth_std, depth_mean = torch.std_mean(depths, dim=0, keepdim=True)
    if subtract_min_depth:
        depth_mean = depth_mean - torch.min(depth_mean)
        depth_mean[~mask.repeat(1, depth_mean.shape[1], 1, 1)] = 1
    visuals['depth pred'] = depth_mean

    return visuals

def prepare_datasets(opt):
    val_dataset_dir = os.path.join(opt.dataroot, opt.val_dataset_dir)
    val_dataset = ValDataset(val_dataset_dir)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=opt.num_workers)

    return val_dataloader

def main():
    fps = []
    # start_time = time.time()
    opt = Options(train=False).parse()
    ch1 = "check"
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda')
    net = get_model('decomposer', weights_init=opt.weights_decomposer, output_last_ft=True, out_range='0,1').to(device)
    net.eval()
    ch2 = "check"
    val_dataloader = prepare_datasets(opt)

    image_dir = opt.val_dataset_dir + ("_final" if opt.test_time_aug else "")
    os.makedirs(os.path.join(opt.output, image_dir, "grid"), exist_ok=True)
    print_ious = deque()
    iou_file_name = os.path.join(opt.output, image_dir + "_iou.txt")
    f = open(iou_file_name, "a")

    for data in val_dataloader:
        start_time = time.time()
        image, mask, shoeprint, albedo_segmentation, name, pad_h_before, pad_h_after, pad_w_before, pad_w_after = data

        # Directly transfer to device without make_variable
        image = image.to(device)
        mask = mask.to(device)
        shoeprint = shoeprint.to(device)

        visuals = OrderedDict()
        visuals[name[0]] = image
        if valid_tensor(shoeprint):
            visuals['GT print'] = shoeprint
        visuals['mask'] = mask
        visuals = get_average_visuals(net, image, mask, visuals=visuals, test_time_aug=opt.test_time_aug)

        real_gt_print_pred = get_print(visuals['depth pred'], mask, shoeprint)
        
        # 배경 흑백 전환 
        real_gt_print_pred[~mask] = False
        
        print_iou = iou(real_gt_print_pred, shoeprint, mask).item()
        print_ious.append(print_iou)
        visuals['print pred, iou: {:0.2f}'.format(print_iou)] = real_gt_print_pred
        visuals['depth pred'] = get_color_mapped_images(visuals['depth pred'], mask).to(image.device, torch.float32)

        save_path = os.path.join(opt.output, image_dir, "grid", name[0])
        save_tensor_grid(visuals, save_path, fig_shape=[2, 3], figsize=(10, 5))

        del visuals[name[0]]
        visuals['real image'] = image
        del visuals['print pred, iou: {:0.2f}'.format(print_iou)]
        visuals['print pred'] = real_gt_print_pred
        end_time = time.time()
        # 소요 시간 계산 
        elapsed_time = end_time - start_time

        print(f"Saving {name[0]}, {elapsed_time: .2f} seconds")
        fps.append(elapsed_time)

        save_individual_images(visuals, os.path.join(opt.output, image_dir), name,
                               pad_h_before=pad_h_before, pad_h_after=pad_h_after,
                               pad_w_before=pad_w_before, pad_w_after=pad_w_after)

        f.write(name[0].rsplit('.', 1)[0] + '\t{:0.2f}\n'.format(print_iou))

    print(f'Mean iou: {np.mean(print_ious)}, Average seconds:  {np.mean(fps):.2f}')
    f.write('mean iou: {:0.2f}\n'.format(np.mean(print_ious)))
    f.close()

if __name__ == '__main__':
    main()