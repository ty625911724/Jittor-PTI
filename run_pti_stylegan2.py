from random import choice
from string import ascii_uppercase
import jittor.transform as transform
import os
from configs import global_config, paths_config
#import wandb

from training_stylegan2.coaches.multi_id_coach import MultiIDCoach
from training_stylegan2.coaches.single_id_coach import SingleIDCoach
from utils.ImagesDataset import ImagesDataset

def run_PTI(run_name='', use_wandb=False, use_multi_id_training=False):
    #os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    #os.environ['CUDA_VISIBLE_DEVICES'] = global_config.cuda_visible_devices

    if run_name == '':
        global_config.run_name = ''.join(choice(ascii_uppercase) for i in range(12))
    else:
        global_config.run_name = run_name

    #if use_wandb:
    #    run = wandb.init(project=paths_config.pti_results_keyword, reinit=True, name=global_config.run_name)
    global_config.pivotal_training_steps = 1
    global_config.training_step = 1

    embedding_dir_path = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}/{paths_config.pti_results_keyword}'
    os.makedirs(embedding_dir_path, exist_ok=True)
    
    img_size = 1024
    transform_image = transform.Compose([
        transform.Resize(size = img_size),
        transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = ImagesDataset(paths_config.input_data_path, transform_image).set_attrs(batch_size=1, shuffle=False)

    if use_multi_id_training:
        coach = MultiIDCoach(dataset, use_wandb)
    else:
        coach = SingleIDCoach(dataset, use_wandb)

    coach.train()

    return global_config.run_name


if __name__ == '__main__':
    run_PTI(run_name='', use_wandb=False, use_multi_id_training=False)
    #run_PTI(run_name='', use_wandb=False, use_multi_id_training=True)


