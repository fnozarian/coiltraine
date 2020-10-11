import os
import time
import sys
import random


import torch
import traceback
import dlib

# What do we define as a parameter what not.

from configs import g_conf, set_type_of_process, merge_with_yaml
from network import CoILModel
from input import CoILDataset, Augmenter
from logger import coil_logger
from coilutils.checkpoint_schedule import get_latest_evaluated_checkpoint, is_next_checkpoint_ready,\
    maximun_checkpoint_reach, get_next_checkpoint

import numpy as np
import glob

def write_waypoints_output(iteration, output):

    for i in range(g_conf.BATCH_SIZE):
        steer = 0.7 * output[i][3]

        if steer > 0:
            steer = min(steer, 1)
        else:
            steer = max(steer, -1)

        coil_logger.write_on_csv(iteration, [steer,
                                            output[i][1],
                                            output[i][2]])


def write_regular_output(iteration, output):
    for i in range(len(output)):
        coil_logger.write_on_csv(iteration, [output[i][0],
                                            output[i][1],
                                            output[i][2]])




# The main function maybe we could call it with a default name
def execute(gpu, exp_batch, exp_alias, dataset_name, suppress_output):
    latest = None
    try:
        # We set the visible cuda devices
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

        # At this point the log file with the correct naming is created.
        merge_with_yaml(os.path.join('configs', exp_batch, exp_alias+'.yaml'))
        # The validation dataset is always fully loaded, so we fix a very high number of hours
        g_conf.NUMBER_OF_HOURS = 10000
        set_type_of_process('validation', dataset_name)

        if not os.path.exists('_output_logs'):
            os.mkdir('_output_logs')

        if suppress_output:
            sys.stdout = open(os.path.join('_output_logs',
                                           exp_alias + '_' + g_conf.PROCESS_NAME + '_'
                                           + str(os.getpid()) + ".out"),
                              "a", buffering=1)
            sys.stderr = open(os.path.join('_output_logs',
                              exp_alias + '_err_' + g_conf.PROCESS_NAME + '_'
                                           + str(os.getpid()) + ".out"),
                              "a", buffering=1)


        # Define the dataset. This structure is has the __get_item__ redefined in a way
        # that you can access the HDFILES positions from the root directory as a in a vector.
        full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], dataset_name)
        augmenter = Augmenter(None)
        # Definition of the dataset to be used. Preload name is just the validation data name
        dataset = CoILDataset(full_dataset, transform=augmenter,
                              preload_name=dataset_name)

        # Creates the sampler, this part is responsible for managing the keys. It divides
        # all keys depending on the measurements and produces a set of keys for each bach.

        # The data loader is the multi threaded module from pytorch that release a number of
        # workers to get all the data.
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=g_conf.BATCH_SIZE,
                                                  shuffle=False,
                                                  num_workers=g_conf.NUMBER_OF_LOADING_WORKERS,
                                                  pin_memory=True)

        checkpoints_path = os.path.join('_logs', exp_batch, exp_alias, 'checkpoints/*')
        checkpoints = []
        models = []
        checkpoint_iterations = []
        for i, ckpt_pth in enumerate(glob.glob(checkpoints_path)):
            checkpoint = torch.load(ckpt_pth)
            checkpoints.append(checkpoint)
            checkpoint_iterations.append(checkpoint['iteration'])
            print("loading model {iter}/{total}".format(iter=i, total=len(models)))
            model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
            models.append(model)

        n_models_float = float(len(models))


        accumulated_mse = 0
        accumulated_error = 0
        iteration_on_checkpoint = 0
        all_info = dict()

        for data in data_loader:

            # Compute the forward pass on a batch from  the validation dataset
            controls = data['directions']

            means = []
            sigma2_aleas = []
            for model in models:
                mean, log_var = model.forward_branch(torch.squeeze(data['rgb']).cuda(),
                                          dataset.extract_inputs(data).cuda(),
                                          controls)
                means.append(mean)
                sigma2_aleas.append(torch.exp(log_var))

            mean = torch.zeros(means[0].size()).cuda()  # shape: [batch_size, 3]
            for value in means:
                mean = mean + value / n_models_float

            sigma2_alea = torch.zeros(means[0].size()).cuda()  # shape: [batch_size, 3]
            for value in sigma2_aleas:
                sigma2_alea = sigma2_alea + value / n_models_float

            sigma2_epi = torch.zeros(means[0].size()).cuda()  # shape: [batch_size, 3]
            for value in means:
                sigma2_epi = sigma2_epi + torch.pow(mean - value, 2) / n_models_float

            # It could be either waypoints or direct control
            if 'waypoint1_angle' in g_conf.TARGETS:
                write_waypoints_output(checkpoint_iterations, mean)
            else:
                write_regular_output(checkpoint_iterations, mean)

            mse = torch.mean((mean -
                              dataset.extract_targets(data).cuda()) ** 2).data.tolist()
            mean_error = torch.mean(
                torch.abs(mean -
                          dataset.extract_targets(data).cuda())).data.tolist()

            accumulated_error += mean_error
            accumulated_mse += mse
            # error = torch.abs(mean - dataset.extract_targets(data).cuda())

            all_info_mean = mean.detach().cpu().numpy()
            all_info_sigma2_alea = sigma2_alea.detach().cpu().numpy()
            all_info_sigma2_epi = sigma2_epi.detach().cpu().numpy()
            # all_info_log_vars = log_vars.detach().cpu().numpy()
            all_info_speeds = dataset.extract_inputs(data).detach().cpu().numpy()
            all_info_targets = dataset.extract_targets(data).detach().cpu().numpy()
            all_info_controls = controls.detach().cpu().numpy()

            if iteration_on_checkpoint == 0:
                all_info['means'] = all_info_mean
                all_info['sigma2_alea'] = all_info_sigma2_alea
                all_info['sigma2_epi'] = all_info_sigma2_epi
                # all_info['log_vars'] = all_info_log_vars
                all_info['speeds'] = all_info_speeds
                all_info['targets'] = all_info_targets
                all_info['controls'] = all_info_controls
                all_info['img_paths'] = data['img_path']
            else:
                all_info['means'] = np.concatenate([all_info['means'], all_info_mean], axis=0)
                all_info['sigma2_alea'] = np.concatenate([all_info['sigma2_alea'], all_info_sigma2_alea], axis=0)
                all_info['sigma2_epi'] = np.concatenate([all_info['sigma2_epi'], all_info_sigma2_epi], axis=0)
                # all_info['log_vars'] = np.concatenate([all_info['log_vars'], all_info_log_vars], axis=0)
                all_info['speeds'] = np.concatenate([all_info['speeds'], all_info_speeds], axis=0)
                all_info['targets'] = np.concatenate([all_info['targets'], all_info_targets], axis=0)
                all_info['controls'] = np.concatenate([all_info['controls'], all_info_controls], axis=0)
                all_info['img_paths'] = np.concatenate([all_info['img_paths'], data['img_path']], axis=0)

            iteration_on_checkpoint += 1
            print("Iteration %d  on Checkpoints %s : Error %f" % (iteration_on_checkpoint,
                                                                 str(checkpoint_iterations), mean_error))

        """
            ########
            Finish a round of validation, write results, wait for the next
            ########
        """
        checkpoint_average_mse = accumulated_mse / (len(data_loader))
        checkpoint_average_error = accumulated_error / (len(data_loader))
        print('checkpoint_average_mse: ', checkpoint_average_mse)
        print('checkpoint_average_error: ', checkpoint_average_error)


        mae_errors = np.mean(np.abs(all_info['means'] - all_info['targets']), axis=0)
        print('MAE Steering Angle for Ensemble Model: ', mae_errors[0])
        print('MAE Throttle for Ensemble Model: ', mae_errors[1])
        print('MAE Break for Ensemble Model: ', mae_errors[2])
        print('MAE Total: ', np.mean(mae_errors))

        filename = 'all_info_chpt_' + '_'.join(str(ch) for ch in checkpoint_iterations)
        print("writing all info to file {}".format(filename))
        np.save(filename, all_info)

        coil_logger.add_message('Finished', {})

    except KeyboardInterrupt:
        coil_logger.add_message('Error', {'Message': 'Killed By User'})
        # We erase the output that was unfinished due to some process stop.
        if latest is not None:
            coil_logger.erase_csv(latest)

    except RuntimeError as e:
        if latest is not None:
            coil_logger.erase_csv(latest)
        coil_logger.add_message('Error', {'Message': str(e)})

    except:
        traceback.print_exc()
        coil_logger.add_message('Error', {'Message': 'Something Happened'})
        # We erase the output that was unfinished due to some process stop.
        if latest is not None:
            coil_logger.erase_csv(latest)
