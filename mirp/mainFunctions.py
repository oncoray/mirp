import logging
import multiprocessing as mp

import time
from inspect import stack, getmodule
from typing import List

import numpy as np
import pandas as pd

from mirp.settings.importDataSettings import import_data_settings
from mirp.settings.importConfigurationSettings import import_configuration_settings
from mirp.experimentClass import ExperimentClass


def get_roi_labels(**kwargs):
    raise RuntimeError(
        f"The get_roi_labels function has been replaced by "
        f"mirp.extractMaskLabels.extract_mask_labels."
    )


def get_image_acquisition_parameters(**kwargs):
    raise RuntimeError(
        f"The get_image_acquisition_parameters function has been replaced by "
        f"mirp.extractImageParameters.extract_image_parameters."
    )


def get_file_structure_parameters(**kwargs):
    raise RuntimeError(
        f"The get_file_structure_parameters function has been fully deprecated, without replacement."
    )


def parse_file_structure(**kwargs):
    raise RuntimeError(
        f"The parse_file_structure function has been fully deprecated, without replacement."
    )


def extract_images_for_deep_learning(**kwargs):
    raise RuntimeError(
        f"The extract_images_for_deep_learning function has been replaced by "
        f"mirp.deepLearningPreprocessing.deep_learning_preprocessing."
    )


def extract_features(**kwargs):
    raise RuntimeError(
        f"The extract_features function has been replaced by mirp.extractFeaturesAndImage.extract_features."
    )


def extract_images_to_nifti(**kwargs):
    raise RuntimeError(
        f"The extract_images_to_nifti function has been replaced by mirp.extractFeaturesAndImage.extract_images."
    )


def process_images(data_config, settings_config, n_processes=1, keep_images_in_memory=False, compute_features=True,
                   extract_images=False, plot_images=False):
    """
    Process images for various tasks.

    :param data_config: full path to a data configuration xml file.
    :param settings_config: full path to a settings configuration xml file.
    :param n_processes: number of simultaneous processes. For n_processes > 1, parallel processes are started.
    :param keep_images_in_memory: flag to keep images in memory. This avoids repeated loading of images, but at the expense of memory.
    :param compute_features: flag to compute features (default: False)
    :param extract_images: flag to extract images and mask in Nifti format (default: False)
    :param plot_images: flag to plot images and masks as .png (default: False)
    :return:
    """

    if not compute_features and not extract_images:
        return None

    # Extract features
    if n_processes > 1:

        # Usually, we would only run this code if __name__ == "__main__".
        # However, as this function is nested in a module, the __name__ is always "mirp.mainFunctions" instead of "__main__", thus prohibiting any multiprocessing.
        # Therefore we have to see if "__main__" is the name that appears in one of the parent modules. This is not always the top module, and therefore we search
        # the stack. The trick is that on windows OS "__main__" does appear in the calling environment from where the script is executed, but is called "__mp_main__"
        # in the multiprocessing spawning process. We perform this check to prevent infinite spawning on Windows OS, as it doesn't fork neatly but spawns identical
        # separate process by repeating the stack call in each process. The alternative would be to include a __name__ == "__main__" check in the calling script,
        # and a switch variable (e.g. as_master) in the function call, but that puts the onus on the end-user, and is a terrible idea due to the principle of least
        # astonishment.
        module_names = ["none"]
        for stack_entry in stack():
            current_module = getmodule(stack_entry[0])
            if current_module is not None:
                module_names += [current_module.__name__]

        if "__main__" in module_names:

            # Parse data
            settings_list = import_configuration_settings(compute_features=compute_features,
                                                          path=settings_config)
            data_obj_list: List[ExperimentClass] = import_data_settings(path=data_config,
                                                                        config_settings=settings_list,
                                                                        keep_images_in_memory=keep_images_in_memory,
                                                                        compute_features=compute_features, extract_images=extract_images,
                                                                        plot_images=plot_images)

            # Initiate process manager
            df_mngr = pd.DataFrame({"job_id": np.arange(len(data_obj_list)),
                                    "job_processed": np.zeros(shape=len(data_obj_list), dtype=bool),
                                    "job_in_process": np.zeros(shape=len(data_obj_list), dtype=bool),
                                    "assigned_worker": -np.ones(shape=len(data_obj_list), dtype=int),
                                    "error_iteration": np.zeros(shape=len(data_obj_list), dtype=int)})

            # Initiate worker list
            worker_list = []
            for ii in np.arange(n_processes):

                # Check if enough jobs are available
                if ii >= len(data_obj_list):
                    break

                # Add job to worker
                process_name = "_".join([data_obj_list[ii].subject] + data_obj_list[ii].data_str +
                                        [data_obj_list[ii].settings.general.config_str])
                worker_list.append(mp.Process(target=_parallel_process, args=(data_obj_list[ii],), name=process_name))
                worker_list[ii].daemon = True
                df_mngr.loc[ii, "assigned_worker"] = ii

            # Initiate a list that keeps track of repeated errors and skips those samples.
            error_skip_list = []

            # Iterate and process all jobs
            while np.any(~df_mngr.job_processed):

                # Start jobs
                for ii in np.arange(len(worker_list)):
                    # Check if worker is assigned
                    if ~np.any(df_mngr.assigned_worker == ii):
                        continue

                    # Get current job id
                    curr_job_id = df_mngr.job_id[df_mngr.assigned_worker == ii]

                    # Check if job is still in progress or was completed
                    if df_mngr.job_processed[curr_job_id].values or df_mngr.job_in_process[curr_job_id].values:
                        continue

                    # Start process
                    df_mngr.loc[curr_job_id, "job_in_process"] = True
                    worker_list[ii].start()

                # No more workers are available
                free_workers = []

                # Check finished jobs - every 5 seconds
                while len(free_workers) == 0:
                    time.sleep(5)
                    for ii in np.arange(len(worker_list)):

                        # Check if worker is assigned
                        if ~np.any(df_mngr.assigned_worker == ii):
                            free_workers.append(ii)
                            continue

                        # Get current job id
                        curr_job_id = df_mngr.job_id[df_mngr.assigned_worker == ii].values[0]

                        # Check if worker is still processing
                        if worker_list[ii].is_alive():
                            continue

                        # Check exit code of the stopped worker
                        if worker_list[ii].exitcode == 0:
                            # Normal exit - update table and set worker
                            df_mngr.loc[curr_job_id, "job_processed"] = True
                            df_mngr.loc[curr_job_id, "job_in_process"] = False
                            df_mngr.loc[curr_job_id, "assigned_worker"] = -1

                            free_workers.append(ii)

                        else:
                            # This indicates some fault (e.g. segmentation fault)
                            df_mngr.loc[curr_job_id, "error_iteration"] += 1

                            # Stop after 2 iterations that produce errors
                            if df_mngr.loc[curr_job_id, "error_iteration"] < 2:
                                df_mngr.loc[curr_job_id, "job_in_process"] = False
                                df_mngr.loc[curr_job_id, "assigned_worker"] = -1

                                logging.warning("Process ended prematurely, attempting to restart.")
                            else:
                                df_mngr.loc[curr_job_id, "job_processed"] = True
                                df_mngr.loc[curr_job_id, "job_in_process"] = False
                                df_mngr.loc[curr_job_id, "assigned_worker"] = -1

                                error_skip_list.append(curr_job_id)

                                logging.warning("Process ended prematurely, no attempt to restart again.")

                            # Free up the worker
                            free_workers.append(ii)

                # Check remaining available jobs
                available_jobs = df_mngr.job_id[~np.logical_or(df_mngr.job_processed, df_mngr.job_in_process)]

                # Add new jobs to workers
                for jj in np.arange(len(free_workers)):

                    # Check if enough jobs are available
                    if jj >= len(available_jobs):
                        break

                    # Add job to worker
                    sel_job_id = available_jobs.values[jj]
                    process_name = "_".join([data_obj_list[sel_job_id].subject] + data_obj_list[sel_job_id].data_str
                                            + [data_obj_list[sel_job_id].settings.general.config_str])

                    worker_list[free_workers[jj]] = mp.Process(target=_parallel_process,
                                                               args=(data_obj_list[sel_job_id],), name=process_name)
                    worker_list[free_workers[jj]].daemon = True
                    df_mngr.loc[sel_job_id, "assigned_worker"] = free_workers[jj]

            # Exit statement
            logging.info("Feature calculation has been completed.")

            if len(error_skip_list) > 0:
                names = ["_".join(data_obj_list[ii].data_str) + " of " + data_obj_list[ii].subject + " (" +
                         data_obj_list[ii].cohort + ")" for ii in error_skip_list]
                logging.info("No features could be calculated for %s due to errors.",
                             ", ".join(sample_name for sample_name in names))

    else:
        # Parse data
        settings_list = import_configuration_settings(compute_features=compute_features,
                                                      path=settings_config)
        data_obj_list = import_data_settings(path=data_config,
                                             config_settings=settings_list,
                                             keep_images_in_memory=keep_images_in_memory,
                                             compute_features=compute_features,
                                             extract_images=extract_images)

        for data_obj in data_obj_list:
            data_obj.process()


def _parallel_process(data_obj):
    """
    Function for parallel feature extraction

    :param data_obj: input ExperimentClass object.
    :return:

    This function is used internally.
    """

    data_obj.process()
