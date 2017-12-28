import json
import logging
import os

from src import helper

logger = logging.getLogger(__name__)


def merge(game_name, existing_data='log.json'):
    # merge new data with master dataset for the game
    output_dir = helper.get_output_folder()
    dataset_dir = helper.get_dataset_folder()
    image_dir = os.path.join(output_dir, "images")
    master_data_path = os.path.join(dataset_dir, game_name)
    os.makedirs(master_data_path, exist_ok=True)
    master_images = os.path.join(master_data_path, 'images')
    os.makedirs(master_images, exist_ok=True)

    master_log_file = os.path.join(master_data_path, game_name + ".json")
    # create master dataset if doesn't exist
    if not os.path.exists(master_log_file):
        with open(master_log_file, 'w') as f:
            json.dump(dict(), f)
    # get existing dataset
    with open(master_log_file, 'r') as f:
        json_file = json.load(f)
        dataset_size = json_file.get('size', 0)
        data = json_file.get('data', [])
    # get new data
    with open(os.path.join(output_dir, existing_data), 'r') as f:
        new_log = json.load(f).get('data')

    # merge data
    logger.info("Initial dataset size: {}".format(dataset_size))
    count = dataset_size + 1
    for log in new_log:
        try:
            image_file_name = "{}.png".format(log['count'])
            img_path = os.path.join(image_dir, image_file_name)
            os.rename(img_path, os.path.join(master_images, "{}.png".format(count)))
            data.append({
                "count": count,
                "presses": log['presses']
            })
            count += 1
        except FileNotFoundError:
            logger.error("image not found, skipping ".format(count))

    # save dataset
    with open(master_log_file, 'w') as f:
        json.dump({
            "size": len(data),
            "data": data
        }, f)

    logger.info("Resulting dataset size after merge: {}".format(len(data)))
