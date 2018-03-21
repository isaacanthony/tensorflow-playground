import csv
import json
import os.path

PWD  = 'kaggle/imaterialist'
SET  = 'validation' # 'train' # 'test'
data = json.load(open("{}/{}.json".format(PWD, SET)))

# Generate train, validation CSVs
with open("{}/{}.csv".format(PWD, SET), 'w') as csv_path:
    csv_writer = csv.writer(csv_path)
    csv_writer.writerow(['image_id', 'label_id'])

    for annotation in data['annotations']:
        filepath = "{}/{}/{}".format(PWD, SET, annotation['image_id'])
        if os.path.isfile(filepath):
            csv_writer.writerow([annotation['image_id'], annotation['label_id']])

# Generate test CSV
# with open("{}/{}.csv".format(PWD, SET), 'w') as f:
#     f.write("image_id\n")
#
#     for image in data['images']:
#         filepath = "{}/{}/{}".format(PWD, SET, annotation['image_id'])
#         if os.path.isfile(filepath):
#             f.write("{}\n".format(image['image_id']))
