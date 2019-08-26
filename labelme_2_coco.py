import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys

import numpy as np
import PIL.Image
import copy

import labelme


try:
    import pycocotools.mask
except ImportError:
    print('Please install pycocotools:\n\n    pip install pycocotools\n')
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_dir', help='input annotated directory')
    parser.add_argument('output_dir', help='output dataset directory')
    parser.add_argument('--labels', help='labels file', required=True)
    args = parser.parse_args()


    print(args.input_dir)

    if osp.exists(args.output_dir):
        print('Output directory already exists:', args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    # os.makedirs(osp.join(args.output_dir, 'train'))
    print('Creating dataset:', args.output_dir)


    region = {
        "shape_attributes": {
            "name": "polygon",
            "all_points_x": [],
            "all_points_y": [],
        },
        "region_attributes": {}
    }

    image = {
            "fileref":"",
            "base64_img_data":"",
            "file_attributes":{

            },
            "size": 0,
            "filename": "",
            "regions":{
                "0":{},
            }
        }



    data = {

    }


    out_ann_file = osp.join(args.output_dir, 'via_region_data.json')
    label_files = glob.glob(osp.join(args.input_dir, '*.json'))[:100]
    for image_id, label_file in enumerate(label_files):
        print('Generating dataset from:', label_file)
        with open(label_file) as f:
            label_data = json.load(f)

        base = osp.splitext(osp.basename(label_file))[0]
        out_img_file = osp.join(
            args.output_dir, base + '.jpg'
        )

        img_file = osp.join(
            osp.dirname(label_file), label_data['imagePath']
        )
        print(img_file)
        img = np.asarray(PIL.Image.open(img_file))
        PIL.Image.fromarray(img).save(out_img_file)

        current_image = copy.deepcopy(image)
        current_image["size"] = os.path.getsize(out_img_file)
        current_image["filename"] = "%s.jpg" % base
        current_region = copy.deepcopy(region)
        # print(region)



        for shape in label_data['shapes']:
            points = shape['points']
            # print("len-----",len(points))
            for x,y in points:
                current_region["shape_attributes"]["all_points_x"].append(int(x))
                current_region["shape_attributes"]["all_points_y"].append(int(y))
            current_image["regions"]["0"] = current_region
        data["%s.jpg%d" % (base, os.path.getsize(out_img_file))] = current_image

    with open(out_ann_file, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    main()