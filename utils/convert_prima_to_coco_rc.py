# This script was adapted from https://github.com/Layout-Parser/layout-model-training/blob/master/tools/convert_prima_to_coco.py

import os
import re
import json
import imagesize
from glob import glob
from bs4 import BeautifulSoup
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import sys
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
from cocosplit import cocosplit


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def cvt_coords_to_array(obj):
    if obj.has_attr('points'):
        points_str = obj['points']
        points = np.array([tuple(map(float, p.strip().split(','))) for p in points_str.strip().split()])
        return points
    elif obj.find_all("Point"):
        return np.array(
            [(float(pt['x']), float(pt['y']))
             for pt in obj.find_all("Point")]
        )
    else:
        return np.array([])


def cal_ployarea(points):
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def _create_category(schema=0):
    if schema == 0:
        categories = [
            {"supercategory": "layout", "id": 1, "name": "TableRow"},
            {"supercategory": "layout", "id": 2, "name": "TableColumn"},
            # {"supercategory": "layout", "id": 3, "name": "TableRegion"},
            # {"supercategory": "layout", "id": 4, "name": "TableCell"}
        ]

        find_categories = lambda name: \
            [val["id"] for val in categories if val['name'] == name][0]

        conversion = {
            'TableRow': find_categories("TableRow"),
            'TableColumn': find_categories("TableColumn"),
            # 'TableRegion': find_categories("TableRegion"),
            # 'TableCell': find_categories("TableCell")
        }

        return categories, conversion

_categories, _categories_conversion = _create_category(schema=0)

_info = {
    "description": "Dutch East India Company table recognition dataset",
    "version": "1.0",
    "year": 2025,
    "contributor": "Gerhard de Kok",
    "date_created": "2025/01/06",
}

def group_cells_by_row_and_column(cells):
    rows = {}
    columns = {}

    for cell in cells:
        row_idx = int(cell['row'])
        col_idx = int(cell['col'])

        if row_idx not in rows:
            rows[row_idx] = []
        rows[row_idx].append(cell)

        if col_idx not in columns:
            columns[col_idx] = []
        columns[col_idx].append(cell)

    return rows, columns


def calculate_group_boundaries(group):
    boundaries = []
    for group_idx, cell_list in group.items():
        polygons = []
        for cell in cell_list:
            coords = cvt_coords_to_array(cell.Coords)
            if len(coords) >= 3:
                polygon = Polygon(coords)
                if not polygon.is_valid:
                    polygon = make_valid(polygon)
                polygons.append(polygon)
            else:
                continue  # Handle cells with less than 3 points appropriately
        if not polygons:
            continue
        union_polygon = unary_union(polygons)
        if union_polygon.geom_type == 'Polygon':
            boundary_pts = np.array(union_polygon.exterior.coords)
            boundaries.append((group_idx, boundary_pts))
        elif union_polygon.geom_type == 'MultiPolygon':
            largest_polygon = max(union_polygon.geoms, key=lambda p: p.area)
            boundary_pts = np.array(largest_polygon.exterior.coords)
            boundaries.append((group_idx, boundary_pts))
    return boundaries


def _load_soup(filename):
    with open(filename, "r") as fp:
        soup = BeautifulSoup(fp.read(), 'xml')
    return soup

def _image_template(image_id, image_path):
    width, height = imagesize.get(image_path)
    return {
        "file_name": os.path.basename(image_path),
        "height": height,
        "width": width,
        "id": int(image_id)
    }

def _anno_template(anno_id, image_id, pts, obj_tag):
    x_1, x_2 = pts[:, 0].min(), pts[:, 0].max()
    y_1, y_2 = pts[:, 1].min(), pts[:, 1].max()
    height = y_2 - y_1
    width = x_2 - x_1

    return {
        "segmentation": [pts.flatten().tolist()],
        "area": cal_ployarea(pts),
        "iscrowd": 0,
        "image_id": image_id,
        "bbox": [x_1, y_1, width, height],
        "category_id": _categories_conversion[obj_tag],
        "id": anno_id
    }

class PRIMADataset():
    def __init__(self, base_path, anno_path='XML', image_path='Images'):
        self.base_path = base_path
        self.anno_path = os.path.join(base_path, anno_path)
        self.image_path = os.path.join(base_path, image_path)
        self._ids = self.find_all_image_ids()

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, idx):
        return self.load_image_and_annotation(idx)

    def find_all_annotation_files(self):
        return glob(os.path.join(self.anno_path, '*.xml'))

    def find_all_image_ids(self):
        replacer = lambda s: os.path.basename(s).replace('pc-', '').replace('.xml', '')
        return [replacer(s) for s in self.find_all_annotation_files()]

    def load_image_and_annotation(self, idx):
        image_id = self._ids[idx]
        image_path = os.path.join(self.image_path, f'{image_id}.jpg')
        image = Image.open(image_path)
        anno = self.load_annotation(idx)
        return image, anno

    def load_annotation(self, idx):
        image_id = self._ids[idx]
        anno_path = os.path.join(self.anno_path, f'pc-{image_id}.xml')
        # A dirty hack to load the files w/wo pc- simultaneously
        if not os.path.exists(anno_path):
            anno_path = os.path.join(self.anno_path, f'{image_id}.xml')
            assert os.path.exists(anno_path), "Invalid path"
        anno = _load_soup(anno_path)
        return anno

    def convert_to_COCO(self, save_path):
        all_image_infos = []
        all_anno_infos = []
        anno_id = 0

        for idx, image_id in enumerate(tqdm(self._ids)):
            image_path = os.path.join(self.image_path, f'{image_id}.jpg')
            if not os.path.exists(image_path):
                print(f"Image file {image_path} does not exist. Skipping.")
                continue
            image_info = _image_template(idx, image_path)
            all_image_infos.append(image_info)

            anno = self.load_annotation(idx)

            for item in anno.find_all('TableRegion'):
                # # Process TableRegion
                # pts = cvt_coords_to_array(item.Coords)
                # if 0 not in pts.shape and pts.shape[0] >= 3:
                #     anno_info = _anno_template(anno_id, idx, pts, "TableRegion")
                #     all_anno_infos.append(anno_info)
                #     anno_id += 1

                # Collect all TableCells for row/column grouping and annotation
                table_cells = item.find_all('TableCell')
                rows, columns = group_cells_by_row_and_column(table_cells)

                # Process rows
                row_boundaries = calculate_group_boundaries(rows)
                for row_idx, row_pts in row_boundaries:
                    if len(row_pts) >= 3:
                        anno_info = _anno_template(anno_id, idx, row_pts, "TableRow")
                        all_anno_infos.append(anno_info)
                        anno_id += 1

                # Process columns
                column_boundaries = calculate_group_boundaries(columns)
                for col_idx, col_pts in column_boundaries:
                    if len(col_pts) >= 3:
                        anno_info = _anno_template(anno_id, idx, col_pts, "TableColumn")
                        all_anno_infos.append(anno_info)
                        anno_id += 1

                # # Process TableCells
                # for cell in table_cells:
                #     cell_coords = cvt_coords_to_array(cell.Coords)
                #     if 0 not in cell_coords.shape and cell_coords.shape[0] >= 3:
                #         anno_info = _anno_template(anno_id, idx, cell_coords, "TableCell")
                #         all_anno_infos.append(anno_info)
                #         anno_id += 1

        final_annotation = {
            "info": _info,
            "licenses": [],
            "images": all_image_infos,
            "annotations": all_anno_infos,
            "categories": _categories
        }

        with open(save_path, 'w') as fp:
            json.dump(final_annotation, fp, cls=NpEncoder)

        return final_annotation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prima_datapath', type=str, default='../data/voc', help='path to the PRIMA dataset')
    parser.add_argument('--anno_savepath', type=str, default='../data/voc/annotations.json', help='path to save COCO annotations')
    args = parser.parse_args()

    print("Start running the conversion script")

    print(f"Loading the information from the path {args.prima_datapath}")
    dataset = PRIMADataset(args.prima_datapath)

    print(f"Saving the annotation to {args.anno_savepath}")
    res = dataset.convert_to_COCO(args.anno_savepath)

    cocosplit.main(
        args.anno_savepath,
        split_ratio=0.9,
        having_annotations=True,
        train_save_path=args.anno_savepath.replace('.json', '-train.json'),
        test_save_path=args.anno_savepath.replace('.json', '-val.json'),
        random_state=24
    )
