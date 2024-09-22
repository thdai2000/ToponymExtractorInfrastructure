import json
import os
import matplotlib.pyplot as plt
import numpy as np
import bezier
import bezier_utils as butils
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageEnhance
import random
import english_encoding as Encoding
from tqdm import tqdm
import DeepFont
from sklearn.model_selection import train_test_split

class MapKuratorDataset:
    def __init__(self, image_root_folder) -> None:
        self.images = {}
        self.annos = {}
        self.image_root_folder = image_root_folder

    def clear(self):
        self.images = {}
        self.annos = {}

    def read_anno(self, json_anno_path):
        with open(json_anno_path, "r") as f:
            data_dict = json.load(f)

        images = data_dict['images']
        self.images = {record['id']: record for record in images}
        self.annos = {record['id']: [] for record in images}
        annotations = data_dict['annotations']
        for annotation in annotations:
            image_id = annotation['image_id']
            self.annos[image_id].append(annotation)

    def draw_annotations(self, image_id, text_decoder=Encoding.decode_text_96):
        annos = self.annos[image_id]
        image = self.images[image_id]
        image_path = os.path.join(self.image_root_folder, image['file_name'])
        image = plt.imread(image_path)
        plt.imshow(image)
        for anno in annos:
            bbox = anno['bbox']
            # plt.plot([bbox[0], bbox[0], bbox[0] + bbox[2], bbox[0] + bbox[2], bbox[0]], [bbox[1], bbox[1] + bbox[3], bbox[1] + bbox[3], bbox[1], bbox[1]])
            if text_decoder is not None:
                text = text_decoder(anno['rec'])
            else:
                text = ','.join([rec for rec in anno['rec'] if rec < 96])
            # plt.text(bbox[0], bbox[1], text)
            polys = anno['polys']  # [x1, y1, x2, y2, ...]
            polys = np.array(polys).reshape(-1, 2)
            # print(len(polys))
            plt.plot(polys[:, 0], polys[:, 1], color='orange')
            plt.plot(polys[:, 0][:len(polys) // 2], polys[:, 1][:len(polys) // 2], color='b')
            plt.plot(polys[:, 0][len(polys) // 2:], polys[:, 1][len(polys) // 2:], color='b')

            # Plot the first point as small red circle
            plt.scatter(polys[0, 0], polys[0, 1], color='red', s=5)

            if text_decoder is not None:
                text = text_decoder(anno['rec'])
                plt.text(bbox[0], bbox[1], text)
        plt.show()


'''
RUMSEY JSON FORMAT

image: str
groups: list
    [group]: list
        [entry]: dict
            vertices: list, [[x1, y1], [x2, y2], ...], 16 points
            text: str
            illegible: bool
            truncated: bool
'''


class RumseyDataset:
    def __init__(self, image_folder_path) -> None:
        self.images = []
        self.groups = []
        self.image_folder_path = image_folder_path

    def clear(self):
        self.images = []
        self.groups = []

    def __len__(self):
        return len(self.groups)

    def read_anno(self, json_anno_path):
        with open(json_anno_path) as f:
            data = json.load(f)
        for entry in data:
            self.images.append(os.path.join(self.image_folder_path, entry['image']))
            self.groups.append(entry['groups'])

    def draw_annotations(self, image_index, thickness=2, scatter=False):
        image_path = self.images[image_index]

        # Draw with plt
        plt.clf()
        image = Image.open(image_path)
        plt.imshow(image)
        # plt.show()
        for group in self.groups[image_index]:

            for entry in group:
                text = entry['text']

                vertices = np.array(entry["vertices"]).reshape(-1, 2)
                # plt.text(vertices_upper[0, 0], vertices_upper[0, 1], text, color=color, fontsize=10, ha='center', va='center')

                if scatter:
                    plt.scatter(vertices[:, 0], vertices[:, 1], color=np.random.rand(3), s=2)
                else:
                    plt.plot(vertices[:, 0], vertices[:, 1], color="orange", linewidth=thickness)
                    plt.plot(vertices[:, 0][:len(vertices) // 2], vertices[:, 1][:len(vertices) // 2], color="blue",
                             linewidth=thickness)
                    plt.plot(vertices[:, 0][len(vertices) // 2:], vertices[:, 1][len(vertices) // 2:], color="blue",
                             linewidth=thickness)

                    plt.scatter(vertices[0, 0], vertices[0, 1], color='red', s=5)

        # plt.show()
        plt.savefig("vis/{}_vertices.png".format(image_index))
        plt.close()

        return

    def get_images(self):
        return self.images

    def get_groups(self):
        return self.groups

    def get_instances(self):
        instances = []
        for group in self.groups:
            for entry in group:
                instances.append(entry)
        return instances

    def get_polygons(self):
        polygons = []
        for group in self.groups:
            for entry in group:
                polygons.append(entry['vertices'])
        return polygons

    def get_texts(self):
        texts = []
        for group in self.groups:
            for entry in group:
                texts.append(entry['text'])
        return texts


'''
CVAT JSON FORMAT

image: str
groups: list
    [group]: list
        [entry]: dict
            vertices: list, [[[x,y],...] (upper line in reading order), [[x, y], ...] (lower line in reverse reading order)]
            text: str
            illegible: bool
            truncated: bool
'''
class CvatDataset:
    def __init__(self, image_folder_path) -> None:
        self.images = []
        self.groups = []
        self.image_folder_path = image_folder_path

    def clear(self):
        self.images = []
        self.groups = []

    def __len__(self):
        return len(self.groups)

    def read_anno(self, json_anno_path):
        with open(json_anno_path) as f:
            data = json.load(f)
        for entry in data:
            self.images.append(os.path.join(self.image_folder_path, entry['image']))
            self.groups.append(entry['groups'])

    def draw_annotations(self, image_index, thickness=2, scatter=False):
        image_path = self.images[image_index]

        # Draw with plt
        plt.clf()
        image = Image.open(image_path)
        plt.imshow(image)
        # plt.show()
        for group in self.groups[image_index]:

            if len(group) == 1:
                color = "gray"
            else:
                color = np.random.rand(3)
            for i, entry in enumerate(group):
                text = entry['text']
                # print(entry)
                vertices_upper = np.array(entry["vertices"][0]).reshape(-1, 2)
                vertices_lower = np.array(entry["vertices"][1]).reshape(-1, 2)
                vertices = np.concatenate([vertices_upper, vertices_lower])
                # plt.text(vertices_upper[0, 0], vertices_upper[0, 1], text, color=color, fontsize=10, ha='center', va='center')

                if scatter:
                    plt.scatter(vertices[:, 0], vertices[:, 1], color=color, s=2)
                else:
                    plt.plot(vertices[:, 0], vertices[:, 1], color="orange", linewidth=1)
                    plt.plot(vertices_upper[:, 0], vertices_upper[:, 1], color=color, linewidth=2)
                    plt.plot(vertices_lower[:, 0], vertices_lower[:, 1], color=color, linewidth=2)
                    plt.scatter(vertices_upper[0, 0], vertices_upper[0, 1], color='red', s=3)
                    plt.text(vertices_lower[-1, 0], vertices_lower[-1, 1], str(i), fontsize=5)

        # plt.show()
        plt.savefig("vis/{}.png".format(image_index))
        plt.close()

        return

    def get_images(self):
        return self.images

    def get_groups(self):
        return self.groups

    def get_instances(self):
        instances = []
        for group in self.groups:
            for entry in group:
                instances.append(entry)
        return instances

    def get_polygons(self):
        polygons = []
        for group in self.groups:
            for entry in group:
                polygons.append(entry['vertices'])
        return polygons

    def get_texts(self):
        texts = []
        for group in self.groups:
            for entry in group:
                texts.append(entry['text'])
        return texts



'''
DeepSolo dataset
'''
# with open("train_96voc.json", "r") as f:
#     data = json.load(f)
#     pass
#
# data['licenses'] = []
# data['info'] = {}
# data['categories'] = [{'id': 1, 'name': 'text', 'supercategory': 'beverage', 'keypoints': ['mean', 'xmin', 'x2', 'x3', 'xmax', 'ymin', 'y2', 'y3', 'ymax', 'cross']}]
#
# data['images']
#
# data['annotations']

# image_record = {
#     "coco_url": "",
#     "date_captured": "",
#     "file_name": "img_494.jpg",
#     "flickr_url": "",
#     "id": 494,
#     "license": 0,
#     "width": 1280,
#     "height": 720
#     }
#
# anno_record = {
#     "area": 6579.0,
#     "bbox": [174.0, 187.0, 153.0, 43.0],
#     "category_id": 1,
#     "id": 0,
#     "image_id": 346,
#     "iscrowd": 0,
#     "bezier_pts": [174, 193,
#                    222, 191,
#                    271, 189,
#                    320, 187,
#
#                    326, 222,
#                    277, 224,
#                    228, 226,
#                    179, 229
#                    ],
#     "rec": [33, 67, 67, 69, 83, 83, 79, 82, 73, 69, 83, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96]
#     }

def bbox_include(bbox1, bbox2):
    '''
        Check if bbox1 includes bbox2
    '''
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    return x1 <= x2 and y1 <= y2 and x1 + w1 >= x2 + w2 and y1 + h1 >= y2 + h2


def bbox_inflate(bbox, inflate_ratio):
    '''
        Inflate the bounding box by a ratio
    '''
    x, y, w, h = bbox
    x -= w * inflate_ratio
    y -= h * inflate_ratio
    w += w * 2 * inflate_ratio
    h += h * 2 * inflate_ratio
    return [x, y, w, h]


def bbox_overlap(bbox1, bbox2):
    '''
        Check if bbox1 overlaps bbox2
    '''
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    return x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2


def get_bbox_intersection(bbox1, bbox2):
    '''
        Get the intersection of two bounding boxes
    '''
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    x_min = max(x1, x2)
    y_min = max(y1, y2)
    x_max = min(x1 + w1, x2 + w2)
    y_max = min(y1 + h1, y2 + h2)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def bezier_bbox_crop(bezier_pts_deg_4, bbox, segments=41, reversed=False):
    '''
        Crop the bezier curve to the bounding box
    '''
    # Convert the bezier points to polylines
    bezier_pts = [(bezier_pts_deg_4[i], bezier_pts_deg_4[i + 1]) for i in range(0, len(bezier_pts_deg_4), 2)]
    curve = bezier_pts[:4]

    if reversed:
        curve = curve[::-1]

    curve_xx = [p[0] for p in curve]
    curve_yy = [p[1] for p in curve]

    polyline_xx, polyline_yy = butils.bezier_to_polyline(curve_xx, curve_yy, segments)

    # Evaluate how much of the polyline stays within the bounding box
    x_min, y_min, w, h = bbox
    x_max, y_max = x_min + w, y_min + h
    polyline = [(polyline_xx[i], polyline_yy[i]) for i in range(len(polyline_xx))]
    polyline_in_bbox = [p if x_min <= p[0] <= x_max and y_min <= p[1] <= y_max else None for p in polyline]

    # Find the first and last points that are within the bounding box
    first_in_bbox = None
    last_in_bbox = None
    for i, p in enumerate(polyline_in_bbox):
        if p is not None:
            first_in_bbox = i
            break

    # If the polyline is completely outside the bounding box, return None
    if first_in_bbox is None:
        return None, [0, 0]

    for i, p in enumerate(polyline_in_bbox[first_in_bbox:] + [None]):
        if p is None:
            last_in_bbox = i + first_in_bbox
            break

    # If the polyline is completely inside the bounding box, return the original bezier points
    if last_in_bbox is None:
        return bezier_pts_deg_4, [0, 1]

    # If the polyline intersects the bounding box more than once, return None
    if first_in_bbox != 0 and last_in_bbox != len(polyline_in_bbox) - 1:
        return None, [0, 0]

    if last_in_bbox - first_in_bbox < 2:
        return None, [0, 0]

    # If the polyline intersects the bounding box only once, crop the bezier curve
    cropped_polyline = polyline[first_in_bbox:last_in_bbox]

    # Fit a bezier curve to the cropped polyline
    cpts_x, cpts_y = butils.bezier_from_polyline_v2([p[0] for p in cropped_polyline], [p[1] for p in cropped_polyline])

    assert len(cpts_x) == 4
    assert len(cpts_y) == 4

    # bezier_pts = [x1, y1, x2, y2, x3, y3, x4, y4]
    bezier_pts = []
    if reversed:
        cpts_x = cpts_x[::-1]
        cpts_y = cpts_y[::-1]

    for i in range(4):
        bezier_pts.append(cpts_x[i])
        bezier_pts.append(cpts_y[i])

    return bezier_pts, [first_in_bbox / (segments - 1), last_in_bbox / (segments - 1)]


def text_range_crop(text, crop_range):
    '''
        Crop the text to the range (0, 1)
    '''
    import math

    length = len(text)

    crop_start = math.ceil(crop_range[0] * length)
    crop_end = math.floor(crop_range[1] * length)

    # Crop the text
    text = text[crop_start:crop_end]

    return text


class DeepSoloDataset:
    def __init__(self, image_root_folder=None) -> None:
        self.licenses = []
        self.info = {}
        self.categories = [{'id': 1, 'name': 'text', 'supercategory': 'beverage',
                            'keypoints': ['mean', 'xmin', 'x2', 'x3', 'xmax', 'ymin', 'y2', 'y3', 'ymax', 'cross']}]
        self.images = {}
        self.annos = {}
        self.image_root_folder = image_root_folder

    def load_annotations_from_file(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
            self.load_annotations(data)

    def load_annotations(self, data_dict):
        images = data_dict['images']
        self.images = {record['id']: record for record in images}
        self.annos = {record['id']: [] for record in images}
        annotations = data_dict['annotations']
        for annotation in annotations:
            image_id = annotation['image_id']
            self.annos[image_id].append(annotation)

    def scale(self, scale_ratio):
        # for image in self.images.values():
        #    image['width'] = int(image['width'] * scale_ratio)
        #    image['height'] = int(image['height'] * scale_ratio)
        for anno in self.annos.values():
            for annotation in anno:
                annotation['area'] = round(annotation['area'] * scale_ratio * scale_ratio, 2)
                annotation['bbox'] = [round(val * scale_ratio, 2) for val in annotation['bbox']]
                bezier_pts = annotation['bezier_pts']
                bezier_pts = [round(val * scale_ratio, 2) for val in bezier_pts]
                annotation['bezier_pts'] = bezier_pts

    def make_tile(self, m, tiled_image_folder, datasetname, save_images=True):
        '''
            Crop the annotations on each image to mxm tiles
        '''
        new_images = {}
        original_m = m
        # Tile images
        for image_id, image in tqdm(self.images.items()):
            image_path = os.path.join(self.image_root_folder, image['file_name'])

            # special case: the only whole map
            if "catherwood_1835" in image["file_name"]:
                m = 6
            elif "12148_btv1b53027745sf1_0" in image["file_name"]:
                m = 4
            else:
                m = original_m

            image = Image.open(image_path)
            width, height = image.size
            tile_width = width // m
            tile_height = height // m
            for h_tile_id in range(m):
                for v_tile_id in range(m):
                    x_min = tile_width * h_tile_id
                    y_min = tile_height * v_tile_id
                    x_max = x_min + tile_width
                    y_max = y_min + tile_height
                    tile_image = image.crop((x_min, y_min, x_max, y_max))
                    new_image_name = f"{datasetname}_{image_id}_{v_tile_id}_{h_tile_id}.png"
                    if save_images:
                        if not os.path.exists(tiled_image_folder):
                            os.makedirs(tiled_image_folder)
                        tile_image.save(os.path.join(tiled_image_folder, new_image_name))

                    # Create a new image record
                    new_image_id = (v_tile_id * m + h_tile_id) * len(self.images) + image_id
                    new_image_record = {
                        "coco_url": "",
                        "date_captured": "",
                        "file_name": new_image_name,
                        "flickr_url": "",
                        "id": new_image_id,
                        "license": 0,
                        "width": tile_width,
                        "height": tile_height
                    }
                    new_images[new_image_id] = new_image_record

        new_annos = {}
        for image_id, annos in self.annos.items():
            for anno in annos:
                for h_tile_id in range(m):
                    for v_tile_id in range(m):
                        original_image_id = image_id
                        new_image_id = (v_tile_id * m + h_tile_id) * len(self.images) + original_image_id
                        # Calculate the bounding box of the tile
                        image = self.images[image_id]
                        width, height = image['width'], image['height']
                        tile_width = width // m
                        tile_height = height // m
                        x_min = tile_width * h_tile_id
                        y_min = tile_height * v_tile_id
                        x_max = x_min + tile_width
                        y_max = y_min + tile_height
                        tile_bbox = [x_min, y_min, tile_width, tile_height]
                        # tile_bbox = bbox_inflate(tile_bbox, 0.1)
                        orig_anno_bbox = anno['bbox']

                        # if the bbox of annotation is not in the tile, skip
                        if not bbox_overlap(tile_bbox, orig_anno_bbox):
                            continue

                        # Calculate the new bezier points
                        bezier_pts = anno['bezier_pts']

                        new_bbox = get_bbox_intersection(tile_bbox, orig_anno_bbox)
                        new_bbox[0] = new_bbox[0] - x_min
                        new_bbox[1] = new_bbox[1] - y_min
                        new_bbox = [round(val, 2) for val in new_bbox]

                        if not bbox_include(tile_bbox, orig_anno_bbox):
                            # Crop the bezier curve to the bounding box
                            bezier_pts_upper, crop_range_upper = bezier_bbox_crop(bezier_pts[:8], tile_bbox)
                            bezier_pts_lower, crop_range_lower = bezier_bbox_crop(bezier_pts[8:], tile_bbox,
                                                                                  reversed=True)

                            if bezier_pts_upper is None or bezier_pts_lower is None:
                                continue

                            # Average the crop ranges
                            crop_range = [(crop_range_upper[i] + crop_range_lower[i]) / 2 for i in range(2)]

                            text_rec = [rec for rec in anno['rec'] if rec < anno['rec'][-1]]

                            # Crop the text to the range
                            rec = text_range_crop(text_rec, crop_range)

                            if len(rec) == 0 or min(rec) == anno['rec'][-1]:
                                continue

                            new_rec = rec + [anno['rec'][-1]] * (25 - len(rec))

                            new_bezier_pts = bezier_pts_upper + bezier_pts_lower
                        else:
                            new_bezier_pts = bezier_pts
                            new_rec = anno['rec']

                        new_bezier_pts = [(new_bezier_pts[i], new_bezier_pts[i + 1]) for i in
                                          range(0, len(new_bezier_pts), 2)]
                        new_bezier_pts = [(p[0] - x_min, p[1] - y_min) for p in new_bezier_pts]
                        new_bezier_pts = [round(val, 2) for p in new_bezier_pts for val in p]

                        # Create a new annotation record
                        new_anno_record = {
                            "area": new_bbox[2] * new_bbox[3],
                            "bbox": new_bbox,
                            "category_id": 1,
                            "id": None,  # Generated at output stage
                            "image_id": new_image_id,
                            "iscrowd": 0,
                            "bezier_pts": new_bezier_pts,
                            "rec": new_rec
                        }

                        if new_image_id not in new_annos:
                            new_annos[new_image_id] = []

                        new_annos[new_image_id].append(new_anno_record)

        self.images = new_images
        self.annos = new_annos
        self.image_root_folder = tiled_image_folder

    def save_annotations(self, file_path, round_to=2):
        annotations = [anno for annos in self.annos.values() for anno in annos]
        for i, anno in enumerate(annotations):
            anno['id'] = i

        def pretty_floats(obj):
            if isinstance(obj, float):
                return round(obj, round_to)
            elif isinstance(obj, dict):
                return dict((k, pretty_floats(v)) for k, v in obj.items())
            return obj

        data = {
            "licenses": self.licenses,
            "info": self.info,
            "categories": self.categories,
            "images": list(self.images.values()),
            "annotations": annotations
        }

        with open(file_path, "w") as f:
            json.dump(pretty_floats(data), f)

    def register_image(self, image_name):
        # Check if the image is already registered
        for image in self.images.values():
            if image['file_name'] == image_name:
                print("Image already registered")
                return None

        # Check file exists
        image_path = os.path.join(self.image_root_folder, image_name)
        # print(image_path)
        if not os.path.exists(image_path):
            print("Image file not found")
            return None

        # Check width and height
        image = Image.open(image_path)
        width, height = image.size

        # Register the image
        # print(len(self.images))
        image_id = len(self.images)
        image_record = {
            "coco_url": "",
            "date_captured": "",
            "file_name": image_name,
            "flickr_url": "",
            "id": image_id,
            "license": 0,
            "width": width,
            "height": height
        }
        # if image_name == "mk_2_0_1.png":
        #     print(image_id)
        #     print(self.image_root_folder)
        self.images[image_id] = image_record
        self.annos[image_id] = []

        return image_id

    def _approx_bezier_v2(self, poly_upper_x, poly_upper_y, poly_lower_x, poly_lower_y):
        import bezier_utils as butils
        cpts_upper_x, cpts_upper_y = butils.bezier_from_polyline_v2(poly_upper_x, poly_upper_y)
        # print(poly_lower_x)
        # print(poly_lower_y)
        cpts_lower_x, cpts_lower_y = butils.bezier_from_polyline_v2(poly_lower_x, poly_lower_y)
        return True, cpts_upper_x, cpts_upper_y, cpts_lower_x, cpts_lower_y

    # def _approx_bezier(self, poly_upper_x, poly_upper_y, poly_lower_x, poly_lower_y, max_error_ratio = 0.5, max_attempts = 10):
    #     import bezier_utils as butils
    #
    #     success = False
    #     attempt = 0
    #     cpts_upper_x, cpts_upper_y = [], []
    #     cpts_lower_x, cpts_lower_y = [], []
    #     while not success and attempt < max_attempts:
    #         success = True
    #         error_ratio = max_error_ratio/max_attempts*(attempt + 1)
    #         poly_length_upper = butils.polyline_length(poly_upper_x, poly_upper_y)
    #         poly_length_lower = butils.polyline_length(poly_lower_x, poly_lower_y)
    #         avg_length = (poly_length_upper + poly_length_lower) / 2
    #         max_error = avg_length * error_ratio
    #
    #         cpts_upper_x, cpts_upper_y = butils.bezier_from_polyline(poly_upper_x, poly_upper_y, max_error)
    #
    #         cpts_lower_x, cpts_lower_y = butils.bezier_from_polyline(poly_lower_x, poly_lower_y, max_error)
    #
    #         bezier_length_upper = butils.bezier_length(cpts_upper_x, cpts_upper_y)
    #         bezier_length_lower = butils.bezier_length(cpts_lower_x, cpts_lower_y)
    #
    #         # Check if two bezier curves have similar length, if not, raise an error
    #         thresh = 1.2
    #         ratio_upper = poly_length_upper/bezier_length_upper
    #         ratio_lower = poly_length_lower/bezier_length_lower
    #         if ratio_upper > thresh or ratio_upper < 1/thresh or ratio_lower > thresh or ratio_lower < 1/thresh:
    #             success = False
    #             attempt += 1
    #
    #     return success, cpts_upper_x, cpts_upper_y, cpts_lower_x, cpts_lower_y

    def register_annotation_plain(self, image_id, _bezier_pts: 'list[float]', rec, bbox):
        # Check if the image_id is valid
        if image_id not in self.images:
            print("Image not found")
            return False

        # # Calculate the bounding box from the bezier points
        # bezier_pts = [(_bezier_pts[i], _bezier_pts[i + 1]) for i in range(0, len(_bezier_pts), 2)]
        # curve1 = bezier_pts[:4]
        # curve1_xx = [p[0] for p in curve1]
        # curve1_yy = [p[1] for p in curve1]
        # curve2 = bezier_pts[4:]
        # curve2_xx = [p[0] for p in curve2]
        # curve2_yy = [p[1] for p in curve2]
        #
        # nodes1 = np.asfortranarray([curve1_xx, curve1_yy])
        # nodes2 = np.asfortranarray([curve2_xx, curve2_yy])
        # bezier_curve1 = bezier.Curve(nodes1, degree=3)
        # bezier_curve2 = bezier.Curve(nodes2, degree=3)
        # x1, y1 = bezier_curve1.evaluate_multi(np.linspace(0, 1, 10))
        # x2, y2 = bezier_curve2.evaluate_multi(np.linspace(0, 1, 10))
        # x = np.concatenate([x1, x2])
        # y = np.concatenate([y1, y2])
        # x_min, x_max = np.min(x), np.max(x)
        # y_min, y_max = np.min(y), np.max(y)
        # bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

        # Register the annotation
        anno_record = {
            "area": bbox[2] * bbox[3],
            "bbox": bbox,
            "category_id": 1,
            "id": None,  # Generated at output stage
            "image_id": image_id,
            "iscrowd": 0,
            "bezier_pts": _bezier_pts,
            "rec": rec
        }
        self.annos[image_id].append(anno_record)

        return True

    def register_annotation_polygon(self, image_id, polygon: 'list[tuple]', rec):
        if min(rec) == rec[-1]:
            return False

        # Check if the image_id is valid
        if image_id not in self.images:
            print("Image not found")
            return False

        # Calculate the bounding box from polygon
        x = [p[0] for p in polygon]
        y = [p[1] for p in polygon]
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        bbox = [round(x_min, 2),
                round(y_min, 2),
                round(x_max - x_min, 2),
                round(y_max - y_min, 2)
                ]

        # Split polygon into upper and lower halves
        upper_half = polygon[:len(polygon) // 2]
        lower_half = polygon[len(polygon) // 2:]

        poly_upper_x, poly_upper_y = [p[0] for p in upper_half], [p[1] for p in upper_half]
        poly_lower_x, poly_lower_y = [p[0] for p in lower_half], [p[1] for p in lower_half]

        # Convert polyline to bezier curve
        success, cpts_upper_x, cpts_upper_y, cpts_lower_x, cpts_lower_y = self._approx_bezier_v2(poly_upper_x,
                                                                                                 poly_upper_y,
                                                                                                 poly_lower_x,
                                                                                                 poly_lower_y)

        if not success:
            print("Failed to approximate bezier curve")
            raise Exception("Failed to approximate bezier curve")

        # bezier_pts = [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8]
        bezier_pts = []
        for i in range(4):
            bezier_pts.append(round(cpts_upper_x[i], 2))
            bezier_pts.append(round(cpts_upper_y[i], 2))
        for i in range(4):
            bezier_pts.append(round(cpts_lower_x[i], 2))
            bezier_pts.append(round(cpts_lower_y[i], 2))

        # Register the annotation
        anno_record = {
            "area": bbox[2] * bbox[3],
            "bbox": bbox,
            "category_id": 1,
            "id": None,  # Generated at output stage
            "image_id": image_id,
            "iscrowd": 0,
            "bezier_pts": bezier_pts,
            "rec": rec
        }
        self.annos[image_id].append(anno_record)

        return True

    def register_annotation_polygon_cvat(self, image_id, polygon: 'list[list]', rec):
        if min(rec) == rec[-1]:
            return False

        # Check if the image_id is valid
        if image_id not in self.images:
            print("Image not found")
            return False

        # Calculate the bounding box from polygon
        all_vertices = polygon[0] + polygon[1]
        x = [p[0] for p in all_vertices]
        y = [p[1] for p in all_vertices]
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        bbox = [round(x_min, 2),
                round(y_min, 2),
                round(x_max - x_min, 2),
                round(y_max - y_min, 2)
                ]

        # Split polygon into upper and lower halves
        upper_half = polygon[0]
        lower_half = polygon[1]

        poly_upper_x, poly_upper_y = [p[0] for p in upper_half], [p[1] for p in upper_half]
        poly_lower_x, poly_lower_y = [p[0] for p in lower_half], [p[1] for p in lower_half]

        # Convert polyline to bezier curve
        success, cpts_upper_x, cpts_upper_y, cpts_lower_x, cpts_lower_y = self._approx_bezier_v2(poly_upper_x,
                                                                                                 poly_upper_y,
                                                                                                 poly_lower_x,
                                                                                                 poly_lower_y)

        if not success:
            print("Failed to approximate bezier curve")
            raise Exception("Failed to approximate bezier curve")

        # bezier_pts = [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8]
        bezier_pts = []
        for i in range(4):
            bezier_pts.append(round(cpts_upper_x[i], 2))
            bezier_pts.append(round(cpts_upper_y[i], 2))
        for i in range(4):
            bezier_pts.append(round(cpts_lower_x[i], 2))
            bezier_pts.append(round(cpts_lower_y[i], 2))

        # Register the annotation
        anno_record = {
            "area": bbox[2] * bbox[3],
            "bbox": bbox,
            "category_id": 1,
            "id": None,  # Generated at output stage
            "image_id": image_id,
            "iscrowd": 0,
            "bezier_pts": bezier_pts,
            "rec": rec
        }
        self.annos[image_id].append(anno_record)

        return True

    def show_image(self, image_id):
        image = self.images[image_id]
        print(image)
        annos = self.annos[image_id]
        print(annos)

    def draw_annotations(self, image_id, text_decoder=Encoding.decode_text_96):
        annos = self.annos[image_id]
        image = self.images[image_id]
        image_path = os.path.join(self.image_root_folder, image['file_name'])
        image = Image.open(image_path)
        print(image_path)
        plt.clf()
        plt.imshow(image)
        for anno in annos:
            bbox = anno['bbox']
            # plt.plot([bbox[0], bbox[0], bbox[0] + bbox[2], bbox[0] + bbox[2], bbox[0]], [bbox[1], bbox[1] + bbox[3], bbox[1] + bbox[3], bbox[1], bbox[1]])
            # if text_decoder is not None:
            #     text = text_decoder(anno['rec'])
            # else:
            #     text = ','.join([rec for rec in anno['rec'] if rec < 96])
            # plt.text(bbox[0], bbox[1], text)
            bezier_pts = anno['bezier_pts']
            # Group the points into 4-tuples
            bezier_pts = [(bezier_pts[i], bezier_pts[i + 1]) for i in range(0, len(bezier_pts), 2)]
            curve1 = bezier_pts[:4]
            curve1_xx = [p[0] for p in curve1]
            curve1_yy = [p[1] for p in curve1]
            curve2 = bezier_pts[4:]
            curve2_xx = [p[0] for p in curve2]
            curve2_yy = [p[1] for p in curve2]

            nodes1 = np.asfortranarray([curve1_xx, curve1_yy])
            nodes2 = np.asfortranarray([curve2_xx, curve2_yy])
            bezier_curve1 = bezier.Curve(nodes1, degree=3)
            bezier_curve2 = bezier.Curve(nodes2, degree=3)

            # plt.scatter(curve1_xx+curve2_xx, curve1_yy+curve2_yy, c=np.random.rand(3), s=2)

            x1, y1 = bezier_curve1.evaluate_multi(np.linspace(0, 1, 100))
            x2, y2 = bezier_curve2.evaluate_multi(np.linspace(0, 1, 100))
            plt.plot(np.concatenate([x1, x2]), np.concatenate([y1, y2]), 'orange', linewidth=1)
            plt.plot(x1, y1, 'blue', linewidth=1)
            plt.plot(x2, y2, 'blue', linewidth=1)
            plt.scatter(x1[0], y1[0], color='red', s=3)
            # plt.scatter(x2[0], y2[0], color = 'orange', s=3)

            if text_decoder is not None:
                text = text_decoder(anno['rec'])
                plt.text(x2[-1], y2[-1], text, fontsize=7)
        # plt.show()

        plt.savefig("./vis/{}.png".format(image_id))
        plt.close()


class GrouperDataset:
    def __init__(self, image_root_folder) -> None:
        '''
        images: {image_id: {'file_name': str, 'id': int, 'width': int, 'height': int}}
        annos: {image_id: [{'rec': [[int,]], 'bezier': [[float,]]}]}

        'rec' is a 35-element list of integers representing the text in the polygon, for example:
        self.annos[image_id][toponym_id]['rec'][word_id] = [char_id1, char_id2, char_id3, char_id4, char_id5, char_id6, char_id7, char_id8, ...]

        'bezier' is a list of 16 floats representing 8 control points of the bezier curve
        for example:
        self.annos[image_id][toponym_id]['bezier'][word_id] = [
            upper_x1, upper_y1,
            upper_x2, upper_y2,
            upper_x3, upper_y3,
            upper_x4, upper_y4,

            lower_x1, lower_y1,
            lower_x2, lower_y2,
            lower_x3, lower_y3,
            lower_x4, lower_y4
        ]
        '''
        self.images = {}
        self.annos = {}
        self.words = {}
        self.image_root_folder = image_root_folder

    def register_image(self, image_name):
        # Check if the image is already registered
        for image in self.images.values():
            if image['file_name'] == image_name:
                print("Image already registered")
                return None

        # Check file exists
        image_path = os.path.join(self.image_root_folder, image_name)
        if not os.path.exists(image_path):
            print("Image file not found")
            return None

        # Check width and height
        image = Image.open(image_path)
        width, height = image.size

        # Register the image
        image_id = len(self.images)
        image_record = {
            "file_name": image_name,
            "id": image_id,
            "width": width,
            "height": height,
        }
        self.images[image_id] = image_record
        self.annos[image_id] = []
        self.words[image_id] = []

        return image_id

    def _approx_bezier_v2(self, poly_upper_x, poly_upper_y, poly_lower_x, poly_lower_y):
        import bezier_utils as butils
        cpts_upper_x, cpts_upper_y = butils.bezier_from_polyline_v2(poly_upper_x, poly_upper_y)
        # print(poly_lower_x)
        # print(poly_lower_y)
        cpts_lower_x, cpts_lower_y = butils.bezier_from_polyline_v2(poly_lower_x, poly_lower_y)
        return True, cpts_upper_x, cpts_upper_y, cpts_lower_x, cpts_lower_y

    # def _approx_bezier(self, poly_upper_x, poly_upper_y, poly_lower_x, poly_lower_y, max_error_ratio=0.5,
    #                    max_attempts=10):
    #     import bezier_utils as butils
    #
    #     success = False
    #     attempt = 0
    #     cpts_upper_x, cpts_upper_y = [], []
    #     cpts_lower_x, cpts_lower_y = [], []
    #     while not success and attempt < max_attempts:
    #         success = True
    #         error_ratio = max_error_ratio / max_attempts * (attempt + 1)
    #         poly_length_upper = butils.polyline_length(poly_upper_x, poly_upper_y)
    #         poly_length_lower = butils.polyline_length(poly_lower_x, poly_lower_y)
    #         avg_length = (poly_length_upper + poly_length_lower) / 2
    #         max_error = avg_length * error_ratio
    #
    #         cpts_upper_x, cpts_upper_y = butils.bezier_from_polyline(poly_upper_x, poly_upper_y, max_error)
    #
    #         cpts_lower_x, cpts_lower_y = butils.bezier_from_polyline(poly_lower_x, poly_lower_y, max_error)
    #
    #         bezier_length_upper = butils.bezier_length(cpts_upper_x, cpts_upper_y)
    #         bezier_length_lower = butils.bezier_length(cpts_lower_x, cpts_lower_y)
    #
    #         # Check if two bezier curves have similar length, if not, raise an error
    #         thresh = 1.2
    #         ratio_upper = poly_length_upper / bezier_length_upper
    #         ratio_lower = poly_length_lower / bezier_length_lower
    #         if ratio_upper > thresh or ratio_upper < 1 / thresh or ratio_lower > thresh or ratio_lower < 1 / thresh:
    #             success = False
    #             attempt += 1
    #
    #     return success, cpts_upper_x, cpts_upper_y, cpts_lower_x, cpts_lower_y

    def register_annotation(self, image_id, text_polygon_dict, upper_lower_split):
        import bezier_utils as butils
        if image_id not in self.images:
            print("Image not found")
            return None

        if not text_polygon_dict:
            print("Empty annotation")
            return None

        group = {'rec': [], 'bezier': []}
        for rec, polygon in zip(text_polygon_dict['rec'], text_polygon_dict['polygon']):

            if upper_lower_split:
                upper_half = polygon[0]
                lower_half = list(reversed(polygon[1]))  # the grouper requires upper and lower has the same direction
            else:
                upper_half = polygon[:len(polygon)//2]
                lower_half = list(reversed(polygon[len(polygon)//2:]))
            # print(polygon)
            # print(upper_half)
            # print(lower_half)
            poly_upper_x, poly_upper_y = [p[0] for p in upper_half], [p[1] for p in upper_half]
            poly_lower_x, poly_lower_y = [p[0] for p in lower_half], [p[1] for p in lower_half]

            success, cpts_upper_x, cpts_upper_y, cpts_lower_x, cpts_lower_y = self._approx_bezier_v2(poly_upper_x,
                                                                                                  poly_upper_y,
                                                                                                  poly_lower_x,
                                                                                                  poly_lower_y)

            if not success:
                print("Failed to approximate bezier curve")
                raise Exception("Failed to approximate bezier curve")

            # bezier_pts = [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8]
            bezier_pts = []
            for i in range(4):
                bezier_pts.append(round(cpts_upper_x[i], 2))
                bezier_pts.append(round(cpts_upper_y[i], 2))
            for i in range(4):
                bezier_pts.append(round(cpts_lower_x[i], 2))
                bezier_pts.append(round(cpts_lower_y[i], 2))

            upper_half_xx, upper_half_yy = butils.bezier_to_polyline(cpts_upper_x, cpts_upper_y, num_pts=40)
            lower_half_xx, lower_half_yy = butils.bezier_to_polyline(cpts_lower_x, cpts_lower_y, num_pts=40)

            center = (
                (sum(upper_half_xx) + sum(lower_half_xx)) / (len(upper_half_xx) + len(lower_half_xx)),
                (sum(upper_half_yy) + sum(lower_half_yy)) / (len(upper_half_yy) + len(lower_half_yy)))

            group['rec'].append(rec)
            group['bezier'].append(bezier_pts)
            self.words[image_id].append(
                {'rec': rec, 'bezier': bezier_pts, 'center': center, 'group_id': len(self.annos[image_id]),
                 'id_in_group': len(group['rec']) - 1})

        self.annos[image_id].append(group)

        return True

    def gen_font_embed(self, image_id, deepfont_encoder):
        import DeepFont
        import bezier_utils as butils

        net = deepfont_encoder

        image = self.images[image_id]
        image_path = os.path.join(self.image_root_folder, image['file_name'])
        image = Image.open(image_path)

        words = self.words[image_id]

        # Get word snippets from the image according to the annotation
        snippets = []

        for word in words:
            bezier_pts = word['bezier']
            snippet = butils.get_bezier_bbox(image, bezier_pts)
            snippets.append(snippet)

        # Encode the snippets
        features = DeepFont.EncodeFontBatch(net, snippets)

        self.words[image_id] = [{**word, 'font_embed': feature} for word, feature in zip(words, features)]

    def gen_font_embed_all(self, deepfont_encoder_path, device='cpu'):
        '''
            One minute inference for all 200 full map patches in rumsey dataset
        '''

        net = DeepFont.load_model(deepfont_encoder_path, device)

        for image_id in tqdm(self.images):
            self.gen_font_embed(image_id, net)

    def save_annotations_to_file(self, file_path):
        with open(file_path, 'w') as f:
            f.write(json.dumps({'images': self.images, 'annotations': self.annos, 'words': self.words}))

    def load_annotations_from_file(self, file_path):
        with open(file_path, 'r') as f:
            data = json.loads(f.read())
            self.images = {int(key): value for key, value in data['images'].items()}
            self.annos = {int(key): value for key, value in data['annotations'].items()}
            self.words = {int(key): value for key, value in data['words'].items()}

    def draw_annotations(self, image_id, decode_text):
        import bezier_utils as butils
        image = self.images[image_id]
        image_path = os.path.join(self.image_root_folder, image['file_name'])
        image = plt.imread(image_path)
        plt.imshow(image)
        for group in self.annos[image_id]:
            # Assign random color to each group
            color = (np.random.rand(), np.random.rand(), np.random.rand())
            avg_xs = []
            avg_ys = []
            for rec, bezier_pts in zip(group['rec'], group['bezier']):
                upper_half = [(bezier_pts[i], bezier_pts[i + 1]) for i in range(0, 8, 2)]
                lower_half = [(bezier_pts[i], bezier_pts[i + 1]) for i in range(8, 16, 2)]

                upper_half_x = [p[0] for p in upper_half]
                upper_half_y = [p[1] for p in upper_half]
                lower_half_x = [p[0] for p in lower_half]
                lower_half_y = [p[1] for p in lower_half]

                # Convert bezier curve to polyline
                upper_half_x, upper_half_y = butils.bezier_to_polyline(upper_half_x, upper_half_y)
                lower_half_x, lower_half_y = butils.bezier_to_polyline(lower_half_x, lower_half_y)

                # Plot begin points
                plt.scatter(upper_half_x[0], upper_half_y[0], color='black')
                plt.scatter(lower_half_x[0], lower_half_y[0], color='black')

                plt.plot(upper_half_x, upper_half_y, color=color)
                plt.plot(lower_half_x, lower_half_y, color=color)
                text = decode_text(rec)
                plt.text(upper_half[0][0], upper_half[0][1], text, color=color)

                avg_x = (upper_half_x[0] + upper_half_x[-1] + lower_half_x[0] + lower_half_x[-1]) / 4
                avg_y = (upper_half_y[0] + upper_half_y[-1] + lower_half_y[0] + lower_half_y[-1]) / 4

                avg_xs.append(avg_x)
                avg_ys.append(avg_y)

            avg_xs = avg_xs
            avg_ys = avg_ys
            plt.plot(avg_xs, avg_ys, color='red')
            plt.scatter(avg_xs[0], avg_ys[0], color='black')

        plt.show()

    def sample(self, image_id, sample_count, closest_pts_count=15, non_overlap=False):
        import bezier_utils as butils
        samples = []
        words = self.words[image_id]

        if sample_count > len(words):
            sample_count = len(words)

        word_samples = np.random.choice(words, sample_count, replace=not non_overlap)

        def _anchors(_word, type='111222333'):
            arr = [
                ((_word['bezier'][0] + _word['bezier'][14]) / 2, (_word['bezier'][1] + _word['bezier'][15]) / 2),
                ((_word['bezier'][6] + _word['bezier'][8]) / 2, (_word['bezier'][7] + _word['bezier'][9]) / 2),
                _word['center']
            ]
            if type == '111222333':
                return [arr[0], arr[0], arr[0], arr[1], arr[1], arr[1], arr[2], arr[2], arr[2]]
            elif type == '123123123':
                return [arr[0], arr[1], arr[2], arr[0], arr[1], arr[2], arr[0], arr[1], arr[2]]

        for word in word_samples:
            word_anchors = _anchors(word, type='111222333')
            _, nabb_l, nabb_s = butils.get_bezier_bbox_params(word['bezier'])

            nabb_l_new = nabb_l * np.linalg.norm(nabb_s) / np.linalg.norm(nabb_l)

            import cv2
            mat_src = np.float32(
                [np.array(word['center']), np.array(word['center']) + nabb_l, np.array(word['center']) + nabb_s])
            mat_dst = np.float32(
                [np.array(word['center']), np.array(word['center']) + nabb_l_new, np.array(word['center']) + nabb_s])
            T = cv2.getAffineTransform(mat_src, mat_dst)

            anchors_to_compare = []
            dictionary_candidates = []
            inliners = []
            for w in words:
                if w['group_id'] == word['group_id']:
                    inliners.append(w)
                anchors_to_compare.append(_anchors(w, type='123123123'))
                dictionary_candidates.append(w)

            dist_vectors = np.array(anchors_to_compare) - np.array(word_anchors)
            dist_vectors_transformed = np.einsum('ji,akj->aki', T[:, :2], dist_vectors)
            distances = np.min(np.linalg.norm(dist_vectors_transformed, axis=2), axis=1)

            closest_indices = np.argsort(distances)[:closest_pts_count + 1]

            dictionary = []
            for i in closest_indices:
                dictionary.append(dictionary_candidates[i])

            complete = True
            for w in inliners:
                if w not in dictionary:
                    # Remove the word from the inliners
                    inliners.remove(w)
                    complete = False

            '''
                word: the word to be sampled around
                dictionary: the closest set of words to the word (non-inclusive of the word itself)
                toponym: the words that are in the same toponym as the word (preserve the order)
                complete: whether the dictionary contains all words in topynym, or, whether the toponym is complete in this scope
            '''
            samples.append({'word': word, 'dictionary': dictionary[1:], 'toponym': inliners, 'complete': complete})

        return samples

    def sample_ratio(self, image_id, sample_ratio, closest_pts_count=15, non_overlap=False):
        sample_count = int(len(self.words[image_id]) * sample_ratio)
        return self.sample(image_id, sample_count, closest_pts_count, non_overlap)

    def draw_sample(self, image_id, sample, decode_text):
        '''
            sample = {'word': word, 'dictionary': dictionary, 'toponym': toponym, 'complete': complete}
        '''
        words = self.words[image_id]

        import bezier_utils as butils
        image = self.images[image_id]
        image_path = os.path.join(self.image_root_folder, image['file_name'])
        image = plt.imread(image_path)
        plt.imshow(image)

        # Draw the sample
        word = sample['word']
        group_id = word['group_id']
        dictionary = sample['dictionary']
        toponym = sample['toponym']
        use_font_embed = False

        # Draw all words
        for aword in words:
            bezier_pts = aword['bezier']
            upper_half = [(bezier_pts[i], bezier_pts[i + 1]) for i in range(0, 8, 2)]
            lower_half = [(bezier_pts[i], bezier_pts[i + 1]) for i in range(8, 16, 2)]

            upper_half_x = [p[0] for p in upper_half]
            upper_half_y = [p[1] for p in upper_half]
            lower_half_x = [p[0] for p in lower_half]
            lower_half_y = [p[1] for p in lower_half]

            # Convert bezier curve to polyline
            upper_half_x, upper_half_y = butils.bezier_to_polyline(upper_half_x, upper_half_y)
            lower_half_x, lower_half_y = butils.bezier_to_polyline(lower_half_x, lower_half_y)

            plt.plot(upper_half_x, upper_half_y, color='grey')
            plt.plot(lower_half_x, lower_half_y, color='grey')
            text = decode_text(aword['rec'])
            aword_group_id = aword['group_id']
            if aword_group_id == group_id:
                plt.text(upper_half[0][0], upper_half[0][1], text, color='magenta')
            else:
                plt.text(upper_half[0][0], upper_half[0][1], text, color='grey')

        if 'font_embed' in word:
            word_font_embed = word['font_embed']
            use_font_embed = True

        for dword in dictionary:
            bezier_pts = dword['bezier']
            upper_half = [(bezier_pts[i], bezier_pts[i + 1]) for i in range(0, 8, 2)]
            lower_half = [(bezier_pts[i], bezier_pts[i + 1]) for i in range(8, 16, 2)]

            upper_half_x = [p[0] for p in upper_half]
            upper_half_y = [p[1] for p in upper_half]
            lower_half_x = [p[0] for p in lower_half]
            lower_half_y = [p[1] for p in lower_half]

            # Convert bezier curve to polyline
            upper_half_x, upper_half_y = butils.bezier_to_polyline(upper_half_x, upper_half_y)
            lower_half_x, lower_half_y = butils.bezier_to_polyline(lower_half_x, lower_half_y)

            similarity_str = ''
            if use_font_embed:
                dword_font_embed = dword['font_embed']
                euclidean_distance = np.linalg.norm(np.array(word_font_embed) - np.array(dword_font_embed))
                similarity_str = f':{euclidean_distance:.2f}'

            plt.plot(upper_half_x, upper_half_y, color='red')
            plt.plot(lower_half_x, lower_half_y, color='red')
            text = decode_text(dword['rec'])
            plt.text(upper_half[0][0], upper_half[0][1], text + similarity_str, color='red')

        for id, tword in enumerate(toponym):
            bezier_pts = tword['bezier']
            upper_half = [(bezier_pts[i], bezier_pts[i + 1]) for i in range(0, 8, 2)]
            lower_half = [(bezier_pts[i], bezier_pts[i + 1]) for i in range(8, 16, 2)]

            upper_half_x = [p[0] for p in upper_half]
            upper_half_y = [p[1] for p in upper_half]
            lower_half_x = [p[0] for p in lower_half]
            lower_half_y = [p[1] for p in lower_half]

            # Convert bezier curve to polyline
            upper_half_x, upper_half_y = butils.bezier_to_polyline(upper_half_x, upper_half_y)
            lower_half_x, lower_half_y = butils.bezier_to_polyline(lower_half_x, lower_half_y)

            plt.plot(upper_half_x, upper_half_y, color='blue')
            plt.plot(lower_half_x, lower_half_y, color='blue')
            text = decode_text(tword['rec'])
            plt.text(lower_half[0][0], lower_half[0][1], text + f': {id}', color='blue')

        for wword in [word]:
            bezier_pts = wword['bezier']
            upper_half = [(bezier_pts[i], bezier_pts[i + 1]) for i in range(0, 8, 2)]
            lower_half = [(bezier_pts[i], bezier_pts[i + 1]) for i in range(8, 16, 2)]

            upper_half_x = [p[0] for p in upper_half]
            upper_half_y = [p[1] for p in upper_half]
            lower_half_x = [p[0] for p in lower_half]
            lower_half_y = [p[1] for p in lower_half]

            # Convert bezier curve to polyline
            upper_half_x, upper_half_y = butils.bezier_to_polyline(upper_half_x, upper_half_y)
            lower_half_x, lower_half_y = butils.bezier_to_polyline(lower_half_x, lower_half_y)

            plt.plot(upper_half_x, upper_half_y, color='yellow')
            plt.plot(lower_half_x, lower_half_y, color='yellow')
            text = decode_text(wword['rec'])
            plt.text(upper_half[0][0], upper_half[0][1], text, color='yellow')

        if sample['complete']:
            plt.title('Complete')
        else:
            plt.title('Incomplete')
        plt.show()

    def bezier_centralized(self, bezier_pts: 'list[float]', center_pt: 'list[float]') -> 'list[float]':
        '''
        :param bezier_pts: a list of 16 floats representing 8 control points of the bezier curve: [
                upper_x1, upper_y1,
                upper_x2, upper_y2,
                upper_x3, upper_y3,
                upper_x4, upper_y4,

                lower_x1, lower_y1,
                lower_x2, lower_y2,
                lower_x3, lower_y3,
                lower_x4, lower_y4
            ]
        '''
        centralized_pts = []
        for i in range(len(bezier_pts)):
            if i % 2 == 0:
                centralized_pts.append(bezier_pts[i] - center_pt[0])
            else:
                centralized_pts.append(bezier_pts[i] - center_pt[1])

        return centralized_pts

    def yield_samples(self, img_ids, sample_ratio, augmentation=True, variant_length=False):

        yielded = []

        for img_id in img_ids:

            samples = self.sample_ratio(img_id, sample_ratio)

            for sample in samples:
                query_text = sample['word']['rec']
                query_bezier_centralized = self.bezier_centralized(sample['word']['bezier'], sample['word']['center'])
                query_bezier = sample['word']['bezier']
                query_font = sample['word']['font_embed']
                query_id_in_group = sample['word']['id_in_group']
                neighbour_text = []
                neighbour_bezier_centralized = []
                neighbour_bezier = []
                neighbour_of_the_same_group = []
                neighbour_font = []
                neighbour_id_in_group = []

                for neighbour in sample['dictionary']:
                    neighbour_text.append(neighbour['rec'])
                    neighbour_bezier.append(neighbour['bezier'])
                    neighbour_bezier_centralized.append(
                        self.bezier_centralized(neighbour['bezier'], sample['word']['center']))
                    neighbour_of_the_same_group.append(int(neighbour['group_id'] == sample['word']['group_id']))
                    neighbour_font.append(neighbour['font_embed'])
                    neighbour_id_in_group.append(neighbour['id_in_group'])

                # blend query and neighbour
                source_text = neighbour_text + [query_text]
                source_bezier_centralized = neighbour_bezier_centralized + [query_bezier_centralized]
                source_bezier = neighbour_bezier + [query_bezier]
                source_toponym_mask = neighbour_of_the_same_group + [1]
                source_font = neighbour_font + [query_font]
                source_id_in_group = neighbour_id_in_group + [query_id_in_group]

                if augmentation:
                    if 1 in neighbour_of_the_same_group:
                        n_augmentations = 20
                    else:
                        n_augmentations = 3
                else:
                    n_augmentations = 1

                if variant_length:

                    # extract the ids of toponym and randomly sample some noisy neighbours
                    toponym_ids_ = [i for i, t in enumerate(source_toponym_mask) if t == 1]
                    non_toponym_ids = [i for i, t in enumerate(source_toponym_mask) if t != 1]

                    # sample a random number of non toponym ids
                    num_to_sample = random.randint(0, len(non_toponym_ids))
                    sampled_non_toponym_ids = random.sample(non_toponym_ids, num_to_sample)

                    synthesized_ids_in_list = toponym_ids_ + sampled_non_toponym_ids

                    source_text_ = []
                    source_bezier_centralized_ = []
                    source_bezier_ = []
                    source_toponym_mask_ = []
                    source_font_ = []
                    source_id_in_group_ = []
                    for idx in synthesized_ids_in_list:
                        source_text_.append(source_text[idx])
                        source_bezier_.append(source_bezier[idx])
                        source_bezier_centralized_.append(source_bezier_centralized[idx])
                        source_toponym_mask_.append(source_toponym_mask[idx])
                        source_font_.append(source_font[idx])
                        source_id_in_group_.append(source_id_in_group[idx])
                    query_id_in_sythesized_list = len(toponym_ids_) - 1

                    indices_source = list(range(len(source_text_)))
                    for _ in range(n_augmentations):
                        random.shuffle(indices_source)

                        indices_no_query = [i for i in indices_source if i != query_id_in_sythesized_list]
                        neighbour_text_ = [source_text_[i] for i in indices_no_query]
                        neighbour_bezier_centralized_ = [source_bezier_centralized_[i] for i in indices_no_query]
                        neighbour_bezier_ = [source_bezier_[i] for i in indices_no_query]
                        neighbour_of_the_same_group_ = [source_toponym_mask_[i] for i in indices_no_query]
                        neighbour_font_ = [source_font_[i] for i in indices_no_query]

                        # blend query and neighbour
                        source_text_ = [source_text_[i] for i in indices_source]
                        source_bezier_centralized_ = [source_bezier_centralized_[i] for i in indices_source]
                        source_bezier_ = [source_bezier_[i] for i in indices_source]
                        source_toponym_mask_ = [source_toponym_mask_[i] for i in indices_source]
                        source_font_ = [source_font_[i] for i in indices_source]
                        source_id_in_group_ = [source_id_in_group_[i] for i in indices_source]

                        # get toponym id in reading order
                        toponym_id_in_source_ = [i for i, v in enumerate(source_toponym_mask_) if v == 1]
                        toponym_id_in_group_ = [source_id_in_group_[i] for i in toponym_id_in_source_]
                        toponym_id_sorted_in_source_ = [x[1] for x in
                                                        sorted(list(zip(toponym_id_in_group_, toponym_id_in_source_)),
                                                               key=lambda x: x[0])]
                        query_id_in_source_ = indices_source.index(query_id_in_sythesized_list)
                        toponym_len = len(toponym_id_sorted_in_source_)
                        # will add sot: 0, eot: 1 to the front in batch processing

                        source_len = len(source_text_)

                        yielded.append({'query_text': query_text,
                                        'query_bezier_centralized': query_bezier_centralized,
                                        'neighbour_text': neighbour_text_,
                                        'neighbour_bezier_centralized': neighbour_bezier_centralized_,
                                        'neighbour_of_the_same_group': neighbour_of_the_same_group_,
                                        'query_bezier': query_bezier,
                                        'neighbour_bezier': neighbour_bezier_,
                                        'query_font': query_font,
                                        'neighbour_font': neighbour_font_,
                                        'source_text': source_text_,
                                        'source_bezier_centralized': source_bezier_centralized_,
                                        'source_bezier': source_bezier_,
                                        'source_toponym_mask': source_toponym_mask_,
                                        'source_font': source_font_,
                                        'toponym_id_sorted_in_source': toponym_id_sorted_in_source_,
                                        'query_id_in_source': query_id_in_source_,
                                        'img_id': img_id,
                                        'toponym_len': toponym_len,
                                        'source_len': source_len})

                else:
                    indices_source = list(range(len(source_text)))
                    for _ in range(n_augmentations):
                        random.shuffle(indices_source)

                        indices_no_query = [i for i in indices_source if i != len(source_text) - 1]
                        neighbour_text_ = [neighbour_text[i] for i in indices_no_query]
                        neighbour_bezier_centralized_ = [neighbour_bezier_centralized[i] for i in indices_no_query]
                        neighbour_bezier_ = [neighbour_bezier[i] for i in indices_no_query]
                        neighbour_of_the_same_group_ = [neighbour_of_the_same_group[i] for i in indices_no_query]
                        neighbour_font_ = [neighbour_font[i] for i in indices_no_query]
                        # blend query and neighbour
                        source_text_ = [source_text[i] for i in indices_source]
                        source_bezier_centralized_ = [source_bezier_centralized[i] for i in indices_source]
                        source_bezier_ = [source_bezier[i] for i in indices_source]
                        source_toponym_mask_ = [source_toponym_mask[i] for i in indices_source]
                        source_font_ = [source_font[i] for i in indices_source]
                        source_id_in_group_ = [source_id_in_group[i] for i in indices_source]
                        # get toponym id in reading order
                        toponym_id_in_source_ = [i for i, v in enumerate(source_toponym_mask_) if v == 1]
                        toponym_id_in_group_ = [source_id_in_group_[i] for i in toponym_id_in_source_]
                        toponym_id_sorted_in_source_ = [x[1] for x in
                                                        sorted(list(zip(toponym_id_in_group_, toponym_id_in_source_)),
                                                               key=lambda x: x[0])]
                        query_id_in_source_ = indices_source.index(len(source_text) - 1)
                        # add sot, eot, and paddings
                        toponym_len = len(toponym_id_sorted_in_source_)
                        toponym_id_sorted_in_source_ = [len(source_text_)] + toponym_id_sorted_in_source_ + [
                            len(source_text_) + 1] + [len(source_text_) + 2] * (
                                                                   len(source_text) - len(toponym_id_sorted_in_source_))

                        yielded.append({'query_text': query_text,
                                        'query_bezier_centralized': query_bezier_centralized,
                                        'neighbour_text': neighbour_text_,
                                        'neighbour_bezier_centralized': neighbour_bezier_centralized_,
                                        'neighbour_of_the_same_group': neighbour_of_the_same_group_,
                                        'query_bezier': query_bezier,
                                        'neighbour_bezier': neighbour_bezier_,
                                        'query_font': query_font,
                                        'neighbour_font': neighbour_font_,
                                        'source_text': source_text_,
                                        'source_bezier_centralized': source_bezier_centralized_,
                                        'source_bezier': source_bezier_,
                                        'source_toponym_mask': source_toponym_mask_,
                                        'source_font': source_font_,
                                        'toponym_id_sorted_in_source': toponym_id_sorted_in_source_,
                                        'query_id_in_source': query_id_in_source_,
                                        'img_id': img_id,
                                        'toponym_len': toponym_len})

        return yielded

    def get_train_test_set(self, train_ratio=0.9, sample_ratio=1.0, random_seed=42):

        # filter out images with no group labels
        valid_image_ids = []
        for i in range(len(self.annos)):
            num_groups = len(self.annos[i])
            num_words = 0
            for group in self.annos[i]:
                num_words += len(group['bezier'])
            if num_groups != num_words:
                valid_image_ids.append(i)

        train_img_ids, test_img_ids = train_test_split(valid_image_ids, train_size=train_ratio,
                                                       test_size=1 - train_ratio, random_state=random_seed)
        self.train_set = self.yield_samples(train_img_ids, sample_ratio=1.0, augmentation=True, variant_length=True)
        self.test_set = self.yield_samples(test_img_ids, sample_ratio=sample_ratio, augmentation=False,
                                           variant_length=True)

        return self.train_set, self.test_set

    def predict_plot(self,
                     query_pts,
                     query_text,
                     neighbour_pts: list,
                     neighbour_text,
                     gt_label: 'list[int]',
                     predicted_label: 'list[int]',
                     img_id: int,
                     img_name: int,
                     predicted_query_order=-1,
                     predicted_neighbour_order=[-1] * 15):
        '''
        :param neighbour_pts: a list of lists, each list consists of 16 floats representing 8 control points
                of the upper and lower bezier curves of a polygon: [
                upper_x1, upper_y1,
                upper_x2, upper_y2,
                upper_x3, upper_y3,
                upper_x4, upper_y4,

                lower_x1, lower_y1,
                lower_x2, lower_y2,
                lower_x3, lower_y3,
                lower_x4, lower_y4
            ]
                gt_label, predicted_label: a list of the same length as bezier_pts, each element is a binary variable,
                indicating the class of the polygon
        :return a plot of bezier curves, the curves of polygons with gt_label 1 are colored yellow;
        the curves of polygons with predicted_label 1 are colored green; others are colored blue
        '''
        # Assuming self.images, self.image_root_folder, query_pts, query_text, neighbour_pts, gt_label, predicted_label, neighbour_text, img_name are defined
        image = self.images[img_id]
        image_path = os.path.join(self.image_root_folder, image['file_name'])
        background = Image.open(image_path).convert("RGBA")

        # Create a drawing context
        alpha = 0.7  # Set the desired transparency level (0.0 to 1.0)
        enhancer = ImageEnhance.Brightness(background.split()[3])
        background.putalpha(enhancer.enhance(alpha))
        draw = ImageDraw.Draw(background)

        # Plot the query
        upper_half = [(query_pts[i], query_pts[i + 1]) for i in range(0, 8, 2)]
        lower_half = [(query_pts[i], query_pts[i + 1]) for i in range(8, 16, 2)]

        upper_half_x = [p[0] for p in upper_half]
        upper_half_y = [p[1] for p in upper_half]
        lower_half_x = [p[0] for p in lower_half]
        lower_half_y = [p[1] for p in lower_half]

        upper_half_x, upper_half_y = butils.bezier_to_polyline(upper_half_x, upper_half_y)
        lower_half_x, lower_half_y = butils.bezier_to_polyline(lower_half_x, lower_half_y)

        draw.line(list(zip(upper_half_x, upper_half_y)), fill="orange", width=3)
        draw.line(list(zip(lower_half_x, lower_half_y)), fill="orange", width=3)
        draw.text((lower_half[0][0], lower_half[0][1]),
                  query_text + " Order: {}".format(predicted_query_order),
                  fill="orange")

        x_min = min(min(upper_half_x), min(lower_half_x))
        x_max = max(max(upper_half_x), max(lower_half_x))
        y_min = min(min(upper_half_y), min(lower_half_y))
        y_max = max(max(upper_half_y), max(lower_half_y))

        for i, pts in enumerate(neighbour_pts):
            upper_half = [(pts[i], pts[i + 1]) for i in range(0, 8, 2)]
            lower_half = [(pts[i], pts[i + 1]) for i in range(8, 16, 2)]

            upper_half_x = [p[0] for p in upper_half]
            upper_half_y = [p[1] for p in upper_half]
            lower_half_x = [p[0] for p in lower_half]
            lower_half_y = [p[1] for p in lower_half]

            upper_half_x, upper_half_y = butils.bezier_to_polyline(upper_half_x, upper_half_y)
            lower_half_x, lower_half_y = butils.bezier_to_polyline(lower_half_x, lower_half_y)

            if gt_label[i] == 1 and predicted_label[i] == 1:
                color = 'green'
            elif gt_label[i] == 1 and predicted_label[i] == 0:
                color = 'blue'
            elif gt_label[i] == 0 and predicted_label[i] == 1:
                color = 'red'
            else:
                color = 'grey'

            draw.line(list(zip(upper_half_x, upper_half_y)), fill=color, width=3)
            draw.line(list(zip(lower_half_x, lower_half_y)), fill=color, width=3)
            draw.text((lower_half[0][0], lower_half[0][1]),
                      neighbour_text[i] + " Order: {}".format(predicted_neighbour_order[i]),
                      fill=color)

            x_min = min(x_min, min(upper_half_x), min(lower_half_x))
            x_max = max(x_max, max(upper_half_x), max(lower_half_x))
            y_min = min(y_min, min(upper_half_y), min(lower_half_y))
            y_max = max(y_max, max(upper_half_y), max(lower_half_y))

        try:
            cropped_image = background.crop((x_min - 10, y_min - 10, x_max + 10, y_max + 10))
        except:
            cropped_image = background.crop((x_min, y_min, x_max, y_max))
        cropped_image.save(f'plots/{img_name}.png')