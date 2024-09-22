from datasets import *


def downscale_images(src_image_folder, dst_image_folder, dst_width, dst_height, to_jpg=False):
    for image in os.listdir(src_image_folder):
        image_path = os.path.join(src_image_folder, image)
        with Image.open(image_path) as img:
            if img.size[0] <= dst_width and img.size[1] <= dst_height:
                pass
            else:
                img = img.resize((dst_width, dst_height))
            if to_jpg:
                image = os.path.splitext(image)[0] + ".jpg"
            img.save(os.path.join(dst_image_folder, image))


def convert_rumsey_to_grouper(rumsey: RumseyDataset, grouper, text_encoder, use_jpg=False):
    for i in range(len(rumsey)):
        image = rumsey.images[i]
        image_name = os.path.basename(image)
        if use_jpg:
            image_name = os.path.splitext(image_name)[0] + ".jpg"
        groups = rumsey.groups[i]

        img_id = grouper.register_image(image_name)
        if img_id is None:
            raise Exception("Failed to register image")

        for group in groups:
            text_polygon_dict = {'rec': [], 'polygon': []}
            for entry in group:
                rec = text_encoder(entry['text'], ignore=['#'])
                polygon = entry['vertices']
                text_polygon_dict['rec'].append(rec)
                text_polygon_dict['polygon'].append(polygon)

            grouper.register_annotation(img_id, text_polygon_dict, upper_lower_split=False)

        # print(f"Processed image {image_name}, {i}/{len(rumsey)}")


def convert_cvat_to_grouper(cvat: CvatDataset, grouper, text_encoder, use_jpg=False):
    for i in range(len(cvat)):
        image = cvat.images[i]
        image_name = os.path.basename(image)
        if use_jpg:
            image_name = os.path.splitext(image_name)[0] + ".jpg"
        groups = cvat.groups[i]

        img_id = grouper.register_image(image_name)
        if img_id is None:
            raise Exception("Failed to register image")

        for group in groups:
            text_polygon_dict = {'rec': [], 'polygon': []}
            for entry in group:
                rec = text_encoder(entry['text'], ignore=['#'])
                polygon = entry['vertices']
                text_polygon_dict['rec'].append(rec)
                text_polygon_dict['polygon'].append(polygon)

            grouper.register_annotation(img_id, text_polygon_dict, upper_lower_split=True)


def convert_mapKurator_to_grouper(mapKurator: MapKuratorDataset, grouper, text_encoder, use_jpg=False):
    for _, image in mapKurator.images.items():
        image_id = image['id']
        image_name = os.path.basename(image['file_name'])
        annos = mapKurator.annos[image_id]

        img_id = grouper.register_image(image_name)
        if img_id is None:
            raise Exception("Failed to register image")

        '''
        annos
        {'image_id': 1,
          'bbox': [109, 762, 69, 28],
          'category_id': 1,
          'area': 1836.0,
          'rec': [list],
          'text': 'AVE.',
          'id': 9564,
          'polys': [list] (xyxyxy),
          'group_id': 'tjRe5G-_YetsXXx7YqjZr',
          'group_order': 2}
        '''

        group_dict = {}
        single_word = []
        for anno in annos:

            word_dict = {}
            if text_encoder == Encoding.encode_text_96:
                rec = anno['rec']
            else:
                rec = text_encoder(anno['text'], ignore=['#'])
            word_dict['rec'] = rec
            word_dict['polys'] = anno['polys']
            word_dict['id_in_group'] = anno['group_order']
            word_dict['text'] = anno['text']

            if anno['group_id'] == "":
                single_word.append(word_dict)
                continue
            if anno['group_id'] not in group_dict.keys():
                group_dict[anno['group_id']] = []

            group_dict[anno['group_id']].append(word_dict)

        for group_id, group in group_dict.items():

            # sort group according to 'id_in_group'
            group.sort(key=lambda x: x['id_in_group'])

            has_minus1 = False
            for word in group:
                if word['id_in_group'] == -1:
                    has_minus1 = True

            text_polygon_dict = {'rec': [], 'polygon': []}

            for entry in group:
                rec = entry['rec']
                polygon = entry['polys']
                text_polygon_dict['rec'].append(rec)
                text_polygon_dict['polygon'].append([polygon[i:i + 2] for i in range(0, len(polygon), 2)])
                if has_minus1:
                    print(entry['text'])
            if has_minus1:
                print('------')

            grouper.register_annotation(img_id, text_polygon_dict, upper_lower_split=False)

        # print(f"Processed image {image_name}, {image_id}/{len(mapKurator.images)}")


from tqdm.notebook import tqdm

def convert_rumsey_to_deepsolo(rumsey: RumseyDataset, deep_solo: DeepSoloDataset, text_encoder):
    for i in tqdm(range(len(rumsey))):
        image = rumsey.images[i]
        # print(image)
        image_name = os.path.basename(image)
        groups = rumsey.groups[i]
        # print(image_name)

        img_id = deep_solo.register_image(image_name)
        if img_id is None:
            raise Exception("Failed to register image")

        for group in groups:
            for entry in group:
                rec = text_encoder(entry['text'], ignore=['#'])
                polygon = entry['vertices']
                deep_solo.register_annotation_polygon(img_id, polygon, rec)

        print(f"Processed image {image_name}, {i}/{len(rumsey)}")


def convert_cvat_to_deepsolo(cvat: CvatDataset, deep_solo: DeepSoloDataset, text_encoder):
    for i in tqdm(range(len(cvat))):
        image = cvat.images[i]
        # print(image)
        image_name = os.path.basename(image)
        groups = cvat.groups[i]
        # print(image_name)

        img_id = deep_solo.register_image(image_name)
        if img_id is None:
            raise Exception("Failed to register image")

        for group in groups:
            for entry in group:
                rec = text_encoder(entry['text'], ignore=['#'])
                polygon = entry['vertices']
                deep_solo.register_annotation_polygon_cvat(img_id, polygon, rec)

        print(f"Processed image {image_name}, {i}/{len(cvat)}")


def convert_mapKurator_to_deepsolo(mapKurator: MapKuratorDataset, deepsolo: DeepSoloDataset, text_encoder,
                                   copy_rec=True):
    for image_id in tqdm(mapKurator.images):
        image = mapKurator.images[image_id]
        image_name = os.path.basename(image['file_name'])
        annos = mapKurator.annos[image_id]

        img_id = deepsolo.register_image(image_name)
        if img_id is None:
            raise Exception("Failed to register image")

        for anno in annos:
            if copy_rec and text_encoder == Encoding.encode_text_96:
                rec = anno['rec']
            else:
                rec = text_encoder(anno['text'], ignore=['#'])

            polys = anno['polys']

            vertices = [polys[i:i + 2] for i in range(0, len(polys), 2)]

            deepsolo.register_annotation_polygon(img_id, vertices, rec)

        print(f"Processed image {image_name}, {image_id}/{len(mapKurator.images)}")


def combine_deepsolo(deepsolo1: DeepSoloDataset, deepsolo2: DeepSoloDataset, new_images_folder):
    # if not os.path.exists(new_images_folder):
    #     os.mkdir(new_images_folder)
    import shutil
    combined = DeepSoloDataset(new_images_folder)
    import copy

    combined.images = copy.deepcopy(deepsolo1.images)
    combined.annos = copy.deepcopy(deepsolo1.annos)

    # Copy images from deepsolo1 and deepsolo2 to new_images_folder
    for image_id in tqdm(deepsolo1.images.keys()):
        image = deepsolo1.images[image_id]
        image_name = os.path.basename(image['file_name'])
        image_path = os.path.join(deepsolo1.image_root_folder, image['file_name'])
        new_image_path = os.path.join(new_images_folder, image_name)
        shutil.copy(image_path, new_image_path)

    for image_id in tqdm(deepsolo2.images.keys()):
        image = deepsolo2.images[image_id]
        image_name = os.path.basename(image['file_name'])
        image_path = os.path.join(deepsolo2.image_root_folder, image['file_name'])
        new_image_path = os.path.join(new_images_folder, image_name)
        shutil.copy(image_path, new_image_path)

    import time
    time.sleep(10)

    for image_id in tqdm(deepsolo2.images.keys()):

        try:
            image = deepsolo2.images[image_id]
            image_name = os.path.basename(image['file_name'])

            img_id = combined.register_image(image_name)

            if img_id is None:
                print(image_name)
                raise Exception("Failed to register image")

            for anno in deepsolo2.annos[image_id]:
                rec = anno['rec']
                bezier_pts = anno['bezier_pts']
                bbox = anno['bbox']
                if not combined.register_annotation_plain(img_id, bezier_pts, rec, bbox):
                    raise Exception("Failed to register annotation")
        except:
            continue

    return combined