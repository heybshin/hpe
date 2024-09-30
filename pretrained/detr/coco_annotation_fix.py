from pycocotools.coco import COCO
import json


# annotation_file = 'dataset/annotations/all_valid.json'
annotation_file = 'dataset/annotations/all_train.json'

coco = COCO(annotation_file)

category_mapping = {
    4: 0,
    6: 1,
    7: 2,
    8: 3,
    9: 4,
    10: 5,
    # add more mappings as needed
}

new_categories = []
for old_cat_id, new_cat_id in category_mapping.items():
    old_cat = coco.loadCats(old_cat_id)[0]
    new_cat = old_cat.copy()
    new_cat['id'] = new_cat_id
    new_categories.append(new_cat)

coco.dataset['categories'] = new_categories

for annotation in coco.dataset['annotations']:
    old_cat_id = annotation['category_id']
    new_cat_id = category_mapping.get(old_cat_id)
    # if new_cat_id:
    annotation['category_id'] = new_cat_id

for image in coco.dataset['images']:
    # also replace the category_id in the 'category_id' field
    old_category_ids = image['category_ids']
    new_ids = []
    for id in old_category_ids:
        new_ids.append(category_mapping.get(id))

    image['category_ids'] = new_ids

# remapped_annotation_file = 'dataset/annotations/remapped_all_valid.json'
remapped_annotation_file = 'dataset/annotations/remapped_all_train.json'

with open(remapped_annotation_file, 'w') as f:
    json.dump(coco.dataset, f)
