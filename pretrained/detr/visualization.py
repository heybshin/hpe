import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import os


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def filter_bboxes_from_outputs(outputs,
                               threshold=0.7):
    # keep only predictions with confidence above threshold
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold

    probas_to_keep = probas[keep]

    # convert boxes from [0; 1] to image scales
    # bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], (1280, 720))
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], (802, 636))

    return probas_to_keep, bboxes_scaled


def plot_finetuned_results(pil_img, prob=None, boxes=None, num=None):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    if prob is not None and boxes is not None:
      for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
          ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=3))
          cl = p.argmax()
          text = f'{finetuned_classes[cl]}: {p[cl]:0.2f}'
          ax.text(xmin, ymin, text, fontsize=15,
                  bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    # plt.show()
    path = 'outputs/vis/robotaxi_valid'
    os.makedirs(path, exist_ok=True)

    plt.savefig(os.path.join(path, '{}.png'.format(num)))
    # plt.savefig('outputs/vis/robotaxi_valid/{}.png'.format(num))


def run_worflow(my_image, my_model, num=None):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(my_image).unsqueeze(0)

    # propagate through the model
    outputs = my_model(img)

    for threshold in [0.9, 0.7]:
        probas_to_keep, bboxes_scaled = filter_bboxes_from_outputs(outputs,
                                                                   threshold=threshold)

        plot_finetuned_results(my_image,
                               probas_to_keep,
                               bboxes_scaled,
                               num=num)


if __name__ == '__main__':

    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # colors for visualization
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    model = torch.hub.load('facebookresearch/detr',
                           'detr_resnet50',
                           pretrained=False,
                           num_classes=3)

    checkpoint = torch.load('outputs/checkpoint.pth', map_location='cpu')

    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    # finetuned_classes = ['', 'Map', 'Time', 'Trial', 'Sphere', 'Decision Point', 'Result Indicator']
    finetuned_classes = ['' , 'pedestrian', 'vehicle']

    # for i in range(2, 93):
    #     img = Image.open('dataset/test/{}.png'.format(i))
    #     run_worflow(img, model, num=i)
    # img = Image.open('dataset/valid/541.png')
    # run_worflow(img, model)

    path = 'dataset/robotaxi_valid'

    for file in os.listdir(path):
        img = Image.open(os.path.join(path, file))
        run_worflow(img, model, num=file.split('.')[0])
