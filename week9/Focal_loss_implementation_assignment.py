import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class DetectionLoss(nn.Module):
    def __init__(self, num_classes, gamma=2, ignore_index=-1):
        super().__init__()
        self.num_classes = num_classes
        
        # gamma will be uses in focal loss
        self.gamma = gamma
        
        # ignore_index will be used classification loss.
        # at the time of encoding, anchor boxes which 0.4 <IoU < 0.5, assign -1 as label.
        # at the time of finding the classification loss these labels should be ignored
        self.ignore_index = ignore_index

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].

        loss:
          (tensor) loss = (SmoothL1Loss(loc_preds, loc_targets),  FocalLoss(cls_preds, cls_targets)).
        '''

        ################################################################
        # loc_loss
        ################################################################
        

        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.long().sum(1, keepdim=True)

        mask = pos.unsqueeze(2).expand_as(loc_preds)  # [N,#anchors,4]
        masked_loc_preds = loc_preds[mask].view(-1, 4)  # [#pos,4]
        masked_loc_targets = loc_targets[mask].view(-1, 4)  # [#pos,4]
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, reduction='none')
        loc_loss = loc_loss.sum() / num_pos.sum().float()

        ################################################################
        # cls_loss with Focal Loss
        ################################################################
        
        cls_loss = 0.0
        #implement focal loss here
        # cls_preds = F.softmax(cls_preds, dim=2)
        # cls_targets = cls_targets.long()
        batch_size, _ = cls_targets.size()
        batch_conf = cls_preds.view(-1, self.num_classes)
        # compute cross entropy loss
        cls_loss = F.cross_entropy(batch_conf, cls_targets.view(-1), ignore_index=self.ignore_index, reduction='none')
        cls_loss = cls_loss.view(batch_size, -1)
        
        
        # Apply focal loss
        gamma = self.gamma
        
        pt = torch.exp(-cls_loss)
        focal_loss = (1 - pt) ** gamma * cls_loss
        #calculate final loss by taking mean
        cls_loss = focal_loss.sum() / (cls_targets != self.ignore_index).sum().item()
        

        ###
        ### YOUR CODE HERE
        ###
        
        return loc_loss, cls_loss
    
    
class DataEncoder:
    def __init__(self, input_size):
        self.input_size = input_size
        self.anchor_areas = [8 * 8, 16 * 16., 32 * 32., 64 * 64., 128 * 128]  # p3 -> p7
        self.aspect_ratios = [0.5, 1, 2]
        self.scales = [1, pow(2, 1 / 3.), pow(2, 2 / 3.)]
        num_fms = len(self.anchor_areas)
        fm_sizes = [math.ceil(self.input_size[0] / pow(2., i + 3)) for i in range(num_fms)]
        self.anchor_boxes = []
        for i, fm_size in enumerate(fm_sizes):
            anchors = self.generate_anchors(self.anchor_areas[i], self.aspect_ratios, self.scales)
            anchor_grid = self.generate_anchor_grid(input_size, fm_size, anchors)
            self.anchor_boxes.append(anchor_grid)
        self.anchor_boxes = torch.cat(self.anchor_boxes, 0)

    def encode(self, boxes, classes):
        iou = self.compute_iou(boxes, self.anchor_boxes)
        iou, ids = iou.max(1)
        loc_targets = self.encode_boxes(boxes[ids], self.anchor_boxes)
        cls_targets = classes[ids]
        cls_targets[iou < 0.5] = -1
        cls_targets[iou < 0.4] = 0

        return loc_targets, cls_targets
    
    def get_num_anchors(self):
        return len(self.aspect_ratios) * len(self.scales)
    
    @staticmethod
    def encode_boxes(boxes, anchors):
        anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
        anchors_ctr = anchors[:, :2] + 0.5 * anchors_wh
        boxes_wh = boxes[:, 2:] - boxes[:, :2] + 1
        boxes_ctr = boxes[:, :2] + 0.5 * boxes_wh
        return torch.cat([(boxes_ctr - anchors_ctr) / anchors_wh, torch.log(boxes_wh / anchors_wh)], 1)
    
    @staticmethod
    def generate_anchor_grid(input_size, fm_size, anchors):
        grid_size = input_size[0] / fm_size
        x, y = torch.meshgrid(torch.arange(0, fm_size) * grid_size, torch.arange(0, fm_size) * grid_size)
        anchors = anchors.view(-1, 1, 1, 4)
        xyxy = torch.stack([x, y, x, y], 2).float()
        boxes = (xyxy + anchors).permute(2, 1, 0, 3).contiguous().view(-1, 4)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, input_size[0])
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, input_size[1])
        return boxes
    
    @staticmethod
    def generate_anchors(anchor_area, aspect_ratios, scales):
        anchors = []
        for scale in scales:
            for ratio in aspect_ratios:
                h = round(math.sqrt(anchor_area) / ratio)
                w = round(ratio * h)
                x1 = (math.sqrt(anchor_area) - scale * w) * 0.5
                y1 = (math.sqrt(anchor_area) - scale * h) * 0.5
                x2 = (math.sqrt(anchor_area) + scale * w) * 0.5
                y2 = (math.sqrt(anchor_area) + scale * h) * 0.5
                anchors.append([x1, y1, x2, y2])
        return torch.Tensor(anchors)
    
    @staticmethod
    def compute_iou(src, dst):
        p1 = torch.max(dst[:, None, :2], src[:, :2])
        p2 = torch.min(dst[:, None, 2:], src[:, 2:])
        inter = torch.prod((p2 - p1 + 1).clamp(0), 2)
        src_area = torch.prod(src[:, 2:] - src[:, :2] + 1, 1)
        dst_area = torch.prod(dst[:, 2:] - dst[:, :2] + 1, 1)
        iou = inter / (dst_area[:, None] + src_area - inter)
        return iou

def main():

    img_height = img_width = 300

    num_classes = 5  # including background

    bounding_boxes = torch.tensor([[100, 100, 150, 150], [120, 200, 160, 250]], dtype=torch.float)
    targets = torch.tensor([2, 4])

    data_encoder = DataEncoder((img_height, img_width))

    num_anchors = data_encoder.get_num_anchors()

    bboxes, labels = data_encoder.encode(bounding_boxes, targets)

    torch.manual_seed(21)

    pred_bboxes = torch.rand((1, labels.size()[0], 4))
    pred = torch.rand((1, labels.size()[0], num_classes))

    gamma = 2
    detection_loss = DetectionLoss(num_classes, gamma)

    bb_loss, cls_pred_loss = detection_loss(pred_bboxes, bboxes.unsqueeze(0), pred, labels.unsqueeze(0))

    print('Bounding Box Loss:\t{0:.5}'.format(bb_loss))
    print('Classification Loss:\t{0:.5}'.format(cls_pred_loss))

# entry point
if __name__ == "__main__":
    main()