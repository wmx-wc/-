# -
# 动手学深度学习书笔记

# 目标检测
## 锚框

⽬标检测算法通常会在输⼊图像中采样⼤量的区域，然后判断这些区域中是否包含我们感兴趣的⽬标，并调整区域边界从而更准确地预测⽬标的真实边界框（ground-truth bounding box）

不同的模型使⽤的区域采样⽅法可能不同。

方法：以每个像素为中⼼，⽣成多个缩放⽐和宽⾼⽐（aspect ratio）
不同的边界框。（不唯一）。这些边界框被称为锚框（anchor box）。

### ⽣成多个锚框

假设输⼊图像的⾼度为h，宽度为w。我们以图像的每个像素为中⼼⽣成不同形状的锚框：缩放⽐为s ∈ (0, 1]，宽⾼⽐为r > 0。那么锚框的宽度和⾼度分别是ws√r和hs/√r。请注意，当中⼼位置给定时，已知宽和⾼的锚框是确定的。

要⽣成多个不同形状的锚框，让我们设置许多缩放⽐（scale）取值1, . . . , sn和许多宽⾼⽐（aspect ratio）取值r1, . . . , rm。当使⽤这些⽐例和⻓宽⽐的所有组合以每个像素为中⼼时，输⼊图像将总有whnm个锚框。计算量太大。在实践中，可以只考虑包含s1或r1的组合：

(s1, r1),(s1, r2), . . . ,(s1, rm),(s2, r1),(s3, r1), . . . ,(sn, r1). 

也就是说，以同⼀像素为中⼼的锚框的数量是n + m − 1。对于整个输⼊图像，我们将共⽣成wh(n + m − 1)个锚框。

*⽣成锚框的⽅法在下⾯的multibox_prior函数中实现。我们指定输⼊图像、尺⼨列表和宽⾼⽐列表，然后此函数将返回所有的锚框。*

```python
import torch
from d2l import torch as d2l

torch.set_printoptions(2) # 精简输出精度

# 输入图像，缩放比，宽高比
def multibox_prior(data, sizes, ratios):
    """⽣成以每个像素为中⼼具有不同形状的锚框"""
    # 获得图像的高度和宽度
    in_height, in_width = data.shape[-2:]
    # 获取图像所在设备，缩放比总数，宽高比总数
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    # 每个像素具有的锚框个数
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    # 将缩放比和宽高比转化为张量
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # 为了将锚点移动到像素的中⼼，需要设置偏移量。
    # 因为⼀个像素的的⾼为1且宽为1，我们选择偏移我们的中⼼0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height # 在y轴上缩放步长
    steps_w = 1.0 / in_width # 在x轴上缩放步长
    # ⽣成锚框的所有中⼼点
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w)
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # ⽣成“boxes_per_pixel”个⾼和宽，
    # 之后⽤于创建锚框的四⻆坐标(xmin,xmax,ymin,ymax)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   size_tensor[0] * torch.sqrt(ratio_tensor[1:])))\
                    * in_height / in_width # 处理矩形输⼊
    h = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   size_tensor[0] * torch.sqrt(ratio_tensor[1:])))

    # 除以2来获得半⾼和半宽
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2
    print(anchor_manipulations)
    # 每个中⼼点都将有“boxes_per_pixel”个锚框，
    # 所以⽣成含所有锚框中⼼的⽹格，重复了“boxes_per_pixel”次
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)

img = d2l.plt.imread("../img/catdog.jpg")
h, w = img.shape[:2]
print(h, w)
X = torch.rand(size=(1, 3, h, w))
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print(Y)
```

将锚框变量Y的形状更改为(图像⾼度,图像宽度,以同⼀像素为中⼼的锚框的数量,4)后，我们可以获得以指定像素的位置为中⼼的所有锚框。

在接下来的内容中，我们访问以（250,250）为中⼼的第⼀个锚框。它有四个
元素：锚框左上⻆的(x, y)轴坐标和右下⻆的(x, y)轴坐标。将两个轴的坐标各分别除以图像的宽度和⾼度后，所得的值介于0和1之间。

为了显⽰以图像中以某个像素为中⼼的所有锚框，我们定义了下⾯的show_bboxes函数来在图像上绘制多个边界框。

```python
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """显⽰所有边界框"""
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj
    # 锚框的标签
    labels = _make_list(labels)
    # 锚框的颜色
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    # 遍历以该像素为中心的每个锚框，分配颜色
    for i, bbox in enumerate(bboxes):
        # 选择锚框的颜色
        color = colors[i % len(colors)]
        # 画出锚框
        rect = d2l.bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))
```

变量boxes中x轴和y轴的坐标值已分别除以图像的宽度和⾼度。绘制锚框时，我们需要恢复它们原始的坐标值。因此，我们在下⾯定义了变量bbox_scale。现在，我们可以绘制出图像中所有以(250,250)为中⼼的锚框了。如下所⽰，缩放⽐为0.75且宽⾼⽐为1的蓝⾊锚框很好地围绕着图像中的狗。

```python
d2l.set_figsize()
bbox_scale = torch.tensor((w, h, w, h))
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale, ['s=0.75, r=1', 's=0.5, r=1','s=0.25, r=1',
                 's=0.75, r=2','s=0.75, r=0.5'])
d2l.plt.show()
```

## 交并⽐（IoU）

如果已知⽬标的真实边界框，那么这⾥的“好”该如何如何量化呢？直观地说，我们可以衡量锚框和真实边界框之间的相似性。我们知道杰卡德系数（Jaccard）可以衡量两组之间的相似性。给定集合A和B，他们的杰卡德系数是他们交集的⼤小除以他们并集的⼤小.

交并⽐的取值范围在0和1之间：0表⽰两个边界框⽆重合像素，1表⽰两个边界框完全重合。

![这是图片](/深度学习/计算机视觉/img/iou.png)

使⽤交并⽐来衡量锚框和真实边界框之间、以及不同锚框之间的相似度。给定两个锚框或边界框的列表，以下box_iou函数将在这两个列表中计算它们成对的交并⽐。

```python
def box_iou(boxes1, boxes2):
    """计算两个锚框或边界框列表中成对的交并⽐"""
    boxes_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                                (boxes[:, 3] - boxes[:, 1]))
    # boxes1,boxes2,areas1,areas2的形状:
    # boxes1：(boxes1的数量,4),
    # boxes2：(boxes2的数量,4),
    # areas1：(boxes1的数量,),
    # areas2：(boxes2的数量,)
    areas1 = boxes_area(boxes1)
    areas2 = boxes_area(boxes2)
    # inter_upperlefts,inter_lowerrights,inters的形状:
    # (boxes1的数量,boxes2的数量,2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    # 交集不能为负数
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # inter_areasandunion_areas的形状:(boxes1的数量,boxes2的数量)
    # 求交集
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return  inter_areas / union_areas
```

### 在训练数据中标注锚框

为了训练⽬标检测模型，我们需要每个锚框的类别（class）和偏移量（offset）标签，其中前者是与锚框相关的对象的类别，后者是真实边界框相对于锚框的偏移量。在预测时，我们为每个图像⽣成多个锚框，预测所有锚框的类别和偏移量，根据预测的偏移量调整它们的位置以获得预测的边界框，最后只输出符合特定条件的预测边界框。

⽬标检测训练集带有“真实边界框”的位置及其包围物体类别的标签。要标记任何⽣成的锚框，我们可以参考分配到的最接近此锚框的真实边界框的位置和类别标签。在下⽂中，我们将介绍⼀个算法，它能够把最接近的真实边界框分配给锚框。

#### 将真实边界框分配给锚框

给定图像，假设锚框是A1, A2, . . . , Ana，真实边界框是B1, B2, . . . , Bnb，其中na ≥ nb。让我们定义⼀个矩阵X ∈ Rna×nb，其中第i⾏、第j列的元素xij是锚框Ai和真实边界框Bj的IoU。该算法包含以下步骤：

1. 在矩阵X中找到最⼤的元素，并将它的⾏索引和列索引分别表⽰为i1和j1。然后将真实边界框Bj1分配给锚框Ai1。这很直观，因为Ai1和Bj1是所有锚框和真实边界框配对中最相近的。在第⼀个分配完成后，丢弃矩阵中i1th⾏和j1th列中的所有元素。
2. 在矩阵X中找到剩余元素中最⼤的元素，并将它的⾏索引和列索引分别表⽰为i2和j2。我们将真实边界框Bj2分配给锚框Ai2，并丢弃矩阵中i2th⾏和j2th列中的所有元素。
3. 此时，矩阵X中两⾏和两列中的元素已被丢弃。我们继续，直到丢弃掉矩阵X中nb列中的所有元素。此时，我们已经为这nb个锚框各⾃分配了⼀个真实边界框。
4. 只遍历剩下的na − nb个锚框。例如，给定任何锚框Ai，在矩阵X的第ith⾏中找到与Ai的IoU最⼤的真实边界框Bj，只有当此IoU⼤于预定义的阈值时，才将Bj分配给Ai。

```python
# 将真实边界框分配给锚框（真实边界框，锚框，设备，分配的最低交并比）
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """将最接近的真实边界框分配给锚框"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位于第i⾏和第j列的元素x_ij是锚框i和真实边界框j的IoU, 二维矩阵
    jaccard = box_iou(anchors, ground_truth)
    # 对于每个锚框，分配的真实边界框的张量
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # 根据阈值，决定是否分配真实边界框
    # 返回每行的最大值，一维向量，还有最大值在列上的索引
    max_ious, indices = torch.max(jaccard, dim=1)
    # 求出jaccard矩阵中最大元素所在位置
    anc_i = torch.nonzero(max_ious >=0.5).reshape(-1) # 去掉小于阈值的元素索引，行索引
    box_j = indices[max_ious >= 0.5]  # 去掉小于阈值的元素索引，列索引
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        # 距离
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long() # 列索引
        anc_idx = (max_idx / num_gt_boxes).long() # 行索引
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map
```

