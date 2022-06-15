#coding: utf-8
#----- 標準ライブラリ -----#
# None
#----- 専用ライブラリ -----#
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#----- 自作モジュール -----#
# None


def one_hot_changer(tensor, vector_dim, dim=-1, bool_=False):
    """index tensorをone hot vectorに変換する関数

    Args:
        tensor (torch.tensor,dtype=torch.long): index tensor
        vector_dim (int): one hot vectorの次元。index tensorの最大値以上の値でなくてはならない
        dim (int, optional): one hot vectorをどこの次元に組み込むか. Defaults to -1.
        bool_ (bool, optional): Trueにするとbool型になる。Falseの場合はtorch.float型. Defaults to False.

    Raises:
        TypeError: index tensor is not torch.long
        ValueError: index tensor is greater than vector_dim

    Returns:
        torch.tensor: one hot vector
    """
    if bool_:
        data_type = bool
    else:
        data_type = torch.float

    if tensor.dtype != torch.long:
        raise TypeError("入力テンソルがtorch.long型ではありません")
    if tensor.max() >= vector_dim:
        raise ValueError(f"入力テンソルのindex番号がvector_dimより大きくなっています\ntensor.max():{tensor.max()}")

    # one hot vector用単位行列
    one_hot = torch.eye(vector_dim, dtype=data_type, device=tensor.device)
    vector = one_hot[tensor]

    # one hot vectorの次元変更
    dim_change_list = list(range(tensor.dim()))
    # もし-1ならそのまま出力
    if dim == -1:
        return vector
    # もしdimがマイナスならスライス表記と同じ性質にする
    if dim < 0:
        dim += 1  # omsertは-1が最後から一つ手前

    dim_change_list.insert(dim, tensor.dim())
    vector = vector.permute(dim_change_list)
    return vector

class IoULoss(nn.Module):
    def __init__(self, n_class, weight=None):
        super().__init__()
        self.n_class = n_class
        if weight is None:
            self.weight = torch.ones([self.n_class])
        else:
            self.weight = weight

    def forward(self, outputs, targets):
        outputs = F.softmax(outputs, dim=1)
        # targets=>[batch,256,256]
        targets = one_hot_changer(targets, self.n_class, dim=1)

        intersectoin = (outputs * targets).sum(dim=(2, 3))
        total = (outputs + targets).sum(dim=(2, 3))
        union = total - intersectoin
        loss = (intersectoin + 1e-24) / (union + 1e-24)
        loss = 1 - loss
        loss = self.weight.to(loss)[None] * loss
        loss = loss.mean()
        return loss


class Edge_IoULoss(nn.Module):
    def __init__(self, n_class, edge_range=3, lamda=1.0, weight=None):
        super().__init__()
        self.n_class = n_class
        
        self.avgPool = nn.AvgPool2d(2 * edge_range + 1, stride=1, padding=edge_range)

        if weight is None:
            self.weight = torch.ones([self.n_class])
        else:
            self.weight = weight
        self.lamda = lamda

    def edge_decision(self, seg_map):
        # seg_map is segmentation map [batch,class,H,W] value:0 or 1
        # Non differentiable
        smooth_map = self.avgPool(seg_map)

        # 物体の曲線付近内側までを1.とするフラグを作成
        object_edge_inside_flag = seg_map * (smooth_map != seg_map)
        return object_edge_inside_flag

    def forward(self, outputs, targets):
        outputs = F.softmax(outputs, dim=1)
        # targets [batch,H,W] => [batch,class,H,W]
        targets = one_hot_changer(targets, self.n_class, dim=1)

        intersectoin = (outputs * targets).sum(dim=(2, 3))
        total = (outputs + targets).sum(dim=(2, 3))
        union = total - intersectoin
        IoU_loss = (intersectoin + 1e-24) / (union + 1e-24)

        IoU_loss = 1 - IoU_loss
        IoU_loss = self.weight.to(IoU_loss)[None] * IoU_loss

        # edge IoU Loss
        predicts_idx = outputs.argmax(dim=1)
        predicts_seg_map = one_hot_changer(predicts_idx, self.n_class, dim=1)
        predict_edge = self.edge_decision(predicts_seg_map)
        targets_edge = self.edge_decision(targets)

        outputs = outputs * predict_edge
        targets = targets * targets_edge

        intersectoin = (outputs * targets).sum(dim=(2, 3))
        union = targets.sum(dim=(2, 3))
        edge_IoU_loss = (intersectoin + 1e-24) / (union + 1e-24)
        edge_IoU_loss = 1 - edge_IoU_loss
        edge_IoU_loss = self.weight.to(edge_IoU_loss)[None] * edge_IoU_loss

        return IoU_loss.mean() + edge_IoU_loss.mean() * self.lamda


############## main ##############
if __name__ == '__main__':
    device = torch.device("cuda:0")

    batch_size = 8
    class_num = 4
    H, W = 256, 256
    
    #モデルの出力のshapeは[batchサイズ, class数, H, W]
    #softmaxをかける前の値である必要があります
    model_predict = torch.rand([batch_size, class_num, H, W]).cuda(device)
    
    #セグメンテーションラベルのshapeは[batchサイズ, H, W]
    #ラベルの値はクラス番号を示しており、long型でなければなりません。
    segmentation_label = torch.randint(class_num, [batch_size, H, W]).cuda(device)

    print("model predicted shape:", model_predict.shape)
    print("model predicted type:", model_predict.dtype)
    print("segmentation label shape:", segmentation_label.shape)
    print("segmentation label type:", segmentation_label.dtype)
    
    #損失関数の定義
    loss_func = Edge_IoULoss(n_class=class_num, edge_range=1, lamda=1.0)

    loss = loss_func(model_predict, segmentation_label)
    
    print("Loss value:", loss)