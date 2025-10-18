import torch
import torch.nn as nn
import torch.nn.functional as F

def modified_cos(theta):
  """
  Custom cosine modification.
  """
  theta = theta.clone().detach().float()
  return torch.where(theta <= 3.14159, torch.cos(theta), -torch.cos(theta) - 2)

def get_shifted_pred(y_true, y_pred, a_pred, weight_last_layer, bias_last_layer, maj_wt, min_wt):
  """
  Calculates angle shifted predictions.
  """
  num_labels = y_pred.shape[1]
  y_true_oh = F.one_hot(y_true, num_labels)
  
  norm_a_pred_ = torch.linalg.norm(a_pred, dim=1, keepdim=True)
  norm_weight_last_layer_ = torch.linalg.norm(weight_last_layer, dim=0, keepdim=True)
  
  pr_ = torch.matmul(a_pred, weight_last_layer)
  cos_t = ((pr_ / norm_a_pred_) / norm_weight_last_layer_)
  th_ = torch.acos(cos_t)
  
  M = torch.where(y_true_oh == 1, maj_wt, min_wt)
  th_ = M * th_
  
  cos_t = modified_cos(th_)
  
  z = (norm_weight_last_layer_ * norm_a_pred_) * cos_t + bias_last_layer
  soft_prediction = F.softmax(z, dim=-1)
  
  return soft_prediction

def sparse_categorical_maaw_loss(y_true, y_pred, a_pred, weight_last_layer, bias_last_layer, maj_wt=1.0, min_wt=1.0):
  """
  Computes the Sparse Categorical MAAW Loss.
  """
  num_labels = y_pred.shape[1]
  y_true_oh = F.one_hot(y_true, num_labels)
  
  fin_preds = get_shifted_pred(y_true, y_pred, a_pred, weight_last_layer, bias_last_layer, maj_wt, min_wt)
  fin_preds = torch.clamp(fin_preds, 1e-7, 1 - 1e-7)
  
  f = torch.tensor([torch.count_nonzero(y_true == i) for i in range(num_labels)], dtype=torch.float32) + 1e-9
  f = torch.tensor([float(f[i]) for i in y_true])
  
  wt = 1 - 1 / f
  wt_scaled_pow = torch.pow(1.0 - fin_preds, wt.unsqueeze(1) * y_true_oh)
  
  temp = y_true_oh * torch.log(fin_preds)
  loss = wt_scaled_pow * temp
  
  xcent = -1. / f * torch.sum(loss, dim=-1)
  xcent_mean = torch.mean(xcent)
  
  return xcent_mean

class SparseCategoricalMaawLoss(nn.Module):
  """
  Sparse Categorical MAAW Loss for PyTorch.
  """
  def __init__(self, maj_wt=1.0, min_wt=1.0):
    super(SparseCategoricalMaawLoss, self).__init__()
    self.maj_wt = maj_wt
    self.min_wt = min_wt

  def forward(self, y_true, y_pred, a_pred, weight_last_layer, bias_last_layer):
    return sparse_categorical_maaw_loss(y_true, y_pred, a_pred, weight_last_layer, bias_last_layer, self.maj_wt, self.min_wt)
