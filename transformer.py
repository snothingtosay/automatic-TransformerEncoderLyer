import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, query_size, key_dim, num_units, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_units = num_units  # 每个头的特征维度
        self.num_heads = num_heads  # 注意力头的数量
        self.key_dim = key_dim  # Key 的维度

        # 定义线性变换，用于将输入的 query、key 和 value 转换为 num_units 的维度
        self.W_query = nn.Linear(in_features=query_size, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key, mask=None):
        # 将输入的 query、key 和 value 通过线性变换
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)  # [N, T_k, num_units]

        # 计算每个头的大小
        split_size = self.num_units // self.num_heads

        # 将 query、key 和 value 按照头的数量进行分割
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # 计算注意力分数
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)  # 缩放分数，以避免数值过大

        # 如果提供了 mask，则应用 mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(0).repeat(self.num_heads, 1, querys.shape[2], 1)
            scores = scores.masked_fill(mask, -float('inf'))  # 将被遮挡的部分设置为负无穷

        # 对分数进行 softmax 归一化，得到注意力权重
        scores = F.softmax(scores, dim=-1)  # 在最后一个维度上进行 softmax

        # 计算最终的输出
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # 合并所有头的输出，形状变为 [N, T_q, num_units]

        return out, scores  # 返回最终输出和注意力分数


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(query_size=d_model, key_dim=d_model, num_units=d_model, num_heads=num_heads)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
###这里可能需要修改的地方就是这里的SRC可能需要进行拆分
    def forward(self, src, mask=None):
        attn_output, attn_scores = self.self_attention(src, src, mask)  # 获取注意力输出和注意力分数
        src = self.norm1(src + attn_output)  # 残差连接 + 层归一化
        ff_output = self.linear2(self.dropout2(F.relu(self.linear1(src))))
        src = self.norm2(src + ff_output)  # 残差连接 + 层归一化
        return src, attn_scores  # 返回最终输出和注意力分数


