from torch import nn
import torch

class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        with torch.no_grad():
            dummy = torch.zeros(1,
                                configs.input_channels,
                                configs.features_len)
            out = self.conv_block1(dummy)
            out = self.conv_block2(out)
            out = self.conv_block3(out)
            flat_dim = out.numel()  # = 1 * C_out * T_after

        # now build head with exact in_features
        self.logits = nn.Linear(flat_dim, configs.num_classes)

        # model_output_dim = configs.features_len
        # self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x_flat = x.reshape(x.shape[0], -1)
        # print("shape of logits",x_flat.shape)
        # #currently with adiac UCR, shape: [64,7680]
        # #maybe it needs to accept, it as [7680,64]
        # print(">>> after conv blocks:")
        # print("  x      :", x.shape)                # (B, C_out, T_after)
        # print("  x_flat :", x_flat.shape)           # (B, C_out * T_after)
        # print("  logits :", self.logits)            # shows in_features & out_features
        # print("  weight :", self.logits.weight.shape)  # (out_features, in_features)
        # print("=============")
        # x_flat = x.view(x.size(0), -1)
        # D_actual   = x_flat.size(1)
        # W           = self.logits.weight
        # D_expected = W.size(1)
        # K           = W.size(0)

        # print(f">>> x after conv:    {x.shape}        → D_actual={D_actual}")
        # print(f">>> logits.weight:   {W.shape}        → D_expected={D_expected}, num_classes={K}")    
        logits = self.logits(x_flat)
        return logits, x
