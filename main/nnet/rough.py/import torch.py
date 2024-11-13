import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint  # Import adjoint method

class ODEFunc(nn.Module):
    def __init__(self, input_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, input_dim)
        )

    def forward(self, t, x):
        return self.net(x)

class MS_SL2_split_model(nn.Module):
    def __init__(self,
                 L=16,
                 N=512,
                 X=8,
                 R=1,
                 B=256,
                 Sc=256,
                 Slice=1,
                 H=512,
                 P=3,
                 norm="gLN",
                 num_spks=2,
                 non_linear="sigmoid",
                 causal=False):
        super(MS_SL2_split_model, self).__init__()
        supported_nonlinear = {
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
            "softmax": F.softmax
        }
        if non_linear not in supported_nonlinear:
            raise RuntimeError("Unsupported non-linear function: {}".format(
                non_linear))
        self.non_linear = supported_nonlinear[non_linear]
        self.non_linear_type = non_linear

        self.encoder_1d = Conv1D(1, N, L, stride=L // 2, bias=False)
        self.receptive = 0
        self.m = Slice
        self.l = L
        self.r = R
        self.b = B
        self.p = P
        self.strides = L // 2
        self.mask_ensemble = nn.ModuleList([])
        for i in range(Slice):
            self.mask_ensemble.append(Conv1DBlock(N, H, 16, 1, Sc, P, norm, causal))
        self.spks = num_spks
        self.spks_conv = nn.Conv1d(N * Slice, N * num_spks, 1, bias=False)
        self.decoder_1d = ConvTrans1D(N, 1, L, stride=L // 2, bias=False)
        self.merge_decoder_1d = nn.Conv1d(4, 2, 1)
        self.merge_decoder_1d2 = nn.Conv1d(2, 2, 1)
        self.receptive = 0
        self._reset_parameters()

        # Neural ODE components
        self.ode_func = ODEFunc(input_dim=N)
        self.solver_method = 'adjoint'  # Use adjoint sensitivity method

    def _reset_parameters(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)

    def forward(self, x):
        """
        x: [N x S]
        """
        if x.dim() != 2:
            raise RuntimeError("{} expects 2D tensor as input".format(self.__name__))
        batch_size, nsample = x.size()
        x = self.encoder_1d(x)  # N x C x L

        # Split the input into different slices
        sliced_outputs = []
        input_chunks = x.chunk(self.m, dim=-1)
        for i in range(self.m):
            input_chunk = input_chunks[i]
            skip_connection = 0
            conv_chunk = input_chunk
            for j in range(self.r):
                skip, conv_chunk = self.mask_ensemble[i](conv_chunk)
                skip_connection = skip_connection + skip
            sliced_outputs.append(skip_connection)

        # Concatenate the sliced outputs along the channel dimension
        mask_ensemble_out = torch.cat(sliced_outputs, dim=1)
        print(f"mask_ensemble_out shape: {mask_ensemble_out.shape}")

        # Pass through the 1x1 conv layer
        mask = self.spks_conv(mask_ensemble_out)
        print(f"mask shape after spks_conv: {mask.shape}")

        # Reshape the mask and apply a softmax non-linearity if required
        out_channels, nframes = mask.size(1), mask.size(2)
        mask = mask.view(batch_size, self.spks, out_channels // self.spks, nframes)
        print(f"mask shape after reshape: {mask.shape}")

        # Integrate Neural ODE after mask estimation and before non-linear activation
        mask_reshaped = mask.permute(0, 1, 3, 2).reshape(-1, out_channels // self.spks)
        print(f"mask_reshaped shape: {mask_reshaped.shape}")
        t = torch.tensor([0.0, 1.0]).to(mask_reshaped.device)
        ode_out = odeint(self.ode_func, mask_reshaped, t, method='adjoint')[1]
        mask = ode_out.view(batch_size, self.spks, nframes, out_channels // self.spks).permute(0, 1, 3, 2)
        print(f"mask shape after ODE integration: {mask.shape}")

        if self.non_linear_type == "softmax":
            mask = self.non_linear(mask, dim=1)
        else:
            mask = self.non_linear(mask)

        est_source = []
        for i in range(self.spks):
            temp = mask[:, i] * x
            temp = self.decoder_1d(temp, squeeze=True)
            est_source.append(temp)

        est_source = torch.stack(est_source, dim=1)
        return est_source

# Placeholder classes for Conv1D, ConvTrans1D, and Conv1DBlock to make the code run
class Conv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias):
        super(Conv1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, bias=bias)

    def forward(self, x):
        return self.conv(x)

class ConvTrans1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias):
        super(ConvTrans1D, self).__init__()
        self.conv_trans = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, bias=bias)

    def forward(self, x, squeeze=False):
        return self.conv_trans(x)

class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding, norm, causal):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = norm
        self.causal = causal

    def forward(self, x):
        skip = self.conv(x)
        return skip, skip  # Just return skip for simplicity

# Example usage
if __name__ == "__main__":
    # Create a dummy input
    dummy_input = torch.randn(4, 1024)

    # Initialize the model
    model = MS_SL2_split_model()

    # Forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
