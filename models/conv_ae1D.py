import torch.nn as nn
from collections import OrderedDict
from models.utils.layers import conv_block1D, deconv_block1D, bottleneck1D_fc, bottleneck1D_convolutional
import torch

from config import activation_dict

torch.manual_seed(0)


class Encoder1D(nn.Module):
    """
    1D Convolutional Encoder with flexible bottleneck compression.

    Architecture:
    Input → Conv blocks (with pooling) → [Optional Bottleneck] → Latent

    Bottleneck modes:
    - skip_bottleneck=True: No bottleneck, encoder output goes directly to decoder
    - skip_bottleneck=False + flattened=True: FC bottleneck
    - skip_bottleneck=False + flattened=False: Convolutional bottleneck
    """

    def __init__(self,
                 in_channels=1,
                 base_filters=32,
                 kernel_size=3,
                 num_layers=2,
                 padding=1,
                 dilation=1,
                 pool_ks=2,
                 pool_stride=2,
                 activation=nn.ReLU(),
                 bottleneck_act=None,
                 compression_factor=2,
                 seq_length=16,
                 flattened=True,
                 compression_type='on_features',
                 bottleneck_conv=True,
                 conv_compression_strategy='balanced',
                 skip_bottleneck=False):
        super().__init__()
        self.in_channels = in_channels
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.padding = padding
        self.pool_ks = pool_ks
        self.pool_stride = pool_stride
        self.compression_factor = compression_factor
        self.compression_type = compression_type
        self.seq_length = seq_length
        self.act = activation
        self.bottleneck_act = bottleneck_act
        self.flattened = flattened
        self.bottleneck_conv_enabled = bottleneck_conv
        self.conv_compression_strategy = conv_compression_strategy
        self.skip_bottleneck = skip_bottleneck

        # Build encoder layers with progressive channel increase
        layers = []
        in_f = in_channels
        for i in range(num_layers):
            out_f = base_filters * (2 ** i)
            layers.append((
                f'enc_lay_{i + 1}',
                conv_block1D(
                    in_f, out_f,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    pool_ks=self.pool_ks,
                    pool_stride=self.pool_stride,
                    activation=self.act,
                    padding=self.padding
                )
            ))
            in_f = out_f

        self.encoder = nn.Sequential(OrderedDict(layers))

        # Get encoder output dimensions
        with torch.no_grad():
            x = torch.zeros(1, self.in_channels, self.seq_length)
            x = self.encoder(x)
            _, enc_channels, seq_enc = x.size()

        self.seq_enc = seq_enc

        # ============================================
        # BOTTLENECK CONFIGURATION
        # ============================================

        if self.skip_bottleneck:
            # ============================================
            # NO BOTTLENECK - Direct connection
            # ============================================
            print("Encoder1D: Skip bottleneck (no compression)")
            self.bottleneck_out_channels = enc_channels
            self.flattened_size = enc_channels * seq_enc
            self.latent_dim = self.flattened_size  # For compatibility
            self.bottleneck = nn.Identity()  # Pass-through

            print(f"   Encoder output: ({enc_channels}, {seq_enc}) = {enc_channels * seq_enc} values")
            print(f"   No compression - direct to decoder")

        else:
            # ============================================
            # WITH BOTTLENECK
            # ============================================

            # Optional 1x1 doubling conv
            if self.bottleneck_conv_enabled:
                doubling_conv = nn.Conv1d(enc_channels, enc_channels * 2, kernel_size=1)
                channels_after_doubling = enc_channels * 2
                print(
                    f"Encoder1D: Using bottleneck conv (doubling channels: {enc_channels} → {channels_after_doubling})")
            else:
                doubling_conv = None
                channels_after_doubling = enc_channels
                print(f"Encoder1D: No bottleneck conv (channels: {enc_channels})")

            if not self.flattened:
                # ============================================
                # CONVOLUTIONAL LATENT SPACE (flattened=False)
                # ============================================
                print("Encoder1D: Convolutional latent space (no FC)")

                # Compute bottleneck channels using compression strategy
                bottleneck_channels = self._compute_bottleneck_channels(
                    in_channels=self.in_channels,
                    enc_channels=channels_after_doubling,
                    seq_length=self.seq_length,
                    seq_enc=seq_enc,
                    compression_factor=self.compression_factor,
                    strategy=self.conv_compression_strategy,
                    compression_type=self.compression_type
                )

                # Create compression conv (1x1 conv to reduce channels)
                compression_conv = nn.Conv1d(channels_after_doubling, bottleneck_channels, kernel_size=1)
                self.bottleneck_out_channels = bottleneck_channels

                # Set dimensions
                self.flattened_size = bottleneck_channels * seq_enc
                self.latent_dim = self.flattened_size  # For compatibility

                print(
                    f"   Input: ({self.in_channels}, {self.seq_length}) = {self.in_channels * self.seq_length} values")
                print(f"   After encoder: ({enc_channels}, {seq_enc}) = {enc_channels * seq_enc} values")
                if self.bottleneck_conv_enabled:
                    print(
                        f"   After doubling conv: ({channels_after_doubling}, {seq_enc}) = {channels_after_doubling * seq_enc} values")
                print(
                    f"   After compression conv: ({bottleneck_channels}, {seq_enc}) = {bottleneck_channels * seq_enc} values")
                print(
                    f"   Compression ratio: {(self.in_channels * self.seq_length) / (bottleneck_channels * seq_enc):.2f}x")

                # Create convolutional bottleneck
                self.bottleneck = bottleneck1D_convolutional(
                    doubling_conv=doubling_conv,
                    compression_conv=compression_conv,
                    activation=self.act,
                    bottleneck_activation=self.bottleneck_act,
                    batch_norm=True
                )

            else:
                # ============================================
                # FC LATENT SPACE (flattened=True)
                # ============================================
                print("Encoder1D: FC latent space")
                self.bottleneck_out_channels = channels_after_doubling
                self.flattened_size = channels_after_doubling * seq_enc

                # Compute latent dimension based on compression_type
                if self.compression_type == 'on_features':
                    self.latent_dim = int(self.flattened_size // self.compression_factor)
                    print(f"   Compression on features: {self.flattened_size} → {self.latent_dim}")
                elif self.compression_type == 'on_inputs':
                    input_size = self.in_channels * self.seq_length
                    self.latent_dim = int(input_size // self.compression_factor)
                    print(f"   Compression on inputs: {input_size} → {self.latent_dim}")
                else:
                    raise ValueError(f"Unknown compression_type: '{self.compression_type}'")

                # Create FC bottleneck
                self.bottleneck = bottleneck1D_fc(
                    doubling_conv=doubling_conv,
                    activation=self.act,
                    flattened_size=self.flattened_size,
                    latent_dim=self.latent_dim,
                    bottleneck_activation=self.bottleneck_act,
                    batch_norm=True
                )

        self._init_weights()

    def _compute_bottleneck_channels(self, in_channels, enc_channels, seq_length,
                                     seq_enc, compression_factor, strategy, compression_type):
        """
        Compute number of channels for convolutional bottleneck compression conv.

        Note: enc_channels here is AFTER optional doubling conv.

        Strategies:
        - 'balanced': Compute channels to achieve exact target compression (RECOMMENDED)
        - 'on_channels': Apply compression_factor only on channels
        - 'on_remaining': Account for temporal compression already done
        - 'sqrt_balanced': Geometric mean approach

        Args:
            in_channels: Input channels
            enc_channels: Channels after encoder (and optional doubling conv)
            seq_length: Input sequence length
            seq_enc: Encoded sequence length
            compression_factor: Target compression factor
            strategy: Compression strategy
            compression_type: 'on_features' or 'on_inputs'

        Returns:
            Number of bottleneck channels (int)
        """
        input_size = in_channels * seq_length
        encoder_size = enc_channels * seq_enc

        # Compute target size based on compression_type
        if compression_type == 'on_features':
            target_size = encoder_size / compression_factor
            reference_size = encoder_size
            print(f"   Compression type: on_features (from {encoder_size} values)")
        elif compression_type == 'on_inputs':
            target_size = input_size / compression_factor
            reference_size = input_size
            print(f"   Compression type: on_inputs (from {input_size} values)")
        else:
            raise ValueError(f"Unknown compression_type: '{compression_type}'")

        temporal_compression = seq_length / seq_enc

        if strategy == 'balanced':
            bottleneck_channels = max(1, int(target_size / seq_enc))
            actual_size = bottleneck_channels * seq_enc
            actual_compression = reference_size / actual_size

            print(f"   Strategy: balanced")
            print(
                f"   Target: {target_size:.1f} values → {bottleneck_channels} channels × {seq_enc} timesteps = {actual_size} values")
            print(f"   Actual compression: {actual_compression:.2f}x (target: {compression_factor}x)")

        elif strategy == 'on_channels':
            bottleneck_channels = max(1, int(enc_channels / compression_factor))
            actual_size = bottleneck_channels * seq_enc
            actual_compression = reference_size / actual_size

            print(f"   Strategy: on_channels")
            print(f"   {enc_channels} channels / {compression_factor} = {bottleneck_channels} channels")
            print(f"   Result: {bottleneck_channels} × {seq_enc} = {actual_size} values")
            print(f"   Actual compression: {actual_compression:.2f}x (target: {compression_factor}x)")

        elif strategy == 'on_remaining':
            if compression_type == 'on_inputs':
                channel_compression = compression_factor / temporal_compression
            else:
                channel_compression = compression_factor

            bottleneck_channels = max(1, int(enc_channels / channel_compression))
            actual_size = bottleneck_channels * seq_enc
            actual_compression = reference_size / actual_size

            print(f"   Strategy: on_remaining")
            print(f"   Temporal compression: {temporal_compression:.1f}x (already done)")
            print(f"   Remaining channel compression: {channel_compression:.1f}x")
            print(f"   Bottleneck: {bottleneck_channels} channels × {seq_enc} timesteps = {actual_size} values")
            print(f"   Actual compression: {actual_compression:.2f}x (target: {compression_factor}x)")

        elif strategy == 'sqrt_balanced':
            if compression_type == 'on_inputs':
                actual_temporal_compression = temporal_compression
                remaining_compression = compression_factor / actual_temporal_compression
                target_channel_compression = remaining_compression ** 0.5
            else:
                target_channel_compression = compression_factor ** 0.5

            bottleneck_channels = max(1, int(enc_channels / target_channel_compression))
            actual_size = bottleneck_channels * seq_enc
            actual_compression = reference_size / actual_size

            print(f"   Strategy: sqrt_balanced")
            print(f"   Target channel compression: {target_channel_compression:.2f}x")
            print(f"   Bottleneck: {bottleneck_channels} channels × {seq_enc} timesteps = {actual_size} values")
            print(f"   Actual compression: {actual_compression:.2f}x (target: {compression_factor}x)")

        else:
            raise ValueError(f"Unknown conv_compression_strategy: '{strategy}'")

        return bottleneck_channels

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        """
        Forward pass through encoder.

        Args:
            x: Input tensor (batch, in_channels, seq_length)

        Returns:
            Latent representation
            - If skip_bottleneck=True: (batch, enc_channels, seq_enc)
            - If flattened=True: (batch, latent_dim)
            - If flattened=False: (batch, bottleneck_channels, seq_enc)
        """
        x = self.encoder(x)
        x = self.bottleneck(x)
        return x


# ==============================
# Decoder 1D
# ==============================

class Decoder1D(nn.Module):
    """
    1D Convolutional Decoder with multiple reconstruction modes.

    Decoder modes:
    - mirror: Symmetric to encoder (last layer outputs in_channels)
    - progressive: Halves channels progressively (may not reach in_channels)
    - base_filters: Halves channels but stops at base_filters

    decoder_head behavior:
    - mirror mode: Optional refinement layer (in_channels → in_channels)
    - other modes: Mandatory reconstruction layer (last_channels → in_channels)

    Input handling:
    - skip_bottleneck=True: Expects (batch, channels, time) directly from encoder
    - flattened=True: Expects (batch, latent_dim), reshapes to (batch, channels, time)
    - flattened=False: Expects (batch, channels, time) from convolutional bottleneck
    """

    def __init__(self,
                 in_channels=1,
                 base_filters=32,
                 kernel_size=2,
                 num_layers=2,
                 stride=2,
                 latent_dim=None,
                 flattened_size=None,
                 seq_length=16,
                 seq_enc=2,
                 activation=nn.ReLU(),
                 flattened=True,
                 double_deconv=False,
                 conv_padding=0,
                 conv_kernel_size=3,
                 conv_stride=1,
                 conv_dilation=1,
                 bottleneck_out_channels=None,
                 decoder_mode='progressive',
                 decoder_head=True,
                 skip_bottleneck=False):
        super().__init__()
        self.in_channels = in_channels
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.seq_enc = seq_enc
        self.act = activation
        self.flattened = flattened
        self.stride = stride
        self.latent_dim = latent_dim
        self.conv_stride = conv_stride
        self.conv_dilation = conv_dilation
        self.conv_kernel_size = conv_kernel_size
        self.conv_padding = conv_padding
        self.double_deconv = double_deconv
        self.decoder_mode = decoder_mode
        self.decoder_head = decoder_head
        self.skip_bottleneck = skip_bottleneck

        if bottleneck_out_channels is None:
            raise ValueError("Provide bottleneck_out_channels from encoder")
        self.bottleneck_out_channels = bottleneck_out_channels

        # Validation: decoder_head must be True for non-mirror modes
        if self.decoder_mode != 'mirror' and not self.decoder_head:
            print('⚠️  WARNING: decoder_head=False only works with decoder_mode=mirror')
            print('   → Forcing decoder_head=True for non-mirror modes')
            self.decoder_head = True

        # Reshape layer only for FC mode (flattened=True) and not skip_bottleneck
        if flattened and not skip_bottleneck:
            self.reshape = nn.Linear(self.latent_dim, flattened_size)
            print(f"Decoder1D: FC mode - Linear reshape ({self.latent_dim} → {flattened_size})")
        else:
            # Convolutional mode or skip_bottleneck: no reshape needed
            self.reshape = None
            if skip_bottleneck:
                print(f"Decoder1D: Skip bottleneck mode - direct from encoder ({bottleneck_out_channels}, {seq_enc})")
            else:
                print(f"Decoder1D: Convolutional mode - direct from bottleneck ({bottleneck_out_channels}, {seq_enc})")

        output_padding = self._compute_output_padding(seq_enc, seq_length, num_layers, stride)

        encoder_filters = [self.base_filters * (2 ** i) for i in range(self.num_layers)]

        layers = []
        in_f = bottleneck_out_channels

        for i in range(num_layers):

            if self.decoder_mode == 'mirror':
                # Mirror mode: symmetric architecture
                if i == self.num_layers - 1:
                    # Last layer always outputs in_channels (natural reconstruction)
                    out_f = self.in_channels
                else:
                    # Mirror the encoder filters in reverse
                    out_f = encoder_filters[self.num_layers - i - 2]

            elif self.decoder_mode == 'base_filters':
                # Halve channels at each layer, but don't go below base_filters
                out_f = in_f // 2
                if out_f < self.base_filters:
                    out_f = self.base_filters

            elif self.decoder_mode == 'progressive':
                # Progressive halving without lower bound
                out_f = in_f // 2

            else:
                raise ValueError(f"Unknown decoder_mode: '{self.decoder_mode}'")

            # Determine if this is the last layer AND no decoder_head
            is_final_layer = (i == num_layers - 1)
            use_activation = self.act if not (is_final_layer and not self.decoder_head) else None

            layers.append((
                f'dec_lay_{i + 1}',
                deconv_block1D(
                    in_f, out_f,
                    kernel_size=self.kernel_size,
                    stride=stride,
                    activation=use_activation,  # ⬅️ None if last layer without decoder_head
                    double_deconv=self.double_deconv,
                    output_padding=output_padding[i],
                    conv_kernel_size=self.conv_kernel_size,
                    conv_padding=self.conv_padding,
                    conv_stride=self.conv_stride,
                    conv_dilation=self.conv_dilation
                )
            ))
            in_f = out_f

        self.decoder = nn.Sequential(OrderedDict(layers))

        # Create decoder_out head
        if self.decoder_mode == 'mirror':
            # Mirror mode: decoder_head is optional (refinement layer)
            if self.decoder_head:
                self.decoder_out = nn.Conv1d(in_channels, in_channels, kernel_size=1)
                print(f"Decoder1D: Mirror mode with decoder_head - refinement layer ({in_channels}→{in_channels})")
            else:
                self.decoder_out = None
                print(f"Decoder1D: Mirror mode without decoder_head - direct reconstruction")
        else:
            # Progressive/base_filters: decoder_head is mandatory
            self.decoder_out = nn.Conv1d(out_f, in_channels, kernel_size=1)
            print(f"Decoder1D: {self.decoder_mode} mode - mandatory decoder_head ({out_f}→{in_channels})")

        self._init_weights()

    def _compute_output_padding(self, Lb, L_target, num_layers, stride, kernel_size=2, padding=0, dilation=1):
        """
        Compute output_padding for each ConvTranspose layer to match target length.

        Args:
            Lb: Encoded sequence length (bottleneck)
            L_target: Target reconstruction length
            num_layers: Number of decoder layers
            stride: Stride for ConvTranspose
            kernel_size: Kernel size for ConvTranspose
            padding: Padding for ConvTranspose
            dilation: Dilation for ConvTranspose

        Returns:
            List of output_padding values for each layer
        """
        lengths = [Lb]
        for _ in range(num_layers):
            L_in = lengths[-1]
            L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
            lengths.append(L_out)

        diff = L_target - lengths[-1]

        ops = [0] * num_layers
        for i in reversed(range(num_layers)):
            if diff <= 0:
                break
            op = min(diff, stride - 1)
            ops[i] = op
            diff -= op * (stride ** (num_layers - 1 - i))

        return ops

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        """
        Forward pass through decoder.

        Args:
            x: Latent representation
               - skip_bottleneck=True: (batch, enc_channels, seq_enc)
               - FC mode (flattened=True): (batch, latent_dim)
               - Convolutional mode (flattened=False): (batch, bottleneck_channels, seq_enc)

        Returns:
            Reconstructed sequence with shape (batch, in_channels, seq_length)
        """
        if self.reshape is not None:
            # FC mode: reshape from latent vector to convolutional format
            x = self.reshape(x)
            x = x.view((-1, self.bottleneck_out_channels, self.seq_enc))
        # else: skip_bottleneck or convolutional mode, x is already (batch, channels, time)

        # Pass through decoder layers
        x = self.decoder(x)

        # Apply decoder_out head if present
        if self.decoder_out is not None:
            out = self.decoder_out(x)
        else:
            out = x

        return out


# ==============================
# Full CONV_AE1D
# ==============================

class CONV_AE1D(nn.Module):
    """
    1D Convolutional Autoencoder with flexible architecture options.

    Features:
    - Multiple decoder modes (mirror, progressive, base_filters)
    - Optional bottleneck doubling conv (doubles channels before latent)
    - Optional decoder_head (refinement layer)
    - Support for transfer learning with adaptive layers
    - FC or convolutional latent space (controlled by flattened parameter)
    - Skip bottleneck option (direct encoder-decoder connection)

    Compression modes:
    - skip_bottleneck=True: No compression, direct connection
    - flattened=True + compression_type='on_features': FC bottleneck, compress encoder output
    - flattened=True + compression_type='on_inputs': FC bottleneck, compress relative to input
    - flattened=False: Convolutional bottleneck, uses conv_compression_strategy (default: balanced)

    bottleneck_conv parameter:
    - Independent of flattened, adds optional 1x1 conv that doubles channels
    - Can be used with both FC and convolutional latent spaces
    - Ignored if skip_bottleneck=True
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        model_cfg = cfg.model

        # Handle fine-tuning mode (transfer learning)
        available_ft_mode = ['adaptive_layer', 'linear_proj']

        if self.cfg.opt.get("fine_tuning", 0) and self.cfg.opt.get("fine_tuning_mode") in available_ft_mode:
            self.in_channels = len(torch.load(cfg.opt.checkpoint_path)['cfg'].dataset.feats)
        elif self.cfg.opt.get("fine_tuning", 0) and self.cfg.opt.get("fine_tuning_mode") not in available_ft_mode:
            raise NotImplementedError("for Conv 1D only adaptive layer available")
        else:
            self.in_channels = cfg.dataset.n_features

        # Model hyperparameters
        self.kernel_size = model_cfg.kernel_size
        self.base_filters = model_cfg.base_filters
        self.double_deconv = model_cfg.double_deconv
        self.num_layers = model_cfg.num_layers
        self.act = activation_dict.get(model_cfg.get("activation", None), None)
        self.bottleneck_act = activation_dict.get(model_cfg.get("bottleneck_activation", None), None)
        self.bottleneck_conv = model_cfg.get('bottleneck_conv', True)
        self.decoder_mode = model_cfg.get('decoder_mode', 'progressive')
        self.decoder_head = model_cfg.get('decoder_head', True)
        self.stride = model_cfg.stride
        self.pool = model_cfg.pool
        self.increasing = model_cfg.increasing
        self.dilation = model_cfg.dilation
        self.flattened = model_cfg.flattened
        self.skip_bottleneck = model_cfg.get('skip_bottleneck', False)

        # Compression configuration
        self.compression_factor = model_cfg.compression_factor if model_cfg.get('compression_factor_on_inputs',
                                                                                None) is None else model_cfg.get(
            'compression_factor_on_inputs', None)
        self.compression_type = 'on_inputs' if model_cfg.get('compression_factor_on_inputs',
                                                             None) is not None else 'on_features'

        # Convolutional compression strategy (only used when flattened=False)
        self.conv_compression_strategy = model_cfg.get('conv_compression_strategy', 'balanced')

        self.seq_length = cfg.dataset.seq_in_length
        self.pool_ks = 2
        self.pool_stride = 2

        # ============================================
        # ARCHITECTURE VALIDATIONS
        # ============================================

        # Validation 0: Skip bottleneck overrides other bottleneck settings
        if self.skip_bottleneck:
            if self.bottleneck_conv:
                print('ℹ️  NOTE: skip_bottleneck=True overrides bottleneck_conv (will be ignored)')
            print('ℹ️  NOTE: skip_bottleneck=True - no compression, direct encoder→decoder connection')

        # Validation 1: Mirror mode requires bottleneck_conv=False for true symmetry
        if self.decoder_mode == 'mirror' and self.bottleneck_conv and not self.skip_bottleneck:
            print('⚠️  WARNING: Mirror mode requires bottleneck_conv=False for true symmetry')
            print('   → Forcing bottleneck_conv=False')
            self.bottleneck_conv = False

        # Validation 2: Progressive and base_filters converge when bottleneck_conv=True
        if self.decoder_mode in ['progressive',
                                 'base_filters'] and self.bottleneck_conv and self.flattened and not self.skip_bottleneck:
            last_encoder_filters = self.base_filters * (2 ** (self.num_layers - 1))
            bottleneck_filters = last_encoder_filters * 2
            final_filters = bottleneck_filters // (2 ** self.num_layers)

            if final_filters >= self.base_filters:
                print('ℹ️  NOTE: Progressive and base_filters modes are equivalent with bottleneck_conv=True')
                print(f'   → Both will halt at base_filters={self.base_filters}')

        # Validation 3: decoder_head behavior
        if self.decoder_mode != 'mirror' and not self.decoder_head:
            print('⚠️  WARNING: decoder_head=False only works with decoder_mode=mirror')
            print('   → Forcing decoder_head=True for non-mirror modes')
            self.decoder_head = True

        self.padding = (self.kernel_size - 1) // 2

        # ============================================
        # CREATE ENCODER
        # ============================================

        self.encoder = Encoder1D(
            in_channels=self.in_channels,
            base_filters=self.base_filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            num_layers=self.num_layers,
            seq_length=self.seq_length,
            activation=self.act,
            flattened=self.flattened,
            compression_factor=self.compression_factor,
            bottleneck_act=self.bottleneck_act,
            compression_type=self.compression_type,
            bottleneck_conv=self.bottleneck_conv,
            conv_compression_strategy=self.conv_compression_strategy,
            skip_bottleneck=self.skip_bottleneck
        )

        # Store encoder outputs in config
        self.flattened_size = self.encoder.flattened_size
        self.cfg.model.flattened_size = self.flattened_size
        self.latent_dim = self.encoder.latent_dim
        self.cfg.model.latent_dim = self.latent_dim

        # ============================================
        # CREATE DECODER
        # ============================================

        self.decoder = Decoder1D(
            in_channels=self.in_channels,
            base_filters=self.base_filters,
            num_layers=self.num_layers,
            kernel_size=self.pool_ks,
            stride=self.pool_stride,
            latent_dim=self.latent_dim,
            flattened_size=self.flattened_size,
            seq_length=self.seq_length,
            seq_enc=self.encoder.seq_enc,
            activation=self.act,
            flattened=self.flattened,
            double_deconv=self.double_deconv,
            conv_padding=self.padding,
            conv_kernel_size=self.kernel_size,
            conv_stride=self.stride,
            conv_dilation=self.dilation,
            bottleneck_out_channels=self.encoder.bottleneck_out_channels,
            decoder_mode=self.decoder_mode,
            decoder_head=self.decoder_head,
            skip_bottleneck=self.skip_bottleneck
        )

    def forward(self, x):
        """
        Forward pass through autoencoder.

        Args:
            x: Input tensor (batch, in_channels, seq_length)

        Returns:
            Reconstructed tensor (batch, in_channels, seq_length)
        """
        # Optional input adapter for transfer learning
        if hasattr(self, "input_adapter") and self.input_adapter is not None:
            x = self.input_adapter(x)

        # Encode
        enc = self.encoder(x)

        # Decode
        out = self.decoder(enc)

        # Optional output adapter for transfer learning
        if hasattr(self, "output_adapter") and self.output_adapter is not None:
            out = self.output_adapter(out)

        return out