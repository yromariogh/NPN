import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ReconNetwork, self).__init__()

        # Define linear layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Define the forward pass
        x = torch.relu(self.fc2(torch.relu(self.fc1(x))))
        x = self.fc3(x)
        return x
    

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        x = self.net(x)
        return x
    



class MLP_Lipschitz(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_Lipschitz, self).__init__()

        self.net = torch.nn.Sequential(
            torch.nn.utils.spectral_norm(torch.nn.Linear(input_size, hidden_size)),
            torch.nn.ReLU(),
            torch.nn.utils.spectral_norm(torch.nn.Linear(hidden_size, output_size)),
        )

    def forward(self, x):
        x = self.net(x)
        return x
    



class MyAutoencoder(nn.Module):
    def __init__(self, dims, residual=True):
        super().__init__()
        if not isinstance(dims, list) or len(dims) < 2:
            raise ValueError("La lista de dimensiones debe tener al menos dos valores: [dim_entrada, dim_latente].")
        
        self.residual = residual
        
        # Construcción del encoder a partir de la lista de dimensiones.
        encoder_layers = []
        for i in range(len(dims)-1):
            encoder_layers.append(nn.Linear(dims[i], dims[i+1]))
            # Se añade activación ReLU en todas las capas excepto la última.
            if i != len(dims)-2:
                encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Construcción del decoder de forma simétrica: se recorre la lista de dimensiones en orden inverso.
        decoder_layers = []
        for i in reversed(range(1, len(dims))):
            decoder_layers.append(nn.Linear(dims[i], dims[i-1]))
            if i != 1:
                decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x, sigma=None, **kwargs):
        B, *S = x.shape
        x_flat = x.view(B, -1)
        
        # Se obtiene la representación latente mediante el encoder.
        latent = self.encoder(x_flat)
        # Se reconstruye la entrada a partir del latent usando el decoder.
        decoded = self.decoder(latent)
        
        if self.residual:
            # Se añade la conexión residual (la forma debe coincidir: se espera que dims[0] == dim_input).
            decoded = decoded + x_flat
        
        decoded = decoded.view(B, *S)
        return decoded

    def get_latent(self, x, layer_idx=None):
        B, *S = x.shape
        latent = x.view(B, -1)
        
        if layer_idx is None:
            return self.encoder(latent)
        else:
            # Se recorre manualmente el encoder.
            for i, module in enumerate(self.encoder):
                latent = module(latent)
                if i == layer_idx:
                    return latent
            # Si layer_idx excede el número de módulos, retorna la salida final.
            return latent
        







class MyAutoencoder_Lipschitz(nn.Module):
    def __init__(self, dims, residual=True):
        super().__init__()
        if not isinstance(dims, list) or len(dims) < 2:
            raise ValueError("La lista de dimensiones debe tener al menos dos valores: [dim_entrada, dim_latente].")
        
        self.residual = residual
        
        # Construcción del encoder a partir de la lista de dimensiones.
        encoder_layers = []
        for i in range(len(dims)-1):
            ################## AÑADIR LA NORMALIZACION ESPECTRAL A VER SI MEJORA
            encoder_layers.append(torch.nn.utils.spectral_norm(nn.Linear(dims[i], dims[i+1])))
            # Se añade activación ReLU en todas las capas excepto la última.
            if i != len(dims)-2:
                encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Construcción del decoder de forma simétrica: se recorre la lista de dimensiones en orden inverso.
        decoder_layers = []
        for i in reversed(range(1, len(dims))):
            decoder_layers.append(torch.nn.utils.spectral_norm(nn.Linear(dims[i], dims[i-1])))
            if i != 1:
                decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x, sigma=None, **kwargs):
        B, *S = x.shape
        x_flat = x.view(B, -1)
        
        # Se obtiene la representación latente mediante el encoder.
        latent = self.encoder(x_flat)
        # Se reconstruye la entrada a partir del latent usando el decoder.
        decoded = self.decoder(latent)
        
        if self.residual:
            # Se añade la conexión residual (la forma debe coincidir: se espera que dims[0] == dim_input).
            decoded = decoded + x_flat
        
        decoded = decoded.view(B, *S)
        return decoded

    def get_latent(self, x, layer_idx=None):
        B, *S = x.shape
        latent = x.view(B, -1)
        
        if layer_idx is None:
            return self.encoder(latent)
        else:
            # Se recorre manualmente el encoder.
            for i, module in enumerate(self.encoder):
                latent = module(latent)
                if i == layer_idx:
                    return latent
            # Si layer_idx excede el número de módulos, retorna la salida final.
            return latent
        


def get_R_linear_module(args):
    if args.R_linear_module == 'MLP':
        return MLP(input_size=args.n**2, hidden_size=args.hidden_size_base, output_size=args.mB).to(args.device).eval()
    elif args.R_linear_module == 'MLP_Lipschitz':
        return MLP_Lipschitz(input_size=args.n**2, hidden_size=args.hidden_size_base, output_size=args.mB).to(args.device).eval()
    elif args.R_linear_module == 'ReconNetwork':
        return ReconNetwork(input_size=args.n**2, hidden_size=args.hidden_size_base, output_size=args.mB).to(args.device).eval()
    elif args.R_linear_module == 'linear':
        return nn.Linear(args.n**2, args.mB).to(args.device).eval()
    elif args.R_linear_module == 'S':
        return lambda x: x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    

class WSConv2d(nn.Conv2d):
    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=[1,2,3], keepdim=True)
        weight = weight - weight_mean
        std = weight.std(dim=[1,2,3], keepdim=True) + 1e-5
        weight = weight / std
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, h, w = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y



class FullyConnected(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, args):
        super(FullyConnected, self).__init__()
        
        self.exp_mode = args.exp_mode

        # Define linear layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        if 'Identity' in args.exp_mode:
            self.initialize_identity()

    def initialize_identity(self):
        # Initialize fc1
        with torch.no_grad():
            if self.fc1.weight.shape[0] >= self.fc1.weight.shape[1]:
                nn.init.eye_(self.fc1.weight[:self.fc1.weight.shape[1]])
            else:
                nn.init.eye_(self.fc1.weight[:, :self.fc1.weight.shape[0]])
            self.fc1.bias.zero_()

        # Initialize fc2
        with torch.no_grad():
            nn.init.eye_(self.fc2.weight[:self.fc2.weight.shape[1]])
            self.fc2.bias.zero_()

        # Initialize fc3
        with torch.no_grad():
            if self.fc3.weight.shape[0] >= self.fc3.weight.shape[1]:
                nn.init.eye_(self.fc3.weight[:self.fc3.weight.shape[1]])
            else:
                nn.init.eye_(self.fc3.weight[:, :self.fc3.weight.shape[0]])
            self.fc3.bias.zero_()

    def forward(self, x):
        # Define the forward pass
        out = self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))
        if 'Residual' in self.exp_mode:
            return out + x
        else:
            return out
        

def init_identity_conv(conv):
    """
    Inicializa una capa Conv2d (o ConvTranspose2d) para que actúe como la identidad.
    Solo funciona si in_channels == out_channels y el kernel es cuadrado e impar.
    """
    with torch.no_grad():
        if conv.in_channels != conv.out_channels:
            # No se puede inicializar como identidad si los canales no coinciden.
            return
        # Rellenamos con ceros.
        conv.weight.zero_()
        # Calculamos el índice central suponiendo un kernel impar.
        k = conv.kernel_size[0]
        center = k // 2
        # Para cada canal, asignamos el 1 en el centro.
        for i in range(conv.out_channels):
            conv.weight[i, i, center, center] = 1.0
        if conv.bias is not None:
            conv.bias.zero_()

    

class CNN(nn.Module):
    def __init__(self, in_channels=1, num_features=64, num_layers=5, args=None):
        super(CNN, self).__init__()
        self.exp_mode = args.exp_mode
        layers = []
        # Primera capa
        layers.append(nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        # Capas intermedias
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
        # Capa final
        layers.append(nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1))
        self.R_linear_module = get_R_linear_module(args)
        self.net = nn.Sequential(*layers)
        
        if 'Identity' in self.exp_mode:
            self.initialize_identity()

    def initialize_identity(self):
        # Iteramos sobre las capas de la secuencia
        for m in self.net.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                init_identity_conv(m)

    def forward(self, x):
        if 'Residual' in self.exp_mode:
            return self.net(x) + x
        else:
            return self.R_linear_module(self.net(x).reshape(x.shape[0], -1))  # Aplicamos la red lineal después de la convolución


        

class SelfAttention(nn.Module):
    def __init__(self, in_dim, args):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        # Si usamos Identity, iniciamos gamma en 0 para que: gamma*out + x = x
        init_gamma = 0.0 if 'Identity' in args.exp_mode else 0.0  # Puedes ajustar este valor
        self.gamma = nn.Parameter(torch.tensor(init_gamma))
        self.softmax = nn.Softmax(dim=-1)
        self.exp_mode = args.exp_mode
        
    def forward(self, x):
        batch, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch, C, width, height)
        if 'Residual' in self.exp_mode:
            return self.gamma * out + x
        else:
            return out



    
class CNNwithAttention(nn.Module):
    def __init__(self, in_channels=1, num_features=64, args=None):
        super(CNNwithAttention, self).__init__()
        self.exp_mode = args.exp_mode
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.attention = SelfAttention(num_features, args)
        self.conv2 = nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1)
        self.R_linear_module = get_R_linear_module(args)
        
        if 'Identity' in self.exp_mode:
            init_identity_conv(self.conv1)
            init_identity_conv(self.conv2)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.attention(out)
        out = self.conv2(out)
        if 'Residual' in self.exp_mode:
            return out + x
        else:
            return self.R_linear_module(out.reshape(x.shape[0], -1))

        

class MultiScale(nn.Module):
    def __init__(self, in_channels=1, num_features=64, args=None):
        super(MultiScale, self).__init__()
        self.down_conv = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up_conv = nn.ConvTranspose2d(num_features, in_channels, kernel_size=2, stride=2)
        self.exp_mode = args.exp_mode
        self.R_linear_module = get_R_linear_module(args)
        
        if 'Identity' in self.exp_mode:
            init_identity_conv(self.down_conv)
            init_identity_conv(self.up_conv)
            for m in self.bottleneck.modules():
                if isinstance(m, nn.Conv2d):
                    init_identity_conv(m)
        
    def forward(self, x):
        out = self.down_conv(x)
        out = self.pool(out)
        out = self.bottleneck(out)
        out = self.up_conv(out)
        if 'Residual' in self.exp_mode:
            return out + x
        else:
            return self.R_linear_module(out.reshape(x.shape[0], -1))

        

class ViT(nn.Module):
    def __init__(self, in_channels=1, embed_dim=64, num_heads=4, num_layers=2, patch_size=4, image_size=32, args=None):
        super(ViT, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        # Proyección de la imagen a un espacio embebido
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Se puede intentar inicializar self.proj como identidad si embed_dim == in_channels, de lo contrario se deja normal.
        if 'Identity' in args.exp_mode and in_channels == embed_dim:
            init_identity_conv(self.proj)
        # Encoder del transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Reconstrucción de la imagen
        self.deproj = nn.ConvTranspose2d(embed_dim, in_channels, kernel_size=patch_size, stride=patch_size)
        if 'Identity' in args.exp_mode and in_channels == embed_dim:
            init_identity_conv(self.deproj)
        self.exp_mode = args.exp_mode
        self.R_linear_module = get_R_linear_module(args)
        
    def forward(self, x):
        x_proj = self.proj(x)
        b, c, h, w = x_proj.shape
        x_proj = x_proj.flatten(2).transpose(1,2)
        x_transformed = self.transformer(x_proj)
        x_transformed = x_transformed.transpose(1,2).view(b, c, h, w)
        x_reconstructed = self.deproj(x_transformed)
        if 'Residual' in self.exp_mode:
            return x_reconstructed + x
        else:
            return self.R_linear_module(x_reconstructed.reshape(x.shape[0], -1))

        


class PositionalEncoding(nn.Module):
    def __init__(self, num_features, max_iter=1000):
        super(PositionalEncoding, self).__init__()
        # Usamos un embedding para mapear el número de iteración a un vector del tamaño de num_features
        self.embedding = nn.Embedding(max_iter, num_features)
        # Inicializamos los pesos (opcionalmente, con valores pequeños)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

    def forward(self, iteration, batch_size, height, width, device):
        # iteration se espera que sea un entero (por ejemplo, el número de iteración actual)
        iter_tensor = torch.tensor(iteration, dtype=torch.long, device=device)
        # Obtener el embedding: (num_features,)
        pe = self.embedding(iter_tensor)
        # Reorganizar a (1, num_features, 1, 1) y expandir para que se sume a la feature map
        pe = pe.view(1, -1, 1, 1).expand(batch_size, -1, height, width)
        return pe


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0, init_identity=False, args=None):
        super(ConvNeXtBlock, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_prob=args.drop_path_prob) if args.drop_path_prob > 0 else nn.Identity()
        self.args = args
        self.res_scaler = nn.Parameter(torch.ones(1) * args.res_scaler)
        self.use_SE = args.use_SE

        if self.use_SE == 'True':
            self.se = SEBlock(dim)
        if init_identity:
            self.initialize_identity()
        
        # Si se activa el uso de cosPE, incorporamos una capa para proyectar el embedding del timestep
        if args.use_cosPE == 'True':
            self.t_emb_layer = nn.Sequential(
                nn.SiLU(),
                nn.Linear(args.time_dim, dim)
            )
    
    def initialize_identity(self):
        nn.init.zeros_(self.pwconv1.weight)
        nn.init.zeros_(self.pwconv1.bias)
        nn.init.zeros_(self.pwconv2.weight)
        nn.init.zeros_(self.pwconv2.bias)
    
    @staticmethod
    def cos_pos_encoding(t, channels):
        # t se espera que sea de forma (batch, 1) y channels es la dimensión objetivo (por ejemplo, args.time_dim)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=t.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
    def forward(self, x, t=None):
        shortcut = x
        x = self.dwconv(x)
        # Reorganizamos para aplicar LayerNorm en el canal final
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        if self.use_SE == 'True':
            x = self.se(x)
        x = shortcut + self.drop_path(self.res_scaler * x)
        
        # Si se proporciona t y se activó el uso de cosPE, se agrega el embedding de t al bloque
        if t is not None and self.args.use_cosPE == 'True':
            # Nos aseguramos de que t sea un tensor de forma (batch, 1) y de tipo float
            if not torch.is_tensor(t):
                t = torch.full((x.shape[0], 1), t, device=x.device, dtype=torch.float)
            else:
                if t.dim() == 1:
                    t = t.unsqueeze(-1)
                t = t.type(torch.float)
            # Calculamos la codificación cos/sin con dimensión args.time_dim
            t_encoded = ConvNeXtBlock.cos_pos_encoding(t, self.args.time_dim)
            # Proyectamos el embedding para que tenga dimensión 'dim' y expandimos para poder sumarlo a x
            emb = self.t_emb_layer(t_encoded)[:, :, None, None]
            x = x + emb
        return x


class ConvNeXt(nn.Module):
    def __init__(self, in_channels=1, num_features=64, num_blocks=4, args=None):
        super(ConvNeXt, self).__init__()
        # Capa inicial: WSConv2d o Conv2d según el flag
        if args.WSConv == 'True':
            self.initial_conv = WSConv2d(in_channels, num_features, kernel_size=3, padding=1)
        else:
            self.initial_conv = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        
        # En lugar de usar nn.Sequential, usamos ModuleList para poder pasar t a cada bloque
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(num_features, init_identity=('Identity' in args.exp_mode), args=args)
            for _ in range(num_blocks)
        ])
        
        # Capa final: WSConv2d o Conv2d según el flag
        if args.WSConv == 'True':
            self.final_conv = WSConv2d(num_features, in_channels, kernel_size=3, padding=1)
        else:
            self.final_conv = nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1)
        
        self.exp_mode = args.exp_mode
        self.args = args
        self.use_PE = args.use_PE        # Opción para usar positional encoding (PE) aprendido
        self.use_cosPE = args.use_cosPE  # Opción para usar el PE basado en sin/cos (como en IndiUnet)

        if 'Identity' in self.exp_mode:
            init_identity_conv(self.initial_conv)
            init_identity_conv(self.final_conv)
        
        # En este caso, ya que vamos a pasar t a cada bloque (como en IndiUnet), no aplicamos PE en la capa inicial.
        # Si se desea conservar la opción de aplicar un PE global (usando embedding aprendido), se puede agregar aquí.
        if self.use_PE == 'True' and self.use_cosPE != 'True':
            self.pos_encoder = PositionalEncoding(num_features, max_iter=args.max_iter)

        self.R_linear_module = get_R_linear_module(args)
    
    def cos_pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=t.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, iteration=None):
        # Se aplica la capa inicial sin sumar ningún PE (como en IndiUnet, donde t se propaga a través de cada módulo)
        out = self.initial_conv(x)
        
        # Si se quisiera aplicar un PE global (con embedding aprendido) antes de los bloques, se podría descomentar:
        # if self.use_PE == 'True' and iteration is not None:
        #     batch_size, _, height, width = out.shape
        #     pe = self.pos_encoder(iteration, batch_size, height, width, out.device)
        #     out = out + pe

        # Se pasa la iteración t a cada bloque (simulando cómo en IndiUnet se inyecta t en cada capa)
        for block in self.blocks:
            out = block(out, iteration)
        
        out = self.final_conv(out)
        if 'Residual' in self.exp_mode:
            return out + x
        else:
            return self.R_linear_module(out.reshape(x.shape[0], -1))  # Aplicamos la red lineal después de la convolución




def double_convLeon(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )


# adapted from https://github.com/usuyama/pytorch-unet/tree/master
class UNetLeon(nn.Module):

    def __init__(self, n_channels, base_channel):
        super().__init__()

        self.dconv_down1 = double_convLeon(n_channels, base_channel)
        self.dconv_down2 = double_convLeon(base_channel, base_channel * 2)
        self.dconv_down3 = double_convLeon(base_channel * 2, base_channel * 4)
        self.dconv_down4 = double_convLeon(base_channel * 4, base_channel * 8)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.dconv_up3 = double_convLeon(base_channel * 12, base_channel * 4)
        self.dconv_up2 = double_convLeon(base_channel * 6, base_channel * 2)
        self.dconv_up1 = double_convLeon(base_channel * 3, base_channel)

        self.conv_last = nn.Conv2d(base_channel, n_channels, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)  # 256x256

        x = self.maxpool(conv1)  # 128x128
        conv2 = self.dconv_down2(x)

        x = self.maxpool(conv2)  # 64x64
        conv3 = self.dconv_down3(x)

        x = self.maxpool(conv3)  # 32x32
        bootle = self.dconv_down4(x)

        x = self.upsample(bootle)  # 64x64
        x = torch.cat([x, conv3], dim=1)
        up1 = self.dconv_up3(x)

        x = self.upsample(up1)  # 128x128
        x = torch.cat([x, conv2], dim=1)
        up2 = self.dconv_up2(x)

        x = self.upsample(up2)  # 256x256
        x = torch.cat([x, conv1], dim=1)
        up3 = self.dconv_up1(x)

        out = self.conv_last(up3)

        return out






class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], args=None):
        super(UNet, self).__init__()
        self.exp_mode = args.exp_mode
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.R_linear_module = get_R_linear_module(args)

        # Encoder: cada etapa reduce la resolución y aumenta las características
        for feature in features:
            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feature, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = feature

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1] * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-1] * 2, features[-1] * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder: configuración según el modo de fusión de las skip connections
        rev_features = list(reversed(features))
        for i, feature in enumerate(rev_features):
            in_ch = features[-1] * 2 if i == 0 else rev_features[i - 1]
            up_conv = nn.ConvTranspose2d(in_ch, feature, kernel_size=2, stride=2)

            if self.exp_mode == "Residual":
                conv_block = nn.Sequential(
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            else:
                conv_block = nn.Sequential(
                    nn.Conv2d(feature * 2, feature, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            self.ups.append(up_conv)
            self.ups.append(conv_block)

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        if 'Identity' in self.exp_mode:
            self.initialize_identity()

    def initialize_identity(self):
        # Inicializa todas las capas convolucionales para que actúen como la identidad cuando sea posible.
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # Solo si el número de canales coincide y el kernel es impar.
                if m.in_channels == m.out_channels and m.kernel_size[0] % 2 == 1:
                    init_identity_conv(m)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i // 2]
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])
            if self.exp_mode == "Residual":
                x = x + skip_connection
            else:
                x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i + 1](x)

        return self.R_linear_module(self.final_conv(x))  # Aplicamos la red lineal después de la convolución


class IndiSelfAttention(nn.Module):
    """
    A Self-Attention module implementing multi-headed attention mechanism.

    This module applies a multi-head attention mechanism on the input feature map,
    followed by layer normalization and a feedforward neural network.

    Attributes:
        channels (int): The number of channels in the input.
        size (int): The size of each attention head.
    """
    def __init__(self, channels, size):
        super(IndiSelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    """
    Normal convolution block, with 2d convolution -> Group Norm -> GeLU -> convolution -> Group Norm
    Possibility to add residual connection providing residual=True
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """
    maxpool reduce size by half -> 2*DoubleConv -> Embedding layer
    
    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear( # linear projection to bring the time embedding to the proper dimension
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1]) # projection
        return x + emb


class Up(nn.Module):
    """
    We take the skip connection which comes from the encoder
    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )
        
    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class IndiUnet(nn.Module):
    def __init__(self, c_in=1, c_out=1, image_size=64, time_dim=256, device='cuda', latent=False, true_img_size=64, num_classes=None, args=None):
        super(IndiUnet, self).__init__()

        # Encoder
        self.true_img_size = true_img_size
        self.image_size = image_size
        self.time_dim = time_dim
        self.device = device
        self.inc = DoubleConv(c_in, self.image_size) # Wrap-up for 2 Conv Layers
        self.down1 = Down(self.image_size, self.image_size*2) # input and output channels
        # self.sa1 = IndiSelfAttention(self.image_size*2,int( self.true_img_size/2)) # 1st is channel dim, 2nd current image resolution
        self.down2 = Down(self.image_size*2, self.image_size*4)
        # self.sa2 = IndiSelfAttention(self.image_size*4, int(self.true_img_size/4))
        self.down3 = Down(self.image_size*4, self.image_size*4)
        # self.sa3 = IndiSelfAttention(self.image_size*4, int(self.true_img_size/8))
        
        # Bootleneck
        self.bot1 = DoubleConv(self.image_size*4, self.image_size*8)
        self.bot2 = DoubleConv(self.image_size*8, self.image_size*8)
        self.bot3 = DoubleConv(self.image_size*8, self.image_size*4)
        
        # Decoder: reverse of encoder
        self.up1 = Up(self.image_size*8, self.image_size*2)
        # self.sa4 = IndiSelfAttention(self.image_size*2, int(self.true_img_size/4))
        self.up2 = Up(self.image_size*4, self.image_size)
        # self.sa5 = IndiSelfAttention(self.image_size, int(self.true_img_size/2))
        self.up3 = Up(self.image_size*2, self.image_size)
        # self.sa6 = IndiSelfAttention(self.image_size, self.true_img_size)
        self.outc = nn.Conv2d(self.image_size, c_out, kernel_size=1) # projecting back to the output channel dimensions
        
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

        if latent == True:
            self.latent = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, 256)).to(device)    
            
        self.R_linear_module = get_R_linear_module(args)
  
    def pos_encoding(self, t, channels):
        """
        Input noised images and the timesteps. The timesteps will only be
        a tensor with the integer timesteps values in it
        """
        inv_freq = 1.0 /  (
            10000 
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc 

    def forward(self, x, lab=None, t=None):
        # Pass the source image through the encoder network
        t = torch.tensor(t).unsqueeze(-1).type(torch.float).to(self.device)
        t = self.pos_encoding(t, self.time_dim) # Encoding timesteps is HERE, we provide the dimension we want to encode

        
        if lab is not None:
            t += self.label_emb(lab)
        
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        # x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        # x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        # x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        
        x = self.up1(x4, x3, t) # We note that upsampling box that in the skip connections from encoder 
        # x = self.sa4(x)
        x = self.up2(x, x2, t)
        # x = self.sa5(x)
        x = self.up3(x, x1, t)
        # x = self.sa6(x)
        output = self.outc(x)

        return self.R_linear_module(output.reshape(x.shape[0], -1))  # Aplicamos la red lineal después de la convolución