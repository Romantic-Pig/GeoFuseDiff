import torch
import torchvision
import numpy as np
import rasterio
import os
import re
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels=3, reduction_ratio=4):
        super().__init__()
        reduced_channels = max(1, in_channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class PFTAttention(nn.Module):
    def __init__(self, in_channels, out_channels=6, reduction_ratio=4):
        super().__init__()
        reduced_channels = max(1, in_channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, out_channels, kernel_size=1),
        )
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        y = self.sigmoid(y)
        x = self.conv(x)
        return x * y


class UpscaleDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir,
                 in_shape=(5, 5), out_shape=(125, 125),
                 constant_variables=["z", "slope", "aspect"],
                 coarse_dir="ERA5_2m",
                 fine_dir="CLDAS_2m",          
                 pft_dir="PFT",
                 constant_variables_filename=None,
                 pft_out_channels=6,
                 leak_free=True,               
                 provided_stats=None):       
        self.data_dir = data_dir
        self.coarse_dir = os.path.join(data_dir, coarse_dir)
        self.fine_dir = os.path.join(data_dir, fine_dir) if fine_dir is not None else None
        self.pft_dir = os.path.join(data_dir, pft_dir)
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.varnames = ["temp"]
        self.n_var = len(self.varnames)
        self.ntime = 0
        self.constant_variables_filename = constant_variables_filename
        self.constant_variables = constant_variables
        self.pft_out_channels = pft_out_channels
        self.leak_free = leak_free
        self.provided_stats = provided_stats

        if not os.path.exists(self.coarse_dir):
            raise FileNotFoundError(f"Coarse resolution directory not found: {self.coarse_dir}")
        if self.fine_dir is not None and not os.path.exists(self.fine_dir):
            raise FileNotFoundError(f"High resolution directory not found: {self.fine_dir}")
        if not os.path.exists(self.pft_dir):
            raise FileNotFoundError(f"Land use directory not found: {self.pft_dir}")

        tiff_pattern = re.compile(r'^\d{10}\.tif$')
        coarse_files = sorted([f for f in os.listdir(self.coarse_dir) if tiff_pattern.match(f)])

        if not coarse_files:
            raise ValueError("No TIFF files were found in the coarse resolution directory.")


        if self.fine_dir is not None:
            fine_files = sorted([f for f in os.listdir(self.fine_dir) if tiff_pattern.match(f)])
            if not fine_files:
                raise ValueError("No TIFF files were found in the high-resolution directory.")
            common_files = sorted(list(set(coarse_files) & set(fine_files)))
            if not common_files:
                raise ValueError("No TIFF files were found in the coarse resolution and high-resolution directory.")
            self.filenames = common_files
            self.has_fine = True
        else:
            self.filenames = coarse_files
            self.has_fine = False

        self.ntime = len(self.filenames)
        if self.ntime == 0:
            raise ValueError(f"No usable data file was found in the directory {data_dir}.")

        
        if self.has_fine:
            with rasterio.open(os.path.join(self.fine_dir, self.filenames[0])) as src:
                self.transform = src.transform
                height, width = src.height, src.width
                if (height, width) != self.out_shape:
                    raise ValueError(
                        f"{self.filenames[0]} TIFF shape mismatch: expect {self.out_shape}, get {(height, width)}")
                self.lon = np.array([self.transform[2] + x * self.transform[0] for x in range(width)])
                self.lat = np.array([self.transform[5] + y * self.transform[4] for y in range(height)])
        else:
           
            with rasterio.open(os.path.join(self.coarse_dir, self.filenames[0])) as src:
                coarse_transform = src.transform

                height_in, width_in = src.height, src.width
                x_res = coarse_transform[0] * (width_in / self.out_shape[1])
                y_res = coarse_transform[4] * (height_in / self.out_shape[0])
                self.transform = rasterio.Affine(
                    x_res, coarse_transform[1], coarse_transform[2],
                    coarse_transform[3], y_res, coarse_transform[5]
                )
            height, width = self.out_shape
            self.lon = np.array([self.transform[2] + x * self.transform[0] for x in range(width)])
            self.lat = np.array([self.transform[5] + y * self.transform[4] for y in range(height)])

        self.nlat = self.H = len(self.lat)
        self.nlon = self.W = len(self.lon)

        self.interp_transform = torchvision.transforms.Resize(self.out_shape,
                                                              interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                                                              antialias=True)
        self.coarsen_transform = torchvision.transforms.Resize(self.in_shape,
                                                               interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                                                               antialias=True)

        if self.constant_variables:
            self.channel_attention = ChannelAttention(in_channels=3, reduction_ratio=4)


        coarse_data = []
        fine_data = [] if self.has_fine else None


        for filename in self.filenames:
            try:
                coarse_path = os.path.join(self.coarse_dir, filename)
                with rasterio.open(coarse_path) as src:
                    data = src.read(1)
                    if np.any(np.isnan(data)):
                        continue
                    if data.shape != self.in_shape:
                        continue
                    coarse_data.append(data[np.newaxis, :, :])

                if self.has_fine:
                    fine_path = os.path.join(self.fine_dir, filename)
                    with rasterio.open(fine_path) as src:
                        fdata = src.read(1)
                        if np.any(np.isnan(fdata)):
                            continue
                        if fdata.shape != self.out_shape:
                            continue
                        fine_data.append(fdata[np.newaxis, :, :])
            except:
                continue

        coarse = torch.from_numpy(np.stack(coarse_data)).float()
        coarse_up = self.interp_transform(coarse)  # [N,1,H_out,W_out]

        if self.has_fine:
            fine = torch.from_numpy(np.stack(fine_data)).float()  # [N,1,H_out,W_out]
            residual = fine - coarse_up
        else:
            fine = None
            residual = None

        self.normalize_rawdata_mean = coarse_up.mean()
        self.normalize_rawdata_std = coarse_up.std()

        if residual is not None:
            self.normalize_residual_mean = residual.mean()
            self.normalize_residual_std = residual.std()
            self.sigma_data = self.normalize_residual_std
        else:
            if self.provided_stats is None:
                raise ValueError("No fine data is available for estimating residual statistics; please inject training-period statistics via provided_stats.")
            self.normalize_residual_mean = torch.tensor(self.provided_stats["normalize_residual_mean"])
            self.normalize_residual_std = torch.tensor(self.provided_stats["normalize_residual_std"])
            self.sigma_data = torch.tensor(self.provided_stats.get("sigma_data", float(self.normalize_residual_std)))

        self.normalize_rawdata_transform = torchvision.transforms.Normalize(
            float(self.normalize_rawdata_mean), float(self.normalize_rawdata_std)
        )
        self.normalize_residual_transform = torchvision.transforms.Normalize(
            float(self.normalize_residual_mean), float(self.normalize_residual_std)
        )

        # （DEM：z、slope、aspect） ===
        if self.constant_variables:
            path = self.constant_variables_filename
            if path is None or not os.path.exists(path):
                raise FileNotFoundError(f"Terrain file not found: {path}")
            with rasterio.open(path) as src:
                topo = src.read(1)
                if topo.shape != self.out_shape:
                    raise ValueError(f"{path} TIFF shape mismatch: expect {self.out_shape}, get {topo.shape}")
                topo = torch.from_numpy(topo).float()
                weights = np.cos(np.radians(self.lat))
                weighted_topo = topo * weights[:, np.newaxis]
                mean_var = weighted_topo.mean()
                std_var = weighted_topo.std()
                topo = (topo - mean_var) / std_var
                self.const_var = {'z': topo.unsqueeze(0)}

                dy, dx = np.gradient(topo.numpy(), self.lat[1] - self.lat[0], self.lon[1] - self.lon[0])
                slope = np.arctan(np.sqrt(dx ** 2 + dy ** 2)) * 180 / np.pi
                aspect = np.arctan2(-dy, dx) * 180 / np.pi
                aspect = np.where(aspect < 0, aspect + 360, aspect)
                slope = torch.from_numpy(slope).float()
                aspect = torch.from_numpy(aspect).float()
                weighted_slope = slope * weights[:, np.newaxis]
                weighted_aspect = aspect * weights[:, np.newaxis]
                mean_slope = weighted_slope.mean()
                std_slope = weighted_slope.std()
                mean_aspect = weighted_aspect.mean()
                std_aspect = weighted_aspect.std()
                slope = (slope - mean_slope) / std_slope
                aspect = (aspect - mean_aspect) / std_aspect
                self.const_var['slope'] = slope.unsqueeze(0)
                self.const_var['aspect'] = aspect.unsqueeze(0)
        else:
            self.const_var = {}

        # === PFT ===
        years = sorted(set(int(f[:4]) for f in self.filenames))
        self.pft_files = {year: f"PFT_{year}.tif" for year in years}
        unique_pft_values = set()
        for year, pft_file in self.pft_files.items():
            pft_path = os.path.join(self.pft_dir, pft_file)
            if not os.path.exists(pft_path):
                raise FileNotFoundError(f"PFT file not found: {pft_path}")
            with rasterio.open(pft_path) as src:
                pft_data = src.read(1)
                if pft_data.shape != self.out_shape:
                    raise ValueError(f"{pft_path} TIFF shape mismatch: expect {self.out_shape}, get {pft_data.shape}")
                unique_values = np.unique(pft_data)
                unique_pft_values.update(unique_values)

        self.pft_classes = sorted(list(unique_pft_values))
        self.num_pft_classes = len(self.pft_classes)
        self.pft_value_to_index = {val: idx for idx, val in enumerate(self.pft_classes)}
        self.pft_attention = PFTAttention(in_channels=self.num_pft_classes, out_channels=self.pft_out_channels,
                                          reduction_ratio=4)

        for year, pft_file in self.pft_files.items():
            pft_path = os.path.join(self.pft_dir, pft_file)
            with rasterio.open(pft_path) as src:
                pft_data = src.read(1)
                pft_data = torch.from_numpy(pft_data).long()
                pft_one_hot = torch.zeros((self.num_pft_classes, self.out_shape[0], self.out_shape[1]))
                for val, idx in self.pft_value_to_index.items():
                    pft_one_hot[idx] = (pft_data == val).float()
                self.const_var[f'pft_{year}'] = pft_one_hot

        # === time ===
        self.time = [datetime.strptime(f[:-4], '%Y%m%d%H') for f in self.filenames]
        self.month = np.array([t.month for t in self.time]) / 12.0
        self.time_of_day = np.array([t.hour + t.minute / 60.0 for t in self.time]) / 24.0
        self.month_norm = torch.from_numpy(self.month).float()
        self.time_of_day_norm = torch.from_numpy(self.time_of_day).float()

        self.vmin = torch.tensor(-30.0)
        self.vmax = torch.tensor(50.0)

    def __len__(self):
        return self.ntime

    def __getitem__(self, index):
        filename = self.filenames[index]
        coarse_path = os.path.join(self.coarse_dir, filename)
        year = int(filename[:4])
        try:
            with rasterio.open(coarse_path) as src:
                coarse_data = torch.from_numpy(src.read(1)).float().unsqueeze(0)   # [1,H_in,W_in]
        except Exception as e:
            raise ValueError(f"Error loading {filename} coarse resolution: {e}")

        if coarse_data.shape[1:] != self.in_shape:
            raise ValueError(f"{coarse_path} TIFF shape mismatch: expect {self.in_shape}, get {coarse_data.shape[1:]}")


        coarse_up = self.interp_transform(coarse_data)  # [1,H_out,W_out]

        # targets
        fine_data = None
        targets = None
        if self.has_fine:
            fine_path = os.path.join(self.fine_dir, filename)
            try:
                with rasterio.open(fine_path) as src:
                    fine_data = torch.from_numpy(src.read(1)).float().unsqueeze(0)  # [1,H_out,W_out]
            except Exception as e:
                raise ValueError(f"Error loading {filename} high-resolution: {e}")
            if fine_data.shape[1:] != self.out_shape:
                raise ValueError(f"{fine_path} TIFF shape mismatch: expect {self.out_shape}, get {fine_data.shape[1:]}")

            residual = fine_data - coarse_up
            targets = self.normalize_residual_transform(residual)

        # ===Normalize(coarse_up) + DEM（CA）
        inputs = self.normalize_rawdata_transform(coarse_up)  # 1

        if self.constant_variables:
            static_vars = torch.cat([self.const_var['z'], self.const_var['slope'], self.const_var['aspect']], dim=0)
            static_vars = self.channel_attention(static_vars)  # 3
            inputs = torch.cat([inputs, static_vars], dim=0)

        # PFT
        pft_key = f'pft_{year}'
        if pft_key not in self.const_var:
            fallback_year = next(iter(self.pft_files.keys()))
            pft_key = f'pft_{fallback_year}'
        pft_data = self.const_var[pft_key]
        pft_data = self.pft_attention(pft_data)  # => [pft_out_channels, H, W]

        return {
            "inputs": inputs,                 # 4
            "targets": targets,
            "pft_data": pft_data,             # pft_out_channels
            "fine": fine_data,
            "coarse": coarse_up,              # [1,H,W]
            "month": self.month_norm[index],
            "time_of_day": self.time_of_day_norm[index],
            "topo": self.const_var.get('z', torch.zeros(1, *self.out_shape)),
            "filename": filename
        }

    def residual_to_fine_image(self, residual, coarse_image):

        return coarse_image + (residual * self.normalize_residual_std + self.normalize_residual_mean)

    def plot_fine(self, image_fine, ax, vmin=None, vmax=None):
        plt.sca(ax)
        vmin = image_fine.min().item() if vmin is None else vmin
        vmax = image_fine.max().item() if vmax is None else vmax
        plt.imshow(image_fine, vmin=vmin, vmax=vmax, origin='upper',
                   extent=(self.lon.min(), self.lon.max(), self.lat.min(), self.lat.max()))
        plt.colorbar()

    def plot_batch(self, coarse_image, fine_image, fine_image_pred, N=3):
        batch_size = coarse_image.shape[0]
        N = min(N, batch_size)
        fig, axs = plt.subplots(N, 3, figsize=(15, N * 5))
        if N == 1:
            axs = [axs]
        for j in range(N):
            self.plot_fine(coarse_image[j, 0], axs[j][0])
            axs[j][0].set_title("Coarse resolution input (after interpolation)")
            self.plot_fine(fine_image_pred[j, 0], axs[j][1])
            axs[j][1].set_title("Model prediction (mean)")
            self.plot_fine(fine_image[j, 0], axs[j][2])
            axs[j][2].set_title("True high resolution")
        plt.tight_layout()
        return fig, axs
