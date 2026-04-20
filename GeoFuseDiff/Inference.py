import torch
from Network import EDMPrecond
import numpy as np
import matplotlib.pyplot as plt
import os
import rasterio
from rasterio.crs import CRS
from affine import Affine
from DatasetUS import UpscaleDataset
from torch.utils.data import DataLoader
import sys
import io
import logging
from TrainDiffusion import compute_metrics, compute_crps

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
logging.basicConfig(filename='inference.log', level=logging.INFO, format='%(asctime)s - %(message)s', encoding='utf-8')

def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
        logging.info(' '.join(map(str, args)))
    except OSError as e:
        logging.error(f"Console Error: {e}. message: {' '.join(map(str, args))}")

@torch.no_grad()
def sample_model_EDS(input_batch, model, device, dataset, num_steps=40,
                     sigma_min=0.002, sigma_max=80, rho=7, S_churn=40,
                     S_min=0, S_max=float('inf'), S_noise=1, num_samples=30):

    images_input = input_batch["inputs"].to(device, dtype=torch.float32)
    pft_input = input_batch["pft_data"].to(device, dtype=torch.float32)
    coarse = input_batch["coarse"].to(device, dtype=torch.float32)
    fine = input_batch["fine"].to(device, dtype=torch.float32)


    condition_params = torch.stack((input_batch["month"].to(device, dtype=torch.float32),
                                    input_batch["time_of_day"].to(device, dtype=torch.float32)), dim=1)

    sigma_min = max(sigma_min, model.sigma_min)
    sigma_max = min(sigma_max, model.sigma_max)


    predicted_samples = []
    for i in range(num_samples):


        init_noise = torch.randn((images_input.shape[0], 1, images_input.shape[2], images_input.shape[3]),
                                 dtype=torch.float32, device=device)


        step_indices = torch.arange(num_steps, dtype=torch.float32, device=init_noise.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) *
                   (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([model.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])


        x_next = init_noise * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = model.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)


            denoised = model(x_hat, t_hat, condition_img=images_input, pft_data=pft_input, class_labels=condition_params).to(torch.float32)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur


            if i < num_steps - 1:

                denoised_prime = model(x_next, t_next, condition_img=images_input, pft_data=pft_input, class_labels=condition_params).to(torch.float32)
                d_prime = (x_next - denoised_prime) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)


        predicted = dataset.residual_to_fine_image(x_next, coarse)
        predicted_samples.append(predicted.unsqueeze(1))  # (batch_size, 1, channels, H, W)

    predicted_samples = torch.cat(predicted_samples, dim=1)  # (batch_size, num_samples, channels, H, W)



    return coarse.detach().cpu(), fine.detach().cpu(), predicted_samples.detach().cpu()


def plot_results(coarse, fine, predicted, dataset, vmin=None, vmax=None, output_path="inference_result.png"):

    if predicted.dim() == 5:

        predicted_mean = predicted.mean(dim=1)

    batch_size = coarse.shape[0]
    N = min(3, batch_size)
    fig, axs = plt.subplots(N, 3, figsize=(15, N * 5))
    if N == 1:
        axs = [axs]

    titles = ['Coarse resolution input (after interpolation)', 'True high resolution', 'Model prediction (mean)']
    for j in range(N):
        dataset.plot_fine(coarse[j, 0], axs[j][0], vmin=vmin, vmax=vmax)
        axs[j][0].set_title(titles[0], fontsize=12, pad=10)

        dataset.plot_fine(fine[j, 0], axs[j][1], vmin=vmin, vmax=vmax)
        axs[j][1].set_title(titles[1], fontsize=12, pad=10)

        dataset.plot_fine(predicted_mean[j, 0], axs[j][2], vmin=vmin, vmax=vmax)
        axs[j][2].set_title(titles[2], fontsize=12, pad=10)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    safe_print(f"The result image has been saved to: {output_path}")


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    safe_print(f"Using equipment: {device}")


    base_data_path = "H:/GeoFuseDiff/results/"
    test_data_path = os.path.join(base_data_path, "H:/GeoFuseDiff/results/")
    dem_file_path = os.path.join(base_data_path, "DEM/dem_1km.tif")
    pft_base_dir = os.path.join(base_data_path, "Land_use/LinYi")
    model_path = "H:/GeoFuseDiff/results/results/best_model.pth"
    output_dir = "H:/GeoFuseDiff/results/predicted_results"


    if not os.path.exists(model_path):
        safe_print(f"Error: Model file not found at '{model_path}'. Please specify the correct path.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)


    dataset_test = UpscaleDataset(
        data_dir=test_data_path,
        use_all_files=True,
        constant_variables=["z", "slope", "aspect"],
        coarse_dir="ERA5_2m_predict",
        fine_dir="CLDAS_2m_predict",
        pft_dir=pft_base_dir,
        constant_variables_filename=dem_file_path,
        pft_out_channels=6
    )
    safe_print(f"The test dataset has been loaded and contains {len(dataset_test)} samples.")
    safe_print(f"The dataset's sigma_data: {dataset_test.sigma_data}")

    in_channels_cpem = 5
    diff_model = EDMPrecond(
        img_resolution=[125, 125],
        in_channels=in_channels_cpem,
        out_channels=1,
        label_dim=2,
        sigma_data=dataset_test.sigma_data,
        mid_channels=64,
        channel_mult=[1, 2, 3],
        attn_resolutions=[25, 5]
    ).to(device)

    # 加载预训练权重
    try:
        diff_model.load_state_dict(torch.load(model_path, map_location=device))
        safe_print(f"The model weights have been successfully loaded from {model_path}.")
    except Exception as e:
        safe_print(f"Error loading model weights from {model_path}: {e}")
        sys.exit(1)
    diff_model.eval()

    dataloader_test = DataLoader(dataset_test, batch_size=2, shuffle=False)


    for batch_idx, input_batch in enumerate(dataloader_test):
        safe_print(f"\nBatch being processed {batch_idx+1}/{len(dataloader_test)}...")
        coarse, fine, predicted_samples = sample_model_EDS(input_batch, diff_model, device, dataset_test)


        mae, rmse, crps = compute_metrics(predicted_samples, fine)
        safe_print(f"batch {batch_idx+1} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, CRPS: {crps:.4f}")


        predicted_mean = predicted_samples.mean(dim=1)
        filenames = input_batch.get("filename", [f"predicted_{i}" for i in range(predicted_mean.size(0))])

        for i, filename in enumerate(filenames):
            base_filename = os.path.basename(filename)
            output_path = os.path.join(output_dir, base_filename)
            try:
                with rasterio.open(
                        output_path, 'w', driver='GTiff',
                        height=predicted_mean.shape[2], width=predicted_mean.shape[3],
                        count=1, dtype=predicted_mean[i, 0].numpy().dtype,
                        crs=CRS.from_epsg(4326), transform=dataset_test.transform
                ) as dst:
                    dst.write(predicted_mean[i, 0].numpy(), 1)
                safe_print(f"The TIFF file has been saved: {output_path}")
            except Exception as e:
                safe_print(f"An error occurred while saving {output_path} using rasterio: {e}")

        if batch_idx == 0:
            vmin = dataset_test.vmin.item()
            vmax = dataset_test.vmax.item()
            plot_output_path = os.path.join(output_dir, f"inference_result_batch_{batch_idx}.png")
            plot_results(coarse, fine, predicted_samples, dataset_test, vmin=vmin, vmax=vmax, output_path=plot_output_path)

    safe_print(f"\nReasoning complete. Results saved to: {output_dir}")