import torch
import torch.amp
import torchvision
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
from DatasetUS import UpscaleDataset
from Network import EDMPrecond
from scipy.stats import norm
import scipy.fft


class VPLoss:

    def __init__(self, sigma_data=1.0):
        self.sigma_data = sigma_data

    def __call__(self, net, images, conditional_img, pft_data, labels, augment_labels=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * 1.2 + 0.0).exp()
        weight = 1 / (self.sigma_data ** 2 + sigma ** 2)
        y, augment_labels = images, augment_labels
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, conditional_img, pft_data, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss.mean()


def compute_crps(pred_samples, target):

    pred_samples = pred_samples.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    batch_size, num_samples, _, height, width = pred_samples.shape
    crps = 0.0
    for i in range(batch_size):
        for h in range(height):
            for w in range(width):
                samples = pred_samples[i, :, 0, h, w]
                y = target[i, 0, h, w]
                samples = np.sort(samples)
                cdf = np.linspace(0, 1, num_samples)
                crps += np.trapz(np.abs(cdf - (samples >= y).astype(float)), samples)
    crps /= (batch_size * height * width)
    return crps


def compute_metrics(pred_samples, target):

    pred_mean = pred_samples.mean(dim=1)
    target_for_metrics = target.cpu()
    pred_mean_for_metrics = pred_mean.detach().cpu()

    mae = np.mean(np.abs(pred_mean_for_metrics.numpy() - target_for_metrics.numpy()))
    rmse = np.sqrt(np.mean((pred_mean_for_metrics.numpy() - target_for_metrics.numpy()) ** 2))
    crps = compute_crps(pred_samples.cpu(), target.cpu())

    return mae, rmse, crps


def compute_power_spectrum(image):
    image = image.detach().cpu().numpy()
    f_transform = scipy.fft.fft2(image)
    f_shift = scipy.fft.fftshift(f_transform)
    power_spectrum = np.abs(f_shift) ** 2
    h, w = image.shape
    center = (h // 2, w // 2)
    y, x = np.indices((h, w))
    radii = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    radii = radii.astype(int)
    max_radius = min(h, w) // 2
    radial_mean = np.array([power_spectrum[radii == r].mean() for r in range(max_radius) if np.any(radii == r)])
    return radial_mean


def training_step(network, loss_fn, optimiser, dataloader, scaler, step, accum, writer, device):
    network.train()
    losses = []
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"train :: Epoch: {step}")):
        image_input = batch["inputs"].to(device)
        image_output = batch["targets"].to(device)
        pft_data = batch["pft_data"].to(device)
        condition_params = torch.cat(
            (batch["month"].to(device).unsqueeze(1), batch["time_of_day"].to(device).unsqueeze(1)), dim=1)
        with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
            loss = loss_fn(net=network, images=image_output, conditional_img=image_input, pft_data=pft_data,
                           labels=condition_params)
        scaler.scale(loss).backward()
        if accum > 1:
            if (batch_idx + 1) % accum == 0 or (batch_idx + 1 == len(dataloader)):
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad()
        else:
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad()
        losses.append(loss.detach().cpu().numpy())
        writer.add_scalar("loss", loss.detach().cpu().numpy(), batch_idx + step * len(dataloader))
    return np.mean(losses)


@torch.no_grad()
def validation_step(network, loss_fn, dataloader, writer, step, device, save_dir, best_mae, best_model_path,
                    best_metrics):

    network.eval()
    losses = []
    maes, rmses, crpss = [], [], []
    temp_model_path = os.path.join(save_dir, f"temp_model_{step}.pth")
    temp_image_path = os.path.join(save_dir, f"temp_Epoch_{step}.png")
    temp_uncertainty_path = os.path.join(save_dir, f"uncertainty_Epoch_{step}.png")
    temp_spectrum_path = os.path.join(save_dir, f"spectrum_Epoch_{step}.png")

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"test :: Epoch: {step}")):
        image_input = batch["inputs"].to(device)
        image_output = batch["targets"].to(device)
        pft_data = batch["pft_data"].to(device)
        condition_params = torch.cat(
            (batch["month"].to(device).unsqueeze(1), batch["time_of_day"].to(device).unsqueeze(1)), dim=1)
        topo = batch["topo"].to(device)

        num_samples = 30
        pred_samples = []
        sigma = torch.ones(image_output.shape[0], device=device)
        for _ in range(num_samples):
            noise = torch.randn_like(image_output)
            pred = network(noise, sigma, image_input, pft_data, condition_params)
            pred = dataloader.dataset.residual_to_fine_image(pred, batch["coarse"].to(device))
            pred_samples.append(pred.unsqueeze(1))
        pred_samples = torch.cat(pred_samples, dim=1)

        with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
            loss = loss_fn(net=network, images=image_output, conditional_img=image_input, pft_data=pft_data,
                           labels=condition_params)

        mae, rmse, crps = compute_metrics(pred_samples, batch["fine"].to(device))
        maes.append(mae)
        rmses.append(rmse)
        crpss.append(crps)
        losses.append(loss.detach().cpu().numpy())

        if batch_idx == 0:
            fine_image = batch["fine"].to(device)
            coarse_image = batch["coarse"].to(device)
            pred_mean = pred_samples.mean(dim=1)
            fig, _ = dataloader.dataset.plot_batch(coarse_image.detach().cpu(), fine_image.detach().cpu(),
                                                   pred_mean.detach().cpu())
            plt.savefig(temp_image_path)
            plt.close()

            uncertainty = pred_samples.std(dim=1) * (1 + topo / topo.std())
            fig, ax = plt.subplots()
            dataloader.dataset.plot_fine(uncertainty[0, 0].cpu(), ax, vmin=uncertainty.min().item(),
                                         vmax=uncertainty.max().item())
            plt.savefig(temp_uncertainty_path)
            plt.close()

            pred_spectrum = compute_power_spectrum(pred_mean[0, 0].cpu())
            true_spectrum = compute_power_spectrum(fine_image[0, 0].cpu())
            fig, ax = plt.subplots()
            freqs = np.arange(len(pred_spectrum))
            ax.plot(freqs, pred_spectrum, label="prediction")
            ax.plot(freqs, true_spectrum, label="true")
            ax.set_xlabel("frequency")
            ax.set_ylabel("power")
            ax.set_title("Power spectrum")
            ax.legend()
            plt.savefig(temp_spectrum_path)
            plt.close()

    epoch_loss = np.mean(losses)
    epoch_mae = np.mean(maes)
    epoch_rmse = np.mean(rmses)
    epoch_crps = np.mean(crpss)

    writer.add_scalar("Loss/Verification", epoch_loss, step)
    writer.add_scalar("Indicator/MAE", epoch_mae, step)
    writer.add_scalar("Indicator/RMSE", epoch_rmse, step)
    writer.add_scalar("Indicator/CRPS", epoch_crps, step)

    updated_best_metrics = best_metrics.copy()
    if epoch_mae < best_metrics["mae"]["value"]:
        updated_best_metrics["mae"]["value"] = epoch_mae
        updated_best_metrics["mae"]["epoch"] = step
    if epoch_rmse < best_metrics["rmse"]["value"]:
        updated_best_metrics["rmse"]["value"] = epoch_rmse
        updated_best_metrics["rmse"]["epoch"] = step
    if epoch_crps < best_metrics["crps"]["value"]:
        updated_best_metrics["crps"]["value"] = epoch_crps
        updated_best_metrics["crps"]["epoch"] = step

    best_mae_updated = best_mae
    if epoch_mae < best_mae:
        best_mae_updated = epoch_mae
        torch.save(network.state_dict(), temp_model_path)
        best_model_path_new = os.path.join(save_dir, "best_model.pth")
        best_image_path = os.path.join(save_dir, "best_result.png")
        best_uncertainty_path = os.path.join(save_dir, "best_uncertainty.png")
        best_spectrum_path = os.path.join(save_dir, "best_spectrum.png")
        os.replace(temp_model_path, best_model_path_new)
        os.replace(temp_image_path, best_image_path)
        os.replace(temp_uncertainty_path, best_uncertainty_path)
        os.replace(temp_spectrum_path, best_spectrum_path)
    else:
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        if os.path.exists(temp_uncertainty_path):
            os.remove(temp_uncertainty_path)
        if os.path.exists(temp_spectrum_path):
            os.remove(temp_spectrum_path)

    return epoch_loss, epoch_mae, epoch_rmse, epoch_crps, best_mae_updated, updated_best_metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Use equipment: {device}")
    pft_channels = 6

    dataset_train = UpscaleDataset(
        "H:/GeoFuseDiff/", train=True, use_all_files=False, constant_variables=["z", "slope", "aspect"],
        coarse_dir="ERA5_2m", fine_dir="CLDAS_2m", pft_dir="土地利用数据/LinYi",
        constant_variables_filename="H:/GeoFuseDiff//DEM/dem_1km.tif",
        pft_out_channels=pft_channels
    )
    dataset_test = UpscaleDataset(
        "H:/GeoFuseDiff/", train=False, use_all_files=False, constant_variables=["z", "slope", "aspect"],
        coarse_dir="ERA5_2m", fine_dir="CLDAS_2m", pft_dir="Land_use/LinYi",
        constant_variables_filename="H:/GeoFuseDiff/DEM/dem_1km.tif",
        pft_out_channels=pft_channels
    )
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=2, shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=2, shuffle=False)

    # CPEM inputs
    in_channels_cpem = 1 + 1 + 3
    network = EDMPrecond(
        img_resolution=[125, 125],
        in_channels=in_channels_cpem,
        out_channels=1,
        label_dim=2,
        sigma_data=dataset_train.sigma_data,
        mid_channels=64
    ).to(device)

    optimiser = torch.optim.Adam(network.parameters(), lr=1e-4)
    loss_fn = VPLoss(sigma_data=dataset_train.sigma_data)
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')

    writer = SummaryWriter()
    save_dir = "H:/GeoFuseDiff/results/"
    os.makedirs(save_dir, exist_ok=True)
    num_epochs = 100
    best_mae = float('inf')
    best_model_path = os.path.join(save_dir, "best_model.pth")

    best_metrics = {
        "mae": {"value": float('inf'), "epoch": -1},
        "rmse": {"value": float('inf'), "epoch": -1},
        "crps": {"value": float('inf'), "epoch": -1}
    }

    for step in range(num_epochs):
        epoch_loss = training_step(network, loss_fn, optimiser, dataloader_train, scaler, step, accum=1, writer=writer,
                                   device=device)
        print(f"train :: Epoch {step}, loss: {epoch_loss:.4f}")
        val_loss, val_mae, val_rmse, val_crps, best_mae, updated_best_metrics = validation_step(
            network, loss_fn, dataloader_test, writer, step, device, save_dir, best_mae, best_model_path, best_metrics
        )
        best_metrics = updated_best_metrics
        print(
            f"test :: Epoch {step}, loss: {val_loss:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, CRPS: {val_crps:.4f}")

    writer.close()


if __name__ == "__main__":
    main()