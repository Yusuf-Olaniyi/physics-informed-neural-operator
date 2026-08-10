"""Physics-informed training loop for the forward operator G_f: q(x) -> w(x)."""
import os
import torch
import torch.optim as optim

from beam_no.physics import physics_loss


def train_pino(model, train_loader, epochs, E, I, lambda_phys=1e-6, lr=1e-3,
                device="cuda", val_loader=None, save_dir="outputs/checkpoints"):
    """Train the forward FNO with a combined data + physics loss.

    L = L_data + lambda_phys * L_physics

    Batch layout (from BeamDataset): X = [q(x), x], Y = [w(x)].
    """
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "best_forward_model.pth")
    last_path = os.path.join(save_dir, "last_forward_model.pth")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    mse = torch.nn.MSELoss()

    history = {"train_loss": [], "data_loss": [], "physics_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        running_loss = running_data = running_phys = 0.0

        for X_input, Y_target, x_coords in train_loader:
            X_input = X_input.to(device)
            Y_target = Y_target.to(device)

            optimizer.zero_grad()

            q = X_input[:, :, 0]
            # Rebuild x as a leaf tensor that requires grad, so the physics
            # residual can be computed via autograd on the model output.
            x_physics = X_input[:, :, 1].clone().detach().requires_grad_(True)

            model_input = torch.stack([q, x_physics], dim=-1)
            prediction = model(model_input)

            loss_data = mse(prediction, Y_target)
            loss_phys = physics_loss(prediction[:, :, 0], q, x_physics, E, I)
            loss = loss_data + lambda_phys * loss_phys

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_data += loss_data.item()
            running_phys += loss_phys.item()

        scheduler.step()

        n_batches = len(train_loader)
        train_loss = running_loss / n_batches
        data_loss = running_data / n_batches
        phys_loss = running_phys / n_batches

        history["train_loss"].append(train_loss)
        history["data_loss"].append(data_loss)
        history["physics_loss"].append(phys_loss)

        log = f"Epoch [{epoch+1:3d}/{epochs}] | Train: {train_loss:.6e} | Data: {data_loss:.6e} | Physics: {phys_loss:.6e}"

        if val_loader is not None:
            model.eval()
            val_running = 0.0
            with torch.no_grad():
                for X_input, Y_target, x_coords in val_loader:
                    X_input = X_input.to(device)
                    Y_target = Y_target.to(device)
                    prediction = model(X_input)
                    val_running += mse(prediction, Y_target).item()

            val_loss = val_running / len(val_loader)
            history["val_loss"].append(val_loss)
            log += f" | Val: {val_loss:.6e}"

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "E": E, "I": I, "lambda_phys": lambda_phys,
                }, best_path)

        print(log)

    torch.save({"model_state_dict": model.state_dict()}, last_path)
    print("\nTraining completed.")
    print(f"Best model saved to: {best_path}")
    print(f"Last epoch model saved to: {last_path}")

    return model, history
