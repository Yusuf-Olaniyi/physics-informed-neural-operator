"""Training loop for the inverse operator G_i: w(x) -> q(x).

Purely data-driven (no physics residual — the physics constraint is
defined on the forward map EI*w''''=q; enforcing it here would require
differentiating the *input* signal w(x) rather than the model output,
which is a different formulation left for future work).
"""
import os
import torch
import torch.optim as optim


def train_inverse_fno(model, train_loader, epochs, lr=1e-3, device="cpu",
                       val_loader=None, save_dir="outputs/checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "best_inverse_model.pth")
    last_path = os.path.join(save_dir, "last_inverse_model.pth")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for X_inverse, load_target, x_coords in train_loader:
            X_inverse = X_inverse.to(device)
            load_target = load_target.to(device)
            x_coords = x_coords.to(device)

            model_input = torch.cat([X_inverse, x_coords.unsqueeze(-1)], dim=-1)
            prediction = model(model_input)

            loss = criterion(prediction, load_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        history["train_loss"].append(train_loss)
        log = f"Epoch [{epoch+1:3d}/{epochs}] | Train: {train_loss:.6e}"

        if val_loader is not None:
            model.eval()
            val_running = 0.0
            with torch.no_grad():
                for X_inverse, load_target, x_coords in val_loader:
                    X_inverse = X_inverse.to(device)
                    load_target = load_target.to(device)
                    x_coords = x_coords.to(device)
                    model_input = torch.cat([X_inverse, x_coords.unsqueeze(-1)], dim=-1)
                    prediction = model(model_input)
                    val_running += criterion(prediction, load_target).item()

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
                }, best_path)

        print(log)

    torch.save({"model_state_dict": model.state_dict()}, last_path)
    print("\nTraining completed.")
    print(f"Best model saved to: {best_path}")
    print(f"Last epoch model saved to: {last_path}")

    return model, history
