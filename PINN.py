# Save this as pinn_poisson_1d.py and run: python pinn_poisson_1d.py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ---- Config ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(1234)
torch.manual_seed(1234)


# ---- Neural Net Definition ----
class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        layer_list = []
        for i in range(len(layers) - 1):
            layer_list.append(nn.Linear(layers[i], layers[i + 1]))
        self.net = nn.ModuleList(layer_list)

        # Xavier initialization
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        z = torch.cat([x, t], dim=1)  # Combine x and t as input
        for i, layer in enumerate(self.net[:-1]):
            z = layer(z)
            z = self.activation(z)
        z = self.net[-1](z)
        return z


# ---- Problem: Diffusion Equation ----
def diffusion_coefficient(x):
    """Discontinuous diffusion coefficient D(x)."""
    D = torch.ones_like(x)
    D[x >= 0.5] = 0.1
    return D


def f_source(x, t):
    """Source term Q(x, t)."""
    return torch.zeros_like(x)


# ---- Collocation & Boundary Sampling ----
def sample_interior(n):
    """Sample collocation points in the interior of the domain."""
    x = np.random.rand(n, 1)  # x in (0, 1)
    t = np.random.rand(n, 1)  # t in (0, 1)
    return torch.tensor(x, dtype=torch.float32, requires_grad=True).to(
        device
    ), torch.tensor(t, dtype=torch.float32, requires_grad=True).to(device)


def sample_boundary(n):
    """Sample boundary points (x=0, x=1, t=0)."""
    x0 = torch.zeros((n, 1), dtype=torch.float32, requires_grad=True).to(device)
    x1 = torch.ones((n, 1), dtype=torch.float32, requires_grad=True).to(device)
    t = torch.rand((n, 1), dtype=torch.float32, requires_grad=True).to(device)
    return x0, x1, t


def sample_initial(n):
    """Sample initial condition points (t=0)."""
    x = torch.rand((n, 1), dtype=torch.float32, requires_grad=True).to(device)
    t0 = torch.zeros((n, 1), dtype=torch.float32, requires_grad=True).to(device)
    return x, t0


# ---- PDE Residual ----
def physics_residual(model, x, t):
    """Compute the residual of the diffusion equation."""
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[
        0
    ]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[
        0
    ]
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0]

    D = diffusion_coefficient(x)
    flux = x**2 * D * u_x
    flux_x = torch.autograd.grad(flux, x, torch.ones_like(flux), create_graph=True)[0]

    # r = u_t - (D * u_xx) - f_source(x, t)
    r = x**2 * u_t - flux_x - x**2 * f_source(x, t)  # Modified PDE for testing

    return r


def boundary_conditions(model, x_b0, x_b1, t_b, f_b1):
    u_b0 = model(x_b0, t_b)
    u_b1 = model(x_b1, t_b)

    # compute du/dr at r=0
    u_r = torch.autograd.grad(
        u_b0, x_b0, grad_outputs=torch.ones_like(u_b0), create_graph=True
    )[0]

    # enforce u_r(0,t) = 0
    bc_loss_origin = torch.mean(u_r**2)

    bc_loss_end = torch.mean((u_b1 - f_b1) ** 2)  # Dirichlet BC at x=1

    bc_loss = bc_loss_origin + bc_loss_end
    return bc_loss


# ---- Training Loop ----
def train(
    model,
    epochs=40000,
    n_collocation=200,
    n_boundary=500,
    n_initial=500,
    lr=1e-3,
    print_every=2000,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Collocation points
        x_coll, t_coll = sample_interior(n_collocation)

        # Physics loss (residual)
        r = physics_residual(model, x_coll, t_coll)
        loss_phys = torch.mean(r**2)

        # Boundary loss
        x_b0, x_b1, t_b = sample_boundary(n_boundary)
        f_b1 = 0.2 * torch.ones_like(x_b1)  # Assuming Dirichlet BC at x=1
        loss_b = boundary_conditions(model, x_b0, x_b1, t_b, f_b1)

        # Initial condition loss
        x_i, t_i = sample_initial(n_initial)
        u_i = model(x_i, t_i)
        u_exact_i = torch.exp(
            -((x_i - 0.3) ** 2) / (2 * 0.05**2)
        )  # Example initial condition
        loss_i = torch.sum((u_i - u_exact_i) ** 2)

        # Total loss
        if epoch < 2000:
            loss = loss_phys + loss_b + 10 * loss_i
        else:
            loss = 10 * loss_phys + loss_b + loss_i

        loss.backward()
        optimizer.step()

        if epoch % print_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:5d} | Loss: {loss.item():.3e} | Loss_phys: {loss_phys.item():.3e} | Loss_b: {loss_b.item():.3e} | Loss_i: {loss_i.item():.3e}"
            )

    return model


# ---- Run ----
if __name__ == "__main__":
    train_flag = True  # Set True to train the model
    layers = [2, 50, 50, 50, 1]  # Input: (x, t), Output: u(x, t)

    if train_flag:
        print("Training PINN for Diffusion Equation...")
        model = PINN(layers).to(device)

        model = train(
            model,
            epochs=4000,
            n_collocation=500,
            n_boundary=100,
            n_initial=500,
            lr=1e-3,
            print_every=500,
        )

        # Evaluate and plot
        model.eval()
        x_plot = torch.linspace(0, 1, 300).unsqueeze(1).to(device)
        t_values = [0.0, 0.1, 0.2, 0.5, 1.0]  # Time values to evaluate the solution

        plt.figure(figsize=(10, 6))
        # Plot PINN predictions for different times
        for t in t_values:
            t_plot = torch.full_like(x_plot, t).to(
                device
            )  # Create a constant t for all x
            with torch.no_grad():
                u_pred = model(x_plot, t_plot).cpu().numpy()
            plt.plot(x_plot.cpu().numpy(), u_pred, label=f"t = {t:.2f}")

        # Plot initial condition as dashed line
        x_init = x_plot.cpu().numpy()
        u_init = np.exp(-((x_init - 0.3) ** 2) / (2 * 0.05**2))
        plt.plot(x_init, u_init, "k--", label="Initial Condition")

        plt.xlabel("x")
        plt.ylabel("u(x, t)")
        plt.title("PINN Solution for Diffusion Equation at Different Times")
        plt.legend()
        plt.grid()
        plt.show()
