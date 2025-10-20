import numpy as np
import astropy.units as u
import astropy.constants as const
import math


def get_theoretical_profile(r, masks, v_w, D_values, R_TS, R_b, f_gal, f_TS):
    """
    Calculate the theoretical Giovanni profile for cosmic rays.

    Parameters:
    -----------
    r : array-like
        Radial coordinates (pc)
    r_bubble : array-like (bool)
        Boolean mask for bubble region
    r_ISM : array-like (bool)
        Boolean mask for ISM region
    v_w : astropy Quantity
        Wind velocity
    D_values : array-like
        Diffusion coefficient values
    R_TS : astropy Quantity
        Termination shock radius
    R_b : astropy Quantity
        Bubble radius
    f_gal : float
        Galactic background level
    f_TS : float
        Termination shock level

    Returns:
    --------
    np.array
        Theoretical profile
    """
    # masks
    r_bubble = masks["r_bubble"]
    r_ISM = masks["r_ISM"]

    # Ensure correct units for calculations
    v_b = v_w.to("pc/Myr") / 4
    # D inside bubble at mid-point
    D_b = D_values[r_bubble][5].to("pc**2/Myr")
    D_out = D_values[r_ISM][0].to("pc**2/Myr")

    # α(r,p)
    alpha_pre = v_b * R_TS / D_b
    alpha = (v_b * R_TS / D_b) * (1.0 - R_TS / (r[r_bubble] * u.pc))

    print(
        f"Debug: v_b={v_b.decompose().value:.3e}, D_b={D_b.to('pc**2/Myr').value:.3e} ({D_b.to('cm**2/s').value:.3e} cm2/s)"
    )
    print(f"Debug: alpha_pre={alpha_pre.decompose().value:.3e}")

    # α_b = α(r=R_b,p)
    alpha_b = (v_b * R_TS / D_b) * (1.0 - R_TS / R_b)

    print(f"Debug: alpha_b={alpha_b.decompose().value:.3e}")

    # β(p)
    beta = (D_out * R_b) / (v_b * R_TS**2)

    print(f"Warning: beta={beta.decompose().value:.4g} < 1.")
    print(
        "D_out = {:.3e}, v_b = {:.3e}, R_b = {:.3e}, R_TS = {:.3e}".format(
            D_out.to("pc**2/Myr").value,
            v_b.to("pc/Myr").value,
            R_b.to("pc").value,
            R_TS.to("pc").value,
        )
    )

    EXP_MAX = 700.0
    EXP_MIN = -700.0
    alpha_clip = np.clip(alpha, EXP_MIN, EXP_MAX)

    if np.all(alpha == alpha_clip):  # Normal case
        # f_b(r,p) / f_TS
        numerator = (
            np.exp(alpha) + beta * (np.exp(alpha_b) - np.exp(alpha))
        ) + f_gal / f_TS * beta * (np.exp(alpha) - 1.0)
        denominator = 1.0 + beta * (np.exp(alpha_b) - 1.0)
        f_b_over_ts = numerator / denominator

        f_b_over_ts_RB = (
            (np.exp(alpha_b) + f_gal / f_TS * beta * (np.exp(alpha_b) - 1.0))
        ) / denominator

        print(f"Debug: f_b_over_ts_RB={f_b_over_ts_RB.decompose().value:.3e}")

    else:  # Extreme case to avoid overflow

        f_b_over_ts = 1 + (1 - beta) / beta * np.exp(alpha - alpha_b)
        f_b_over_ts_RB = 1 / beta

    # f_out(r,p) / f_TS
    f_out_over_ts = f_b_over_ts_RB * (R_b / (r[r_ISM] * u.pc)) + f_gal / f_TS * (
        1.0 - R_TS / (r[r_ISM] * u.pc)
    )

    # Concatenate the profiles
    f_b = f_b_over_ts
    f_out = f_out_over_ts

    # TEMPORARY FIX
    f_w = np.zeros(masks["r_wind"].sum())

    return np.concatenate([f_w, f_b, f_out])


def get_velocity_profile(r, v_w, R_TS, R_b, masks):
    """
    Calculate velocity profile for Giovanni model.

    Parameters:
    -----------
    r : array-like
        Radial coordinates (pc)
    v_w : astropy Quantity
        Wind velocity
    R_TS : astropy Quantity
        Termination shock radius
    R_b : astropy Quantity
        Bubble radius

    Returns:
    --------
    np.array
        Velocity profile in pc/Myr
    """
    r_wind = masks["r_wind"]
    r_bubble = masks["r_bubble"]
    r_ISM = masks["r_ISM"]

    v_field = np.zeros_like(r)
    v_field[r_wind] = v_w.to("pc/Myr").value
    v_field[r_bubble] = (
        v_w.to("pc/Myr").value / 4 * (R_TS.to("pc").value / r[r_bubble]) ** 2
    )
    # v_field[r_ISM] = 0  # Already zero

    return v_field


def get_diffusion_profile(
    r,
    v_p,
    r_L,
    r_Inj,
    R_b,
    D_ISM=3e28 * u.cm**2 / u.s,
    diffusion_model="kolmogorov",
    masks=None,
):
    """
    Calculate diffusion coefficient profile.

    Parameters:
    -----------
    r : array-like
        Radial coordinates (pc)
    v_p : astropy Quantity
        Particle velocity
    r_L : array-like
        Larmor radius values
    r_Inj : astropy Quantity
        Injection radius
    R_b : astropy Quantity
        Bubble radius
    D_ISM : astropy Quantity, optional
        Diffusion coefficient in ISM
    diffusion_model : str, optional
        Diffusion model ('kolmogorov', 'kraichnan' or 'bohm')

    Returns:
    --------
    astropy Quantity array
        Diffusion coefficient profile
    """
    r_ISM = masks["r_ISM"]

    match diffusion_model.lower():
        case "bohm":
            # Bohm-like diffusion inside bubble
            D_values = 1 / 3 * v_p * r_L
        case "kraichnan":
            # Kraichnan-like diffusion inside bubble
            D_values = 1 / 3 * v_p * r_L ** (1 / 2) * r_Inj ** (1 / 2)
        case "kolmogorov":
            # Kolmogorov-like diffusion inside bubble
            D_values = 1 / 3 * v_p * r_L ** (1 / 3) * r_Inj ** (2 / 3)
        case _:
            raise ValueError(
                "Invalid diffusion model. Choose 'kolmogorov', 'kraichnan' or 'bohm'."
            )

    # Constant diffusion in ISM
    D_values[r_ISM] = D_ISM

    return D_values


def get_magnetic_field_profile(r, eta_B, M_dot, v_w, R_TS, R_b, masks):
    """
    Calculate magnetic field profile.

    Parameters:
    -----------
    r : array-like
        Radial coordinates (pc)
    eta_B : float
        Magnetic field efficiency
    M_dot : astropy Quantity
        Mass loss rate
    v_w : astropy Quantity
        Wind velocity
    R_TS : astropy Quantity
        Termination shock radius
    R_b : astropy Quantity
        Bubble radius

    Returns:
    --------
    astropy Quantity array
        Magnetic field profile in Gauss
    """
    r_wind = masks["r_wind"]
    r_bubble = masks["r_bubble"]
    r_ISM = masks["r_ISM"]

    delta_B = np.zeros_like(r) * u.G

    # Wind region
    delta_B[r_wind] = (
        1
        / ((r[r_wind] * u.pc).to("cm").value)
        * np.sqrt(0.5 * eta_B * M_dot.to("g/s").value * v_w.to("cm/s").value)
    ) * u.G

    # Handle r=0 case
    if len(r) > 1:
        delta_B[0] = (
            1
            / ((r[1] * u.pc).to("cm").value)
            * np.sqrt(0.5 * eta_B * M_dot.to("g/s").value * v_w.to("cm/s").value)
        ) * u.G

    # Bubble region
    delta_B[r_bubble] = (
        np.sqrt(11)
        / R_TS.to("cm").value
        * np.sqrt(0.5 * eta_B * M_dot.to("g/s").value * v_w.to("cm/s").value)
    ) * u.G

    # ISM region (zero by default)

    return delta_B


def get_larmor_radius(p, delta_B):
    """
    Calculate Larmor radius profile.

    Parameters:
    -----------
    p : astropy Quantity
        Particle momentum
    delta_B : astropy Quantity array
        Magnetic field profile

    Returns:
    --------
    astropy Quantity array
        Larmor radius profile
    """
    r_L = np.zeros_like(delta_B.value) * u.cm

    # Only calculate where B > 0
    mask = delta_B.value > 0
    r_L[mask] = (p / (const.e.si * delta_B[mask])).to("pc")

    return r_L


def get_source_term(r, R_TS, eta_inj, rho_w, v_w, p, Q_amplitude=1000):
    """
    Calculate source term for cosmic ray injection.

    Parameters:
    -----------
    r : array-like
        Radial coordinates (pc)
    R_TS : astropy Quantity
        Termination shock radius
    eta_inj : float
        Injection efficiency
    rho_w : astropy Quantity
        Wind density
    v_w : astropy Quantity
        Wind velocity
    p : astropy Quantity
        Particle momentum
    Q_amplitude : float, optional
        Source amplitude (simplified)

    Returns:
    --------
    np.array
        Source term
    """
    Q = np.zeros_like(r)

    # Injection at termination shock
    injection_mask = (r >= 0.99 * R_TS.to("pc").value) & (
        r <= 1.01 * R_TS.to("pc").value
    )

    # Simplified source term
    Q[injection_mask] = Q_amplitude

    return Q


def calculate_giovanni_parameters(L_wind, M_dot, rho_0, t_b):
    """
    Calculate key Giovanni model parameters.

    Parameters:
    -----------
    L_wind : astropy Quantity
        Wind luminosity
    M_dot : astropy Quantity
        Mass loss rate
    rho_0 : astropy Quantity
        Ambient density
    t_b : astropy Quantity
        Bubble age

    Returns:
    --------
    dict
        Dictionary with R_TS, R_b, v_w, rho_w
    """
    # Wind velocity
    v_w = np.sqrt(2 * L_wind / M_dot)

    # Bubble radius
    R_b = (
        (250 / (308 * math.pi)) ** (1 / 5)
        * L_wind ** (1 / 5)
        * rho_0 ** (-1 / 5)
        * t_b ** (3 / 5)
    )

    # Termination shock radius
    R_TS = (
        np.sqrt((3850 * math.pi) ** (2 / 5) / (28 * math.pi) * M_dot * v_w)
        * L_wind ** (-1 / 5)
        * rho_0 ** (-3 / 10)
        * t_b ** (2 / 5)
    )

    # Wind density at termination shock
    rho_w = 3 * M_dot / (4 * math.pi * (R_TS**2) * v_w)

    return {"R_TS": R_TS, "R_b": R_b, "v_w": v_w, "rho_w": rho_w}


def get_cosmic_ray_sea():
    """
    Calculate cosmic ray sea background values.

    Returns:
    --------
    dict
        Dictionary with f_sea and related values
    """
    Mp = const.m_p
    c = const.c
    E = lambda Ek: Ek + Mp * c**2
    p = lambda Ek: np.sqrt((E(Ek) ** 2 - (Mp * c**2) ** 2) / c**2)
    LorenzBeta = lambda Ek: np.sqrt(1 - (Mp * c**2 / E(Ek)) ** 2)

    # Cosmic ray sea
    Kcr = (
        0.4544
        * 10**-4
        / (45**2 * const.c.value)
        * (u.cm * u.GeV / const.c.to("cm s-1")) ** -3
        / 100
    )
    f_sea = (
        4
        * np.pi
        / c.to("cm s-1")
        * (1 * u.GeV / c.to("cm s-1")) ** 2
        * Kcr
        * (LorenzBeta(1 * u.GeV) ** -1)
        * (1 * u.GeV / (45 * u.GeV)) ** -4.85
        * (1 + (1 * u.GeV / (336 * u.GeV)) ** 5.54) ** 0.024
    )

    C = 1.882 * 10**-9 * (u.eV**-1 * u.cm**-2 * u.s**-1 * u.sr**-1)
    Jsea_Voy = (
        C
        * (1 * u.GeV / (1 * u.MeV)) ** 0.129
        * (1 + 1 * u.GeV / (624.5 * u.MeV)) ** -2.829
    ).to("eV-1 cm-2 s-1 sr-1")
    fsea_Voy = Jsea_Voy * (4 * np.pi * u.sr) / c
    f_sea_Mix = f_sea * (1 * u.GeV > 90 * u.GeV) + fsea_Voy * (1 * u.GeV < 90 * u.GeV)

    return {"f_sea": f_sea, "fsea_Voy": fsea_Voy, "f_sea_Mix": f_sea_Mix}


def get_particle_velocity(E_k):
    """
    Calculate particle velocity from kinetic energy using relativistic formula.

    Parameters:
    -----------
    E_k : astropy Quantity
        Particle kinetic energy

    Returns:
    --------
    astropy Quantity
        Particle velocity
    """
    p = (
        np.sqrt((E_k + const.m_p * const.c**2) ** 2 - (const.m_p * const.c**2) ** 2)
        / const.c
    )
    v_p = p * const.c**2 / np.sqrt((p * const.c) ** 2 + (const.m_p * const.c**2) ** 2)
    return v_p, p


def create_giovanni_setup(
    r_0=0.0 * u.pc,
    r_end=500.0 * u.pc,
    num_points=4000,
    L_wind=1e38 * u.erg / u.s,
    M_dot=1e-4 * const.M_sun / u.yr,
    rho_0=const.m_p / u.cm**3,
    t_b=1 * u.Myr,
    eta_B=0.1,
    eta_inj=0.1,
    E_k=1 * u.GeV,
    diffusion_model="kolmogorov",
    include_shocks=False,
):
    """
    Create a complete Giovanni model setup with all profiles.

    Parameters:
    -----------
    r_0 : astropy Quantity, optional
        Inner radius
    r_end : astropy Quantity, optional
        Outer radius
    num_points : int, optional
        Number of grid points
    L_wind : astropy Quantity, optional
        Wind luminosity
    M_dot : astropy Quantity, optional
        Mass loss rate
    rho_0 : astropy Quantity, optional
        Ambient density
    t_b : astropy Quantity, optional
        Bubble age
    eta_B : float, optional
        Magnetic field efficiency
    eta_inj : float, optional
        Injection efficiency
    E_p : astropy Quantity, optional
        Particle energy
    diffusion_model : str, optional
        Diffusion model ('kolmogorov', 'kraichnan' or 'bohm')

    Returns:
    --------
    dict
        Dictionary with all profiles and parameters
    """

    # Spatial grid
    r = np.linspace(r_0.value, r_end.value, num_points)
    if include_shocks:
        r = np.linspace(r_0.value, r_end.value, num_points - 2)

    # Calculate basic parameters
    params = calculate_giovanni_parameters(L_wind, M_dot, rho_0, t_b)
    R_TS = params["R_TS"]
    R_b = params["R_b"]
    v_w = params["v_w"]
    rho_w = params["rho_w"]

    if include_shocks:
        # Include shock positions in grid
        r = np.sort(np.concatenate(([R_TS.to("pc").value, R_b.to("pc").value], r)))

    # Region masks
    r_wind = r < R_TS.to("pc").value
    r_bubble = (r >= R_TS.to("pc").value) & (r <= R_b.to("pc").value)
    r_ISM = r > R_b.to("pc").value

    masks = {"r_wind": r_wind, "r_bubble": r_bubble, "r_ISM": r_ISM}

    # Particle velocity
    v_p, p = get_particle_velocity(E_k)

    # Magnetic field profile
    delta_B = get_magnetic_field_profile(r, eta_B, M_dot, v_w, R_TS, R_b, masks=masks)

    # Larmor radius
    r_L = get_larmor_radius(p, delta_B)

    # Diffusion coefficient
    r_Inj = 1.0 * u.pc
    D_ISM = (3 * 10**28 * (E_k / (1 * u.GeV)) ** (1 / 3)) * u.cm**2 / u.s
    D_values = get_diffusion_profile(
        r,
        v_p,
        r_L,
        r_Inj,
        R_b,
        diffusion_model=diffusion_model,
        masks=masks,
        D_ISM=D_ISM,
    )

    # Velocity profile
    v_field = get_velocity_profile(r, v_w, R_TS, R_b, masks=masks)

    # Source term
    Q = get_source_term(r, R_TS, eta_inj, rho_w, v_w, p)

    return {
        "r": r,
        "r_0": r_0,
        "r_end": r_end,
        "num_points": num_points,
        "L_wind": L_wind,
        "M_dot": M_dot,
        "rho_0": rho_0,
        "t_b": t_b,
        "eta_B": eta_B,
        "eta_inj": eta_inj,
        "R_TS": R_TS,
        "R_b": R_b,
        "v_w": v_w,
        "rho_w": rho_w,
        "v_p": v_p,
        "delta_B": delta_B,
        "r_L": r_L,
        "D_values": D_values,
        "v_field": v_field,
        "Q": Q,
        "E_k": E_k,
        "p": p,
        "params": params,
        "masks": masks,
    }
