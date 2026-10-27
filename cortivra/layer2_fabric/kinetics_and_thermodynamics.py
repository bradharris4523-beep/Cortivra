"""
Chemical reaction networks, statistical thermodynamics, and cooperative binding
(`cortivra.layer2_fabric.kinetics_and_thermodynamics`).

Provides:
    - `StoichiometricReactionBrick`: Port-Hamiltonian chemical reaction network satisfying
      detailed balance (`k_r^+ * prod(c_eq^{nu^+}) == k_r^- * prod(c_eq^{nu^-})`), mass
      conservation (`m^T S = 0`), charge electroneutrality (`z^T S = 0`), and non-negative
      entropy production via a symmetric positive semi-definite Onsager conductance matrix `R(c)`.
    - `CooperativeBindingBrick`: Cooperative Hill / Langmuir / Michaelis-Menten receptor-ligand
      binding and allosteric state modulation brick.
"""

from __future__ import annotations

from typing import Sequence

import torch

from cortivra.layer1_axioms.hamiltonian import (
    LawBrick,
    ParamSpec,
    PowerPortSpec,
    StateVarSpec,
)
from cortivra.layer1_axioms.units import (
    DIMENSIONLESS,
    J,
    J_per_mol,
    Unit,
    mM,
    mol_per_s,
    s,
)
from cortivra.layer2_fabric.primitives import (
    cooperative_hill_saturation,
    linear_relaxation_rate,
    reversible_mass_action_rates,
)


class StoichiometricReactionBrick(LawBrick):
    """
    Reversible chemical reaction network with stoichiometric matrices `nu_reactants` (`[S, R]`)
    and `nu_products` (`[S, R]`), net stoichiometry `S = nu_products - nu_reactants`.

    Port-Hamiltonian Formulation (Linearized Onsager / Detailed-Balance Free Energy):
        Around thermodynamic equilibrium concentrations `c_eq` (`[S]`) where
        `v_eq,r = k_r^+ * prod_i (c_eq,i)^{nu^+_{ir}} = k_r^- * prod_i (c_eq,i)^{nu^-_{ir}}`:
            - Free energy storage: `H(c) = 0.5 * sum_i (c_i - c_eq,i)^2 / c_eq,i`
            - Chemical effort: `e_i = dH/dc_i = (c_i - c_eq,i) / c_eq,i`
            - Onsager dissipation matrix: `R(c) = S * diag(r_conductance) * S^T` (`R = R^T >= 0`)
        Also provides `nonlinear_mass_action_flow(state, params)` for exact nonlinear mass-action
        integration far from equilibrium.
    """

    def __init__(
        self,
        name: str = "stoichiometric_reaction_network",
        species_names: Sequence[str] = ("A", "B", "AB"),
        reactant_stoich: torch.Tensor | None = None,
        product_stoich: torch.Tensor | None = None,
        molar_masses_g_mol: Sequence[float] = (100.0, 150.0, 250.0),
        valences: Sequence[int] = (1, -1, 0),
        equilibrium_concentrations: Sequence[float] = (1.0, 1.0, 2.0),
        default_conductances: Sequence[float] | None = None,
        concentration_unit: Unit = mM,
        time_unit: Unit = s,
        energy_unit: Unit = J,
    ) -> None:
        num_species = len(species_names)
        if reactant_stoich is None or product_stoich is None:
            # Default reversible bimolecular association: A + B <-> AB
            reactant_stoich = torch.tensor([[1.0], [1.0], [0.0]], dtype=torch.float64)
            product_stoich = torch.tensor([[0.0], [0.0], [1.0]], dtype=torch.float64)

        if reactant_stoich.shape != product_stoich.shape or reactant_stoich.shape[0] != num_species:
            raise ValueError(
                f"Stoichiometric matrix shape mismatch: reactant_stoich {tuple(reactant_stoich.shape)}, "
                f"product_stoich {tuple(product_stoich.shape)}, num_species={num_species}."
            )

        self.reactant_stoich = reactant_stoich.to(torch.float64)
        self.product_stoich = product_stoich.to(torch.float64)
        self.net_stoich = self.product_stoich - self.reactant_stoich  # [S, R]
        self.num_reactions = int(self.net_stoich.shape[1])
        self.equilibrium_conc = torch.tensor(equilibrium_concentrations, dtype=torch.float64)
        self.molar_masses_tensor = torch.tensor(molar_masses_g_mol, dtype=torch.float64)
        self.valences_tensor = torch.tensor(valences, dtype=torch.float64)

        state_specs = [
            StateVarSpec(
                name=sp_name,
                unit=concentration_unit,
                default_value=float(equilibrium_concentrations[idx]) * 1.2,
                hard_min=1e-12,
                hard_max=1e6,
                homeostatic_min=1e-9,
                homeostatic_max=1e4,
                valence=int(valences[idx]),
                molar_mass_g_mol=float(molar_masses_g_mol[idx]),
                description=f"Concentration of species {sp_name}",
            )
            for idx, sp_name in enumerate(species_names)
        ]

        conds = (
            list(default_conductances)
            if default_conductances is not None
            else [1.0] * self.num_reactions
        )
        param_specs = [
            ParamSpec(f"k_onsager_{r}", DIMENSIONLESS, default_value=float(conds[r]), min_value=0.0)
            for r in range(self.num_reactions)
        ]

        port_specs = [
            PowerPortSpec(
                name="chemical_flux_port",
                effort_unit=J_per_mol,
                flow_unit=mol_per_s,
                domain="chemistry",
                state_var_name=species_names[0],
            )
        ]

        super().__init__(
            name=name,
            domain="chemistry",
            state_specs=state_specs,
            param_specs=param_specs,
            port_specs=port_specs,
            time_unit=time_unit,
            energy_unit=energy_unit,
        )

    def hamiltonian_energy(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        c_eq = self.equilibrium_conc.to(dtype=state.dtype, device=state.device).unsqueeze(0)
        return 0.5 * torch.sum(((state - c_eq) ** 2) / c_eq, dim=-1)

    def effort(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        c_eq = self.equilibrium_conc.to(dtype=state.dtype, device=state.device).unsqueeze(0)
        return (state - c_eq) / c_eq

    def dissipation_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        # R = S * diag(k_onsager) * S^T, guaranteed symmetric positive semi-definite
        s_mat = self.net_stoich.to(dtype=state.dtype, device=state.device)  # [S, R]
        k_rates = torch.clamp(params, min=0.0)  # [B, R]
        # [B, S, R] * [B, R, S] -> [B, S, S]
        s_batch = s_mat.unsqueeze(0).expand(state.shape[0], -1, -1)
        weighted_s = s_batch * k_rates.unsqueeze(1)
        return torch.bmm(weighted_s, s_batch.transpose(1, 2))

    def nonlinear_mass_action_flow(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Compute exact nonlinear mass-action species derivatives `dc/dt = S * (k^+ c^{nu^+} - k^- c^{nu^-})`
        where `k^+` and `k^-` are derived from `k_onsager` and `c_eq` to satisfy microscopic detailed balance.
        """
        dtype, device = state.dtype, state.device
        nu_f = self.reactant_stoich.to(dtype=dtype, device=device)
        nu_r = self.product_stoich.to(dtype=dtype, device=device)
        c_eq = self.equilibrium_conc.to(dtype=dtype, device=device).unsqueeze(0)
        k_onsager = torch.clamp(params, min=0.0)

        # Detailed balance requires k_f * prod(c_eq^{nu_f}) == k_r * prod(c_eq^{nu_r}) == k_onsager
        log_ceq = torch.log(torch.clamp(c_eq, min=1e-15))
        eq_fwd_act = torch.exp(torch.matmul(log_ceq, nu_f))
        eq_rev_act = torch.exp(torch.matmul(log_ceq, nu_r))

        k_f = k_onsager / torch.clamp(eq_fwd_act, min=1e-15)
        k_r = k_onsager / torch.clamp(eq_rev_act, min=1e-15)
        dc_dt, _ = reversible_mass_action_rates(state, nu_f, nu_r, k_f, k_r)
        return dc_dt

    def nonlinear_free_energy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute exact convex relative-entropy (Kullback-Leibler / Boltzmann-Gibbs) Lyapunov
        free energy for nonlinear mass-action kinetics (`[Batch]`):
            `G(c) = sum_i [ c_i * (ln(c_i / c_eq,i) - 1) + c_eq,i ] >= 0`
        Guaranteed `G(c) == 0` if and only if `c == c_eq`, and `dG/dt = grad_G(c) . dc/dt <= 0`
        along any detailed-balanced nonlinear mass-action trajectory.
        """
        c_safe = torch.clamp(state, min=1e-15)
        c_eq = self.equilibrium_conc.to(dtype=state.dtype, device=state.device).unsqueeze(0)
        return torch.sum(c_safe * (torch.log(c_safe / c_eq) - 1.0) + c_eq, dim=-1)

    def nonlinear_chemical_affinity_dissipation(
        self, state: torch.Tensor, params: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute exact nonlinear entropy production rate `Sigma(c) = -grad_G(c) . dc/dt >= 0` (`[Batch]`):
            `Sigma(c) = sum_r (v_r^+ - v_r^-) * ln(v_r^+ / v_r^-) >= 0`
        """
        c_safe = torch.clamp(state, min=1e-15)
        c_eq = self.equilibrium_conc.to(dtype=state.dtype, device=state.device).unsqueeze(0)
        grad_g = torch.log(c_safe / c_eq)  # [B, S]
        dc_dt = self.nonlinear_mass_action_flow(c_safe, params)  # [B, S]
        return -torch.sum(grad_g * dc_dt, dim=-1)


class CooperativeBindingBrick(LawBrick):
    """
    Port-Hamiltonian cooperative Hill receptor-ligand binding and allosteric modulation brick.

    State vector (`[Batch, 1]`):
        - `theta_bound`: Fractional receptor occupancy (`[0, 1]`).

    Parameters (`[Batch, 4]`):
        - `ligand_conc`: Ligand/modulator concentration (`mM`, `>= 0`).
        - `kd`: Dissociation constant `K_d` (`mM`, `> 0`).
        - `hill_n`: Hill cooperativity coefficient `n` (`> 0`).
        - `tau_bind`: Binding relaxation time constant (`s` or `ms`, `> 0`).
    """

    def __init__(
        self,
        name: str = "cooperative_hill_binding",
        ligand_conc_default: float = 0.05,
        kd_default: float = 0.05,
        hill_n_default: float = 1.5,
        tau_bind_default: float = 1.0,
        conc_unit: Unit = mM,
        time_unit: Unit = s,
        energy_unit: Unit = J,
    ) -> None:
        theta0 = float(
            cooperative_hill_saturation(
                torch.tensor(ligand_conc_default, dtype=torch.float64),
                kd_default,
                hill_n_default,
            ).item()
        )
        state_specs = [
            StateVarSpec(
                name="theta_bound",
                unit=DIMENSIONLESS,
                default_value=theta0,
                hard_min=0.0,
                hard_max=1.0,
                homeostatic_min=0.0,
                homeostatic_max=1.0,
                description="Fractional cooperative receptor occupancy",
            )
        ]
        param_specs = [
            ParamSpec("ligand_conc", conc_unit, default_value=ligand_conc_default, min_value=0.0),
            ParamSpec("kd", conc_unit, default_value=kd_default, min_value=1e-12),
            ParamSpec("hill_n", DIMENSIONLESS, default_value=hill_n_default, min_value=0.1, max_value=10.0),
            ParamSpec("tau_bind", time_unit, default_value=tau_bind_default, min_value=1e-6),
        ]
        super().__init__(
            name=name,
            domain="chemistry",
            state_specs=state_specs,
            param_specs=param_specs,
            port_specs=(),
            time_unit=time_unit,
            energy_unit=energy_unit,
        )

    def target_occupancy(self, params: torch.Tensor) -> torch.Tensor:
        ligand = params[:, 0]
        kd = params[:, 1]
        hill_n = params[:, 2]
        return cooperative_hill_saturation(ligand, kd, hill_n)

    def hamiltonian_energy(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        theta = state[:, 0]
        theta_inf = self.target_occupancy(params)
        return 0.5 * ((theta - theta_inf) ** 2)

    def effort(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        theta = state[:, 0]
        theta_inf = self.target_occupancy(params)
        return (theta - theta_inf).unsqueeze(-1)

    def dissipation_matrix(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        tau = torch.clamp(params[:, 3], min=1e-6)
        r_mat = (1.0 / tau).view(-1, 1, 1)
        return r_mat

    def dissipative_flow(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        theta = state[:, 0]
        theta_inf = self.target_occupancy(params)
        tau = torch.clamp(params[:, 3], min=1e-6)
        return (-linear_relaxation_rate(theta, theta_inf, tau)).unsqueeze(-1)
