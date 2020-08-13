"""
Example setup and run script for the 3d stimulation and long-term cooling example.

Main differences from the example 1 setup are related to geometry, BCs, wells and
gravity.
"""
import scipy.sparse.linalg as spla
import numpy as np
import porepy as pp
import logging
import time
from typing import Tuple, Dict
from porepy.models.contact_mechanics_biot_model import ContactMechanicsBiot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Model(ContactMechanicsBiot):
    """
    This class provides the parameter specification differing from examples 1 and 2.
    """

    def __init__(self, params: Dict):
        super().__init__(params)

        # Set additional case specific fields
        self.scalar_scale = 1e7
        self.length_scale = 15

        self.file_name = self.params["file_name"]
        self.folder_name = self.params["folder_name"]

        self.export_fields = [
            "u_exp",
            "p_exp",
            "p_minus_ph",
            "traction_exp",
            "aperture_exp",
            "u_global",
            "cell_centers",
            "well",
            "u_exp_0",
            "aperture_0",
        ]
        # Initial aperture, a_0
        self.initial_aperture = 1e-3 / self.length_scale

        # Dilation angle
        self._dilation_angle = np.radians(5.0)
        self.params = params

        self.mesh_args = params.get("mesh_args", None)

    def fractures(self):
        """
        Define the two fractures.
        The first fracture is the one where injection takes place.
        """
        n_points = 4
        # Size
        s = 12
        # Major axis rotation
        major = np.pi / 4
        # Dip
        dip_1, dip_2 = np.pi / 4, np.pi / 4
        # Strike:
        # The values below imply dip about the y and x axis, respectively
        strike, strike_2 = np.pi / 2, 0
        f_1 = pp.EllipticFracture(
            np.array([-10, 0, 0]), s, s, major, strike, dip_1, num_points=n_points
        )

        f_2 = pp.EllipticFracture(
            np.array([10, 0, 0]), s, s, major, strike_2, dip_2, num_points=n_points
        )
        self.fracs = [f_1, f_2]

    def create_grid(self):
        """
        Method that creates the GridBucket of a 3D domain with the two fractures
        defined by self.fractures(). 
        The grid bucket is represents the mixed-dimensional grid.
        """
        self.fractures()

        # Define the domain
        size = 80
        self.box = {
            "xmin": -size,
            "xmax": size,
            "ymin": -size,
            "ymax": size,
            "zmin": -size,
            "zmax": size,
        }
        # Make a fracture network
        self.network = pp.FractureNetwork3d(self.fracs, domain=self.box)

        # Generate the mixed-dimensional mesh
        # write_fractures_to_csv(self)
        gb = self.network.mesh(self.mesh_args)

        pp.contact_conditions.set_projections(gb)

        self.gb = gb
        self.Nd = self.gb.dim_max()

        # Tag the wells
        self._tag_well_cells()
        self.n_frac = len(gb.grids_of_dimension(self.Nd - 1))
        self.update_all_apertures(to_iterate=False)
        self.update_all_apertures()

    def set_mechanics_parameters(self):
        """ Mechanical parameters.
        Note that we divide the momentum balance equation by self.scalar_scale. 
        """
        gb = self.gb
        for g, d in gb:
            if g.dim == self.Nd:
                # Rock parameters
                rock = self.rock
                lam = rock.LAMBDA * np.ones(g.num_cells) / self.scalar_scale
                mu = rock.MU * np.ones(g.num_cells) / self.scalar_scale
                C = pp.FourthOrderTensor(mu, lam)

                bc = self.bc_type_mechanics(g)
                bc_values = self.bc_values_mechanics(g)
                sources = self.source_mechanics(g)

                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {
                        "bc": bc,
                        "bc_values": bc_values,
                        "source": sources,
                        "fourth_order_tensor": C,
                        "biot_alpha": self.biot_alpha(g),
                        "time_step": self.time_step,
                    },
                )

            elif g.dim == self.Nd - 1:

                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {
                        "friction_coefficient": 0.5,
                        "contact_mechanics_numerical_parameter": 1e1,
                        "dilation_angle": self._dilation_angle,
                        "time": self.time,
                    },
                )

        for e, d in gb.edges():
            mg = d["mortar_grid"]
            # Parameters for the surface diffusion. Not used as of now.
            pp.initialize_data(
                mg,
                d,
                self.mechanics_parameter_key,
                {"mu": self.rock.MU, "lambda": self.rock.LAMBDA},
            )

    def set_scalar_parameters(self):
        """ Set parameters for the scalar (pressure) equation.
        """
        for g, d in self.gb:

            a = self.aperture(g)
            specific_volumes = self.specific_volumes(g)

            # Define boundary conditions for flow
            bc = self.bc_type_scalar(g)
            # Set boundary condition values
            bc_values = self.bc_values_scalar(g)

            biot_coefficient = self.biot_alpha(g)
            compressibility = self.fluid.COMPRESSIBILITY

            mass_weight = compressibility * self.porosity(g)
            if g.dim == self.Nd:
                mass_weight += (
                    biot_coefficient - self.porosity(g)
                ) / self.rock.BULK_MODULUS

            mass_weight *= self.scalar_scale * specific_volumes
            g_rho = (
                -pp.GRAVITY_ACCELERATION
                * self.density(g)
                / self.scalar_scale
                * self.length_scale
            )
            gravity = np.zeros((self.Nd, g.num_cells))
            gravity[self.Nd - 1, :] = g_rho
            pp.initialize_data(
                g,
                d,
                self.scalar_parameter_key,
                {
                    "bc": bc,
                    "bc_values": bc_values,
                    "mass_weight": mass_weight,
                    "biot_alpha": biot_coefficient,
                    "time_step": self.time_step,
                    "ambient_dimension": self.Nd,
                    "source": self.source_scalar(g),
                    #            + self.dVdt_source(g, d, self.scalar_parameter_key),
                    "vector_source": gravity.ravel("F"),
                },
            )
        for e, data_edge in self.gb.edges():
            g_l, g_h = self.gb.nodes_of_edge(e)
            params_l = self.gb.node_props(g_l)[pp.PARAMETERS][self.scalar_parameter_key]
            mg = data_edge["mortar_grid"]
            a = mg.slave_to_mortar_avg() * self.aperture(g_l)

            grho = (
                mg.slave_to_mortar_avg()
                * params_l["vector_source"][self.Nd - 1 :: self.Nd]
            )

            gravity = np.zeros((self.Nd, mg.num_cells))
            gravity[self.Nd - 1, :] = grho * a / 2

            data_edge = pp.initialize_data(
                e,
                data_edge,
                self.scalar_parameter_key,
                {"vector_source": gravity.ravel("F")},
            )
        self.set_permeability()

    def aperture(self, g, from_iterate=True) -> np.ndarray:
        """
        Obtain the aperture of a subdomain. See update_all_apertures.
        """
        if from_iterate:
            return self.gb.node_props(g)[pp.STATE][pp.ITERATE]["aperture"]
        else:
            return self.gb.node_props(g)[pp.STATE]["aperture"]

    def specific_volumes(self, g, from_iterate=True) -> np.ndarray:
        """
        Obtain the specific volume of a subdomain. See update_all_apertures.
        """
        if from_iterate:
            return self.gb.node_props(g)[pp.STATE][pp.ITERATE]["specific_volume"]
        else:
            return self.gb.node_props(g)[pp.STATE]["specific_volume"]

    def update_all_apertures(self, to_iterate=True):
        """
        To better control the aperture computation, it is done for the entire gb by a
        single function call. This also allows us to ensure the fracture apertures
        are updated before the intersection apertures are inherited.
        """
        gb = self.gb
        for g, d in gb:

            apertures = np.ones(g.num_cells)
            if g.dim == (self.Nd - 1):
                # Initial aperture

                apertures *= self.initial_aperture
                # Reconstruct the displacement solution on the fracture
                g_h = gb.node_neighbors(g)[0]
                data_edge = gb.edge_props((g, g_h))
                if pp.STATE in data_edge:
                    u_mortar_local = self.reconstruct_local_displacement_jump(
                        data_edge, from_iterate=to_iterate
                    )
                    apertures -= u_mortar_local[-1].clip(max=0)
            if to_iterate:
                pp.set_iterate(
                    d,
                    {"aperture": apertures.copy(), "specific_volume": apertures.copy()},
                )
            else:
                state = {
                    "aperture": apertures.copy(),
                    "specific_volume": apertures.copy(),
                }
                pp.set_state(d, state)

        for g, d in gb:
            parent_apertures = []
            num_parent = []
            if g.dim < (self.Nd - 1):
                for edges in gb.edges_of_node(g):
                    e = edges[0]
                    g_h = e[0]

                    if g_h == g:
                        g_h = e[1]

                    if g_h.dim == (self.Nd - 1):
                        d_h = gb.node_props(g_h)
                        if to_iterate:
                            a_h = d_h[pp.STATE][pp.ITERATE]["aperture"]
                        else:
                            a_h = d_h[pp.STATE]["aperture"]
                        a_h_face = np.abs(g_h.cell_faces) * a_h
                        mg = gb.edge_props(e)["mortar_grid"]
                        # Assumes g_h is master
                        a_l = (
                            mg.mortar_to_slave_avg()
                            * mg.master_to_mortar_avg()
                            * a_h_face
                        )
                        parent_apertures.append(a_l)
                        num_parent.append(np.sum(mg.mortar_to_slave_int().A, axis=1))
                    else:
                        raise ValueError("Intersection points not implemented in 3d")
                parent_apertures = np.array(parent_apertures)
                num_parents = np.sum(np.array(num_parent), axis=0)

                apertures = np.sum(parent_apertures, axis=0) / num_parents

                specific_volumes = np.power(apertures, self.Nd - g.dim)
                if to_iterate:
                    pp.set_iterate(
                        d,
                        {
                            "aperture": apertures.copy(),
                            "specific_volume": specific_volumes.copy(),
                        },
                    )
                else:
                    state = {
                        "aperture": apertures.copy(),
                        "specific_volume": specific_volumes.copy(),
                    }
                    pp.set_state(d, state)

        return apertures

    def set_permeability(self):
        """
        Cubic law in fractures, rock permeability in the matrix.
        If "blocking_perm" is present in self.params, this value is used for
        Fracture 2.
        """
        # Viscosity has units of Pa s, and is consequently divided by the scalar scale.
        viscosity = self.fluid.dynamic_viscosity() / self.scalar_scale
        gb = self.gb
        key = self.scalar_parameter_key
        from_iterate = True
        blocking_perm = self.params.get("blocking_perm", None)
        for g, d in gb:
            if g.dim < self.Nd:
                # Set fracture permeability
                specific_volumes = self.specific_volumes(g, from_iterate)

                if d["node_number"] == 1 or blocking_perm is None:
                    # Use cubic law in fractures. First compute the unscaled
                    # permeability
                    apertures = self.aperture(g, from_iterate=from_iterate)
                    apertures_unscaled = apertures * self.length_scale
                    k = np.power(apertures_unscaled, 2) / 12 / viscosity
                else:
                    # Blocking and intersection
                    k = blocking_perm
                d[pp.PARAMETERS][key]["perm_nu"] = k
                # Multiply with the cross-sectional area
                k = k * specific_volumes
                # Divide by fluid viscosity and scale back
                kxx = k / self.length_scale ** 2

            else:
                # Use the rock permeability in the matrix
                kxx = (
                    self.rock.PERMEABILITY
                    / viscosity
                    * np.ones(g.num_cells)
                    / self.length_scale ** 2
                )
            K = pp.SecondOrderTensor(kxx)
            d[pp.PARAMETERS][key]["second_order_tensor"] = K

        # Normal permeability inherited from the neighboring fracture g_l
        for e, d in gb.edges():
            mg = d["mortar_grid"]
            g_l, _ = gb.nodes_of_edge(e)
            data_l = gb.node_props(g_l)
            a = self.aperture(g_l, from_iterate)
            V = self.specific_volumes(g_l, from_iterate)
            # We assume isotropic permeability in the fracture, i.e. the normal
            # permeability equals the tangential one
            k_s = data_l[pp.PARAMETERS][key]["second_order_tensor"].values[0, 0]
            # Division through half the aperture represents taking the (normal) gradient
            kn = mg.slave_to_mortar_int() * np.divide(k_s, a * V / 2)
            pp.initialize_data(mg, d, key, {"normal_diffusivity": kn})

    def biot_alpha(self, g) -> float:
        if g.dim == self.Nd:
            return self.params.get("biot_alpha", 0.7)
        else:
            # Used for the volume change term in the fracture. See DivU
            return 1

    def porosity(self, g) -> float:
        if g.dim == self.Nd:
            return 0.01
        else:
            return 1.0

    def density(self, g, dp=None) -> np.ndarray:
        """ Density computed from current pressure solution
        taken from the previous iterate.
        """
        if dp is None:
            p_0 = self.scalar_scale * self.initial_scalar(g)
            _, p_k, p_n = self._variable_increment(
                g, self.scalar_variable, self.scalar_scale,
            )
            dp = p_k - p_0

        rho_0 = 1e3 * (pp.KILOGRAM / pp.METER ** 3) * np.ones(g.num_cells)
        rho = rho_0 * np.exp(dp * self.fluid.COMPRESSIBILITY)

        return rho

    def faces_to_fix(self, g):
        """
        Identify three boundary faces to fix (u=0). This should allow us to assign
        Neumann "background stress" conditions on the rest of the boundary faces.
        """
        all_bf, *_ = self.domain_boundary_sides(g)
        point = np.array(
            [
                [(self.box["xmin"] + self.box["xmax"]) / 2],
                [(self.box["ymin"] + self.box["ymax"]) / 2],
                [self.box["zmax"]],
            ]
        )
        distances = pp.distances.point_pointset(point, g.face_centers[:, all_bf])
        indexes = np.argsort(distances)
        faces = all_bf[indexes[: self.Nd]]
        return faces

    def _tag_well_cells(self):
        """
        Tag well cells with unitary values, positive for injection cells and negative
        for production cells.
        """
        for g, d in self.gb:
            tags = np.zeros(g.num_cells)
            if g.dim < self.Nd:
                point = np.array(
                    [
                        [(self.box["xmin"] + self.box["xmax"]) / 2],
                        [self.box["ymin"]],
                        [0],
                    ]
                )
                distances = pp.distances.point_pointset(point, g.cell_centers)
                indexes = np.argsort(distances)
                if d["node_number"] == 1:
                    tags[indexes[-1]] = 1  # injection

            g.tags["well_cells"] = tags
            pp.set_state(d, {"well": tags.copy()})

    def source_flow_rates(self) -> Tuple[int, int]:
        """
        The rate is given in l/s = m^3/s e-3. Length scaling also needed to convert from
        the scaled length to m.
        The values returned depend on the simulation phase.
        """
        t = self.time
        tol = 1e-10
        injection, production = 0, 0
        if t > self.phase_limits[1] + tol and t < self.phase_limits[2] + tol:
            injection = 60
            production = 0
        elif t > self.phase_limits[2] + tol:
            injection, production = 0, 0
        w = pp.MILLI * (pp.METER / self.length_scale) ** self.Nd
        return injection * w, production * w

    def bc_type_mechanics(self, g) -> pp.BoundaryConditionVectorial:
        """
        We set Neumann values imitating an anisotropic background stress regime on all
        but three faces, which are fixed to ensure a unique solution.
        """
        all_bf, *_, bottom = self.domain_boundary_sides(g)
        faces = self.faces_to_fix(g)
        # write_fixed_faces_to_csv(g, faces, self)
        bc = pp.BoundaryConditionVectorial(g, faces, "dir")
        frac_face = g.tags["fracture_faces"]
        bc.is_neu[:, frac_face] = False
        bc.is_dir[:, frac_face] = True
        return bc

    def bc_type_scalar(self, g) -> pp.BoundaryCondition:
        """
        We prescribe the pressure value at all external boundaries.
        """
        # Define boundary regions
        all_bf, *_ = self.domain_boundary_sides(g)
        # pdb.set_trace()
        return pp.BoundaryCondition(g, all_bf, "dir")

    def bc_values_mechanics(self, g) -> np.ndarray:
        """
        Lithostatic mechanical BC values.
        """
        bc_values = np.zeros((g.dim, g.num_faces))
        if np.isclose(self.time, self.phase_limits[0]):
            return bc_values.ravel("F")

        # Retrieve the boundaries where values are assigned
        all_bf, east, west, north, south, top, bottom = self.domain_boundary_sides(g)
        A = g.face_areas
        # Domain centred at 1 km below surface

        # Gravity acceleration
        gravity = (
            pp.GRAVITY_ACCELERATION
            * self.rock.DENSITY
            * self._depth(g.face_centers)
            / self.scalar_scale
        )

        we, sn, bt = 1.3, 0.6, 1
        bc_values[0, west] = (we * gravity[west]) * A[west]
        bc_values[0, east] = -(we * gravity[east]) * A[east]
        bc_values[1, south] = (sn * gravity[south]) * A[south]
        bc_values[1, north] = -(sn * gravity[north]) * A[north]
        if self.Nd > 2:
            bc_values[2, bottom] = (bt * gravity[bottom]) * A[bottom]
            bc_values[2, top] = -(bt * gravity[top]) * A[top]

        faces = self.faces_to_fix(g)
        bc_values[:, faces] = 0

        return bc_values.ravel("F")

    def bc_values_scalar(self, g) -> np.ndarray:
        """
        Hydrostatic pressure BC values.
        """
        # Retrieve the boundaries where values are assigned
        all_bf, *_ = self.domain_boundary_sides(g)
        bc_values = np.zeros(g.num_faces)
        depth = self._depth(g.face_centers[:, all_bf])
        bc_values[all_bf] = self.fluid.hydrostatic_pressure(depth) / self.scalar_scale
        return bc_values

    def source_mechanics(self, g) -> np.ndarray:
        """
        Gravity term.
        """
        values = np.zeros((self.Nd, g.num_cells))
        values[2] = (
            pp.GRAVITY_ACCELERATION
            * self.rock.DENSITY
            * g.cell_volumes
            * self.length_scale
            / self.scalar_scale
        )
        return values.ravel("F")

    def source_scalar(self, g) -> np.ndarray:
        """
        Source term for the scalar equation.
        For slightly compressible flow in the present formulation, this has units of m^3.
        
        Sources are handled by ScalarSource discretizations.
        The implicit scheme yields multiplication of the rhs by dt, but
        this is not incorporated in ScalarSource, hence we do it here.
        """
        injection, production = self.source_flow_rates()
        wells = (
            injection
            * g.tags["well_cells"]
            * self.time_step
            * g.tags["well_cells"].clip(min=0)
        )
        wells += (
            production
            * g.tags["well_cells"]
            * self.time_step
            * g.tags["well_cells"].clip(max=0)
        )

        return wells

    def _set_time_parameters(self):
        """
        Specify time parameters.
        """
        # For the initialization run, we use the following
        # start time
        self.time = -5e2 * pp.YEAR
        # and time step
        self.time_step = -self.time / 1

        # We use
        t_1 = 5 * pp.DAY
        self.end_time = t_1 + 2 * pp.DAY
        self.max_time_step = self.end_time
        self.phase_limits = [self.time, 0, t_1, self.end_time]
        self.phase_time_steps = [self.time_step, pp.DAY * 1, pp.DAY / 2, 1]

    def adjust_time_step(self):
        """
        Adjust the time step so that smaller time steps are used when the driving forces
        are changed. Also make sure to exactly reach the start and end time for
        each phase. 
        """
        # Default is to just increase the time step somewhat
        self.time_step = getattr(self, "time_step_factor", 1.0) * self.time_step

        # We also want to make sure that we reach the end of each simulation phase
        for dt, lim in zip(self.phase_time_steps, self.phase_limits):
            diff = self.time - lim
            if diff < 0 and -diff <= self.time_step:
                self.time_step = -diff

            if np.isclose(self.time, lim):
                self.time_step = dt
        # And that the time step doesn't grow too large after the equilibration phase
        if self.time > 0:
            self.time_step = min(self.time_step, self.max_time_step)

    def _depth(self, coords) -> np.ndarray:
        """
        Unscaled depth. We center the domain at 1 km below the surface.
        """
        return 1.0 * pp.KILO * pp.METER - self.length_scale * coords[2]

    def set_rock_and_fluid(self):
        """
        Set rock and fluid properties to those of granite and water.
        The matrix permeability may be adjusted by prescribing a "permeability"
        value in the parameters during model construction.
        """
        self.rock = Granite()
        self.rock.BULK_MODULUS = pp.params.rock.bulk_from_lame(
            self.rock.LAMBDA, self.rock.MU
        )
        self.fluid = Water()
        self.rock.PERMEABILITY = self.params.get("permeability", 2.5e-15)

    def _variable_increment(self, g, variable, scale=1, x0=None):
        """ Extracts the variable solution of the current and previous time step and
        computes the increment.
        """
        d = self.gb.node_props(g)
        if x0 is None:
            x0 = d[pp.STATE][variable] * scale
        x1 = d[pp.STATE][pp.ITERATE][variable] * scale
        dx = x1 - x0
        return dx, x1, x0

    def initial_condition(self) -> None:
        """
        Initial value for the Darcy fluxes. TODO: Add to THM.
        """
        for g, d in self.gb:
            d[pp.PARAMETERS] = pp.Parameters()
            d[pp.PARAMETERS].update_dictionaries(
                [self.mechanics_parameter_key, self.scalar_parameter_key,]
            )
        self.update_all_apertures(to_iterate=False)
        self.update_all_apertures()
        super().initial_condition()

        for g, d in self.gb:

            d[pp.STATE]["cell_centers"] = g.cell_centers.copy()
            p0 = self.initial_scalar(g)
            state = {
                self.scalar_variable: p0,
                "u_exp_0": np.zeros(g.num_cells),
                "aperture_0": self.aperture(g) * self.length_scale,
            }
            iterate = {
                self.scalar_variable: p0,
            }  # For initial flux

            pp.set_state(d, state)
            pp.set_iterate(d, iterate)

    def initial_scalar(self, g) -> np.ndarray:
        depth = self._depth(g.cell_centers)
        return self.fluid.hydrostatic_pressure(depth) / self.scalar_scale

    def set_exporter(self):
        self.exporter = pp.Exporter(
            self.gb, self.file_name, folder_name=self.viz_folder_name + "_vtu"
        )
        self.export_times = []

    def export_step(self):
        """
        Export the current solution to vtu. The method sets the desired values in d[pp.STATE].
        For some fields, it provides zeros in the dimensions where the variable is not defined,
        or pads the vector values with zeros so that they have three components, as required
        by ParaView.
        We use suffix _exp on all exported variables, to separate from scaled versions also
        stored in d.
        """
        if "exporter" not in self.__dict__:
            self.set_exporter()
        for g, d in self.gb:
            if g.dim == self.Nd:
                pad_zeros = np.zeros((3 - g.dim, g.num_cells))
                u = d[pp.STATE][self.displacement_variable].reshape(
                    (self.Nd, -1), order="F"
                )
                u_exp = np.vstack((u * self.length_scale, pad_zeros))
                d[pp.STATE]["u_exp"] = u_exp
                d[pp.STATE]["u_global"] = u_exp
                d[pp.STATE]["traction_exp"] = np.zeros(d[pp.STATE]["u_exp"].shape)
            elif g.dim == (self.Nd - 1):
                pad_zeros = np.zeros((2 - g.dim, g.num_cells))
                g_h = self.gb.node_neighbors(g)[0]
                data_edge = self.gb.edge_props((g, g_h))

                u_mortar_local = self.reconstruct_local_displacement_jump(
                    data_edge, from_iterate=False
                )
                mortar_u = data_edge[pp.STATE][self.mortar_displacement_variable]
                mg = data_edge["mortar_grid"]
                displacement_jump_global_coord = (
                    mg.mortar_to_slave_avg(nd=self.Nd)
                    * mg.sign_of_mortar_sides(nd=self.Nd)
                    * mortar_u
                )
                u_mortar_global = displacement_jump_global_coord.reshape(
                    (self.Nd, -1), order="F"
                )
                u_exp = np.vstack((u_mortar_local * self.length_scale, pad_zeros))
                d[pp.STATE]["u_exp"] = u_exp
                d[pp.STATE]["u_global"] = np.vstack(
                    (u_mortar_global * self.length_scale, pad_zeros)
                )
                traction = d[pp.STATE][self.contact_traction_variable].reshape(
                    (self.Nd, -1), order="F"
                )

                d[pp.STATE]["traction_exp"] = (
                    np.vstack((traction, pad_zeros)) * self.scalar_scale
                )
            else:
                d[pp.STATE]["traction_exp"] = np.zeros((3, g.num_cells))
                u_exp = np.zeros((3, g.num_cells))
                d[pp.STATE]["u_exp"] = u_exp
                d[pp.STATE]["u_global"] = np.zeros((3, g.num_cells))

            d[pp.STATE]["aperture_exp"] = self.aperture(g) * self.length_scale
            if np.isclose(self.time, 0):
                d[pp.STATE]["aperture_0"] = self.aperture(g) * self.length_scale
                d[pp.STATE]["u_exp_0"] = u_exp
            p = d[pp.STATE][self.scalar_variable]
            d[pp.STATE]["p_exp"] = p * self.scalar_scale
            d[pp.STATE]["p_minus_ph"] = (p - self.initial_scalar(g)) * self.scalar_scale

        self.exporter.write_vtk(self.export_fields, time_step=self.time)
        self.export_times.append(self.time)

    def export_pvd(self):
        """
        At the end of the simulation, after the final vtu file has been exported, the
        pvd file for the whole simulation is written by calling this method.
        """
        self.exporter.write_pvd(np.array(self.export_times))

    def prepare_simulation(self):
        self.create_grid()
        self._set_time_parameters()
        self.set_rock_and_fluid()
        self.initial_condition()
        self.set_parameters()
        self.assign_variables()
        self.assign_discretizations()

        self.discretize()
        self.initialize_linear_solver()

        self.export_step()

    def before_newton_iteration(self):
        """ Rediscretize. Should the parent be updated?
        """
        self.update_all_apertures(to_iterate=True)
        self.set_parameters()
        self.assembler.discretize(
            term_filter=["!mpsa", "!stabilization", "!div_u", "!grad_p", "!diffusion"]
        )
        for dim in range(self.Nd - 1):
            for g in self.gb.grids_of_dimension(dim):
                self.assembler.discretize(
                    term_filter=["diffusion"], grid=g, edges=False
                )

    def after_newton_convergence(self, solution, errors, iteration_counter):
        super().after_newton_convergence(solution, errors, iteration_counter)
        self.update_all_apertures(to_iterate=False)
        self.update_all_apertures(to_iterate=True)

        self.export_step()
        self.adjust_time_step()

    def assemble_and_solve_linear_system(self, tol):
        use_umfpack = self.params.get("use_umfpack", True)

        A, b = self.assembler.assemble_matrix_rhs()
        logger.debug("Max element in A {0:.2e}".format(np.max(np.abs(A))))
        logger.debug(
            "Max {0:.2e} and min {1:.2e} A sum.".format(
                np.max(np.sum(np.abs(A), axis=1)), np.min(np.sum(np.abs(A), axis=1))
            )
        )
        if use_umfpack:
            A.indices = A.indices.astype(np.int64)
            A.indptr = A.indptr.astype(np.int64)
        t_0 = time.time()
        x = spla.spsolve(A, b, use_umfpack=use_umfpack)
        logger.info("Solved in {} s.".format(time.time() - t_0))
        return x


class Water:
    """
    Fluid phase.
    """

    def __init__(self, theta_ref=None):
        if theta_ref is None:
            self.theta_ref = 20 * (pp.CELSIUS)
        else:
            self.theta_ref = theta_ref
        self.VISCOSITY = 1 * pp.MILLI * pp.PASCAL * pp.SECOND
        self.COMPRESSIBILITY = 1e-10 / pp.PASCAL
        self.BULK_MODULUS = 1 / self.COMPRESSIBILITY

    def thermal_expansion(self, delta_theta):
        """ Units: m^3 / m^3 K, i.e. volumetric """
        return 4e-4

    def thermal_conductivity(self, theta=None):  # theta in CELSIUS
        """ Units: W / m K """
        if theta is None:
            theta = self.theta_ref
        return 0.6

    def specific_heat_capacity(self, theta=None):  # theta in CELSIUS
        """ Units: J / kg K """
        return 4200

    def dynamic_viscosity(self, theta=None):  # theta in CELSIUS
        """Units: Pa s"""
        return 0.001

    def hydrostatic_pressure(self, depth, theta=None):
        rho = 1e3 * (pp.KILOGRAM / pp.METER ** 3)
        return rho * depth * pp.GRAVITY_ACCELERATION + pp.ATMOSPHERIC_PRESSURE


class Granite(pp.Granite):
    """
    Solid phase.
    """

    def __init__(self, theta_ref=None):
        super().__init__(theta_ref)
        self.BULK_MODULUS = pp.params.rock.bulk_from_lame(self.LAMBDA, self.MU)

        self.PERMEABILITY = 1e-15

    def thermal_conductivity(self, theta=None):
        return 3.0

    def specific_heat_capacity(self, theta=None):  # theta in CELSIUS
        c_ref = 790.0
        return c_ref


def run(params):
    logger.info("\n\n" + params["file_name"])
    m = Model(params)
    pp.run_time_dependent_model(m, params)
    m.export_pvd()


def base_params(non_base=None):
    # Define mesh sizes for grid generation
    mesh_size = 1.3
    mesh_args = {
        "mesh_size_frac": mesh_size,
        "mesh_size_min": 0.5 * mesh_size,
        "mesh_size_bound": 10 * mesh_size,
    }
    params = {
        "folder_name": "seg_examples/",
        "nl_convergence_tol": 1e-10,
        "max_iterations": 200,
        "mesh_args": mesh_args,
        "max_memory": 7e7,
        "use_umfpack": True,
    }
    if non_base is not None:
        params.update(non_base)
    return params


if __name__ == "__main__":
    run(base_params({"file_name": "base_case"}))
    run(base_params({"file_name": "low_biot", "biot_alpha": 0.6}))
    run(base_params({"file_name": "high_perm", "permeability": 4e-15}))
    run(base_params({"file_name": "blocking", "blocking_perm": 1e-18}))
