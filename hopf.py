import numpy as np
from firedrake import *
from irksome import Dt, ContinuousPetrovGalerkinScheme, TimeStepper

# ---------------------------------------------------------------------
# mesh & spaces
# ---------------------------------------------------------------------
periodic = True
Lx = Ly = Lz = 2
Nx = Ny = Nz = 10
k = 2
fname = 'borrom_periodic_6_vars'
num_vars = 6        # 4, 5, or 6 - BEjH, BEjHp, or BEjHpu
field_strength = 1

dt = Constant(1e-4)
T = 10000
old_helicity = 0

if periodic:
    base = PeriodicRectangleMesh(Nx, Ny, Lx, Ly, quadrilateral=True)
    mesh = ExtrudedMesh(base, Nz, Lz / Nz, periodic=periodic)
    dirichlet_ids = ()
else:
    base = RectangleMesh(Nx, Ny, Lx, Ly, quadrilateral=True)
    mesh = ExtrudedMesh(base, Nz, Lz / Nz, periodic=periodic)
    dirichlet_ids = ("on_boundary", "top", "bottom")

mesh.coordinates.dat.data[:, 0] -= Lx / 2
mesh.coordinates.dat.data[:, 1] -= Ly / 2
mesh.coordinates.dat.data[:, 2] -= Lz / 2

# spatial spaces
Vg = VectorFunctionSpace(mesh, "Q", k)
Vc = FunctionSpace(mesh, "NCE", k)
Vd = FunctionSpace(mesh, "NCF", k)
Vn = FunctionSpace(mesh, "DQ", k-1)
Vp = FunctionSpace(mesh, "Q", k)
Vu = FunctionSpace(mesh, "NCE", k)


if num_vars == 4:
    spaces = [Vd, Vc, Vc, Vc]
elif num_vars == 5:
    spaces = [Vd, Vc, Vc, Vc, Vp]
elif num_vars == 6:
    # spaces = [Vd, Vc, Vc, Vc, Vc, Vp]
    spaces = [Vd, Vc, Vc, Vc, Vu, Vp]

Z = MixedFunctionSpace(spaces)
z = Function(Z)

# ---------------------------------------------------------------------
# STEP 1: calibrate every facet DOF's axis + sign/scale empirically
# ---------------------------------------------------------------------
cal_fields = [as_vector([1.0, 0.0, 0.0]),
              as_vector([0.0, 1.0, 0.0]),
              as_vector([0.0, 0.0, 1.0])]
cal = [Function(Vd).project(v) for v in cal_fields]
cal_data = np.stack([c.dat.data_ro for c in cal], axis=1)  # (ndofs, 3)

ndofs = cal_data.shape[0]
axis_of_dof = np.argmax(np.abs(cal_data), axis=1)
scale_of_dof = cal_data[np.arange(ndofs), axis_of_dof]

dominance = np.abs(scale_of_dof)
off_axis = np.abs(cal_data).sum(axis=1) - dominance
bad = np.where(off_axis > 1e-8)[0]
print(f"[calibration] {len(bad)} / {ndofs} DOFs failed the clean-axis "
      f"check (want 0).")
print(f"[calibration] |scale| range: {np.abs(scale_of_dof).min():.5f} to "
      f"{np.abs(scale_of_dof).max():.5f} (expect ~ h^2 = "
      f"{(Lx/Nx)**2:.5f} if DOF = total flux, not averaged).")

# ---------------------------------------------------------------------
# STEP 2: facet centroids from (base_cell, layer) adjacency, with
# correct extruded-mesh offset handling
# ---------------------------------------------------------------------
coord_fs = mesh.coordinates.function_space()
coord_map_base = coord_fs.cell_node_map().values     # (nbase, ndof_coord_local)
coord_offset = coord_fs.cell_node_map().offset        # (ndof_coord_local,)

dof_map_base = Vd.cell_node_map().values              # (nbase, ndof_local)
dof_offset = Vd.cell_node_map().offset                 # (ndof_local,)

nbase = dof_map_base.shape[0]
nlayers = Nz
print(f"[mesh] nbase={nbase}, nlayers={nlayers}, "
      f"total 3D cells={nbase * nlayers} (expect {Nx*Ny*Nz}), "
      f"dofs/cell={dof_map_base.shape[1]} (expect 6), "
      f"coord-dofs/cell={coord_map_base.shape[1]} (expect 8).")

coords_data = mesh.coordinates.dat.data_ro
ncoord_dofs = coords_data.shape[0]


def wrap_layer_indices(base_row, offset_row, lay, nlayers, total):
    """base_row + lay*offset_row, but wrapped back into [0, total) by a
    full periodic cycle when vertical periodicity makes it overshoot
    (or undershoot) the valid DOF range -- see explanation above."""
    g = base_row + lay * offset_row
    g = np.where(g >= total, g - nlayers * offset_row, g)
    g = np.where(g < 0, g + nlayers * offset_row, g)
    return g


dof_to_cells = {}          # global dof -> list of (bc, lay)
cell_verts_cache = {}       # (bc, lay) -> (8,3) physical vertex coords

for bc in range(nbase):
    for lay in range(nlayers):
        gdofs = wrap_layer_indices(dof_map_base[bc], dof_offset, lay, nlayers, ndofs)
        gcoord_dofs = wrap_layer_indices(coord_map_base[bc], coord_offset, lay, nlayers, ncoord_dofs)
        verts = coords_data[gcoord_dofs]
        cell_verts_cache[(bc, lay)] = verts
        for g in gdofs:
            dof_to_cells.setdefault(int(g), []).append((bc, lay))

from collections import Counter
card_counts = Counter(len(v) for v in dof_to_cells.values())
print(f"[geometry] facet-ownership cardinality histogram: {dict(card_counts)} "
      f"(fully periodic mesh -> expect only {{2: {ndofs}}}, no 1's).")

cell_centers = {key: v.mean(axis=0) for key, v in cell_verts_cache.items()}
cell_bboxmin = {key: v.min(axis=0) for key, v in cell_verts_cache.items()}
cell_bboxmax = {key: v.max(axis=0) for key, v in cell_verts_cache.items()}

domain_min = np.array([-Lx / 2, -Ly / 2, -Lz / 2])
domain_max = np.array([Lx / 2, Ly / 2, Lz / 2])
tol = 1e-9

centroids = np.zeros((ndofs, 3))
valid = np.zeros(ndofs, dtype=bool)

for g, cells in dof_to_cells.items():
    ax = axis_of_dof[g]
    if len(cells) == 2:
        c1, c2 = cells
        b1max, b1min = cell_bboxmax[c1][ax], cell_bboxmin[c1][ax]
        b2max, b2min = cell_bboxmax[c2][ax], cell_bboxmin[c2][ax]
        centroid = cell_centers[c1].copy()
        if abs(b1max - b2min) < tol:
            # genuine neighbours: c1 below c2 along `ax`
            centroid[ax] = b1max
        elif abs(b2max - b1min) < tol:
            # genuine neighbours: c2 below c1 along `ax`
            centroid[ax] = b2max
        else:
            # NOT physically adjacent along this axis -- this is a
            # periodic-wrap facet (c1, c2 sit at opposite ends of the
            # domain). The true shared location is the domain boundary,
            # not the midpoint of their centers.
            if abs(b1max - domain_max[ax]) < tol:
                centroid[ax] = domain_max[ax]
            elif abs(b1min - domain_min[ax]) < tol:
                centroid[ax] = domain_min[ax]
            else:
                continue  # unexpected topology; leave invalid, worth flagging
        # off-axis coordinates: c1 and c2 should already agree here
        for other_ax in range(3):
            if other_ax != ax:
                centroid[other_ax] = 0.5 * (cell_centers[c1][other_ax]
                                             + cell_centers[c2][other_ax])
    elif len(cells) == 1:
        c = cells[0]
        cmin, cmax = cell_bboxmin[c][ax], cell_bboxmax[c][ax]
        centroid = cell_centers[c].copy()
        if abs(cmin - domain_min[ax]) < tol:
            centroid[ax] = cmin
        elif abs(cmax - domain_max[ax]) < tol:
            centroid[ax] = cmax
        else:
            continue
    else:
        continue
    centroids[g] = centroid
    valid[g] = True

print(f"[geometry] resolved centroids for {valid.sum()} / {ndofs} DOFs "
      f"(want at/near {ndofs}).")

# ---------------------------------------------------------------------
# STEP 3: numpy port of make_loop -- evaluate directly at centroids
# ---------------------------------------------------------------------
def rotate_coords(P, axis):
    cols = [P[:, (i - axis) % 3] for i in range(3)]
    return cols[0], cols[1], cols[2]


def rotate_vec(v, axis):
    return [v[(i + axis) % 3] for i in range(3)]


def make_loop_numpy(x, y, z, axis):
    h = Lx / Nx
    eps = 0.01

    yb1, yb2 = -(2 + eps) * h, -(2 - eps) * h
    yt1, yt2 = (2 - eps) * h, (2 + eps) * h
    xl1, xl2 = -(3 + eps) * h, -(3 - eps) * h
    xr1, xr2 = (1 - eps) * h, (1 + eps) * h
    zl, zu = -eps * h, eps * h

    xl_bounds = (x > xl1) & (x < xl2)
    xr_bounds = (x > xr1) & (x < xr2)
    x_tot_bounds = (x > xl2) & (x < xr1)
    yt_bounds = (y > yb1) & (y < yb2)
    yb_bounds = (y > yt1) & (y < yt2)
    y_tot_bounds = (y > yb2) & (y < yt1)
    z_bounds = (z > zl) & (z < zu)

    f1 = xl_bounds & y_tot_bounds & z_bounds
    f2 = xr_bounds & y_tot_bounds & z_bounds
    f3 = yt_bounds & x_tot_bounds & z_bounds
    f4 = yb_bounds & x_tot_bounds & z_bounds

    v1, v2, v3, v4 = [0., 1., 0.], [0., -1., 0.], [-1., 0., 0.], [1., 0., 0.]
    r1, r2, r3, r4 = (rotate_vec(v, axis) for v in (v1, v2, v3, v4))

    out = np.zeros((x.shape[0], 3))
    out[f1] = r1
    out[f2] = r2
    out[f3] = r3
    out[f4] = r4
    return out


def make_loop_numpy_2(x, y, z, axis):
    h = Lx / Nx
    eps = 0.01

    zb1, zb2 = -(2 + eps) * h, -(2 - eps) * h
    zt1, zt2 = (2 - eps) * h, (2 + eps) * h
    xl1, xl2 = -(1 + eps) * h, -(1 - eps) * h
    xr1, xr2 = (3 - eps) * h, (3 + eps) * h
    yl, yu = -eps * h, eps * h

    xl_bounds = (x > xl1) & (x < xl2)
    xr_bounds = (x > xr1) & (x < xr2)
    x_tot_bounds = (x > xl2) & (x < xr1)
    zt_bounds = (z > zb1) & (z < zb2)
    zb_bounds = (z > zt1) & (z < zt2)
    z_tot_bounds = (z > zb2) & (z < zt1)
    y_bounds = (y > yl) & (y < yu)

    f1 = xl_bounds & z_tot_bounds & y_bounds
    f2 = xr_bounds & z_tot_bounds & y_bounds
    f3 = zt_bounds & x_tot_bounds & y_bounds
    f4 = zb_bounds & x_tot_bounds & y_bounds

    v1, v2, v3, v4 = [0., 0., 1.], [0., 0., -1.], [-1., 0., 0.], [1., 0., 0.]
    r1, r2, r3, r4 = (rotate_vec(v, axis) for v in (v1, v2, v3, v4))

    out = np.zeros((x.shape[0], 3))
    out[f1] = r1
    out[f2] = r2
    out[f3] = r3
    out[f4] = r4
    return out


def B_init_numpy(P):
    total = np.zeros((P.shape[0], 3))
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    total += make_loop_numpy(x, y, z, 0)
    total += make_loop_numpy_2(x, y, z, 0)
    return field_strength * total

target_vecs = B_init_numpy(centroids)
target_scalar = target_vecs[np.arange(ndofs), axis_of_dof]
# ---------------------------------------------------------------------
# STEP 4: assemble B directly
# ---------------------------------------------------------------------
B = Function(Vd)
B.dat.data[:] = 0.0
B.dat.data[valid] = scale_of_dof[valid] * target_scalar[valid]

# ---------------------------------------------------------------------
# STEP 5: checks
# ---------------------------------------------------------------------
l2_div = sqrt(assemble(div(B) ** 2 * dx))
print("[check] ||div B||_2 =", l2_div, " (want ~1e-10 or smaller)")

nnz = np.nonzero(np.abs(B.dat.data_ro) > 1e-8)[0]
print(f"[check] {len(nnz)} nonzero facet DOFs assigned.")

print("[check] sample assigned facets (centroid -> value):")
for g in nnz[:8]:
    print(f"  dof {g}: {centroids[g]} -> {B.dat.data_ro[g]:+.4f} "
          f"(axis={axis_of_dof[g]}, calib_scale={scale_of_dof[g]:+.4f})")

# VTKFile("borromean_B_check.pvd").write(B)


z.subfunctions[0].assign(B)
z.subfunctions[1].assign(0)
z.subfunctions[2].assign(0)
z.subfunctions[3].assign(0)
if num_vars == 5:
    z.subfunctions[4].assign(0)
if num_vars == 6:
    z.subfunctions[5].assign(0)

names = ["B", "j", "E", "H"]
if num_vars == 5:
    names += ["p"]
elif num_vars == 6:
    names += ["u", "p"]
for f, name in zip(z.subfunctions, names):
    f.rename(name)

if num_vars == 4:
    B, j, E, H = split(z)
    Bt, jt, Et, Ht = TestFunctions(Z)
elif num_vars == 5:
    B, j, E, H, p = split(z)
    Bt, jt, Et, Ht, pt = TestFunctions(Z)
elif num_vars == 6:
    B, j, E, H, u, p = split(z)
    Bt, jt, Et, Ht, ut, pt = TestFunctions(Z)



tau = Constant(100)
t = Constant(0)


if num_vars == 4:
    F = inner(Dt(B), Bt) * dx \
        + inner(curl(E), Bt) * dx \
        + inner(E, Et) * dx \
        + tau * inner(cross(cross(j, H), H), Et) * dx \
        + inner(H, Ht) * dx \
        - inner(B, Ht) * dx \
        + inner(j, jt) * dx \
        - inner(B, curl(jt)) * dx
elif num_vars == 5:
    F = inner(Dt(B), Bt) * dx \
        + inner(curl(E), Bt) * dx \
        + inner(E, Et) * dx \
        + tau * inner(cross(cross(j, H) - grad(p), H), Et) * dx \
        + inner(H, Ht) * dx \
        - inner(B, Ht) * dx \
        + inner(j, jt) * dx \
        - inner(B, curl(jt)) * dx \
        + inner(cross(j, H), grad(pt)) * dx \
        - inner(grad(p), grad(pt)) * dx
elif num_vars == 6:
    F = inner(Dt(B), Bt) * dx \
        + inner(curl(E), Bt) * dx \
        + inner(E, Et) * dx \
        + inner(cross(u, H), Et) * dx \
        + inner(H, Ht) * dx \
        - inner(B, Ht) * dx \
        + inner(j, jt) * dx \
        - inner(B, curl(jt)) * dx \
        + inner(u, ut) * dx \
        - tau * inner(cross(j, H) - grad(p), ut) * dx \
        + inner(u, grad(pt)) * dx

# Boundary conditions
bcs = [DirichletBC(Z.sub(index), 0, subdomain) for index in range(len(Z)) for subdomain in dirichlet_ids]

# Solver parameters for fieldsplit
sp_fs = {
		"mat_type": "aij",
		"snes_type": "newtonls",
        "snes_monitor": None,
        "ksp_monitor": None,
        "ksp_type":"preonly",
		"pc_type": "lu",
		"pc_factor_mat_solver_type":"mumps",
        "snes_converged_reason": None
}

# sp_fs = {
#     "mat_type": "aij",
#     "snes_type": "newtonls",
#     "snes_rtol": 1e-12,
#     "snes_atol": 1e-14,
#     "snes_stol": 0.0,
#     "snes_monitor": None,
#     "snes_converged_reason": None,
#     "ksp_type": "preonly",
#     # "ksp_converged_reason": None,
#     "pc_type": "lu",
#     "pc_factor_mat_solver_type": "mumps",
# }

pvd = VTKFile("output/" + fname + ".pvd")
pvd.write(*z.subfunctions, time=float(t))


def build_linear_solver(a, L, u_sol, bcs, aP=None, solver_parameters = None, options_prefix=None):
    problem = LinearVariationalProblem(a, L, u_sol, bcs=bcs, aP=aP)
    solver = LinearVariationalSolver(problem,
                                     solver_parameters=solver_parameters,
                                     options_prefix=options_prefix)
    return solver



def helicity_solver():
    # Spaces for magnetic potential computation
    # If using periodic boundary conditions, we need to modify
    # this to account for the harmonic form [0, 0, 1]^T
    # using Yang's solver

    u = TrialFunction(Vc)
    v = TestFunction(Vc)
    u_sol = Function(Vc)

    # weak form of curl-curl problem 
    a = inner(curl(u), curl(v)) * dx
    L = inner(B, curl(v)) * dx
    beta = Constant(0.1)
    Jp_curl = a + inner(beta * u, v) * dx
    bcs_curl = [DirichletBC(Vc, 0, subdomain) for subdomain in dirichlet_ids]

    rtol = 1E-8
    preconditioner = True
    if preconditioner:
        pc_type = "cholesky"
    else:
        pc_type = "none"
    sparams = {
        "snes_type": "ksponly",
        # "ksp_type": "lsqr",
        "ksp_type": "minres",
        "ksp_max_it": 1000,
        "ksp_convergence_test": "skip",
        #"ksp_monitor": None,
        # "ksp_rtol": rtol,
        "pc_type": pc_type,
        "ksp_norm_type": "preconditioned",
        "ksp_minres_nutol": 1E-8,
        }

    solver = build_linear_solver(a, L, u_sol, bcs_curl, Jp_curl, sparams, options_prefix="helicity")
    return solver


helicity_solver = helicity_solver()

def riesz_map(functional):
    function = Function(functional.function_space().dual())
    with functional.dat.vec as x, function.dat.vec as y:
        helicity_solver.snes.ksp.pc.apply(x, y)
    return function


def compute_helicity(B):
    helicity_solver.solve()
    problem = helicity_solver._problem
    if helicity_solver.snes.ksp.getResidualNorm() > 0.01:
        # lifting strategy
        r = assemble(problem.F, bcs=problem.bcs)
        rstar = r.riesz_representation(riesz_map=riesz_map, bcs=problem.bcs)
        rstar.rename("RHS")
        # lft = uh - inner(r, uh)/inner(r, rstar) * rstar
        c = assemble(action(r, problem.u)) / assemble(action(r, rstar))
        ulft = Function(Vc, name="u_lifted")
        ulft.assign(problem.u - c * rstar)
        A = ulft
    else:
        A = problem.u
    diff = norm(curl(A) - B, "L2")
    if mesh.comm.rank == 0:
        print(f"magnetic potential: ||curl(A) - B||_L2 = {diff:.8e}", flush=True)
    A_ = Function(Vc, name="MagneticPotential")
    A_.project(A)
    curlA = Function(Vd, name="CurlA")
    curlA.project(curl(A))
    diff_ = Function(Vd, name="CurlAMinusB")
    diff_.project(B-curlA)
    VTKFile("output/magnetic_potential_" + fname + ".pvd").write(curlA, diff_, A_)
    if periodic:
        # general helicity
        return assemble(inner(A, diff_ + B)*dx), diff
    else:
        return assemble(inner(A, B)*dx), diff


def compute_energy(B):
    return assemble(inner(B, B)*dx)

def compute_Bn(B):
    n = FacetNormal(mesh)
    return sqrt(assemble(inner(dot(B, n), dot(B, n))*ds_v))

def compute_divB(B):
    return norm(div(B), 'L2')

def compute_divu(u):
    return norm(div(u), 'L2')

def compute_divjBp(j, B, p):
    return norm(div(cross(j, B) - grad(p)), 'L2')


def log_timestep(t, z):
    B = z.subfunctions[0]
    j = z.subfunctions[1]
    E = z.subfunctions[2]
    H = z.subfunctions[3]
    divu = ""
    K_u = ""
    global old_helicity
    if num_vars == 5:
        p = z.subfunctions[4]
    elif num_vars == 6:
        u = z.subfunctions[4]
        p = z.subfunctions[5]
    energy = compute_energy(B)
    helicity, diff = compute_helicity(B)
    normalmg = compute_Bn(B)
    divB = compute_divB(B)
    if num_vars == 5:
        divu = compute_divjBp(j, B, p)
        print(f"||div(j x B - grad(p))||: {divu:e}")
    if num_vars == 6:
        divu = compute_divu(u)
        print(f"||div(u)||: {divu:e}")
        K_u = float(tau) * (norm(cross(j, H), 'L2') + norm(grad(p), 'L2')) / norm(u, 'L2')
        print(f"K_u: {K_u}")
    dH = (helicity - old_helicity)/float(dt)
    RE = -2 * assemble(inner(E, H)*dx)
    old_helicity = helicity
    print(f"Solved at t = {float(t):.4f}. Energy: {energy:.8f} ||curlA - B||_L2 {diff:.8e} Helicity: {helicity:.8f} ||B·n||: {normalmg:e} ||div(B)||: {divB:e}")
    return ",".join(map(str, (float(t), energy, diff, helicity, normalmg, divB, divu, dH, RE, K_u))) + "\n"

measurements = []

filename = fname + ".csv"
if mesh.comm.rank == 0:
    with open(filename, "w") as f:
        f.write("time,energy,diff,helicity,normalmg,divB,divu,dH,RE,K_u\n")


method = ContinuousPetrovGalerkinScheme(
    1,
    basis_type="chebyshev"
)

avs = [i for i in range(1, num_vars)]
stepper = TimeStepper(
    F,
    method,
    t,
    dt,
    z,
    aux_indices=avs,
    bcs=bcs,
    bc_type='ODE',
    solver_parameters=sp_fs,
    options_prefix="time_stepper"
    )

print("Timestepping...")
z_backup = Function(Z)
stages_backup = Function(stepper.stages.function_space())

dt_max = 10

dt_min = 1e-15
growth = 1.1
shrink = 0.3

# tolerance on helicity change PER STEP
helicity_step_tol = 1e-15
helicity_step_tol = 1
B_step_tol = 1
B_step_tol = 1e+5
timestep = 0
reject = False
reason = ""
while float(t) <= T:
    timestep += 1

    dt_try = float(dt)

    # Save state before attempting the step
    z_backup.assign(z)
    stages_backup.assign(stepper.stages)

    try:
        # Attempt t_n -> t_n + dt_try
        stepper.advance()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        raise

    except ConvergenceError:
        stepper.stages.assign(stages_backup)
        z.assign(z_backup)
        dt.assign(shrink * dt_try)

        print(
            f"REJECT dt={dt_try:.6e}, "
            f"solver diverged\n"
            f"retrying with dt={float(dt):.6e}"
        )

        # Important: do NOT advance t
        if float(dt) < dt_min:
            raise ValueError("Timestep infeasibly small.")
        continue
        


    B_new = z.subfunctions[0]
    j_new = z.subfunctions[1]
    E_new = z.subfunctions[2]
    H_new = z.subfunctions[3]

    # Midpoint helicity-balance residual:
    Hdot_residual = -2.0 * assemble(inner(E_new, H_new) * dx)

    # Predicted helicity defect over this step
    H_step_error = abs(dt_try * Hdot_residual)

    B_old = z_backup.subfunctions[0]
    B_old_norm = norm(B_old, "L2")
    dB_norm = norm(B_new - B_old, "L2")
    B_step_error = dB_norm / max(B_old_norm, 1e-14)


    if H_step_error > helicity_step_tol:
        reject = True
        reason = (
            f"helicity defect {H_step_error:.3e} "
            f"> {helicity_step_tol:.3e}"
        )
    
    
    if B_step_error > B_step_tol:
        reject = True
        reason = (
            f"relative B change {B_step_error:.3e} "
            f"> {B_step_tol:.3e}"
        )

    if reject:
        z.assign(z_backup)
        stepper.stages.assign(stages_backup)

        new_dt = shrink * dt_try

        if new_dt < dt_min:
            raise ValueError("Timestep infeasibly small.")

        dt.assign(new_dt)

        print(
            f"REJECT dt={dt_try:.6e}: {reason}\n"
            f"retrying with dt={new_dt:.6e}"
        )
        continue

    # ACCEPT
    t.assign(float(t) + dt_try)

    # Only increase dt after a clean accepted step

    print(
        f"ACCEPT t={float(t):.6e}, dt={dt_try:.6e}, "
        f"|dt * Hdot_residual|={H_step_error:.6e}"
        f"|dB|/|B|={B_old_norm:.6e}"
    )

    measurements.append(log_timestep(t, z))
    if mesh.comm.rank == 0:
        print(f"Solving for t = {float(t + dt):.4f} .. ", flush=True)
    if len(measurements) >= 1:
            if PETSc.COMM_WORLD.rank == 0:
                with open(filename, "a") as f:
                    for m in measurements:
                        f.write(m)
            measurements = []
    if timestep % 10 == 0:
        pvd.write(*z.subfunctions, time=float(t))

    dt.assign(min(growth * dt_try, dt_max))
