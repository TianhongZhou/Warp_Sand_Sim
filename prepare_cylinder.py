import numpy as np
import warp as wp
import warp.sim
import warp.sim.render

@wp.kernel
def damp_tangent_on_ground(q: wp.array(dtype=wp.vec3),
                           v: wp.array(dtype=wp.vec3),
                           radius: wp.float32):
    i = wp.tid()
    p = q[i]
    vi = v[i]
    if p[1] <= radius * 1.5:
        vn = wp.vec3(0.0, vi[1], 0.0)
        vt = vi - vn
        v[i] = vn + vt * 0.985

@wp.kernel
def confine_in_cylinder(q: wp.array(dtype=wp.vec3),
                        v: wp.array(dtype=wp.vec3),
                        cylinder_R: wp.float32):
    i = wp.tid()
    p = q[i]
    vi = v[i]
    r = wp.sqrt(p[0]*p[0] + p[2]*p[2])
    if r > cylinder_R:
        nx = p[0] / r
        nz = p[2] / r
        p = wp.vec3(nx * cylinder_R * 0.999, p[1], nz * cylinder_R * 0.999)
        v_rad = vi[0]*nx + vi[2]*nz
        if v_rad > 0.0:
            vi = wp.vec3(vi[0] - v_rad*nx, vi[1], vi[2] - v_rad*nz)
        q[i] = p
        v[i] = vi

class Example:
    def __init__(self, stage_path="prep.usd", out_npz="column_prep_state.npz"):
        self.out_npz = out_npz

        fps = 60
        self.frame_dt = 1.0 / fps
        self.sim_substeps = 64
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # particles
        self.radius = 0.1
        self.mass = 100.0

        # cylinder phase params
        self.cylinder_R = 2.0
        self.settle_seconds = 2.5
        self.keep_cylinder = True  # phase1: True

        # build model (same config will be reused in phase2)
        builder = wp.sim.ModelBuilder()
        builder.default_particle_radius = self.radius

        dim_x, dim_y, dim_z = 16, 32, 16
        cell_x = cell_y = cell_z = self.radius * 2.0
        builder.add_particle_grid(
            dim_x=dim_x, dim_y=dim_y, dim_z=dim_z,
            cell_x=cell_x, cell_y=cell_y, cell_z=cell_z,
            pos=wp.vec3(-(dim_x-1)*cell_x/2, 2.0, -(dim_z-1)*cell_z/2),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            mass=self.mass,
            jitter=self.radius * 0.1,
        )

        self.model = builder.finalize()
        self.model.particle_mu = 0.8
        self.model.soft_contact_mu = 0.1
        self.model.soft_contact_restitution = 0.5

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.integrator = wp.sim.XPBDIntegrator(
            iterations=8,
            soft_contact_relaxation=0.001,
            enable_restitution=True,
        )

        self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=20.0) if stage_path else None

        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as cap:
                self.simulate()
            self.graph = cap.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)

            wp.launch(
                kernel=damp_tangent_on_ground,
                dim=self.model.particle_count,
                inputs=[self.state_1.particle_q, self.state_1.particle_qd, float(self.radius)],
                device=wp.get_device()
            )

            if self.keep_cylinder:
                wp.launch(
                    kernel=confine_in_cylinder,
                    dim=self.model.particle_count,
                    inputs=[self.state_1.particle_q, self.state_1.particle_qd,
                            float(self.cylinder_R)],
                    device=wp.get_device()
                )

            # swap
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def step(self):
        self.model.particle_grid.build(self.state_0.particle_q, self.radius * 2.0)
        if self.use_cuda_graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def save_state_npz(self):
        q_cpu = wp.empty(shape=self.model.particle_count, dtype=wp.vec3, device="cpu")
        qd_cpu = wp.empty(shape=self.model.particle_count, dtype=wp.vec3, device="cpu")
        wp.copy(q_cpu, self.state_0.particle_q)
        wp.copy(qd_cpu, self.state_0.particle_qd)
        q_np = q_cpu.numpy()
        qd_np = qd_cpu.numpy()

        meta = dict(
            radius=self.radius,
            mass=self.mass,
            particle_count=self.model.particle_count,
            frame_dt=self.frame_dt,
            sim_substeps=self.sim_substeps,
        )
        np.savez(self.out_npz, q=q_np, qd=qd_np, **meta)
        print(f"[phase1] saved: {self.out_npz}, N={q_np.shape[0]}")

    def run_until_collapse_and_save(self, num_frames=400):
        while self.sim_time + 1e-9 < self.settle_seconds and num_frames > 0:
            self.step()
            if self.renderer:
                self.renderer.begin_frame(self.sim_time)
                self.renderer.render(self.state_0)
                self.renderer.end_frame()
            num_frames -= 1

        self.save_state_npz()

        if self.renderer:
            self.renderer.save()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--stage_path", type=lambda x: None if x=="None" else str(x), default="prep.usd")
    parser.add_argument("--out_npz", type=str, default="column_prep_state.npz")
    parser.add_argument("--num_frames", type=int, default=400)
    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        ex = Example(stage_path=args.stage_path, out_npz=args.out_npz)
        ex.run_until_collapse_and_save(num_frames=args.num_frames)