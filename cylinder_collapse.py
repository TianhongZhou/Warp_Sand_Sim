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
def air_drag(v: wp.array(dtype=wp.vec3),
             coeff: wp.float32,
             dt: wp.float32):
    i = wp.tid()
    
    v[i] = v[i] * wp.exp(-coeff * dt)

@wp.kernel
def ground_coulomb_like(q: wp.array(dtype=wp.vec3),
                        v: wp.array(dtype=wp.vec3),
                        radius: wp.float32,
                        y_ground: wp.float32,
                        contact_margin: wp.float32,  
                        g: wp.float32,              
                        mu_s: wp.float32,            
                        mu_k: wp.float32,         
                        dt: wp.float32):
    i = wp.tid()
    p = q[i]
    vi = v[i]

    in_contact = (p[1] - y_ground) <= (radius + contact_margin)

    vn = wp.vec3(0.0, vi[1], 0.0)
    vt = vi - vn

    t_len = wp.length(vt)
    if t_len > 0.0:
        dv_s_max = (mu_s if in_contact else mu_s * 7.0) * g * dt
        dv_k = (mu_k if in_contact else 0.0) * g * dt

        if t_len <= dv_s_max:
            vt = wp.vec3(0.0, 0.0, 0.0)
        else:
            new_len = t_len - dv_k
            if new_len < 0.0:
                vt = wp.vec3(0.0, 0.0, 0.0)
            else:
                vt = vt * (new_len / t_len)

    v[i] = vn + vt

class ExamplePhase2:
    def __init__(self, in_npz="column_prep_state.npz", stage_path="collapse.usd"):
        data = np.load(in_npz)
        q_np = data["q"]
        qd_np = data["qd"]
        self.N = q_np.shape[0]

        self.radius = float(data["radius"])
        self.mass = float(data["mass"])
        self.sim_substeps = int(data["sim_substeps"])
        self.frame_dt = float(data["frame_dt"])
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.damp_coef = 3.0

        builder = wp.sim.ModelBuilder()
        builder.default_particle_radius = self.radius

        dim_x, dim_y, dim_z = 16, 32, 16
        assert dim_x*dim_y*dim_z == self.N, "Loaded N must match builder grid count."
        cell = self.radius * 2.0
        builder.add_particle_grid(
            dim_x=dim_x, dim_y=dim_y, dim_z=dim_z,
            cell_x=cell, cell_y=cell, cell_z=cell,
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            mass=self.mass,
            jitter=0.0,
        )

        self.model = builder.finalize()
        self.model.particle_mu = 1.0
        self.model.soft_contact_mu = 0.1
        self.model.soft_contact_restitution = 0.5

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        q_cpu = wp.array(q_np, dtype=wp.vec3, device="cpu")
        qd_cpu = wp.array(qd_np, dtype=wp.vec3, device="cpu")
        wp.copy(self.state_0.particle_q, q_cpu)
        wp.copy(self.state_0.particle_qd, qd_cpu)

        self.integrator = wp.sim.XPBDIntegrator(
            iterations=8,
            soft_contact_relaxation=0.001,
            enable_restitution=True,
        )

        self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=20.0) if stage_path else None

        self.use_cuda_graph = False

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)

            # wp.launch(
            #     kernel=damp_tangent_on_ground,
            #     dim=self.model.particle_count,
            #     inputs=[self.state_1.particle_q, self.state_1.particle_qd, float(self.radius)],
            #     device=wp.get_device()
            # )

            # wp.launch(
            #     kernel=air_drag,
            #     dim=self.model.particle_count,
            #     inputs=[self.state_1.particle_qd,
            #             self.damp_coef,
            #             float(self.sim_dt)],
            #     device=wp.get_device()
            # )

            wp.launch(
                kernel=ground_coulomb_like,
                dim=self.model.particle_count,
                inputs=[self.state_1.particle_q,
                        self.state_1.particle_qd,
                        float(self.radius),
                        0.0,                      # y_ground，地面高度
                        float(self.radius*0.5),   # contact_margin
                        9.81,                     # g
                        0.5,                      # mu_s 静摩擦
                        0.6,                      # mu_k 动摩擦
                        float(self.sim_dt)],
                device=wp.get_device()
            )

            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def step(self):
        self.model.particle_grid.build(self.state_0.particle_q, self.radius * 2.0)
        self.simulate()
        self.sim_time += self.frame_dt

    def run(self, num_frames):
        for _ in range(num_frames):
            self.step()
            if self.renderer:
                self.renderer.begin_frame(self.sim_time)
                self.renderer.render(self.state_0)
                self.renderer.end_frame()
        if self.renderer:
            self.renderer.save()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--in_npz", type=str, default="column_prep_state.npz")
    parser.add_argument("--stage_path", type=lambda x: None if x=="None" else str(x), default="collapse.usd")
    parser.add_argument("--num_frames", type=int, default=2000)
    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        ex2 = ExamplePhase2(in_npz=args.in_npz, stage_path=args.stage_path)
        ex2.run(num_frames=args.num_frames)