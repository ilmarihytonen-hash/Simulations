import threading
import tkinter as tk
from tkinter import ttk, colorchooser
import numpy as np
import moderngl
import moderngl_window as mglw
from moderngl_window import geometry


class ControlPanel(tk.Tk):
    """Tkinter GUI control panel for black hole, accretion disk, and polar jets."""

    def __init__(self, sim):
        super().__init__()
        self.sim = sim
        self.sim.gui = self

        self.title("Kerr Simulation & Polar Jets Controls")
        self.geometry("400x820")
        self.resizable(False, False)

        style = ttk.Style(self)
        style.theme_use("clam")

        # --- Astrophysical Parameters ---
        frame_astro = ttk.LabelFrame(self, text=" Astrophysical Parameters ", padding=10)
        frame_astro.pack(fill="x", padx=10, pady=5)

        ttk.Label(frame_astro, text="Black Hole Mass (M):").pack(anchor="w")
        self.mass_var = tk.DoubleVar(value=sim.bh_mass)
        self.mass_slider = ttk.Scale(
            frame_astro,
            from_=0.2,
            to=5.0,
            variable=self.mass_var,
            command=self.update_params,
        )
        self.mass_slider.pack(fill="x", pady=(0, 5))

        ttk.Label(frame_astro, text="Angular Momentum / Spin (a):").pack(anchor="w")
        self.spin_var = tk.DoubleVar(value=sim.bh_spin)
        self.spin_slider = ttk.Scale(
            frame_astro,
            from_=-0.99,
            to=0.99,
            variable=self.spin_var,
            command=self.update_params,
        )
        self.spin_slider.pack(fill="x", pady=(0, 5))

        ttk.Label(frame_astro, text="Disk Temperature (K):").pack(anchor="w")
        self.temp_var = tk.DoubleVar(value=sim.disk_temp)
        self.temp_slider = ttk.Scale(
            frame_astro,
            from_=1000.0,
            to=40000.0,
            variable=self.temp_var,
            command=self.update_params,
        )
        self.temp_slider.pack(fill="x", pady=(0, 5))

        ttk.Label(frame_astro, text="Disk Density / Brightness:").pack(anchor="w")
        self.bright_var = tk.DoubleVar(value=sim.disk_brightness)
        self.bright_slider = ttk.Scale(
            frame_astro,
            from_=0.1,
            to=4.0,
            variable=self.bright_var,
            command=self.update_params,
        )
        self.bright_slider.pack(fill="x", pady=(0, 5))

        # --- Polar Relativistic Jets ---
        frame_stream = ttk.LabelFrame(self, text=" Polar Relativistic Jets ", padding=10)
        frame_stream.pack(fill="x", padx=10, pady=5)

        self.stream_var = tk.BooleanVar(value=sim.show_streams)
        self.stream_chk = ttk.Checkbutton(
            frame_stream,
            text="Enable Polar Jets",
            variable=self.stream_var,
            command=self.update_params,
        )
        self.stream_chk.pack(anchor="w", pady=2)

        ttk.Label(frame_stream, text="Jet Brightness:").pack(anchor="w")
        self.stream_bright_var = tk.DoubleVar(value=sim.stream_brightness)
        self.stream_bright_slider = ttk.Scale(
            frame_stream,
            from_=0.1,
            to=5.0,
            variable=self.stream_bright_var,
            command=self.update_params,
        )
        self.stream_bright_slider.pack(fill="x", pady=(0, 5))

        # --- Time & Simulation Speed ---
        frame_sim = ttk.LabelFrame(self, text=" Simulation Timing ", padding=10)
        frame_sim.pack(fill="x", padx=10, pady=5)

        ttk.Label(frame_sim, text="Physics Simulation Speed:").pack(anchor="w")
        self.speed_var = tk.DoubleVar(value=sim.sim_speed)
        self.speed_slider = ttk.Scale(
            frame_sim,
            from_=0.0,
            to=4.0,
            variable=self.speed_var,
            command=self.update_params,
        )
        self.speed_slider.pack(fill="x", pady=(0, 5))

        # --- Visual Customization ---
        frame_vis = ttk.LabelFrame(self, text=" Visual Overrides ", padding=10)
        frame_vis.pack(fill="x", padx=10, pady=5)

        self.grid_var = tk.BooleanVar(value=sim.show_grid)
        self.grid_chk = ttk.Checkbutton(
            frame_vis,
            text="Show Celestial Grid (Visualize Lensing)",
            variable=self.grid_var,
            command=self.update_params,
        )
        self.grid_chk.pack(anchor="w", pady=2)

        self.override_var = tk.BooleanVar(value=sim.use_custom_color)
        self.override_chk = ttk.Checkbutton(
            frame_vis,
            text="Use Custom Disk Tint",
            variable=self.override_var,
            command=self.update_params,
        )
        self.override_chk.pack(anchor="w", pady=2)

        self.color_btn = ttk.Button(
            frame_vis, text="Pick Custom Tint Color", command=self.pick_color
        )
        self.color_btn.pack(fill="x", pady=2)

        # --- Camera Controls ---
        frame_cam = ttk.LabelFrame(self, text=" Orbit Camera Controls ", padding=10)
        frame_cam.pack(fill="x", padx=10, pady=5)

        ttk.Label(frame_cam, text="Camera Distance (Zoom):").pack(anchor="w")
        self.dist_var = tk.DoubleVar(value=sim.cam_dist)

        rs = 2.0 * sim.bh_mass
        min_d = max(3.5, rs * 2.2)
        max_d = max(150.0, rs * 40.0)

        self.dist_slider = ttk.Scale(
            frame_cam,
            from_=min_d,
            to=max_d,
            variable=self.dist_var,
            command=self.on_slider_dist_change,
        )
        self.dist_slider.pack(fill="x", pady=(0, 5))

        ttk.Label(frame_cam, text="Field of View (FOV Angle):").pack(anchor="w")
        self.fov_var = tk.DoubleVar(value=sim.cam_fov)
        self.fov_slider = ttk.Scale(
            frame_cam,
            from_=30.0,
            to=110.0,
            variable=self.fov_var,
            command=self.update_params,
        )
        self.fov_slider.pack(fill="x", pady=(0, 5))

        self.reset_cam_btn = ttk.Button(
            frame_cam, text="Reset Camera Position (Key: R)", command=self.reset_camera
        )
        self.reset_cam_btn.pack(fill="x", pady=2)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def pick_color(self):
        color = colorchooser.askcolor(title="Choose Disk Tint Color")
        if color[0]:
            rgb = [c / 255.0 for c in color[0]]
            self.sim.custom_color = rgb
            self.update_params()

    def reset_camera(self):
        self.sim.reset_camera()
        self.dist_var.set(self.sim.cam_dist)
        self.fov_var.set(self.sim.cam_fov)

    def on_slider_dist_change(self, *args):
        self.sim.cam_dist = self.dist_var.get()

    def update_params(self, *args):
        self.sim.bh_mass = self.mass_var.get()
        self.sim.bh_spin = self.spin_var.get()
        self.sim.disk_temp = self.temp_var.get()
        self.sim.disk_brightness = self.bright_var.get()
        self.sim.show_streams = self.stream_var.get()
        self.sim.stream_brightness = self.stream_bright_var.get()
        self.sim.sim_speed = self.speed_var.get()
        self.sim.show_grid = self.grid_var.get()
        self.sim.use_custom_color = self.override_var.get()
        self.sim.cam_fov = self.fov_var.get()

        rs = 2.0 * self.sim.bh_mass
        min_dist = max(3.5, rs * 2.2)
        max_dist = max(150.0, rs * 40.0)

        self.dist_slider.config(from_=min_dist, to=max_dist)

        if self.sim.cam_dist < min_dist:
            self.sim.cam_dist = min_dist
            self.dist_var.set(min_dist)
        elif self.sim.cam_dist > max_dist:
            self.sim.cam_dist = max_dist
            self.dist_var.set(max_dist)

    def on_close(self):
        self.destroy()


class BlackHoleWindowSim(mglw.WindowConfig):
    title = "Kerr Black Hole & Polar Jets Simulator (1080p Orbit Camera)"
    resource_dir = "."
    window_size = (1920, 1080)
    aspect_ratio = 16 / 9
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.quad = geometry.quad_fs()
        self.gui = None

        # Simulation Parameters
        self.bh_mass = 1.0
        self.bh_spin = 0.70
        self.disk_temp = 8500.0
        self.disk_brightness = 1.2
        self.show_streams = True  # Represents Polar Jets now
        self.stream_brightness = 1.8
        self.sim_speed = 1.0
        self.sim_time = 0.0
        self.show_grid = False
        self.custom_color = [1.0, 0.45, 0.15]
        self.use_custom_color = False

        # Orbiting Camera Parameters (Locked to origin (0,0,0))
        self.default_cam_dist = 18.0
        self.default_cam_yaw = 0.45
        self.default_cam_pitch = 0.22
        self.default_cam_fov = 60.0

        self.cam_dist = self.default_cam_dist
        self.cam_yaw = self.default_cam_yaw
        self.cam_pitch = self.default_cam_pitch
        self.cam_fov = self.default_cam_fov

        # Shader Program Setup
        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_position;
                out vec2 v_uv;
                void main() {
                    v_uv = in_position.xy * 0.5 + 0.5;
                    gl_Position = vec4(in_position, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                out vec4 fragColor;
                in vec2 v_uv;

                uniform vec2 u_resolution;
                uniform vec3 u_cam_pos;
                uniform vec3 u_cam_dir;
                uniform vec3 u_cam_up;
                uniform vec3 u_cam_right;
                uniform float u_fov_scale;
                uniform float u_time;

                uniform float u_mass;
                uniform float u_spin;
                uniform float u_temp;
                uniform float u_brightness;
                uniform bool u_show_streams;
                uniform float u_stream_brightness;
                uniform vec3 u_custom_color;
                uniform bool u_use_custom_color;
                uniform bool u_show_grid;

                #define MAX_STEPS 260
                #define PI 3.14159265359

                float hash(vec2 p) {
                    p = fract(p * vec2(123.34, 456.21));
                    p += dot(p, p + 45.32);
                    return fract(p.x * p.y);
                }

                float noise(vec2 p) {
                    vec2 i = floor(p);
                    vec2 f = fract(p);
                    f = f * f * (3.0 - 2.0 * f);
                    float a = hash(i);
                    float b = hash(i + vec2(1.0, 0.0));
                    float c = hash(i + vec2(0.0, 1.0));
                    float d = hash(i + vec2(1.0, 1.0));
                    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
                }

                float disk_turbulence(vec3 p, float time, float M, float a_spin) {
                    float r = length(p.xz);
                    float phi = atan(p.z, p.x);
                    float omega = sqrt(M) / (pow(max(0.1, r), 1.5) + a_spin * sqrt(M));
                    float rotated_phi = phi + omega * time;

                    vec2 uv = vec2(r * 1.2, rotated_phi * 3.5);
                    float n = noise(uv) * 0.55 + noise(uv * 2.2) * 0.3 + noise(uv * 4.5) * 0.15;
                    return clamp(n, 0.0, 1.0);
                }

                // Volumetric Polar Relativistic Jet Field
                float polar_jet_density(vec3 p, float time, float M, float a_spin) {
                    float h = abs(p.y);
                    
                    // Jets originate just outside the inner ISCO region
                    if (h < 1.2 * M) return 0.0;

                    float r_xz = length(p.xz);
                    
                    // Parabolic collimation: Jet expands outward proportionally to sqrt(height)
                    float jet_radius = 0.25 * M + 0.35 * sqrt(h * M);
                    float d = r_xz / jet_radius;

                    // Outward vertical flow speed
                    float flow_speed = 6.0;
                    
                    // Add rotational twist driven by the black hole spin
                    float phi = atan(p.z, p.x);
                    float twist = phi - sign(p.y) * a_spin * h * 0.15;

                    vec2 uv = vec2(twist * 2.0, h - sign(p.y) * time * flow_speed);
                    float n = noise(uv * 1.5) * 0.7 + noise(uv * 3.5) * 0.3;

                    // Gaussian core + smooth edge + vertical fade
                    float core = exp(-(d * d * 5.0));
                    float vertical_falloff = exp(-h * 0.06); // Fades out at extreme heights

                    return core * n * vertical_falloff;
                }

                vec3 get_blackbody_color(float temp_k) {
                    float t = clamp(temp_k, 1000.0, 40000.0) / 100.0;
                    vec3 col;

                    if (t <= 66.0) {
                        col.r = 1.0;
                    } else {
                        col.r = clamp(1.292936 * pow(t - 60.0, -0.1332047), 0.0, 1.0);
                    }

                    if (t <= 66.0) {
                        col.g = clamp(0.3900815 * log(t) - 0.6318414, 0.0, 1.0);
                    } else {
                        col.g = clamp(1.12989 * pow(t - 60.0, -0.0755148), 0.0, 1.0);
                    }

                    if (t >= 66.0) {
                        col.b = 1.0;
                    } else if (t <= 19.0) {
                        col.b = 0.0;
                    } else {
                        col.b = clamp(0.5432067 * log(t - 10.0) - 1.1962545, 0.0, 1.0);
                    }

                    return col;
                }

                vec3 render_background(vec3 dir) {
                    vec3 col = vec3(0.0);

                    vec3 p = dir * 180.0;
                    vec3 cell = floor(p);
                    vec3 f = fract(p) - 0.5;

                    float h = hash(cell.xy + cell.z * 37.0);
                    if (h > 0.965) {
                        float dist = length(f);
                        col += vec3(smoothstep(0.12, 0.0, dist) * (0.6 + 0.4 * sin(h * 100.0))) * 1.4;
                    }

                    if (u_show_grid) {
                        float theta = acos(clamp(dir.y, -1.0, 1.0));
                        float phi = atan(dir.z, dir.x);

                        vec2 grid = abs(fract(vec2(theta * 3.8197, phi * 3.183) + 0.5) - 0.5) / max(vec2(1e-4), fwidth(vec2(theta, phi)));
                        float line = min(grid.x, grid.y);
                        float grid_pattern = 1.0 - min(line, 1.0);

                        vec3 grid_color = vec3(0.15, 0.45, 0.85) * grid_pattern * 0.8;
                        col += grid_color;
                    }

                    return col;
                }

                float get_kerr_isco(float M, float a_spin) {
                    float a = clamp(a_spin / M, -0.99, 0.99);
                    float abs_a = abs(a);
                    float z1 = 1.0 + pow(1.0 - abs_a * abs_a, 1.0/3.0) * (pow(1.0 + abs_a, 1.0/3.0) + pow(1.0 - abs_a, 1.0/3.0));
                    float z2 = sqrt(3.0 * abs_a * abs_a + z1 * z1);
                    float isco_m = 3.0 + z2 - sign(a) * sqrt(max(0.0, (3.0 - z1) * (3.0 + z1 + 2.0 * z2)));
                    return isco_m * M;
                }

                void main() {
                    vec2 st = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y;
                    vec3 ray_dir = normalize(st.x * u_cam_right + st.y * u_cam_up + u_fov_scale * u_cam_dir);

                    vec3 pos = u_cam_pos;
                    vec3 vel = ray_dir;

                    float M = u_mass;
                    float a_spin = u_spin * M;
                    float RS = 2.0 * M;

                    float RH = M + sqrt(max(0.001, M * M - a_spin * a_spin));
                    float ISCO = get_kerr_isco(M, u_spin);
                    float DISK_OUTER = 10.0 * RS;
                    float BOUND_RADIUS = 24.0 * RS;

                    float r_cam = length(pos);
                    if (r_cam > BOUND_RADIUS) {
                        float b = dot(pos, vel);
                        float c = dot(pos, pos) - BOUND_RADIUS * BOUND_RADIUS;
                        float delta = b * b - c;

                        if (delta > 0.0) {
                            float t1 = -b - sqrt(delta);
                            if (t1 > 0.0) {
                                pos += vel * t1;
                            } else {
                                fragColor = vec4(render_background(normalize(vel)), 1.0);
                                return;
                            }
                        } else {
                            fragColor = vec4(render_background(normalize(vel)), 1.0);
                            return;
                        }
                    }

                    vec3 accum_color = vec3(0.0);
                    float alpha = 0.0;

                    for (int i = 0; i < MAX_STEPS; i++) {
                        float r2 = dot(pos, pos);
                        float r = sqrt(r2);

                        if (r > BOUND_RADIUS && dot(pos, vel) > 0.0) {
                            break;
                        }

                        float step_size = max(0.012 * M, min(0.28 * M, r * 0.018));

                        // Schwarzschild Acceleration
                        vec3 L = cross(pos, vel);
                        float h2 = dot(L, L);
                        vec3 acc = -1.5 * RS * h2 / (r2 * r2 * r) * pos;

                        // Kerr Lense-Thirring Frame Dragging Acceleration
                        vec3 spin_axis = vec3(0.0, 1.0, 0.0);
                        vec3 frame_drag = 2.0 * a_spin * M * cross(spin_axis, pos) / (r2 * r2);
                        acc += frame_drag;

                        vec3 delta_v = acc * step_size;
                        if (length(delta_v) > 0.18) {
                            delta_v = normalize(delta_v) * 0.18;
                        }

                        vec3 old_pos = pos;
                        vec3 old_vel = vel;

                        pos += vel * step_size + 0.5 * delta_v * step_size;
                        vel = normalize(vel + delta_v);

                        // Volumetric Polar Jets Sampling
                        if (u_show_streams) {
                            float j_dens = polar_jet_density(pos, u_time, M, u_spin);
                            if (j_dens > 0.005) {
                                vec3 jet_col = vec3(0.40, 0.65, 1.0) * u_stream_brightness * j_dens * step_size * 3.5;
                                accum_color += jet_col * (1.0 - alpha);
                                alpha += j_dens * step_size * 1.2 * (1.0 - alpha);
                            }
                        }

                        // Accretion Disk Mid-Plane Collision Check
                        float dy = old_pos.y - pos.y;
                        if (old_pos.y * pos.y <= 0.0 && abs(dy) > 1e-6) {
                            float t_plane = clamp(old_pos.y / dy, 0.0, 1.0);
                            vec3 hit_pos = mix(old_pos, pos, t_plane);
                            vec3 hit_vel = normalize(mix(old_vel, vel, t_plane));
                            float r_hit = length(hit_pos.xz);

                            if (r_hit >= ISCO && r_hit <= DISK_OUTER) {
                                float radial_falloff = smoothstep(ISCO, ISCO * 1.25, r_hit) * (1.0 - smoothstep(DISK_OUTER * 0.65, DISK_OUTER, r_hit));
                                float turbulence = disk_turbulence(hit_pos, u_time, M, a_spin);
                                float density = radial_falloff * turbulence;

                                if (density > 0.002) {
                                    vec3 tangent = normalize(vec3(-hit_pos.z, 0.0, hit_pos.x));

                                    float omega_kerr = sqrt(M) / (pow(r_hit, 1.5) + a_spin * sqrt(M));
                                    float v_mag = clamp(r_hit * omega_kerr, 0.05, 0.85);
                                    vec3 v_disk = tangent * v_mag;

                                    float v_dot_ray = dot(v_disk, -hit_vel);
                                    float gamma = 1.0 / sqrt(max(0.001, 1.0 - v_mag * v_mag));
                                    float doppler = 1.0 / (gamma * (1.0 - v_dot_ray));
                                    float grav_shift = sqrt(max(0.01, 1.0 - RS / r_hit));

                                    float g_factor = doppler * grav_shift;
                                    float beaming = pow(g_factor, 4.0);
                                    float local_temp = u_temp * pow(ISCO / r_hit, 0.75) * g_factor;

                                    vec3 thermal_color = u_use_custom_color ? u_custom_color : get_blackbody_color(local_temp);

                                    float emission = density * u_brightness * beaming * 1.35;
                                    accum_color += thermal_color * emission * (1.0 - alpha);
                                    alpha += clamp(emission * 0.82, 0.0, 1.0) * (1.0 - alpha);

                                    if (alpha >= 0.98) {
                                        fragColor = vec4(accum_color, 1.0);
                                        return;
                                    }
                                }
                            }
                        }

                        // Event Horizon Shadow Collision
                        if (r < RH * 1.002) {
                            fragColor = vec4(accum_color, 1.0);
                            return;
                        }
                    }

                    vec3 def_dir = normalize(vel);
                    vec3 bg_color = render_background(def_dir);
                    accum_color += bg_color * (1.0 - alpha);

                    accum_color = accum_color / (vec3(1.0) + accum_color * 0.45);
                    accum_color = pow(accum_color, vec3(0.85));

                    fragColor = vec4(accum_color, 1.0);
                }
            """,
        )

        self.gui_thread = threading.Thread(target=self.start_gui, daemon=True)
        self.gui_thread.start()

    def start_gui(self):
        gui = ControlPanel(self)
        gui.mainloop()

    def reset_camera(self):
        self.cam_dist = self.default_cam_dist
        self.cam_yaw = self.default_cam_yaw
        self.cam_pitch = self.default_cam_pitch
        self.cam_fov = self.default_cam_fov

    def on_render(self, time: float, frametime: float):
        self.ctx.clear(0.0, 0.0, 0.0)

        self.sim_time += frametime * self.sim_speed

        # Convert spherical camera coordinates into Cartesian position
        cx = self.cam_dist * np.cos(self.cam_pitch) * np.sin(self.cam_yaw)
        cy = self.cam_dist * np.sin(self.cam_pitch)
        cz = self.cam_dist * np.cos(self.cam_pitch) * np.cos(self.cam_yaw)

        cam_pos = np.array([cx, cy, cz], dtype="f4")

        # Camera ALWAYS points directly at black hole center (0, 0, 0)
        target = np.array([0.0, 0.0, 0.0], dtype="f4")
        cam_dir = target - cam_pos
        cam_dir /= np.linalg.norm(cam_dir)

        # Build orthonormal camera coordinate system basis
        up_temp = np.array([0.0, 1.0, 0.0], dtype="f4")
        cam_right = np.cross(cam_dir, up_temp)
        cam_right /= np.linalg.norm(cam_right)
        cam_up = np.cross(cam_right, cam_dir)

        fov_scale = 1.0 / np.tan(np.radians(self.cam_fov * 0.5))

        self.prog["u_resolution"].value = self.wnd.buffer_size
        self.prog["u_cam_pos"].value = tuple(cam_pos)
        self.prog["u_cam_dir"].value = tuple(cam_dir)
        self.prog["u_cam_up"].value = tuple(cam_up)
        self.prog["u_cam_right"].value = tuple(cam_right)
        self.prog["u_fov_scale"].value = float(fov_scale)
        self.prog["u_time"].value = self.sim_time

        self.prog["u_mass"].value = self.bh_mass
        self.prog["u_spin"].value = self.bh_spin
        self.prog["u_temp"].value = self.disk_temp
        self.prog["u_brightness"].value = self.disk_brightness
        self.prog["u_show_streams"].value = self.show_streams
        self.prog["u_stream_brightness"].value = self.stream_brightness
        self.prog["u_custom_color"].value = tuple(self.custom_color)
        self.prog["u_use_custom_color"].value = self.use_custom_color
        self.prog["u_show_grid"].value = self.show_grid

        self.quad.render(self.prog)

    def mouse_drag_event(self, x, y, dx, dy):
        """Mouse drag strictly orbits camera around center (0,0,0)."""
        self.cam_yaw -= dx * 0.005
        self.cam_pitch += dy * 0.005
        # Lock pitch to prevent gimbal flip over poles
        self.cam_pitch = max(-1.49, min(1.49, self.cam_pitch))

    def mouse_scroll_event(self, x_offset, y_offset):
        rs = 2.0 * self.bh_mass
        min_dist = max(3.5, rs * 2.2)
        max_dist = max(150.0, rs * 40.0)

        self.cam_dist -= y_offset * 1.5
        self.cam_dist = max(min_dist, min(max_dist, self.cam_dist))

        if self.gui and hasattr(self.gui, "dist_var"):
            self.gui.dist_var.set(self.cam_dist)

    def key_event(self, key, action, modifiers):
        if action == self.wnd.keys.ACTION_PRESS or action == self.wnd.keys.ACTION_HOLD:
            if key == self.wnd.keys.W or key == self.wnd.keys.UP:
                self.cam_pitch = min(1.49, self.cam_pitch + 0.04)
            elif key == self.wnd.keys.S or key == self.wnd.keys.DOWN:
                self.cam_pitch = max(-1.49, self.cam_pitch - 0.04)
            elif key == self.wnd.keys.A or key == self.wnd.keys.LEFT:
                self.cam_yaw += 0.04
            elif key == self.wnd.keys.D or key == self.wnd.keys.RIGHT:
                self.cam_yaw -= 0.04
            elif key == self.wnd.keys.R:
                self.reset_camera()
                if self.gui:
                    self.gui.dist_var.set(self.cam_dist)
                    self.gui.fov_var.set(self.cam_fov)


if __name__ == "__main__":
    mglw.run_window_config(BlackHoleWindowSim)