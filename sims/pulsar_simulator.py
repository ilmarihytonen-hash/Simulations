"""
Pulsar Simulator  -  GPU ray-marched neutron star with a rotating, tilted dipole.

Requires:  pip install numpy moderngl moderngl-window pyglet

What is modelled (visually, in units where G = c = 1 and lengths are in M = GM/c^2):
  * A compact neutron star (R ~ 4.8 M) whose gravity bends light (Schwarzschild null
    geodesics, same integrator as the black-hole version), so you can see past the limb
    and often both magnetic poles at once.
  * A tilted magnetic dipole that co-rotates with the star. Field lines are true dipole
    lines (r = L sin^2 theta). Closed lines (L < R_LC) are blue, open lines that cross the
    light cylinder are magenta. Animated plasma flow runs along them.
  * Lighthouse beams from both magnetic poles, optionally hollow-cone, with the
    retarded-time spiral you get because radiation travels at c.
  * A striped, wavy equatorial current sheet ("ballerina skirt") that forms beyond the
    light cylinder.
  * Polar-cap hot spots on the star surface, gravitational redshift, coronal glow.
  * A live pulse-profile plot (what a radio telescope at the camera would record),
    a pulse counter, presets (Crab-like, millisecond, magnetar, ...), auto-orbit.

Controls:  drag = orbit   scroll = zoom   WASD/arrows = orbit   SPACE = pause   R = reset
"""

import collections
import math
import threading
import tkinter as tk
from tkinter import ttk, colorchooser

import numpy as np
import moderngl_window as mglw
from moderngl_window import geometry


# ----------------------------------------------------------------------------------------
# Presets  (period is in *simulated* seconds - slowed way down so you can watch it)
# ----------------------------------------------------------------------------------------
PRESETS = {
    "Classic radio pulsar": dict(
        spin_period=3.0, incl_deg=50.0, beam_width_deg=12.0, hollow=0.3, rlc=22.0,
        beam_bright=1.5, field_bright=1.0, wind_bright=1.0, hotspot=1.2),
    "Crab-like (young, energetic)": dict(
        spin_period=1.6, incl_deg=62.0, beam_width_deg=16.0, hollow=0.0, rlc=18.0,
        beam_bright=2.0, field_bright=1.4, wind_bright=2.2, hotspot=1.8),
    "Orthogonal rotator (interpulse)": dict(
        spin_period=3.5, incl_deg=88.0, beam_width_deg=14.0, hollow=0.15, rlc=24.0,
        beam_bright=1.6, field_bright=1.0, wind_bright=1.2, hotspot=1.2),
    "Aligned rotator (no pulses)": dict(
        spin_period=3.0, incl_deg=4.0, beam_width_deg=15.0, hollow=0.0, rlc=22.0,
        beam_bright=1.2, field_bright=1.0, wind_bright=0.8, hotspot=1.0),
    "Millisecond pulsar": dict(
        spin_period=0.7, incl_deg=40.0, beam_width_deg=22.0, hollow=0.0, rlc=12.0,
        beam_bright=1.3, field_bright=0.8, wind_bright=0.6, hotspot=1.0),
    "Magnetar (extreme field)": dict(
        spin_period=5.0, incl_deg=70.0, beam_width_deg=9.0, hollow=0.5, rlc=34.0,
        beam_bright=1.0, field_bright=3.0, wind_bright=0.7, hotspot=2.5),
}


# ----------------------------------------------------------------------------------------
# Tkinter control panel
# ----------------------------------------------------------------------------------------
class ControlPanel(tk.Tk):
    def __init__(self, sim):
        super().__init__()
        self.sim = sim
        self.vars = {}
        self.labels = {}
        self._after_id = None

        self.title("Pulsar Controls")
        self.geometry("440x850")
        self.resizable(False, False)
        ttk.Style(self).theme_use("clam")

        # --- Preset ---
        top = ttk.LabelFrame(self, text=" Preset ", padding=8)
        top.pack(fill="x", padx=10, pady=(8, 4))
        self.preset_var = tk.StringVar(value="Classic radio pulsar")
        combo = ttk.Combobox(top, textvariable=self.preset_var,
                             values=list(PRESETS.keys()), state="readonly")
        combo.pack(fill="x")
        combo.bind("<<ComboboxSelected>>", self.apply_preset)

        # --- Live pulse profile ---
        lc = ttk.LabelFrame(self, text=" Observed pulse profile (flux toward camera) ", padding=6)
        lc.pack(fill="x", padx=10, pady=4)
        self.cw, self.ch = 400, 110
        self.canvas = tk.Canvas(lc, width=self.cw, height=self.ch, bg="#0b0f1a",
                                highlightthickness=0)
        self.canvas.pack()
        self.info = ttk.Label(lc, text="", anchor="w")
        self.info.pack(fill="x", pady=(4, 0))

        # --- Tabs ---
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=6)
        t_star = ttk.Frame(nb, padding=8)
        t_emit = ttk.Frame(nb, padding=8)
        t_vis = ttk.Frame(nb, padding=8)
        t_cam = ttk.Frame(nb, padding=8)
        nb.add(t_star, text="Star")
        nb.add(t_emit, text="Beams & Field")
        nb.add(t_vis, text="Wind & Visuals")
        nb.add(t_cam, text="Camera")

        # Star tab
        self.slider(t_star, "Neutron star mass (M)", "star_mass", 0.5, 2.0)
        self.slider(t_star, "Radius R/M (compactness, lower = more lensing)",
                    "star_radius_factor", 2.6, 10.0)
        self.slider(t_star, "Surface temperature (K)", "star_temp", 8000.0, 40000.0, "{:.0f}")
        self.slider(t_star, "Polar hot-spot brightness", "hotspot", 0.0, 3.0)
        self.slider(t_star, "Spin period (sim seconds)", "spin_period", 0.3, 12.0)
        self.slider(t_star, "Magnetic inclination α (deg)", "incl_deg", 0.0, 90.0, "{:.0f}°")
        self.slider(t_star, "Light-cylinder radius (M)", "rlc", 10.0, 45.0, "{:.1f}")
        self.slider(t_star, "Simulation speed", "sim_speed", 0.0, 3.0)

        # Beams & field tab
        self.check(t_emit, "Show pulsar beams", "show_beams")
        self.slider(t_emit, "Beam half-width (deg)", "beam_width_deg", 3.0, 40.0, "{:.0f}°")
        self.slider(t_emit, "Hollow-cone amount", "hollow", 0.0, 1.0)
        self.slider(t_emit, "Beam brightness", "beam_bright", 0.0, 4.0)
        self.check(t_emit, "Use custom beam tint", "use_beam_tint")
        ttk.Button(t_emit, text="Pick beam tint color", command=self.pick_color).pack(fill="x", pady=(0, 8))
        self.check(t_emit, "Show magnetic field lines", "show_field")
        self.slider(t_emit, "Field-line brightness", "field_bright", 0.0, 4.0)

        # Wind & visuals tab
        self.check(t_vis, "Show striped wind / current sheet", "show_wind")
        self.slider(t_vis, "Wind brightness", "wind_bright", 0.0, 4.0)
        self.check(t_vis, "Show nebula backdrop", "show_nebula")
        self.check(t_vis, "Show celestial grid (visualize lensing)", "show_grid")
        self.slider(t_vis, "Exposure", "exposure", 0.3, 3.0)

        # Camera tab
        ttk.Label(t_cam, text="Camera distance (zoom)").pack(anchor="w")
        self.dist_var = tk.DoubleVar(value=sim.cam_dist)
        lo, hi = sim.dist_limits()
        self.dist_slider = ttk.Scale(t_cam, from_=lo, to=hi, variable=self.dist_var,
                                     command=self.on_slider_dist_change)
        self.dist_slider.pack(fill="x", pady=(0, 6))
        self.slider(t_cam, "Field of view (deg)", "cam_fov", 30.0, 110.0, "{:.0f}°")
        self.check(t_cam, "Auto-orbit camera", "auto_orbit")
        ttk.Button(t_cam, text="Reset camera (R)", command=sim.reset_camera).pack(fill="x", pady=4)
        ttk.Button(t_cam, text="Pause / resume (Space)", command=sim.toggle_pause).pack(fill="x", pady=4)

        self.update_params()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.poll()

    # ---- widget helpers ----
    def slider(self, parent, text, key, lo, hi, fmt="{:.2f}"):
        row = ttk.Frame(parent)
        row.pack(fill="x")
        ttk.Label(row, text=text).pack(side="left")
        val = ttk.Label(row, width=8, anchor="e")
        val.pack(side="right")
        var = tk.DoubleVar(value=getattr(self.sim, key))
        ttk.Scale(parent, from_=lo, to=hi, variable=var,
                  command=self.update_params).pack(fill="x", pady=(0, 5))
        self.vars[key] = var
        self.labels[key] = (val, fmt)

    def check(self, parent, text, key):
        var = tk.BooleanVar(value=getattr(self.sim, key))
        ttk.Checkbutton(parent, text=text, variable=var,
                        command=self.update_params).pack(anchor="w", pady=2)
        self.vars[key] = var

    # ---- callbacks ----
    def pick_color(self):
        color = colorchooser.askcolor(title="Choose beam tint")
        if color[0]:
            self.sim.beam_tint = [c / 255.0 for c in color[0]]
            self.vars["use_beam_tint"].set(True)
            self.update_params()

    def apply_preset(self, _event=None):
        for k, v in PRESETS[self.preset_var.get()].items():
            self.vars[k].set(v)
        self.update_params()

    def on_slider_dist_change(self, *_):
        self.sim.cam_dist = self.dist_var.get()

    def update_params(self, *_):
        for key, var in self.vars.items():
            val = var.get()
            setattr(self.sim, key, val)
            if key in self.labels:
                lab, fmt = self.labels[key]
                lab.config(text=fmt.format(val))
        lo, hi = self.sim.dist_limits()
        self.dist_slider.config(from_=lo, to=hi)
        if self.sim.cam_dist < lo or self.sim.cam_dist > hi:
            self.sim.cam_dist = min(max(self.sim.cam_dist, lo), hi)

    # ---- periodic refresh (all Tk calls stay on this thread) ----
    def poll(self):
        try:
            # Sync values the render window may have changed (scroll wheel, keys, reset)
            if abs(self.dist_var.get() - self.sim.cam_dist) > 1e-3:
                self.dist_var.set(self.sim.cam_dist)
            if "cam_fov" in self.vars and abs(self.vars["cam_fov"].get() - self.sim.cam_fov) > 1e-3:
                self.vars["cam_fov"].set(self.sim.cam_fov)
                self.update_params()
            self.draw_curve()
        except Exception:
            pass
        self._after_id = self.after(33, self.poll)

    def draw_curve(self):
        c, w, h = self.canvas, self.cw, self.ch
        c.delete("all")
        for f in (0.25, 0.5, 0.75):
            y = h - 8 - f * (h - 20)
            c.create_line(0, y, w, y, fill="#1a2238")
        data = list(self.sim.light_curve)
        P = self.sim.spin_period
        window = max(6.0, 3.0 * P)
        if len(data) > 2:
            t_end = data[-1][0]
            t0 = t_end - window
            pts = []
            for t, inten in data:
                if t >= t0:
                    pts += [(t - t0) / window * w, h - 8 - min(inten, 1.0) * (h - 20)]
            if len(pts) >= 4:
                c.create_line(*pts, fill="#66ccff", width=2)
            inten = data[-1][1]
        else:
            inten = 0.0
        # pulse "LED"
        v = int(60 + 195 * min(inten, 1.0))
        c.create_oval(w - 22, 6, w - 8, 20, fill="#%02x%02x%02x" % (v, v, 255), outline="")
        self.info.config(
            text=f"P = {P:.2f} s   f = {1.0 / P:.2f} Hz   α = {self.sim.incl_deg:.0f}°   "
                 f"pulses: {self.sim.pulse_count}")

    def on_close(self):
        if self._after_id is not None:
            self.after_cancel(self._after_id)
        self.destroy()


# ----------------------------------------------------------------------------------------
# Render window
# ----------------------------------------------------------------------------------------
class PulsarWindowSim(mglw.WindowConfig):
    title = "Pulsar Simulator - Neutron Star, Dipole Field, Lighthouse Beams & Striped Wind"
    resource_dir = "."
    window_size = (1920, 1080)
    aspect_ratio = 16 / 9
    resizable = True
    gl_version = (3, 3)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.quad = geometry.quad_fs()

        # --- Star ---
        self.star_mass = 1.0
        self.star_radius_factor = 4.8
        self.star_temp = 28000.0
        self.hotspot = 1.2

        # --- Rotation / magnetosphere ---
        self.spin_period = 3.0
        self.incl_deg = 50.0
        self.rlc = 22.0
        self.sim_speed = 1.0
        self.paused = False
        self.phase = 0.0
        self.sim_time = 0.0

        # --- Emission ---
        self.show_beams = True
        self.beam_width_deg = 12.0
        self.hollow = 0.3
        self.beam_bright = 1.5
        self.use_beam_tint = False
        self.beam_tint = [0.4, 0.8, 1.0]
        self.show_field = True
        self.field_bright = 1.0
        self.show_wind = True
        self.wind_bright = 1.0

        # --- Visuals ---
        self.show_grid = False
        self.show_nebula = True
        self.exposure = 1.0

        # --- Camera ---
        self.default_cam_dist = 34.0
        self.default_cam_yaw = 0.6
        self.default_cam_pitch = 0.30
        self.default_cam_fov = 60.0
        self.cam_dist = self.default_cam_dist
        self.cam_yaw = self.default_cam_yaw
        self.cam_pitch = self.default_cam_pitch
        self.cam_fov = self.default_cam_fov
        self.auto_orbit = False
        self._held = set()

        # --- Pulse profile ---
        self.light_curve = collections.deque(maxlen=2000)
        self.pulse_count = 0
        self._above = False

        self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)

        self.gui_thread = threading.Thread(target=self.start_gui, daemon=True)
        self.gui_thread.start()

    # ---- helpers ----
    def start_gui(self):
        ControlPanel(self).mainloop()

    def dist_limits(self):
        r = self.star_radius_factor * self.star_mass
        return r * 1.6, 220.0

    def reset_camera(self):
        self.cam_dist = self.default_cam_dist
        self.cam_yaw = self.default_cam_yaw
        self.cam_pitch = self.default_cam_pitch
        self.cam_fov = self.default_cam_fov

    def toggle_pause(self):
        self.paused = not self.paused

    def set_uniform(self, name, value):
        try:
            self.prog[name].value = value
        except KeyError:
            pass  # optimized out by the GLSL compiler

    def cam_position(self):
        return np.array([
            self.cam_dist * math.cos(self.cam_pitch) * math.sin(self.cam_yaw),
            self.cam_dist * math.sin(self.cam_pitch),
            self.cam_dist * math.cos(self.cam_pitch) * math.cos(self.cam_yaw),
        ], dtype="f4")

    def pulse_intensity(self, cam_pos):
        """Approximate flux a distant observer at the camera direction would record."""
        a = math.radians(self.incl_deg)
        m = np.array([math.sin(a) * math.cos(self.phase), math.cos(a),
                      math.sin(a) * math.sin(self.phase)])
        v = cam_pos / np.linalg.norm(cam_pos)
        d = float(np.dot(m, v))
        bw = max(math.radians(self.beam_width_deg), 0.02)
        ang = math.acos(min(1.0, abs(d)))
        a0 = self.hollow * bw * 0.65
        q = (ang - a0) / (bw * 0.6)
        return math.exp(-q * q) * (1.0 if d > 0 else 0.6)

    # ---- frame ----
    def on_render(self, time: float, frametime: float):
        self.ctx.clear(0.0, 0.0, 0.0)

        # Held-key camera motion (frame-rate independent)
        k = self.wnd.keys
        rate = 1.6 * frametime
        if k.W in self._held or k.UP in self._held:
            self.cam_pitch = min(1.49, self.cam_pitch + rate)
        if k.S in self._held or k.DOWN in self._held:
            self.cam_pitch = max(-1.49, self.cam_pitch - rate)
        if k.A in self._held or k.LEFT in self._held:
            self.cam_yaw += rate
        if k.D in self._held or k.RIGHT in self._held:
            self.cam_yaw -= rate
        if self.auto_orbit:
            self.cam_yaw += 0.12 * frametime

        # Advance time and rotor phase
        if not self.paused and self.sim_speed > 0.0:
            dt = frametime * self.sim_speed
            self.sim_time += dt
            self.phase = (self.phase + 2.0 * math.pi / max(self.spin_period, 1e-3) * dt) % (2.0 * math.pi)

        cam_pos = self.cam_position()

        if not self.paused and self.sim_speed > 0.0:
            inten = self.pulse_intensity(cam_pos)
            self.light_curve.append((self.sim_time, inten))
            if inten > 0.5 and not self._above:
                self.pulse_count += 1
                self._above = True
            elif inten < 0.3:
                self._above = False

        # Camera basis (always looking at the star)
        cam_dir = -cam_pos / np.linalg.norm(cam_pos)
        cam_right = np.cross(cam_dir, np.array([0.0, 1.0, 0.0], dtype="f4"))
        cam_right /= np.linalg.norm(cam_right)
        cam_up = np.cross(cam_right, cam_dir)
        fov_scale = 1.0 / math.tan(math.radians(self.cam_fov * 0.5))

        su = self.set_uniform
        su("u_resolution", tuple(float(v) for v in self.wnd.buffer_size))
        su("u_cam_pos", tuple(float(v) for v in cam_pos))
        su("u_cam_dir", tuple(float(v) for v in cam_dir))
        su("u_cam_up", tuple(float(v) for v in cam_up))
        su("u_cam_right", tuple(float(v) for v in cam_right))
        su("u_fov_scale", float(fov_scale))
        su("u_time", float(self.sim_time % 4096.0))

        su("u_mass", float(self.star_mass))
        su("u_star_radius", float(self.star_radius_factor * self.star_mass))
        su("u_star_temp", float(self.star_temp))
        su("u_hotspot", float(self.hotspot))
        su("u_incl", float(math.radians(self.incl_deg)))
        su("u_phase", float(self.phase))
        su("u_rlc", float(self.rlc))

        su("u_show_beams", bool(self.show_beams))
        su("u_beam_width", float(math.radians(self.beam_width_deg)))
        su("u_hollow", float(self.hollow))
        su("u_beam_bright", float(self.beam_bright))
        su("u_use_beam_tint", bool(self.use_beam_tint))
        su("u_beam_tint", tuple(float(c) for c in self.beam_tint))
        su("u_show_field", bool(self.show_field))
        su("u_field_bright", float(self.field_bright))
        su("u_show_wind", bool(self.show_wind))
        su("u_wind_bright", float(self.wind_bright))

        su("u_show_grid", bool(self.show_grid))
        su("u_show_nebula", bool(self.show_nebula))
        su("u_exposure", float(self.exposure))

        self.quad.render(self.prog)

    # ---- input ----
    def mouse_drag_event(self, x, y, dx, dy):
        self.cam_yaw -= dx * 0.005
        self.cam_pitch = max(-1.49, min(1.49, self.cam_pitch + dy * 0.005))

    def mouse_scroll_event(self, x_offset, y_offset):
        lo, hi = self.dist_limits()
        self.cam_dist = max(lo, min(hi, self.cam_dist * (1.0 - 0.06 * y_offset)))

    def key_event(self, key, action, modifiers):
        keys = self.wnd.keys
        if action == keys.ACTION_PRESS:
            self._held.add(key)
            if key == keys.R:
                self.reset_camera()
            elif key == keys.SPACE:
                self.toggle_pause()
        elif action == keys.ACTION_RELEASE:
            self._held.discard(key)


# ----------------------------------------------------------------------------------------
# Shaders
# ----------------------------------------------------------------------------------------
VERTEX_SHADER = """
    #version 330
    in vec3 in_position;
    out vec2 v_uv;
    void main() {
        v_uv = in_position.xy * 0.5 + 0.5;
        gl_Position = vec4(in_position, 1.0);
    }
"""

FRAGMENT_SHADER = """
    #version 330
    out vec4 fragColor;
    in vec2 v_uv;

    uniform vec2  u_resolution;
    uniform vec3  u_cam_pos;
    uniform vec3  u_cam_dir;
    uniform vec3  u_cam_up;
    uniform vec3  u_cam_right;
    uniform float u_fov_scale;
    uniform float u_time;

    uniform float u_mass;
    uniform float u_star_radius;
    uniform float u_star_temp;
    uniform float u_hotspot;
    uniform float u_incl;
    uniform float u_phase;
    uniform float u_rlc;

    uniform bool  u_show_beams;
    uniform float u_beam_width;
    uniform float u_hollow;
    uniform float u_beam_bright;
    uniform bool  u_use_beam_tint;
    uniform vec3  u_beam_tint;
    uniform bool  u_show_field;
    uniform float u_field_bright;
    uniform bool  u_show_wind;
    uniform float u_wind_bright;

    uniform bool  u_show_grid;
    uniform bool  u_show_nebula;
    uniform float u_exposure;

    #define MAX_STEPS 380
    #define BOUND_RADIUS 85.0
    #define TWO_PI 6.28318530718

    // ------------------------------------------------------------ noise
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

    // ------------------------------------------------------------ colour
    vec3 get_blackbody_color(float temp_k) {
        float t = clamp(temp_k, 1000.0, 40000.0) / 100.0;
        vec3 col;
        if (t <= 66.0) col.r = 1.0;
        else col.r = clamp(1.292936 * pow(t - 60.0, -0.1332047), 0.0, 1.0);

        if (t <= 66.0) col.g = clamp(0.3900815 * log(t) - 0.6318414, 0.0, 1.0);
        else col.g = clamp(1.12989 * pow(t - 60.0, -0.0755148), 0.0, 1.0);

        if (t >= 66.0) col.b = 1.0;
        else if (t <= 19.0) col.b = 0.0;
        else col.b = clamp(0.5432067 * log(t - 10.0) - 1.1962545, 0.0, 1.0);
        return col;
    }

    // ------------------------------------------------------------ background
    vec3 render_background(vec3 dir) {
        vec3 col = vec3(0.0);

        if (u_show_nebula) {
            float n1 = noise(dir.xy * 2.2 + dir.z * 5.3);
            float n2 = noise(dir.yz * 4.1 + dir.x * 3.7 + 10.0);
            float neb = smoothstep(0.30, 0.85, n1 * 0.6 + n2 * 0.4);
            col += mix(vec3(0.10, 0.04, 0.20), vec3(0.02, 0.14, 0.20), n2) * neb * 0.45;
        }

        vec3 p = dir * 180.0;
        vec3 cell = floor(p);
        vec3 f = fract(p) - 0.5;
        float h = hash(cell.xy + cell.z * 37.0);
        if (h > 0.965) {
            float dist = length(f);
            vec3 sc = mix(vec3(1.0, 0.85, 0.7), vec3(0.7, 0.85, 1.0), fract(h * 57.0));
            col += sc * smoothstep(0.12, 0.0, dist) * (0.6 + 0.4 * sin(h * 100.0)) * 1.4;
        }

        if (u_show_grid) {
            float theta = acos(clamp(dir.y, -1.0, 1.0));
            float phi = atan(dir.z, dir.x);
            vec2 g = vec2(theta * 3.8197, phi * 3.183);
            vec2 d = abs(fract(g + 0.5) - 0.5);
            float line = 1.0 - smoothstep(0.0, 0.03, min(d.x, d.y));
            col += vec3(0.15, 0.45, 0.85) * line * 0.8;
        }
        return col;
    }

    // ------------------------------------------------------------ pulsar geometry
    vec3 mag_axis(float ph) {
        return vec3(sin(u_incl) * cos(ph), cos(u_incl), sin(u_incl) * sin(ph));
    }

    vec3 beam_color(float t) {
        vec3 core = u_use_beam_tint ? u_beam_tint : vec3(0.80, 0.92, 1.0);
        vec3 edge = u_use_beam_tint ? u_beam_tint * vec3(0.6, 0.7, 1.0) : vec3(0.45, 0.45, 1.0);
        return mix(core, edge, clamp(t, 0.0, 1.0));
    }

    // Lighthouse beam. m_r is the magnetic axis at the *retarded* time, so the beam
    // trails behind the star as a spiral (radiation moves at c).
    vec3 beam_emission(vec3 pn, float r, vec3 m_r) {
        float d = dot(pn, m_r);
        float ang = acos(clamp(abs(d), 0.0, 1.0));
        float side = d > 0.0 ? 1.0 : 0.6;                  // weaker interpulse pole
        float bw = max(u_beam_width, 0.02);
        float a0 = u_hollow * bw * 0.65;                   // hollow cone ring radius
        float q = (ang - a0) / (bw * 0.6);
        float prof = exp(-q * q);
        float rad = smoothstep(u_star_radius * 0.9, u_star_radius * 1.5, r) / (1.0 + r * 0.045);
        float n = 0.7 + 0.3 * noise(vec2(r * 0.45 - u_time * 7.0, ang * 22.0));
        return beam_color(ang / bw) * prof * rad * n * side * u_beam_bright;
    }

    // Dipole field lines: r = L sin^2(theta_m). Lines are the intersection of an
    // L-shell (log spaced) with a magnetic-azimuth plane (N evenly spaced).
    vec3 field_emission(vec3 p, float r) {
        float lag = smoothstep(0.35 * u_rlc, u_rlc, r) * r / u_rlc;   // sweep-back near the LC
        float ph = u_phase - lag;
        float c = cos(ph);
        float s = sin(ph);
        vec3 pb = vec3(p.x * c + p.z * s, p.y, -p.x * s + p.z * c);    // into the rotating frame

        float sa = sin(u_incl);
        float ca = cos(u_incl);
        vec3 m0 = vec3(sa, ca, 0.0);
        vec3 e1 = vec3(-ca, sa, 0.0);
        vec3 e2 = vec3(0.0, 0.0, 1.0);

        float cz = dot(pb, m0) / r;
        float sin2 = max(1.0 - cz * cz, 1e-4);
        float L = r / sin2;
        float phi_m = atan(dot(pb, e2) + 1e-6, dot(pb, e1));

        float q = log(L / u_star_radius) / log(1.5);
        float dq = abs(fract(q + 0.5) - 0.5);

        const float N = 10.0;
        float dphi = abs(fract(phi_m / TWO_PI * N + 0.5) - 0.5) * TWO_PI / N;
        float lat = dphi * sqrt(sin2);

        float e = exp(-(dq * dq) / (0.16 * 0.16)) * exp(-(lat * lat) / (0.06 * 0.06));

        float mlat = atan(abs(cz), sqrt(sin2));
        float flow = 0.65 + 0.35 * sin(mlat * 7.0 - u_time * 4.0 + q * 1.7);

        float open_t = smoothstep(0.85 * u_rlc, 1.15 * u_rlc, L);
        vec3 col = mix(vec3(0.25, 0.60, 1.00), vec3(1.00, 0.45, 0.90), open_t);

        float fade = exp(-(r - u_star_radius) / (1.6 * u_rlc));
        return col * e * flow * fade;
    }

    // Wavy equatorial current sheet ("ballerina skirt") + striped wind.
    vec3 wind_emission(vec3 pn, float r, vec3 m_r, float psi) {
        float sd = dot(pn, m_r);                           // sheet lies where B_r changes sign
        float w = 0.10;
        float dens = exp(-(sd * sd) / (w * w));
        float rad = smoothstep(0.55 * u_rlc, 1.25 * u_rlc, r) * exp(-r / (3.2 * u_rlc));
        float n = noise(vec2(r * 0.30 - u_time * 3.5, dot(pn, vec3(3.1, 4.7, -2.3)) * 3.0));
        float stripes = 0.65 + 0.35 * cos(2.0 * psi);
        vec3 col = mix(vec3(1.0, 0.45, 0.15), vec3(1.0, 0.78, 0.50), n);
        return col * dens * rad * (0.35 + 0.9 * n) * stripes * u_wind_bright;
    }

    vec3 shade_star(vec3 pos, vec3 vel) {
        vec3 n = normalize(pos);
        float mu = clamp(dot(n, -vel), 0.0, 1.0);
        float cp = abs(dot(n, mag_axis(u_phase)));
        float hot = smoothstep(cos(0.32), cos(0.12), cp);
        float halo = smoothstep(cos(0.80), 1.0, cp);

        vec3 base = get_blackbody_color(u_star_temp) * (0.10 + 0.22 * pow(mu, 0.45));
        vec3 col = base
                 + vec3(0.25, 0.45, 1.0) * halo * 0.35
                 + vec3(1.0, 0.93, 0.85) * 4.0 * hot * u_hotspot * (0.5 + 0.5 * mu);
        float redshift = sqrt(max(0.05, 1.0 - 2.0 * u_mass / u_star_radius));
        return col * redshift;
    }

    vec3 tonemap(vec3 c) {
        c *= u_exposure;
        c = vec3(1.0) - exp(-c * 1.15);
        return pow(c, vec3(0.92));
    }

    // ------------------------------------------------------------ main
    void main() {
        vec2 st = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y;
        vec3 ray_dir = normalize(st.x * u_cam_right + st.y * u_cam_up + u_fov_scale * u_cam_dir);

        vec3 pos = u_cam_pos;
        vec3 vel = ray_dir;

        float RS = 2.0 * u_mass;
        float R = u_star_radius;

        // Jump to the bounding sphere if the camera is outside it
        if (length(pos) > BOUND_RADIUS) {
            float b = dot(pos, vel);
            float c = dot(pos, pos) - BOUND_RADIUS * BOUND_RADIUS;
            float disc = b * b - c;
            if (disc > 0.0 && (-b - sqrt(disc)) > 0.0) {
                pos += vel * (-b - sqrt(disc));
            } else {
                fragColor = vec4(tonemap(render_background(vel)), 1.0);
                return;
            }
        }

        vec3 accum = vec3(0.0);

        for (int i = 0; i < MAX_STEPS; i++) {
            float r2 = dot(pos, pos);
            float r = sqrt(r2);

            if (r > BOUND_RADIUS && dot(pos, vel) > 0.0) break;

            float step_size = clamp(r * 0.02, 0.06, 0.8);

            // Schwarzschild light bending
            vec3 L = cross(pos, vel);
            float h2 = dot(L, L);
            vec3 acc = -1.5 * RS * h2 / (r2 * r2 * r) * pos;

            vec3 delta_v = acc * step_size;
            float dl = length(delta_v);
            if (dl > 0.18) delta_v *= 0.18 / dl;

            pos += vel * step_size + 0.5 * delta_v * step_size;
            vel = normalize(vel + delta_v);

            r = max(length(pos), 0.001);

            // Opaque neutron star
            if (r < R) {
                accum += shade_star(pos, vel);
                fragColor = vec4(tonemap(accum), 1.0);
                return;
            }

            // Optically thin emission along the ray
            vec3 emit = vec3(0.0);
            vec3 pn = pos / r;

            if (u_show_field) {
                emit += field_emission(pos, r) * 0.45 * u_field_bright;
            }
            if (u_show_beams || u_show_wind) {
                float psi = u_phase - r / u_rlc;           // retarded rotor angle
                vec3 m_r = mag_axis(psi);
                if (u_show_beams) emit += beam_emission(pn, r, m_r) * 0.06;
                if (u_show_wind)  emit += wind_emission(pn, r, m_r, psi) * 0.16;
            }
            // thin hot corona hugging the surface
            emit += vec3(0.25, 0.45, 1.0) * exp(-(r - R) * 1.5) * 0.5;

            accum += emit * step_size;
        }

        accum += render_background(normalize(vel));
        vec3 c = tonemap(accum);
        c *= 1.0 - 0.30 * dot(st, st);
        c += (hash(gl_FragCoord.xy + fract(u_time)) - 0.5) / 255.0;   // dither
        fragColor = vec4(c, 1.0);
    }
"""


if __name__ == "__main__":
    mglw.run_window_config(PulsarWindowSim)
