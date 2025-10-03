import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.widgets import TextBox, Button, Slider
import math

# ----------------- yardımcı -----------------
def ensure_last_tick(ax, t_start, t_end):
    ax.set_xlim(t_start, t_end)
    ticks = ax.get_xticks()
    ticks = [t for t in ticks if t_start <= t <= t_end]
    if len(ticks) == 0 or abs(ticks[-1] - t_end) > 1e-9:
        ticks.append(t_end)
    ax.set_xticks(sorted(ticks))

# ----------------- kinematik -----------------
def leg_calculator(a2, a3, a4, b, alfa, x, y):
    r = math.hypot(x, y)
    if r == 0:
        r = 1e-9

    c = (b**2 + x**2 + y**2 - a4**2) / (2 * b * r)
    c = max(-1.0, min(1.0, c))

    theta1_rad = math.atan2(y, x) + math.acos(c) + math.radians(alfa)
    theta1_deg = math.degrees(theta1_rad) % 360
    alfa_rad = math.radians(alfa)

    beta_rad = math.atan2(y, x) + math.radians(90)
    sinteta3 = (-r*math.cos(beta_rad) - b*math.sin(theta1_rad - alfa_rad)) / a4
    costeta3 = ( r*math.sin(beta_rad) - b*math.cos(theta1_rad - alfa_rad)) / a4
    teta3_rad = math.atan2(sinteta3, costeta3)

    arg = (a3*math.sin(theta1_rad - teta3_rad) - b*math.sin(alfa_rad)) / a2
    arg = max(-1.0, min(1.0, arg))
    teta2_rad = math.asin(arg) + theta1_rad

    denom = math.sin(teta2_rad - theta1_rad)
    if abs(denom) < 1e-9:
        denom = 1e-9
    s23 = (b*math.sin(teta2_rad - theta1_rad + alfa_rad) +
           a3*math.sin(teta3_rad - teta2_rad)) / denom

    return (theta1_deg,
            math.degrees(teta2_rad) % 360,
            math.degrees(teta3_rad) % 360,
            s23)

# ----------------- app -----------------
class SimApp:
    N1 = 300
    N2 = 80
    INTERVAL_MS_DEFAULT = 40  # ~25 fps
    LEN_TOL = 1e-2            # a4 uzunluk toleransı
    ANG_TOL_DEG = 1.0         # BC ile CD yön farkı toleransı (derece)

    def __init__(self):
        self.params = {
            "a2": 2.0, "a3": 2.0, "a4": 4.0, "b": 9.0, "alfa": 15.0,
            "a_ellipse": 6.0, "b_ellipse": 2.0, "Y_SHIFT": -10.0
        }
        self.speed_step = 1
        self._frame_idx = 0

        self.fig = plt.figure(figsize=(15, 10), constrained_layout=True)
        outer = GridSpec(1, 2, width_ratios=[5.4, 1.0], wspace=0.25, figure=self.fig)

        self.gs_left = GridSpecFromSubplotSpec(
            3, 1, subplot_spec=outer[0],
            height_ratios=[10, 1.2, 1.2], hspace=0.55
        )
        self.ax_mech = self.fig.add_subplot(self.gs_left[0])
        self.ax_th1  = self.fig.add_subplot(self.gs_left[1])
        self.ax_s23  = self.fig.add_subplot(self.gs_left[2])

        self.gs_right = GridSpecFromSubplotSpec(
            12, 1, subplot_spec=outer[1],
            height_ratios=[0.9, 1, 1, 1, 1, 1, 1, 1, 1, 1.2, 0.8, 1.2], hspace=0.22
        )
        self.ax_title = self.fig.add_subplot(self.gs_right[0]); self.ax_title.axis("off")
        self.ax_boxes = [self.fig.add_subplot(self.gs_right[i]) for i in range(1, 9)]
        self.ax_btn   = self.fig.add_subplot(self.gs_right[9])
        self.ax_speed_label = self.fig.add_subplot(self.gs_right[10]); self.ax_speed_label.axis("off")
        self.ax_speed = self.fig.add_subplot(self.gs_right[11])

        self._alloc_arrays()
        self._artists = {}

        self.build_controls()
        self.compute()
        self.build_plots()
        self.start_animation()

    # -------- veri alanları --------
    def _alloc_arrays(self):
        self.X = self.Y = self.T_phys = None
        self.Ax = self.Ay = self.Bx = self.By = self.Cx = self.Cy = None
        self.TH1_unwrapped_deg = None
        self.S23 = None
        self.TOTAL = 0

    # -------- hesap --------
    def compute(self):
        a2 = self.params["a2"]; a3 = self.params["a3"]; a4 = self.params["a4"]
        b  = self.params["b"];  alfa = self.params["alfa"]
        a_ellipse = self.params["a_ellipse"]; b_ellipse = self.params["b_ellipse"]
        Y_SHIFT = self.params["Y_SHIFT"]

        # yörünge
        t0 = np.arcsin(-1.0 / b_ellipse)
        t1 = np.pi - t0
        t = np.linspace(t0, t1, self.N1)
        x_arc = a_ellipse * np.cos(t)
        y_arc = b_ellipse * np.sin(t) + Y_SHIFT
        x_back = np.linspace(x_arc[-1], x_arc[0], self.N2)
        y_back = np.full(self.N2, y_arc[0])

        self.X = np.r_[x_arc, x_back]
        self.Y = np.r_[y_arc, y_back]
        self.TOTAL = self.X.size

        dt = self.INTERVAL_MS_DEFAULT / 1000.0
        self.T_phys = np.arange(self.TOTAL) * dt

        self.Ax = np.zeros(self.TOTAL); self.Ay = np.zeros(self.TOTAL)
        self.Bx = np.zeros(self.TOTAL); self.By = np.zeros(self.TOTAL)
        self.Cx = np.zeros(self.TOTAL); self.Cy = np.zeros(self.TOTAL)
        TH1_rad = np.zeros(self.TOTAL)
        self.S23 = np.zeros(self.TOTAL)

        alfa_rad = math.radians(alfa)

        for i in range(self.TOTAL):
            th1_deg, th2_deg, th3_deg, s23 = leg_calculator(
                a2, a3, a4, b, alfa, float(self.X[i]), float(self.Y[i])
            )
            th1 = math.radians(th1_deg)
            th2 = math.radians(th2_deg)

            # A
            self.Ax[i] = s23 * math.cos(th1)
            self.Ay[i] = s23 * math.sin(th1)
            # B
            self.Bx[i] = self.Ax[i] + a2 * math.cos(th2)
            self.By[i] = self.Ay[i] + a2 * math.sin(th2)
            # C
            c_ang = th1 - alfa_rad
            self.Cx[i] = b * math.cos(c_ang)
            self.Cy[i] = b * math.sin(c_ang)

            TH1_rad[i] = th1
            self.S23[i] = s23

        self.TH1_unwrapped_deg = np.degrees(np.unwrap(TH1_rad))

    # -------- çizimler --------
    def build_plots(self):
        self._artists = {}  # kritik: eski handle'ları bırak
        ax = self.ax_mech
        ax.clear()
        ax.set_aspect("equal")
        a_ellipse = self.params["a_ellipse"]; Y_SHIFT = self.params["Y_SHIFT"]
        ax.set_xlim(-(a_ellipse + 5), (a_ellipse + 5))
        ax.set_ylim(Y_SHIFT - 5, 5)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title("Bacak Mekanizması Hareket Analizi", pad=12)
        ax.grid(True)

        self._artists["trace_line"] = ax.plot([], [], linewidth=2, label="yörünge")[0]
        self._artists["marker"]     = ax.plot([], [], 'o', markersize=6, label="bacak uç")[0]
        self._artists["A0A_line"]   = ax.plot([], [], 'b-', linewidth=2, label="A₀A")[0]
        self._artists["AB_line"]    = ax.plot([], [], 'g-', linewidth=2, label="AB")[0]
        self._artists["BC_line"]    = ax.plot([], [], 'm-', linewidth=2, label="BC")[0]
        # CD’nin rengi hızla değişebileceği için iki sanatçı: çizgi ve rengi
        self._artists["CD_line"]    = ax.plot([], [], '-', linewidth=2, color='c', label="a4 (CD)")[0]
        self._artists["A0_point"]   = ax.plot([], [], 'ko', markersize=5)[0]
        self._artists["A_point"]    = ax.plot([], [], 'ko', markersize=5)[0]
        self._artists["B_point"]    = ax.plot([], [], 'ko', markersize=5)[0]
        self._artists["C_point"]    = ax.plot([], [], 'ko', markersize=5)[0]

        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  borderaxespad=0.0, framealpha=0.95)

        # θ1 paneli
        self._setup_time_axis(self.ax_th1, "θ1 - t", "θ1 (°)",
                              self.T_phys, self.TH1_unwrapped_deg,
                              key_line="theta1_line", key_dot="theta1_dot")

        # s23 paneli
        axs = self.ax_s23
        axs.clear()
        axs.set_title("s23 - t"); axs.set_xlabel("t(s)"); axs.set_ylabel("s23 (cm)")
        axs.grid(True)
        self._apply_time_xlim(axs)
        smin, smax = self.S23.min(), self.S23.max()
        pad = 0.5*(abs(smax-smin)+1e-9)
        axs.set_ylim(smin - pad, smax + pad)
        self._artists["s23_line"] = axs.plot([], [], linewidth=2)[0]
        self._artists["s23_dot"]  = axs.plot([], [], 'o', markersize=5)[0]

        self.fig.canvas.draw_idle()

    def _setup_time_axis(self, ax, title, ylabel, T, Y, key_line, key_dot):
        ax.clear()
        ax.set_title(title); ax.set_xlabel("t(s)"); ax.set_ylabel(ylabel)
        ax.grid(True)
        self._apply_time_xlim(ax)
        ymin, ymax = Y.min(), Y.max()
        ax.set_ylim(ymin - 10, ymax + 10)
        self._artists[key_line] = ax.plot([], [], linewidth=2)[0]
        self._artists[key_dot]  = ax.plot([], [], 'o', markersize=5)[0]

    def _apply_time_xlim(self, ax):
        t_disp = self.T_phys / max(1, int(self.speed_step))
        ensure_last_tick(ax, t_disp[0], t_disp[-1])

    # -------- UI --------
    def build_controls(self):
        self.ax_title.text(0.5, 0.95, "Parametreler", ha="center", va="top",
                           fontsize=12, weight="bold", transform=self.ax_title.transAxes)
        labels = [
            ("a2", "2.0"), ("a3", "2.0"), ("a4", "4.0"),
            ("b", "9.0"), ("alfa", "15.0"),
            ("a_ellipse", "6.0"), ("b_ellipse", "2.0"), ("Y_SHIFT", "-10.0")
        ]
        self.textboxes = {}
        for (name, default), axb in zip(labels, self.ax_boxes):
            tb = TextBox(axb, f"{name}: ", initial=default)
            for spine in axb.spines.values():
                spine.set_edgecolor("0.7")
            self.textboxes[name] = tb

        self.btn_apply = Button(self.ax_btn, "Apply", hovercolor="0.92")
        self.btn_apply.on_clicked(self.on_apply)

        self.ax_speed_label.text(0.5, 0.7, "Speed (step/frame)", ha="center", va="center", fontsize=10)
        self.slider_speed = Slider(self.ax_speed, "", 1, 10, valinit=1, valstep=1)
        self.slider_speed.on_changed(self.on_speed_change)

    def _read_params_from_ui(self):
        return {k: float(tb.text.strip()) for k, tb in self.textboxes.items()}

    # -------- animasyon --------
    def init_anim(self):
        for art in self._artists.values():
            art.set_data([], [])
        self._frame_idx = 0
        return tuple(self._artists.values())

    def update_anim(self, _i):
        i = self._frame_idx % self.TOTAL
        self._frame_idx += int(self.speed_step)
        t_disp = self.T_phys / max(1, int(self.speed_step))

        A = self._artists
        a4 = self.params["a4"]

        x_now, y_now = self.X[i], self.Y[i]
        A["trace_line"].set_data(self.X[:i+1], self.Y[:i+1])
        A["marker"].set_data([x_now], [y_now])

        # --- Linkler A0A, AB, BC ---
        A["A0A_line"].set_data([0, self.Ax[i]], [0, self.Ay[i]])
        A["AB_line"].set_data([self.Ax[i], self.Bx[i]], [self.Ay[i], self.By[i]])
        A["BC_line"].set_data([self.Bx[i], self.Cx[i]], [self.By[i], self.Cy[i]])

        # --- CD: BC ile aynı doğrultuda ve sabit uzunlukta a4 ---
        v_bc_x = self.Cx[i] - self.Bx[i]
        v_bc_y = self.Cy[i] - self.By[i]
        norm_bc = math.hypot(v_bc_x, v_bc_y)
        if norm_bc < 1e-9:
            ux, uy = 1.0, 0.0  # dejenere durumda keyfi yön
        else:
            ux, uy = v_bc_x / norm_bc, v_bc_y / norm_bc

        Dx_exp = self.Cx[i] + a4 * ux
        Dy_exp = self.Cy[i] + a4 * uy
        A["CD_line"].set_data([self.Cx[i], Dx_exp], [self.Cy[i], Dy_exp])

        # doğrulama: gerçek uç nokta (x_now,y_now) ile kıyasla
        v_cd_real_x = x_now - self.Cx[i]
        v_cd_real_y = y_now - self.Cy[i]
        len_err = abs(math.hypot(v_cd_real_x, v_cd_real_y) - a4)

        # yön farkı (derece)
        norm_real = math.hypot(v_cd_real_x, v_cd_real_y)
        if norm_real < 1e-9 or norm_bc < 1e-9:
            ang_err_deg = 0.0
        else:
            dot = (v_cd_real_x * ux + v_cd_real_y * uy) / norm_real
            dot = max(-1.0, min(1.0, dot))
            ang_err_deg = abs(math.degrees(math.acos(dot)))

        # tolerans aşılırsa CD'yi kırmızı renkle göster
        if (len_err > self.LEN_TOL) or (ang_err_deg > self.ANG_TOL_DEG):
            A["CD_line"].set_color('r')
            # bir kerelik uyarı yaz (çok sık yazmasın diye i % 50 kontrolü)
            if i % 50 == 0:
                print(f"[UYARI] a4 koşulu tutmuyor: |ΔL|={len_err:.3f}, Δθ={ang_err_deg:.2f}°")
        else:
            A["CD_line"].set_color('c')

        # Noktalar
        A["A0_point"].set_data([0], [0])
        A["A_point"].set_data([self.Ax[i]], [self.Ay[i]])
        A["B_point"].set_data([self.Bx[i]], [self.By[i]])
        A["C_point"].set_data([self.Cx[i]], [self.Cy[i]])

        # zaman grafikleri
        A["theta1_line"].set_data(t_disp[:i+1], self.TH1_unwrapped_deg[:i+1])
        A["theta1_dot"].set_data([t_disp[i]], [self.TH1_unwrapped_deg[i]])
        self._apply_time_xlim(self.ax_th1)

        A["s23_line"].set_data(t_disp[:i+1], self.S23[:i+1])
        A["s23_dot"].set_data([t_disp[i]], [self.S23[i]])
        self._apply_time_xlim(self.ax_s23)

        return tuple(A.values())

    def start_animation(self):
        if hasattr(self, "ani") and self.ani is not None:
            try:
                self.ani.event_source.stop()
            except Exception:
                pass
            self.ani = None
        self.ani = animation.FuncAnimation(
            self.fig, self.update_anim, frames=self.TOTAL,
            init_func=self.init_anim, interval=self.INTERVAL_MS_DEFAULT,
            blit=True, repeat=True
        )
        self.fig._ani_ref = self.ani
        self.fig.canvas.draw_idle()

    # -------- callbacks --------
    def on_speed_change(self, val):
        self.speed_step = max(1, int(val))
        self._apply_time_xlim(self.ax_th1)
        self._apply_time_xlim(self.ax_s23)
        self.fig.canvas.draw_idle()

    def on_apply(self, event):
        try:
            self.params.update(self._read_params_from_ui())
            self.compute()
            self.build_plots()
            self._frame_idx = 0
            self.start_animation()
        except Exception as e:
            print("Hata (Apply):", e)

def main():
    SimApp()
    plt.show()

if __name__ == "__main__":
    main()
