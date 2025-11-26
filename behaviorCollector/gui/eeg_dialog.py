import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout,
    QCheckBox, QDoubleSpinBox, QSizePolicy
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .utils_gui import error2messagebox


class EEGDialog(QDialog):
    def __init__(self, eeg_data, controller=None, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.raw_data = np.asarray(eeg_data["data"])
        self.times = np.asarray(eeg_data["times"]).squeeze()
        self.tdelay = float(np.asarray(eeg_data.get("tdelay_video (s)", 0)).squeeze())

        self._validate_data()

        self.num_channels = self.raw_data.shape[0]
        self.num_cbrains = self.raw_data.shape[2]

        self.setWindowTitle("EEG Viewer")
        self.setMinimumWidth(800)

        self._init_ui()
        self._connect_signals()
        self._update_time_label(self._current_video_time_s())
        self.update_plot()

    def _validate_data(self):
        if self.raw_data.ndim != 3:
            raise ValueError("EEG data should have 3 dimensions: (channels, time, CBRAIN).")
        if self.times.ndim != 1 and not (self.times.ndim == 2 and 1 in self.times.shape):
            raise ValueError("EEG times should be a 1-D array.")
        self.times = self.times.reshape(-1)
        if self.raw_data.shape[1] != self.times.shape[0]:
            raise ValueError("EEG time dimension does not match the provided times array.")

    def _init_ui(self):
        layout = QVBoxLayout()

        max_time = float(self.times.max() - self.tdelay)
        shape_text = f"EEG loaded | Max time: {max_time:.2f} s | tdelay_video: {self.tdelay:.3f}"
        self.shape_label = QLabel(shape_text)
        layout.addWidget(self.shape_label)

        layout.addLayout(self._build_selection_grid())
        layout.addLayout(self._build_control_row())

        self.canvas = FigureCanvas(Figure(figsize=(6, 4)))
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas, stretch=1)

        self.setLayout(layout)

    def _build_selection_grid(self):
        grid = QGridLayout()
        grid.addWidget(QLabel("Channel ID"), 0, 0, alignment=Qt.AlignCenter)
        grid.addWidget(QLabel("CBRAIN ID"), 1, 0, alignment=Qt.AlignCenter)

        self.channel_checks = []
        max_channel = min(5, self.num_channels)
        for idx in range(max_channel):
            chk = QCheckBox(str(idx + 1))
            chk.setChecked(idx == 0)
            chk.stateChanged.connect(self.update_plot)
            self.channel_checks.append(chk)
            grid.addWidget(chk, 0, idx + 1, alignment=Qt.AlignCenter)

        self.cbrain_checks = []
        max_cbrain = min(5, self.num_cbrains)
        for idx in range(max_cbrain):
            chk = QCheckBox(str(idx + 1))
            chk.setChecked(idx == 0)
            chk.stateChanged.connect(self.update_plot)
            self.cbrain_checks.append(chk)
            grid.addWidget(chk, 1, idx + 1, alignment=Qt.AlignCenter)

        return grid

    def _build_control_row(self):
        row = QHBoxLayout()

        self.ymin_box = QDoubleSpinBox()
        self.ymin_box.setRange(-1e3, 1e3)
        self.ymin_box.setDecimals(3)
        self.ymin_box.setSingleStep(0.05)
        self.ymin_box.setValue(-0.2)
        self.ymin_box.valueChanged.connect(self.update_plot)

        self.ymax_box = QDoubleSpinBox()
        self.ymax_box.setRange(-1e3, 1e3)
        self.ymax_box.setDecimals(3)
        self.ymax_box.setSingleStep(0.05)
        self.ymax_box.setValue(0.2)
        self.ymax_box.valueChanged.connect(self.update_plot)

        self.window_box = QDoubleSpinBox()
        self.window_box.setRange(0.05, 30)
        self.window_box.setDecimals(3)
        self.window_box.setSingleStep(0.05)
        self.window_box.setValue(0.5)
        self.window_box.valueChanged.connect(self.update_plot)

        self.time_label = QLabel("")
        row.addWidget(QLabel("ymin"))
        row.addWidget(self.ymin_box)
        row.addWidget(QLabel("ymax"))
        row.addWidget(self.ymax_box)
        row.addWidget(QLabel("+/- seconds around video time"))
        row.addWidget(self.window_box)
        row.addStretch(1)
        row.addWidget(self.time_label)

        return row

    def _connect_signals(self):
        if self.controller is not None:
            self.controller.position_updated.connect(self._on_video_position)

    def _disconnect_signals(self):
        if self.controller is not None:
            try:
                self.controller.position_updated.disconnect(self._on_video_position)
            except Exception:
                pass

    def selected_channels(self):
        return [idx + 1 for idx, chk in enumerate(self.channel_checks) if chk.isChecked()]

    def selected_cbrains(self):
        return [idx + 1 for idx, chk in enumerate(self.cbrain_checks) if chk.isChecked()]

    def selected_cbrain(self):
        """Backward-compatible helper for callers expecting a single selection."""
        cbrains = self.selected_cbrains()
        return cbrains[0] if cbrains else None

    def _current_video_time_s(self):
        if self.controller is None:
            return 0.0
        return float(self.controller.current) / 1000.0

    def _on_video_position(self, time_ms):
        self._update_time_label(time_ms / 1000.0)
        self.update_plot()

    def _update_time_label(self, time_s):
        self.time_label.setText(f"Video time: {time_s:.3f} s")

    @error2messagebox(to_warn=True)
    def update_plot(self, *args, **kwargs):
        channels = self.selected_channels()
        cbrain_ids = self.selected_cbrains()

        fig = self.canvas.figure
        fig.clf()

        if not channels or not cbrain_ids:
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, "Select Channel ID(s) and CBRAIN ID(s)", ha="center", va="center")
            ax.set_axis_off()
            self.canvas.draw_idle()
            return

        time_sec = self.times - self.tdelay
        center = self._current_video_time_s()
        window = self.window_box.value()
        mask = (time_sec >= center - window) & (time_sec <= center + window)

        ymin, ymax = self.ymin_box.value(), self.ymax_box.value()
        if ymin >= ymax:
            raise ValueError("ymin must be smaller than ymax.")
        subplot_total = len(channels)
        colors = ["#00509e", "#d1495b", "#2b9348", "#ff7b00", "#6a4c93"]

        for idx, ch in enumerate(channels):
            ax = fig.add_subplot(subplot_total, 1, idx + 1)
            for cb_idx, cbrain_id in enumerate(cbrain_ids):
                signal = self.raw_data[ch - 1, :, cbrain_id - 1].squeeze()
                color = colors[cb_idx % len(colors)]
                label = f"CBRAIN {cbrain_id}"
                if mask.any():
                    ax.plot(time_sec[mask], signal[mask], color=color, label=label, linewidth=1.2)
                else:
                    ax.text(0.5, 0.5, "No data in range", ha="center", va="center", transform=ax.transAxes)
            ax.set_ylabel(f"Ch {ch}")
            ax.set_ylim(ymin, ymax)
            if idx == subplot_total - 1:
                ax.set_xlabel("Time (s, video aligned)")
            else:
                ax.tick_params(labelbottom=False)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_xlim(center - window, center + window)
            ax.axvline(center, color="black", linestyle="-", linewidth=1)
            if len(cbrain_ids) > 1:
                ax.legend(loc="upper right", fontsize="small")

        fig.tight_layout()
        self.canvas.draw_idle()

    def closeEvent(self, event):
        self._disconnect_signals()
        return super().closeEvent(event)
