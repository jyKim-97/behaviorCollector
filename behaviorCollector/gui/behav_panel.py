
from PyQt5.QtWidgets import (
    QGraphicsView, QGraphicsScene, QWidget,
    QHBoxLayout, QVBoxLayout, QComboBox, QPlainTextEdit,
    QFormLayout, QPushButton, QLineEdit, QLabel,
    QScrollArea, QGraphicsLineItem, QSizePolicy,
    QFileDialog, QGraphicsTextItem,
    QMessageBox, QDialog, QCheckBox
)

from PyQt5.QtCore import Qt, pyqtSignal, QLineF, QRectF
from PyQt5.QtGui import QPen, QColor, QKeySequence, QFont, QPainter
from collections import OrderedDict


pyqt_KEY_MAP = OrderedDict({  
            Qt.Key_Q: 0,
            Qt.Key_W: 1,
            Qt.Key_E: 2,
            Qt.Key_R: 3,
            Qt.Key_T: 4,
            Qt.Key_A: 5,
            Qt.Key_S: 6,
            Qt.Key_D: 7,
            Qt.Key_F: 8,
            Qt.Key_G: 9,
            Qt.Key_1: 10,
            Qt.Key_2: 11,
            Qt.Key_3: 12,
            Qt.Key_4: 13,
            Qt.Key_5: 14,
            Qt.Key_6: 15,
            Qt.Key_7: 16,
            Qt.Key_8: 17,
            Qt.Key_Z: 18, # -> quit collecting
            Qt.Key_X: 19 # -> remove collecting
        })


from .video_controller import Controller
from .behav_viewer import BehavViewer
from .utils_gui import ColorPicker, tqdm_qt, error2messagebox
from .config_menu import MenuBuilder

from ..processing.behav_container import BehavCollector, BEHAV_TYPES, EVENT, STATE
from ..processing.behav_extractor import BehavExtractor
import re


CURRENT_KEY_ID = 0
KEEP_TIME_MS = []
LAST_ACTIVE_KEY = []        
        
class BehavItemRow(QPushButton):
    
    clicked_with_key = pyqtSignal(int)
    
    def __init__(self, behav_key: str, behav_name: str, behav_type: str, behav_color: str, parent=None):
        
        global CURRENT_KEY_ID
        self.key_id = CURRENT_KEY_ID
        CURRENT_KEY_ID += 1
        KEEP_TIME_MS.append(-1)
        
        super().__init__(parent)
        self.setCheckable(True)
        self.setLayout(QHBoxLayout())
        self.setFixedHeight(60)

        self.behav_name = behav_name
        self.behav_type = behav_type
        self.behav_key = behav_key
        self.clicked.connect(self.on_clicked)
        
        self.label_name = QLabel(f"({self.behav_key}) {behav_name} [{behav_type}]")
        
        self.color_box = QLabel()
        self.color_box.setFixedSize(30, 20)
        self.color_box.setStyleSheet(f"background-color: {behav_color}; border: 1px solid black;")

        self.layout().addWidget(self.label_name)
        self.layout().addWidget(self.color_box)

        self.setCheckable(True)
        self.finding_timepoints = False
        
    def modify_info(self, behav_name, behav_type, behav_color):
        self.label_name.setText(f"({self.behav_key}) {behav_name} [{behav_type}]")
        self.color_box.setStyleSheet(f"background-color: {behav_color}; border: 1px solid black;")
        
    def on_clicked(self):
        self.clicked_with_key.emit(self.key_id)


class BehaviorSelectionDialog(QDialog):
    """Dialog for selecting which behaviors (all epochs) to export."""

    def __init__(self, bcollector: BehavCollector, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Behaviors to Export")
        self.setMinimumSize(400, 300)
        self.checkboxes = []

        layout = QVBoxLayout()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        content_layout = QVBoxLayout()

        has_epoch = False
        for behav_idx, behav in enumerate(bcollector.behav_set):
            if not behav.time_ms:
                continue

            # show total count for quick overview
            label = f"{behav.name} ({behav.type}) - {len(behav.time_ms)} epoch(s)"
            cb = QCheckBox(label)
            cb.setChecked(True)
            content_layout.addWidget(cb)
            self.checkboxes.append((cb, behav_idx))
            has_epoch = True

        content_layout.addStretch()
        scroll_content.setLayout(content_layout)
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)

        button_row = QHBoxLayout()
        self.button_select_all = QPushButton("Select All")
        self.button_clear_all = QPushButton("Clear All")
        self.button_export = QPushButton("Export")
        self.button_cancel = QPushButton("Cancel")

        self.button_select_all.clicked.connect(self._select_all)
        self.button_clear_all.clicked.connect(self._clear_all)
        self.button_export.clicked.connect(self._accept_if_any)
        self.button_cancel.clicked.connect(self.reject)

        button_row.addWidget(self.button_select_all)
        button_row.addWidget(self.button_clear_all)
        button_row.addStretch()
        button_row.addWidget(self.button_cancel)
        button_row.addWidget(self.button_export)
        layout.addLayout(button_row)

        self.setLayout(layout)

        if not has_epoch:
            self.button_export.setEnabled(False)
            self.button_select_all.setEnabled(False)
            self.button_clear_all.setEnabled(False)

    def _select_all(self):
        for cb, _ in self.checkboxes:
            cb.setChecked(True)

    def _clear_all(self):
        for cb, _ in self.checkboxes:
            cb.setChecked(False)

    def _accept_if_any(self):
        if any(cb.isChecked() for cb, _ in self.checkboxes):
            self.accept()
        else:
            QMessageBox.warning(self, "No Selection", "Please select at least one behavior to export.")

    def selected_epochs(self):
        selections = {}
        for cb, behav_idx in self.checkboxes:
            if cb.isChecked():
                # select all epochs for this behavior
                selections[behav_idx] = None
        return selections


class BehavPanel(QWidget):
    
    signal_add_line = pyqtSignal(int, str, int, int) # key_id, color code, time_ms_start, time_ms_end
    signal_saved = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self._init_ui()
        self.bcollector = None
        self.video_controller = None
        self.behav_viewer = None
        self.is_modifying = False
        self.current_selection = -1
        self.duration_ms = 0
        # self._reset_keep()
        
    def _init_ui(self):
        def _set_label(lb):
            label = QLabel(lb)
            return label
        
        layout = QHBoxLayout()
        
        layout_v = QVBoxLayout()
        
        layout_form = QFormLayout()
        self.text_name = QLineEdit()
        self.comb_type = QComboBox()
        self.color_picker = ColorPicker()
        self.color_picker.setFixedSize(80, 20)
        self.button_add = QPushButton("Add Behavior")
        self.button_clear = QPushButton("Clear Input")
        self.text_note = QPlainTextEdit()
        
        layout_form.addRow(_set_label("Behavior Name"), self.text_name)
        layout_form.addRow(_set_label("Behavior type"), self.comb_type)
        layout_form.addRow(_set_label("Color identifier"), self.color_picker)
        layout_form.addRow(_set_label("Note"), self.text_note)
        self._reset_input()
        
        self.comb_type.addItems(BEHAV_TYPES)
        self.button_add.clicked.connect(self.add_behav)
        self.button_clear.clicked.connect(self.clear_behav)
        
        layout_v.addLayout(layout_form)
        layout_v.addWidget(self.button_add)
        layout_v.addWidget(self.button_clear)
        layout.addLayout(layout_v)
        
        layout_v2 = QVBoxLayout()
        self.label = QLabel("Behavior List")
        layout_v2.addWidget(self.label)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout()
        self.scroll_widget.setLayout(self.scroll_layout)

        self.scroll_area.setWidget(self.scroll_widget)
        layout_v2.addWidget(self.scroll_area, stretch=7)
        self.behav_rows = []
        
        layout.addLayout(layout_v2)
        self.setLayout(layout)
        
    def connect_menubar(self, menubar: MenuBuilder):
        menubar.load_header_requested.connect(self.load_behavior_header)
        menubar.load_behav_requested.connect(self.load_behavior)
        menubar.save_header_requested.connect(self.export_behavior_header)
        menubar.save_behav_requested.connect(self.export_behavior)
        menubar.export_epochs_requested.connect(self.export_epochs)
        
    def connect_controller(self, video_control_obj: Controller):
        self.video_controller = video_control_obj
        self.video_controller.duration_updated.connect(self._update_duration)
        
    def connect_behav_viewer(self, behave_viewer_obj: BehavViewer):
        self.behav_viewer = behave_viewer_obj        
    
    @error2messagebox(to_warn=True)
    def add_behav(self, checked=False):
        if self.bcollector is None:
            if self.video_controller.num_video == 0:
                raise ValueError("Please load the video first")
            else:
                self.bcollector = BehavCollector()
        
        name = self.text_name.text()
        type = self.comb_type.currentText()
        note = self.text_note.toPlainText()
        color_code = self.color_picker.color()
        color_hex = color_code.name()
        
        if name == "":
            raise ValueError("Behavior name cannot be empty")
        
        if bool(re.search(r'[\\/:*?"<>|]', name)):
            raise ValueError("Behavior name cannot contain special characters: \\ / : * ? \" < > |")
        
        if self.is_modifying:
            for bid, row in enumerate(self.behav_rows):
                if row.isChecked():
                    break
                
            self.bcollector.set_value(bid, "name", name)
            self.bcollector.set_value(bid, "type", type)
            self.bcollector.set_value(bid, "note", note)
            self.bcollector.set_value(bid, "color_code", color_hex)
            row.modify_info(type, name, color_hex)
            
            self.button_add.setText("Add Behavior")
            self.button_clear.setText("Clear Input")
            
            self.is_modifying = False
            row.setChecked(False)
            
        else:    
            bid = self.bcollector.num
            if bid == len(pyqt_KEY_MAP)-2:
                raise ValueError(f"Maximum number of behaviors reached ({bid}). Cannot add more.")
            
            self.bcollector.add_behav(
                name=name,
                type=type,
                note=note,
                color_code=color_hex
            )
            
            key = list(pyqt_KEY_MAP.keys())[bid]
            key_str = QKeySequence(key).toString()

            row = BehavItemRow(
                key_str, name, type, color_hex
            )
            self.scroll_layout.addWidget(row)
            self.behav_rows.append(row)
            row.clicked_with_key.connect(self.modify_behav)
            
        self._reset_input()
        
    def _toggle_modifying(self, key_id):
        if self.is_modifying:
            self.button_add.setText("Add Behavior")
            self.button_clear.setText("Clear Input")
            self.is_modifying = False
            self.current_selection = -1
            self._reset_input()
        else:
            self.button_add.setText("Modify Behavior")
            self.button_clear.setText("Remove Behavior")
            self.is_modifying = True
            self.current_selection = key_id
    
    @error2messagebox(to_warn=True)
    def modify_behav(self, key_id: int):
        if self.is_modifying:
            if key_id != self.current_selection:
                self.behav_rows[key_id].setChecked(False)
                raise ValueError("Please apply the modification first")
            else:
                self._toggle_modifying(key_id)
            
        else:
            self._toggle_modifying(key_id)
            self.text_name.setText(self.bcollector.get_name(key_id))
            self.comb_type.setCurrentText(self.bcollector.get_type(key_id))
            self.text_note.setPlainText(self.bcollector.get_note(key_id))
            self.color_picker.setColor(QColor(self.bcollector.get_color(key_id)))
            
    @error2messagebox(to_warn=True)
    def clear_behav(self, checked=False):
        if self.is_modifying:
            
            key_id = self.current_selection
            row = self.behav_rows[key_id]
            
            self.scroll_layout.removeWidget(row)
            row.setParent(None)
            self.behav_rows.remove(row)
            self.bcollector.delete_behav(key_id)
            # TODO: need to reorganize the key and mapped shortcut
            
            self._toggle_modifying(key_id)
        else:
            self._reset_input()
        
    @error2messagebox(to_warn=True)
    def load_behavior(self):
        path_dir = QFileDialog.getExistingDirectory(self, "Select behavior directory")
        if path_dir:
            # if self.bcollector is not None:
            #     raise ValueError("Behavior collector already loaded. Please create a new instance.")
            
            if self.video_controller.num_video == 0:
                raise ValueError("Please load the video first")

            self.bcollector = BehavCollector.load(path_dir)
            if self.bcollector.num == 0:
                self.bcollector = None
                raise ValueError("No behavior data found in the selected directory.")
                
            self._add_behav_set()
            for n in range(self.bcollector.num):
                for time_ms in self.bcollector.get_value(n, "time_ms"):
                    self._add_behav_time(n, time_ms, add_to_collector=False)    

            self._compare_item_number()
    
    def _compare_item_number(self):
        # double-check if the number of items in behav_viewer matches the bcollector
        for n in range(self.bcollector.num):
            if self.bcollector.behav_set[n].num != self.behav_viewer.num_items[n]:
                raise ValueError(f"Behavior {n} is not loaded correctly (%d/%d). Please check the data."%(
                    self.bcollector.behav_set[n].num, self.behav_viewer.num_items[n]
                ))
    
    @error2messagebox(to_warn=True)
    def load_behavior_header(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Behavior header",
                                                   "", "Behavior headers (*.json)")
        if file_path:
            # if self.bcollector is not None:
            #     raise ValueError("Behavior collector already loaded. Please create a new instance.")
            self.bcollector = BehavCollector.load_header(file_path)
            self._add_behav_set()
    
    @error2messagebox(to_warn=True)
    def export_behavior_header(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Behavior header", "", "Behavior headers (*.json)")
        if file_path:
            if self.bcollector.save_header(file_path):
                QMessageBox.information(self, "Success", "Behavior header saved successfully.")
    
    @error2messagebox(to_warn=True)
    def export_behavior(self):
        path_dir = QFileDialog.getExistingDirectory(self, "Select behavior directory")
        if path_dir:
            self.bcollector.update_video_path(self.video_controller.current_video_path)
            if self.bcollector.save(path_dir):
                self.signal_saved.emit()
                QMessageBox.information(self, "Success", "Behavior data saved successfully.")
    
    @error2messagebox(to_warn=True)
    def export_epochs(self):
        if self.bcollector is None or self.bcollector.num == 0:
            raise ValueError("No behavior data to export. Please load or create behaviors first.")

        if not any(b.time_ms for b in self.bcollector.behav_set):
            raise ValueError("There are no behavior epochs to export.")

        dialog = BehaviorSelectionDialog(self.bcollector, parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return

        selections = dialog.selected_epochs()
        if not selections:
            raise ValueError("No behaviors selected for export.")

        path_dir = QFileDialog.getExistingDirectory(self, "Select export directory")
        if path_dir:
            self.bcollector.update_video_path(self.video_controller.current_video_path)
            extractor = BehavExtractor(self.bcollector)
            if extractor.extract_epochs(path_dir, selections=selections, tqdm_fn=tqdm_qt):
                QMessageBox.information(self, "Success", "Selected behavior epochs exported successfully.")
        
    def _add_behav_set(self):
        existing_keys = [b.behav_key for b in self.behav_rows]
        for n in range(self.bcollector.num):
            # add behavior
            key = list(pyqt_KEY_MAP.keys())[n]
            key_str = QKeySequence(key).toString()
            if key_str in existing_keys:
                continue
            existing_keys.append(key_str)
            
            row = BehavItemRow(
                key_str,
                self.bcollector.get_name(n),
                self.bcollector.get_type(n),
                self.bcollector.get_color(n)
            )
            
            self.scroll_layout.addWidget(row)
            self.behav_rows.append(row)
            row.clicked_with_key.connect(self.modify_behav)
    
    def _reset_input(self):
        self.text_name.clear()
        self.text_note.clear()
        self.color_picker.setColor(QColor(255,255,255))
        
    def _add_behav_time(self, key_id, time_ms, add_to_collector=True):
        if add_to_collector:
            # TODO: synchronize behavior time rather than updating each time
            self.bcollector.add_behav_time(key_id, time_ms) 
            
        if not isinstance(time_ms, list):
            _time_ms = [time_ms, time_ms+1]
        else:
            _time_ms = time_ms
        
        self.behav_viewer.add_item(key_id, 
            self.bcollector.get_color(key_id),
            _time_ms[0], _time_ms[1])
        
        if add_to_collector:
            self._compare_item_number() # check
    
    def _keep_behav_time(self, key_id):
        if key_id >= CURRENT_KEY_ID:
            raise ValueError("Unexpected key_id")
        
        tp = self.bcollector.get_type(key_id)
        t0 = self.current
        if tp == EVENT:
            self._add_behav_time(key_id, t0)
        elif tp == STATE:
            if KEEP_TIME_MS[key_id] == -1: # stack new time_ms
                KEEP_TIME_MS[key_id] = t0
                LAST_ACTIVE_KEY.append(key_id)
                self.behav_rows[key_id].setChecked(True)
                self.behav_rows[key_id].finding_timepoints = True
            else: # add time range
                tr = [KEEP_TIME_MS[key_id], t0]
                if tr[1] < tr[0]: tr[0], tr[1] = tr[1], tr[0] 
                self._add_behav_time(key_id, tr)
                self._undo_keep(key_id=key_id)
        else:
            raise ValueError(f"Unexpected type {tp}")    
    
    def handle_key_input(self, event):
        key = event.key()
        if key == Qt.Key_Z: # Undo
            self._undo_keep()
        elif key == Qt.Key_X:
            self._delete_behav(self.current)
        else:
            key_id = pyqt_KEY_MAP[key]
            self._keep_behav_time(key_id)
            
    def _undo_keep(self, key_id=-1):
        if len(LAST_ACTIVE_KEY) == 0:
            return
        
        if key_id == -1:
            key_id = LAST_ACTIVE_KEY.pop()
        else:
            LAST_ACTIVE_KEY.remove(key_id)
            
        KEEP_TIME_MS[key_id] = -1
        if self.is_modifying:
            self._toggle_modifying(-1)
            
        self.behav_rows[key_id].setChecked(False)
        self.behav_rows[key_id].finding_timepoints = False
            
    def _reset_keep(self):
        # self.keep_time_ms = -1
        # self.key_id_activate = -1
        raise ValueError("Deprecated")
        
    def _delete_behav(self, time_ms):
        # TODO: tracking both bcollector and behavior viewer is dangerous.
        self.bcollector.delete_behav_time(time_ms)
        self.behav_viewer.delete_item(time_ms)
        
    def _update_duration(self, duration_ms):
        self.duration_ms = duration_ms
        if self.behav_viewer is not None:
            self.behav_viewer.update_duration(duration_ms)
        
    @property
    def current(self):
        return self.video_controller.current
                
