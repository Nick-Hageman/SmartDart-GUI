import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import cv2
import subprocess
import numpy as np
from predict import bboxes_to_xy
from dataset.annotate import draw
from train import build_model
from yacs.config import CfgNode as CN
import os.path as osp
import itertools

from dataset.annotate import draw, get_dart_scores
from firebase_client import FirebaseClient
import uuid

class DartScoringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dart Scoring System")
        self.root.geometry("800x1280")
        self.root.configure(bg="black")
        self.game_id = ""

        self.menu_frame = None
        self.game_frame = None
        self.logo_label = None
        self.flash_label = None
        self.flash_colors = itertools.cycle(["white", "grey"])
        self.flash_job = None

        self.root.bind("<KeyPress-l>", self.start_game)
        self.root.bind("<KeyPress-r>", self.reset_to_menu)
        self.root.bind("<KeyPress-m>", self.on_score_submit)

        self.show_main_menu()

    def show_main_menu(self):
        if self.game_frame:
            self.game_frame.destroy()

        self.menu_frame = tk.Frame(self.root, bg="black")
        self.menu_frame.pack(expand=True, fill="both")

        title_label = tk.Label(self.menu_frame, text="SMARTDART", font=("Arial", 48, "bold"), fg="white", bg="black")
        title_label.pack(pady=(60, 20))

        # Display animated GIF
        self.gif_frames = [ImageTk.PhotoImage(img) for img in self.load_gif("darts.gif")]
        self.gif_label = tk.Label(self.menu_frame, bg="black")
        self.gif_label.pack()
        self.animate_gif(0)

        self.flash_label = tk.Label(self.menu_frame, text="Press L to start", font=("Arial", 24), fg="white", bg="black")
        self.flash_label.pack()

        self.flash_text()

    def flash_text(self):
        if self.flash_label:
            new_color = next(self.flash_colors)
            self.flash_label.config(fg=new_color)
            self.flash_job = self.root.after(500, self.flash_text)

    def load_gif(self, filepath):
        from PIL import Image, ImageSequence
        gif = Image.open(filepath)
        return [frame.copy().resize((1000, 1000)) for frame in ImageSequence.Iterator(gif)]

    def animate_gif(self, frame_index):
        if hasattr(self, 'gif_frames') and self.gif_frames:
            frame = self.gif_frames[frame_index]
            self.gif_label.config(image=frame)
            self.root.after(100, self.animate_gif, (frame_index + 1) % len(self.gif_frames))

    def stop_flashing(self):
        if self.flash_job:
            self.root.after_cancel(self.flash_job)
            self.flash_job = None

    def reset_to_menu(self, event=None):
        self.running = False
        try:
            self.firebase.stop_all_streams()
        except:
            pass
        self.stop_flashing()
        self.show_main_menu()

    def start_game(self, event=None):
        self.menu_frame.destroy()
        self.flash_label = None
        self.logo_label = None
        self.flash_job = None

        self.current_player = 1
        self.round_counter = 1

        self.cfg = self.load_config()
        self.yolo = build_model(self.cfg)
        # self.yolo.load_weights(osp.join('old-models', self.cfg.model.name, 'weights'), self.cfg.model.weights_type)
        # self.yolo.load_weights(osp.join('models', self.cfg.model.name, 'weights'), self.cfg.model.weights_type)
        self.yolo.load_weights(osp.join('final-final-models', self.cfg.model.name, 'weights'), self.cfg.model.weights_type)

        self.game_frame = tk.Frame(self.root, bg="black")
        self.game_frame.pack(fill="both", expand=True)

        # game_id = str(uuid.uuid4())
        self.game_id = str(uuid.uuid4())[:8]

        self.setup_ui()
        self.running = True
        self.update_video_stream()

        self.firebase = FirebaseClient(self.game_id)

        self.firebase = FirebaseClient(self.game_id)
        self.firebase.stream_scores("player1", self.update_table_from_firebase)
        self.firebase.stream_scores("player2", self.update_table_from_firebase)

    def setup_ui(self):
        self.logo_side = tk.Label(self.game_frame, bg="black")
        self.logo_img_small = ImageTk.PhotoImage(Image.open("placeholder.png").resize((80, 80)))
        self.logo_side.config(image=self.logo_img_small)
        self.logo_side.place(x=10, y=10)

        self.player1_frame = tk.Frame(self.game_frame, bg="#222")
        self.player1_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(self.player1_frame, text="Player 1", font=("Arial", 16), fg="white", bg="#222").pack()
        self.p1_darts_frame = tk.Frame(self.player1_frame, bg="#222")
        self.p1_darts_frame.pack()

        self.dart1_p1 = tk.Label(self.p1_darts_frame, text="Dart 1: 0", font=("Arial", 14), bg="#444", fg="white", width=20)
        self.dart2_p1 = tk.Label(self.p1_darts_frame, text="Dart 2: 0", font=("Arial", 14), bg="#444", fg="white", width=20)
        self.dart3_p1 = tk.Label(self.p1_darts_frame, text="Dart 3: 0", font=("Arial", 14), bg="#444", fg="white", width=20)
        self.dart1_p1.pack(side="left", padx=5, pady=5)
        self.dart2_p1.pack(side="left", padx=5, pady=5)
        self.dart3_p1.pack(side="left", padx=5, pady=5)

        self.table_p1 = ttk.Treeview(self.player1_frame, columns=("D1", "D2", "D3", "Total"), show="headings", height=4)
        for col in ("D1", "D2", "D3", "Total"):
            self.table_p1.heading(col, text=col)
            self.table_p1.column(col, width=80)
        self.table_p1.pack(padx=10, pady=5)

        self.video_frame = tk.Label(self.game_frame, bg="black")
        self.video_frame.pack(pady=5)

        self.player2_frame = tk.Frame(self.game_frame, bg="#222")
        self.player2_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(self.player2_frame, text="Player 2", font=("Arial", 16), fg="white", bg="#222").pack()
        self.p2_darts_frame = tk.Frame(self.player2_frame, bg="#222")
        self.p2_darts_frame.pack()

        self.dart1_p2 = tk.Label(self.p2_darts_frame, text="Dart 1: 0", font=("Arial", 14), bg="#444", fg="white", width=20)
        self.dart2_p2 = tk.Label(self.p2_darts_frame, text="Dart 2: 0", font=("Arial", 14), bg="#444", fg="white", width=20)
        self.dart3_p2 = tk.Label(self.p2_darts_frame, text="Dart 3: 0", font=("Arial", 14), bg="#444", fg="white", width=20)
        self.dart1_p2.pack(side="left", padx=5, pady=5)
        self.dart2_p2.pack(side="left", padx=5, pady=5)
        self.dart3_p2.pack(side="left", padx=5, pady=5)

        self.table_p2 = ttk.Treeview(self.player2_frame, columns=("D1", "D2", "D3", "Total"), show="headings", height=4)
        for col in ("D1", "D2", "D3", "Total"):
            self.table_p2.heading(col, text=col)
            self.table_p2.column(col, width=80)
        self.table_p2.pack(padx=10, pady=5)
        
        self.uuid_label = tk.Label(
            self.game_frame,
            text=f"Game code: {self.game_id}",
            font=("Arial", 14),
            fg="white",
            bg="black"
        )
        self.uuid_label.pack(side="bottom", pady=10)

    def update_video_stream(self):
        def capture_frame():
            while self.running:
                try:
                    subprocess.run(["/home/pi/Desktop/Automatic-Darts-Scoring/Server/myenv/bin/python3", "capture.py"])
                    frame = cv2.imread("frame.jpg")
                    if frame is None:
                        continue

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    bboxes = self.yolo.predict(frame_rgb)
                    preds = bboxes_to_xy(bboxes, max_darts=3)
                    xy = preds[preds[:, -1] == 1]
                    frame_annotated = draw(frame_rgb.copy(), xy[:, :2], self.cfg, circles=False, score=True)

                    dart_scores = get_dart_scores(preds[:, :2], self.cfg, numeric=True)
                    self.update_dart_score_labels(dart_scores)

                    img = Image.fromarray(frame_annotated).resize((600, 600))
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_frame.imgtk = imgtk
                    self.video_frame.configure(image=imgtk)

                except Exception as e:
                    print(f"Error in video stream: {e}")

        threading.Thread(target=capture_frame, daemon=True).start()

    def update_dart_score_labels(self, scores):
        scores = scores[:3] + [0] * (3 - len(scores))

        active_fg = "white"
        inactive_fg = "#888"

        # Move logo_side near the current player's label
        if self.current_player == 1:
            self.logo_side.place(x=700, y=60)  # Near Player 1 label
        else:
            self.logo_side.place(x=700, y=830)  # Near Player 2 label

        if self.current_player == 1:
            self.dart1_p1.config(text=f"Dart 1: {scores[0]}", fg=active_fg)
            self.dart2_p1.config(text=f"Dart 2: {scores[1]}", fg=active_fg)
            self.dart3_p1.config(text=f"Dart 3: {scores[2]}", fg=active_fg)
            self.dart1_p2.config(fg=inactive_fg)
            self.dart2_p2.config(fg=inactive_fg)
            self.dart3_p2.config(fg=inactive_fg)
        else:
            self.dart1_p2.config(text=f"Dart 1: {scores[0]}", fg=active_fg)
            self.dart2_p2.config(text=f"Dart 2: {scores[1]}", fg=active_fg)
            self.dart3_p2.config(text=f"Dart 3: {scores[2]}", fg=active_fg)
            self.dart1_p1.config(fg=inactive_fg)
            self.dart2_p1.config(fg=inactive_fg)
            self.dart3_p1.config(fg=inactive_fg)

    def on_score_submit(self, event=None):
        if self.current_player == 1:
            d1 = int(self.dart1_p1.cget("text").split(": ")[1])
            d2 = int(self.dart2_p1.cget("text").split(": ")[1])
            d3 = int(self.dart3_p1.cget("text").split(": ")[1])
            self.firebase.add_score("player1", [d1, d2, d3])
            self.current_player = 2
        else:
            d1 = int(self.dart1_p2.cget("text").split(": ")[1])
            d2 = int(self.dart2_p2.cget("text").split(": ")[1])
            d3 = int(self.dart3_p2.cget("text").split(": ")[1])
            self.firebase.add_score("player2", [d1, d2, d3])
            self.current_player = 1
            self.round_counter += 1

    def update_table_from_firebase(self, player, scores_data):
        table = self.table_p1 if player == "player1" else self.table_p2
        table.delete(*table.get_children())
        if not scores_data:
            return
        scores_list = []
        if isinstance(scores_data, dict):
            sorted_keys = sorted(scores_data.keys(), key=lambda x: int(x) if x.isdigit() else x)
            for key in sorted_keys:
                score = scores_data[key]
                if isinstance(score, list) and len(score) == 3:
                    scores_list.append(score)
                elif isinstance(score, dict) and all(k in score for k in ['0', '1', '2']):
                    scores_list.append([score['0'], score['1'], score['2']])
        elif isinstance(scores_data, list):
            for score in scores_data:
                if isinstance(score, list) and len(score) == 3:
                    scores_list.append(score)
                elif isinstance(score, dict) and all(k in score for k in ['0', '1', '2']):
                    scores_list.append([score['0'], score['1'], score['2']])
        for score in scores_list:
            if all(isinstance(x, (int, float)) for x in score):
                total = sum(score)
                table.insert('', 'end', values=(score[0], score[1], score[2], total))

    def load_config(self):
        cfg = CN(new_allowed=True)
        cfg.merge_from_file(osp.join('configs', 'deepdarts_utrecht.yaml'))
        cfg.model.name = 'deepdarts_utrecht'
        return cfg

    def on_closing(self):
        self.running = False
        try:
            self.firebase.stop_all_streams()
        except:
            pass
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DartScoringApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

