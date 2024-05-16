import customtkinter as ctk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import webbrowser
import openpyxl
import time
# import pyautogui
# import pygetwindow as gw

# スクリプトのディレクトリを取得
script_dir = os.path.dirname(os.path.abspath(__file__))

# アイコンファイルのフルパスを作成
icon_path = os.path.join(script_dir, 'ico.ico')

# CustomTkinterの外観モードをダークモードに設定
ctk.set_appearance_mode("dark")

# CTkアプリケーションを初期化
app = ctk.CTk()
app.title("Elgo_App")
# app.state("zoomed")
# app.attributes('-fullscreen', True)
app.geometry("{0}x{1}+0+0".format(app.winfo_screenwidth(), app.winfo_screenheight()))
# app.iconbitmap(icon_path)

# app.update()

# time.sleep(2)

# # Elgo_Appウィンドウを取得
# windows = gw.getWindowsWithTitle("Elgo_App")
# if windows:
#     elgo_app_window = windows[0]
#     elgo_app_window.activate()
#     time.sleep(0.5)  # ウィンドウがアクティブになるまで待機
#     # Windowsキーと↑キーを同時に押す
#     pyautogui.hotkey('win', 'up')
# else:
#     print("Elgo_Appウィンドウが見つかりませんでした")

# フレームの作成
frame1 = ctk.CTkFrame(master=app, fg_color='black')
frame2 = ctk.CTkFrame(master=app, fg_color='black')

# フレームをグリッドに配置して画面を2つに分割
frame1.grid(row=0, column=0, rowspan=2, sticky="nsew")
frame2.grid(row=0, column=1, rowspan=2, sticky="nsew")

# グリッドの行と列の比率を設定して、フレームがウィンドウ全体に広がるようにする
app.grid_rowconfigure(0, weight=1)
app.grid_columnconfigure(0, weight=1)
app.grid_columnconfigure(1, weight=1)

# frame1 の中に frame3 を作成して配置
frame3 = ctk.CTkFrame(master=frame1, height=80)
frame3.pack(side='bottom', fill='x')

frame4 = ctk.CTkFrame(master=frame2, height=80)
frame4.pack(side='bottom', fill='x')

frame5 = ctk.CTkFrame(master=frame1, height=50)
frame5.pack(side='bottom', fill='x')

frame6 = ctk.CTkFrame(master=frame2, height=50)
frame6.pack(side='bottom', fill='x')

frame7 = ctk.CTkFrame(master=frame1, height=50)
frame7.pack(side='top', fill='x')

frame8 = ctk.CTkFrame(master=frame2, height=50)
frame8.pack(side='top', fill='x')

# タイトルを追加する関数
title_label_1 = ctk.CTkLabel(master=frame7, text="Video Player", font=("Helvetica", 20, "bold"))
title_label_1.pack(side='left', pady=10, padx=5)

title_label_2 = ctk.CTkLabel(master=frame8, text="Joint Angle Plot", font=("Helvetica", 20, "bold"))
title_label_2.pack(side='left', pady=10, padx=5)

# MediaPipe Poseのセットアップ
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# グローバル変数
processed_frames = []
joint_angle_data = {
    'left_shoulder_angle': [],
    'right_shoulder_angle': [],
    'left_elbow_angle': [],
    'right_elbow_angle': [],
    'left_waist_angle': [],
    'right_waist_angle': [],
    'left_knee_angle': [],
    'right_knee_angle': []
}
is_playing = False
current_frame_index = 0

# 関節角度を計算する関数
def get_joint_angles(results):
    angles = {}

    # ランドマークの存在を確認
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # 関節の角度を計算するための関数を定義
        def calculate_angle(a, b, c):
            # ベクトルを計算
            ba = np.array([a.x, a.y, a.z]) - np.array([b.x, b.y, b.z])
            bc = np.array([c.x, c.y, c.z]) - np.array([b.x, b.y, b.z])

            # 内積とノルムを計算
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)

            # ラジアンから度に変換
            angle = np.degrees(angle)
            return angle

        # 各関節の角度を計算
        angles['left_shoulder_angle'] = 180 - calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

        angles['right_shoulder_angle'] = 180 - calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

        angles['left_elbow_angle'] = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

        angles['right_elbow_angle'] = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

        angles['left_waist_angle'] = 180 - calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

        angles['right_waist_angle'] = 180 - calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                                    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

        angles['left_knee_angle'] = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

        angles['right_knee_angle'] = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

    return angles

# 動画をアップロードして処理
def upload_and_process_video():
    global processed_frames, joint_angle_data, current_frame_index, is_playing
    
    # ファイルダイアログを表示して動画ファイルを選択
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mkv;*.mov")])
    if not file_path:
        return
    
    # ファイル名をEntryに表示
    show_file_name(file_path)

    # 動画キャプチャの初期化
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # 動画の全フレーム数を取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # フレームスキップ間隔
    frame_skip = 1
    # データの初期化
    processed_frames.clear()
    joint_angle_data = {
        'left_shoulder_angle': [],
        'right_shoulder_angle': [],
        'left_elbow_angle': [],
        'right_elbow_angle': [],
        'left_waist_angle': [],
        'right_waist_angle': [],
        'left_knee_angle': [],
        'right_knee_angle': []
    }
    current_frame_index = 0

    # フレームを前処理して保存
    while current_frame_index < total_frames:
        # フレームスキップ
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)

        ret, frame = cap.read()
        if not ret:
            break

        # フレームをRGBに変換
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 骨格を描画
        results = pose.process(frame_rgb)

        # フレームデータと結果を保存
        processed_frames.append((frame_rgb, results))

        # 関節角度を計算してリストに保存
        angles = get_joint_angles(results)
        for joint, angle in angles.items():
            joint_angle_data[joint].append(angle)

        # 現在のフレームインデックスを更新
        current_frame_index += frame_skip

        # 進捗をパーセントで計算
        progress_percentage = (current_frame_index / total_frames) * 100

        # 進捗をラベルに表示
        progress_label.configure(text=f"Processing: {progress_percentage:.2f}%")

        # ウィンドウを更新
        app.update_idletasks()

    # 動画キャプチャを解放
    cap.release()

    # スライダーの範囲を設定
    video_slider_1.configure(from_=0, to=len(processed_frames) - 1)
    
    # スライダーの範囲を設定
    video_slider_2.configure(from_=0, to=len(processed_frames) - 1)

    # グラフの描画
    draw_joint_angle_plot()

    # 再生を開始
    current_frame_index = 0
    is_playing = True
    play_video()

def show_file_name(file_path):
    # ファイルパスからファイル名の部分を取得
    file_name = os.path.basename(file_path)
    # CTkEntryにファイル名を表示
    movie_name_input.delete(0, "end")
    movie_name_input.insert(0, file_name)

# 動画再生ループ
def play_video():
    global current_frame_index, is_playing, processed_frames, video_slider_1, video_slider_2

    if is_playing and current_frame_index < len(processed_frames):
        frame_rgb, results = processed_frames[current_frame_index]
        draw_skeleton(frame_rgb, results)
        show_video_frame(frame_rgb)
        video_slider_1.set(current_frame_index)
        video_slider_2.set(current_frame_index)
        app.update_idletasks()
        current_frame_index += 2
        app.after(60, play_video)

def update_video_frame(frame_index):
    global processed_frames, current_frame_index, is_playing

    if frame_index < len(processed_frames):
        current_frame_index = frame_index
        frame_rgb, results = processed_frames[frame_index]
        draw_skeleton(frame_rgb, results)
        show_video_frame(frame_rgb)
    else:
        is_playing = False

# スライダー1の値が変更されたときのイベントハンドラ
def on_slider1_changed(value):
    global current_frame_index
    current_frame_index = int(value)
    update_video_frame(current_frame_index)
    # スライダー2の値をスライダー1に同期させる
    video_slider_2.set(value)

# スライダー2の値が変更されたときのイベントハンドラ
def on_slider2_changed(value):
    global current_frame_index
    current_frame_index = int(value)
    update_video_frame(current_frame_index)
    # スライダー1の値をスライダー2に同期させる
    video_slider_1.set(value)

# フレーム1にスライダー1を追加
video_slider_1 = ctk.CTkSlider(master=frame3, from_=0, to=1, command=on_slider1_changed)
video_slider_1.set(0)  # ここで初期値を設定
video_slider_1.pack(side='bottom', fill='x', pady=20, padx=70)

# フレーム2にスライダー2を追加
video_slider_2 = ctk.CTkSlider(master=frame4, from_=0, to=1, command=on_slider2_changed)
video_slider_2.set(0)  # ここで初期値を設定
video_slider_2.pack(side='bottom', fill='x', pady=20, padx=70)


# フレーム1に進捗を表示するラベルを追加
progress_label = ctk.CTkLabel(master=frame1, text="", font=("Helvetica", 20, "bold"))
progress_label.pack(expand=True)

upload_button = ctk.CTkButton(master=frame5, text="Upload Video", command=upload_and_process_video, font=("Helvetica", 14, "bold"), width=80)
upload_button.pack(side='left', padx=5)

# 骨格を描画するための関数
def draw_skeleton(frame_rgb, results):
    # 骨格のランドマークを取得
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame_rgb, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style())

def draw_joint_angle_plot():
    global joint_angle_data
    
    plt.style.use('dark_background')

    fig, ax = plt.subplots()

    time_axis = np.arange(len(joint_angle_data['left_shoulder_angle']))

    ax.clear()
    for joint, angle_data in joint_angle_data.items():
        ax.plot(time_axis, angle_data, label=joint)

    ax.set_xlabel('Frame')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('Joint Angles Over Time')

    ax.set_ylim([0, 180])

    ax.set_xlim([0, len(time_axis) - 1])
    
    ax.axhline(y=90, color='red', linestyle='--', label='90 degrees')  # 90度の定線を引く

    ax.legend(loc='upper right')

    canvas = FigureCanvasTkAgg(fig, master=frame2)
    canvas.get_tk_widget().pack(fill='both', expand=True)
    canvas.draw()


def draw_joint_angle_plot():
    global joint_angle_data
    
    plt.style.use('dark_background')

    fig, ax = plt.subplots()

    time_axis = np.arange(len(joint_angle_data['left_shoulder_angle']))

    ax.clear()
    for joint, angle_data in joint_angle_data.items():
        ax.plot(time_axis, angle_data, label=joint)

    ax.set_xlabel('Frame')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('Joint Angles Over Time')

    ax.set_ylim([0, 180])

    ax.set_xlim([0, len(time_axis) - 1])
    
    ax.axhline(y=90, color='red', linestyle='--', label='90 degrees')  # 90度の定線を引く

    ax.legend(loc='upper right')

    canvas = FigureCanvasTkAgg(fig, master=frame2)
    canvas.get_tk_widget().pack(fill='both', expand=True)
    canvas.draw()

# グラフをフィルターの値で更新
def draw_joint_angle_plot_refresh(selected_joint):
    global joint_angle_data
    
    for widget in frame2.winfo_children():
        if widget not in [video_slider_2, frame8, title_label_2, export_button, movie_button, filter_label, filter_angle_combobox, frame4, frame6]:  # スライダーとボタンを除外
            widget.destroy()
    
    plt.style.use('dark_background')

    fig, ax = plt.subplots()

    time_axis = np.arange(len(joint_angle_data['left_shoulder_angle']))

    ax.clear()
    if selected_joint == "all":
        for joint, angle_data in joint_angle_data.items():
            ax.plot(time_axis, angle_data, label=joint)
    else:
        ax.plot(time_axis, joint_angle_data[selected_joint], label=selected_joint)

    ax.set_xlabel('Frame')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('Joint Angles Over Time')

    ax.set_ylim([0, 180])

    ax.set_xlim([0, len(time_axis) - 1])
    
    ax.axhline(y=90, color='red', linestyle='--', label='90 degrees')  # 90度の定線を引く

    ax.legend(loc='upper right')

    canvas = FigureCanvasTkAgg(fig, master=frame2)
    canvas.get_tk_widget().pack(fill='both', expand=True)
    canvas.draw()

def on_combobox_select(event):
    selected_joint = filter_angle_combobox.get()
    draw_joint_angle_plot_refresh(selected_joint)

# フレーム1内に動画フレームを表示
def show_video_frame(frame_rgb):
    global label, slider, upload_button
    
    # フレーム1内のスライダー以外のウィジェットを削除
    for widget in frame1.winfo_children():
        if widget not in [video_slider_1, upload_button, title_label_1, play_pause_button, frame3, frame5, frame7]:  # スライダーとボタンを除外
            widget.destroy()
    
    # フレームのサイズを取得
    frame_width = frame1.winfo_width()
    frame_height = frame1.winfo_height()

    # 動画のアスペクト比を取得
    video_height, video_width, _ = frame_rgb.shape
    video_aspect_ratio = video_width / video_height

    # フレームのアスペクト比を取得
    frame_aspect_ratio = frame_width / frame_height

    # リサイズ方法を決定
    if video_aspect_ratio > frame_aspect_ratio:
        # 幅を基準にリサイズ
        new_width = frame_width
        new_height = int(new_width / video_aspect_ratio)
    else:
        # 高さを基準にリサイズ
        new_height = frame_height
        new_width = int(new_height * video_aspect_ratio)

    # 動画をリサイズ
    resized_frame = cv2.resize(frame_rgb, (new_width, new_height))

    # ImageTkオブジェクトを作成
    image = Image.fromarray(resized_frame)
    image_tk = ImageTk.PhotoImage(image=image)

    # 新しいラベルを作成
    show_video_frame.label = ctk.CTkLabel(master=frame1, width=frame_width, height=frame_height, text="")
    show_video_frame.label.pack(fill='both', expand=True)

    # 新しいラベルに画像を設定
    show_video_frame.label.configure(image=image_tk)
    show_video_frame.label.image = image_tk

# フレームサイズ調整関数
def adjust_frame_sizes():
    # ウィンドウが最大化されているかチェック
    if app.state() == "zoomed":
        # 現在のウィンドウサイズを取得
        window_width = app.winfo_width()
        window_height = app.winfo_height()

        # フレームのサイズを計算
        frame_width = window_width // 2
        frame_height = window_height

        # 各フレームのサイズを設定
        frame1.configure(width=frame_width, height=frame_height)
        frame2.configure(width=frame_width, height=frame_height)

        # フレームのグリッド配置を更新
        frame1.grid(row=0, column=0, rowspan=2, sticky="nsew")
        frame2.grid(row=0, column=1, rowspan=2, sticky="nsew")

        # グリッド行と列の比率設定
        app.grid_rowconfigure(0, weight=1)
        app.grid_columnconfigure(0, weight=1)
        app.grid_columnconfigure(1, weight=1)

# フレーム1で動画を再生または一時停止
def toggle_playback():
    global is_playing
    is_playing = not is_playing
    if is_playing:
        play_video()
    else:
        pause_video()

# 動画再生を一時停止する関数
def pause_video():
    global is_playing
    is_playing = False
    
# 動画とグラフを消す関数
def delete_video_plot():
    # フレーム1内のスライダー以外のウィジェットを削除
    for widget in frame1.winfo_children():
        if widget not in [video_slider_1, upload_button, title_label_1, play_pause_button, frame3, frame5, frame7]:  # スライダーとボタンを除外
            widget.destroy()
    
    # フレーム2内のウィジェットをすべて削除
    for widget in frame2.winfo_children():
        if widget not in [video_slider_2, upload_button, title_label_2, frame4, frame6, frame8]:
            widget.destroy()

    # 新しい進捗表示ラベルを作成して配置
    global progress_label
    progress_label = ctk.CTkLabel(master=frame1, text="", font=("Helvetica", 20, "bold"))
    progress_label.pack(expand=True)
    
    video_slider_1.set(0)  # ここでスライダーを初期値を設定
    video_slider_2.set(0)  # ここでスライダーを初期値を設定
    
    movie_name_input.delete(0, "end")
    
    filter_angle_combobox.set("all")


# Excel出力する関数
def export_to_excel():
    # ファイルダイアログを表示して保存先を選択
    file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")], initialfile="Elgo_Results_Sheet.xlsx")
    if not file_path:
        return

    # Excelワークブックとシートを作成
    wb = openpyxl.Workbook()
    
    # データシートを作成
    ws_data = wb.active
    ws_data.title = "data"

    # ヘッダーを追加
    headers = ['Time', 'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow', 'Left Waist', 'Right Waist', 'Left Knee', 'Right Knee']
    ws_data.append(headers)

    # データを追加
    for i in range(len(joint_angle_data['left_shoulder_angle'])):
        row = [
            i,  # 時間（フレーム番号）
            joint_angle_data['left_shoulder_angle'][i],
            joint_angle_data['right_shoulder_angle'][i],
            joint_angle_data['left_elbow_angle'][i],
            joint_angle_data['right_elbow_angle'][i],
            joint_angle_data['left_waist_angle'][i],
            joint_angle_data['right_waist_angle'][i],
            joint_angle_data['left_knee_angle'][i],
            joint_angle_data['right_knee_angle'][i]
        ]
        ws_data.append(row)

    # 新しいシートを追加してグラフを作成
    ws_result = wb.create_sheet(title="result")
    chart = openpyxl.chart.LineChart()
    chart.title = "Joint Angle Data"
    chart.x_axis.title = "Time"
    chart.y_axis.title = "Angle"
    data = openpyxl.chart.Reference(ws_data, min_col=2, min_row=1, max_col=9, max_row=len(joint_angle_data['left_shoulder_angle'])+1)
    cats = openpyxl.chart.Reference(ws_data, min_col=1, min_row=2, max_row=len(joint_angle_data['left_shoulder_angle'])+1)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(cats)
    ws_result.add_chart(chart, "A1")

    # Excelファイルを保存
    wb.save(file_path)
    print(f"Data exported to {file_path}")

def on_movie_button_click():
    # ファイルダイアログを表示して保存先を選択
    output_file = filedialog.asksaveasfilename(defaultextension=".avi", filetypes=[("AVI files", "*.avi")], initialfile="Elgo_Results_Movie")
    if not output_file:
        return  # キャンセルされた場合は何もしない

    # 出力する動画の幅と高さを取得
    height, width, _ = processed_frames[0][0].shape
    fps = 30  # フレームレート

    # 動画ライターを初期化
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 動画コーデックを設定
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # 処理されたフレームを動画に書き込む
    for frame, _ in processed_frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # RGBからBGRに変換
        out.write(frame_bgr)

    # 動画を解放
    out.release()
    print("Processed video saved successfully.")

# リンクを開く関数を定義
def open_link(url):
    webbrowser.open(url)

url = "https://toyotajp.sharepoint.com/:p:/r/sites/msspo_AL_ergo/_layouts/15/Doc.aspx?sourcedoc=%7BE12777F6-E931-4964-B616-41F05DF58533%7D&file=%25u8cc7%25u6599%25u4e00%25u62ec.pptx&action=edit&mobileredirect=true"

# フレーム1に再生、一時停止ボタンを追加
play_pause_button = ctk.CTkButton(master=frame5, text="Play/Pause", command=toggle_playback, font=("Helvetica", 14, "bold"), width=80)
play_pause_button.pack(side='left', padx=5)

# ファイル名のラベル
file_name_label = ctk.CTkLabel(master=frame5, text="File Name :", font=("Helvetica", 14, "bold"))
file_name_label.pack(side='left', padx=5)

# ファイル名をインプット
movie_name_input = ctk.CTkEntry(master=frame5, font=("Helvetica", 12, "bold"), width=200)
movie_name_input.pack(side='left', padx=5)

# 動画とグラフを消すボタン
delete_button = ctk.CTkButton(master=frame5, text="Delete", command=delete_video_plot, fg_color='magenta', font=("Helvetica", 14, "bold"), width=80)
delete_button.pack(side='right', padx=5)

# Excel出力
export_button = ctk.CTkButton(master=frame6, text="Excel", fg_color='green', font=("Helvetica", 14, "bold"), command=export_to_excel, width=80)
export_button.pack(side='right', padx=5)

# Movie出力
movie_button = ctk.CTkButton(master=frame6, text="Movie", fg_color='blue', font=("Helvetica", 14, "bold"), command=on_movie_button_click, width=80)
movie_button.pack(side='right', padx=5)

# 出力のラベル
export_label = ctk.CTkLabel(master=frame6, text="Export :", font=("Helvetica", 14, "bold"))
export_label.pack(side='right', padx=5)

# フィルターのラベル
filter_label = ctk.CTkLabel(master=frame6, text="Filter :", font=("Helvetica", 14, "bold"))
filter_label.pack(side='left', padx=5)

# コンボボックスのリスト
options = ["all", "left_shoulder_angle", "right_shoulder_angle", "left_elbow_angle", "right_elbow_angle", "left_waist_angle", "right_waist_angle", "left_knee_angle", "right_knee_angle"]

# フィルターコンボボックス
filter_angle_combobox = ctk.CTkComboBox(master=frame6, values=options, font=("Helvetica", 14, "bold"), width=200)
filter_angle_combobox.pack(side='left', padx=5)
filter_angle_combobox.bind("<<ComboboxSelected>>", on_combobox_select)

# Refreshボタン
refresh_button = ctk.CTkButton(master=frame6, text="Plot Refresh", font=("Helvetica", 14, "bold"), command=lambda: draw_joint_angle_plot_refresh(filter_angle_combobox.get()), width=80)
refresh_button.pack(side='left')

link_button = ctk.CTkButton(master=frame7, text="Elgo Manual", command=lambda: open_link(url), fg_color='orange', font=("Helvetica", 14, "bold"), width=80)
link_button.pack(side='right', padx=5)

# ウィンドウサイズ変更時のイベントハンドラを設定
app.bind("<Configure>", lambda event: adjust_frame_sizes())

# アプリケーションを実行
app.mainloop()
