import tkinter as tk
import threading
from record import record_audio, transcribe_audio_chunks, stop_recording

# Create the main window
root = tk.Tk()
root.title("Conversational Human Interactive Robot")
root.configure(bg="light grey")

# Create a frame for the text display
frame = tk.Frame(root, bg="light grey")
frame.pack(padx=10, pady=10)

# Create a text widget for incoming text
incoming_text_label = tk.Label(frame, text="Incoming Text:", bg="light grey")
incoming_text_label.pack(anchor="w")
incoming_text = tk.Text(frame, height=10, width=50)
incoming_text.pack(padx=5, pady=5)

# Create a text widget for outgoing text
outgoing_text_label = tk.Label(frame, text="Outgoing Text:", bg="light grey")
outgoing_text_label.pack(anchor="w")
outgoing_text = tk.Text(frame, height=10, width=50)
outgoing_text.pack(padx=5, pady=5)

# Create buttons
button_frame = tk.Frame(root, bg="light grey")
button_frame.pack(pady=10)

def on_button1_click():
    global stop_recording
    if button1["text"] == "Record Audio":
        button1["text"] = "Stop Recording"
        threading.Thread(target=record_audio).start()
    else:
        stop_recording = True
        button1["text"] = "Record Audio"

def on_button2_click():
    transcribed_text = transcribe_audio_chunks()
    incoming_text.delete(1.0, tk.END)
    incoming_text.insert(tk.END, transcribed_text)

button1 = tk.Button(button_frame, text="Record Audio", command=on_button1_click)
button1.pack(side="left", padx=5)

button2 = tk.Button(button_frame, text="Transcribe Audio", command=on_button2_click)
button2.pack(side="left", padx=5)

# Run the application
root.mainloop()
