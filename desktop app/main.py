from tkinter import Label, Button, Tk, filedialog
from functions import classify_music_folder
  
filenames, out_dir = None, None

# Function for opening the file explorer window
def browseFiles():
    global filenames
    filenames = filedialog.askopenfilenames(
        title="Select music files to classify",
        filetypes = (("Mp3 files", "*.mp3*"), (".Wav files", "*.wav*"))
    )
    label_file_explorer.configure(text=str(len(filenames)) + " music files chosen")

# Function for opening the directory explorer window
def browse_out_dir():
    global out_dir
    out_dir = filedialog.askdirectory(title="Select output folder to put music classified")
    label_dir_explorer.configure(text="Output directory: " + out_dir)

# Function for performing the classification
def classify():
    global filenames
    global out_dir
    if filenames and out_dir:
        classify_music_folder(list(filenames), out_dir, classifier="svm")
        label_done.configure(text="Done.")
        # os.system('explorer.exe ' + out_dir)
    else:
        label_done.configure(text="Make sure you have chosen the music files and an output directory!")


window = Tk() # Create the root window

window.title('Music classification by genre') # Set window title

window.geometry("700x500") # Set window size

window.config(background = "white") # Set window background color
  
# Create a File and Directory Explorer labels
label_file_explorer = Label(window, text="Music files: Not chosen", width=100, height=2, fg="black")
label_dir_explorer = Label(window, text="Output directory: Not chosen", width=100, height=2, fg="black")
label_done = Label(window, text="", width=100, height=2, fg="black")

# Create the Buttons
button_explore = Button(window, text = "Choose music files", command = browseFiles)
button_output_folder = Button(window, text = "Choose Ouput directory", command = browse_out_dir)
button_classify = Button(window, text = "Perform classification by genre", command = classify)
  
# Grid method is chosen for placing the widgets at respective positions
# in a table like structure by specifying rows and columns
label_file_explorer.grid(column=1, row=1)
label_dir_explorer.grid(column=1, row=2)
button_explore.grid(column=1, row=3)
button_output_folder.grid(column=1, row=4)
button_classify.grid(column=1, row=5)
label_done.grid(column=1, row=6)
  
# Let the window wait for any events
window.mainloop()