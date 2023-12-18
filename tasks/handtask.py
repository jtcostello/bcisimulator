import numpy as np
import collections
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from simplehand import SimpleHand
import matplotlib as mpl
from matplotlib.widgets import Button
mpl.rcParams['toolbar'] = 'None'

from inputs.hand_tracker import HandTracker
from tasks.utils import visualize_neural_data
from tasks.utils import Clock

# Constants
SCREEN_WIDTH_IN = 10
SCREEN_HEIGHT_IN = 8
NEURAL_SCREEN_WIDTH_IN = 10
NEURAL_SCREEN_HEIGHT_IN = 3
MAX_FPS = 30
DISP_FPS = False
DO_PLOT_NEURAL = True
NUM_CHANS_TO_PLOT = 20
NUM_NEURAL_HISTORY_PLOT = 100   # number of timepoints

CV2_CAMERA_ID = 0               # default camera id for cv2 (usually the webcam)


def hand_task(recorder, decoder, target_type="random"):
    print("\n--- Starting hand task, use ctrl-c to exit ---\n")

    # TODO: weird bug where if you start with a hand in camera, then the gui fails

    # state vars
    recording = False
    online = False

    # init hand tracker
    hand_tracker = HandTracker(camera_id=CV2_CAMERA_ID, show_tracking=True)

    # set up window for hand visualization
    fig_hand = plt.figure(figsize=(SCREEN_WIDTH_IN, SCREEN_HEIGHT_IN), num='Hand Visualization')
    ax_hand = fig_hand.add_subplot(111, projection='3d')
    hand = SimpleHand(fig_hand, ax_hand)
    hand.set_flex(0, 0, 0, 0, 0)
    hand.draw()

    # add button for recording
    ax_record_button = plt.axes((0.05, 0.92, 0.15, 0.05))
    record_button = Button(ax_record_button, 'Start Recording', color="green")

    def toggle_recording():
        nonlocal recording
        recording = not recording
        if recording:
            print("started recording")
            record_button.label.set_text("Stop Recording")
            record_button.color = "red"

        else:
            print("stopped recording")
            record_button.label.set_text("Start Recording")
            record_button.color = "green"
            recorder.save_to_file()

    record_button.on_clicked(lambda _: toggle_recording())

    # add button for online/offline
    if decoder is not None:
        ax_online_button = plt.axes((0.25, 0.92, 0.15, 0.05))
        online_button = Button(ax_online_button, 'Go Online', color="green")

        def toggle_online():
            nonlocal online
            online = not online
            if online:
                online_button.label.set_text("Go Offline")
                online_button.color = "red"
                decoder.set_position(hand_tracker.get_hand_position())
            else:
                online_button.label.set_text("Go Online")
                online_button.color = "green"

        online_button.on_clicked(lambda _: toggle_online())

    # show the hand plot
    plt.show(block=False)

    # set up window for neural visualization
    fig_neural = None
    if DO_PLOT_NEURAL and decoder is not None:
        neural_history = collections.deque(maxlen=NUM_NEURAL_HISTORY_PLOT)
        fig_neural, ax = plt.subplots(figsize=(NEURAL_SCREEN_WIDTH_IN, NEURAL_SCREEN_HEIGHT_IN),
                                      num='Neural Data Visualization (first 20 channels)')
        ani = FuncAnimation(fig_neural,
                            lambda i: visualize_neural_data(ax, neural_history, NUM_CHANS_TO_PLOT),
                            interval=1000 / MAX_FPS,
                            cache_frame_data=False)
        plt.show(block=False)  # non-blocking, continues with script execution

    # main loop
    clock = Clock(disp_fps=DISP_FPS)
    while True:

        # get hand position
        hand_pos_true = hand_tracker.get_hand_position()

        if online:
            # run the decoder to get cursor position
            hand_pos_in = np.array(hand_pos_true)
            hand_pos = decoder.decode(hand_pos_in)
            neural_history.append(decoder.get_recent_neural())

        else:
            # offline - just use the true hand position
            hand_pos = hand_pos_true

        # draw hand
        azim, elev = ax_hand.azim, ax_hand.elev     # get current view
        ax_hand.clear()
        hand.set_flex(*hand_pos)
        hand.draw()
        ax_hand.view_init(elev, azim)               # set view back to what it was
        fig_hand.canvas.draw()
        fig_hand.canvas.flush_events()

        # draw neural data
        if DO_PLOT_NEURAL and fig_neural is not None:
            fig_neural.canvas.draw()
            fig_neural.canvas.flush_events()

        # record data if recording is active
        if recording:
            recorder.record(clock.get_time_ms(),
                            int(clock.get_time_ms() / 1000) + 1,    # dummy trials, once per second
                            hand_pos,
                            [0, 0, 0, 0, 0],                        # dummy target position
                            online)

        # update clock to limit frame rate (usually we're well below this)
        clock.tick(MAX_FPS)
