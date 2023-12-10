import numpy as np
import collections
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from simplehand import SimpleHand
import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'

from inputs.hand_tracker import HandTracker
from tasks.utils import visualize_neural_data
from tasks.utils import Clock

# Constants
SCREEN_WIDTH_IN = 10
SCREEN_HEIGHT_IN = 8
NEURAL_SCREEN_WIDTH_IN = 10
NEURAL_SCREEN_HEIGHT_IN = 3
FPS = 50
DO_PLOT_NEURAL = True      # TODO switch to true
NUM_CHANS_TO_PLOT = 20
NUM_NEURAL_HISTORY_PLOT = 100  # number of timepoints


def hand_task(recorder, decoder, target_type="random"):
    print("\n--- Starting hand task, use ctrl-c to exit ---\n")

    # init hand tracker
    hand_tracker = HandTracker(camera_id=0, show_tracking=True)     # TODO have option for cameraid

    # set up window for hand visualization
    fig_hand, ax_hand = plt.subplots(figsize=(SCREEN_WIDTH_IN, SCREEN_HEIGHT_IN),
                                     num='Hand Visualization')
    # hand = SimpleHand(fig_hand, ax_hand)
    # hand.draw()
    plt.show(block=False)

    # set up window for neural visualization
    fig_neural = None
    if DO_PLOT_NEURAL and decoder is not None:
        neural_history = collections.deque(maxlen=NUM_NEURAL_HISTORY_PLOT)
        fig_neural, ax = plt.subplots(figsize=(NEURAL_SCREEN_WIDTH_IN, NEURAL_SCREEN_HEIGHT_IN),
                                      num='Neural Data Visualization (first 20 channels)')
        ani = FuncAnimation(fig_neural,
                            lambda i: visualize_neural_data(ax, neural_history, NUM_CHANS_TO_PLOT),
                            interval=1000 / FPS,
                            cache_frame_data=False)
        plt.show(block=False)  # non-blocking, continues with script execution

    # state vars
    recording = False
    online = False          # TODO button to toggle online/offline

    # main loop
    clock = Clock()
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
        # hand.set_flex(*hand_pos)
        # hand.draw()
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

        # update clock to limit frame rate
        clock.tick(FPS)


        # TODO spacebar reset position
