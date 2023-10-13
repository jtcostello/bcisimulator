from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import numpy as np
import collections
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
from tasks.utils import TargetGenerator


# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
NEURAL_SCREEN_WIDTH_IN = 10
NEURAL_SCREEN_HEIGHT_IN = 3
FPS = 50
TARGET_RADIUS = 30
CURSOR_RADIUS = 8
HOLD_DURATION = 500                 # in milliseconds
DO_PLOT_NEURAL = True
NUM_CHANS_TO_PLOT = 20
NUM_NEURAL_HISTORY_PLOT = 100       # number of timepoints
TARGET_TYPE = "random"              # "random" or "centerout"

colors = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'white': (255, 255, 255),
    'black': (0, 0, 0)
}
font_size = 24


class Button:
    def __init__(self, x, y, width, height, text, action):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.action = action
        self.color = "green"
        self.visible = True

    def draw(self, screen):
        if self.visible:
            pygame.draw.rect(screen, colors[self.color], self.rect)
            font = pygame.font.SysFont(None, 24)
            label = font.render(self.text, True, colors["black"])
            screen.blit(label, (self.rect.x + 10, self.rect.y + 10))

    def click(self, pos):
        if self.rect.collidepoint(pos):
            self.action()


def normalize_pos(pos):
    return pos[0] / SCREEN_WIDTH, pos[1] / SCREEN_HEIGHT


def unnormalize_pos(pos):
    return pos[0] * SCREEN_WIDTH, pos[1] * SCREEN_HEIGHT


def visualize_neural_data(ax, neural_history):
    ax.clear()  # clear previous data
    if neural_history:
        data = np.array(neural_history)
        ypos = 0
        for ch in range(min(data.shape[1], NUM_CHANS_TO_PLOT)):
            ax.plot(data[:, ch] + ypos)
            ypos += 3
    ax.set_position([0, 0, 1, 1])
    ax.axis('off')


def cursor_task(input_source, recorder, decoder=None, target_type="random"):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Cursor Task")
    pygame.mouse.set_visible(False)
    clock = pygame.time.Clock()
    print("--tip: press spacebar to reset the cursor to your mouse position--")

    # setup targets
    if target_type == "random":
        edge = 0.2  # prevent targets in the outer 20% of the screen
        target_gen = TargetGenerator(num_dof=2, center_out=False, is_discrete=False, continuous_range=[edge, 1 - edge])

    elif target_type == "centerout":
        # 8 circular targets, centered at (0.5, 0.5)
        targets = [(0.8, 0.5), (0.71, 0.71), (0.5, 0.8), (0.29, 0.71),
                   (0.2, 0.5), (0.29, 0.29), (0.5, 0.2), (0.71, 0.29)]
        target_gen = TargetGenerator(num_dof=2, center_out=True, is_discrete=True, discrete_targs=targets)

    # setup mpl figure for neural data visualization
    # DO_PLOT_NEURAL = DO_PLOT_NEURAL if decoder is not None else False
    fig = None
    if DO_PLOT_NEURAL and decoder is not None:
        neural_history = collections.deque(maxlen=NUM_NEURAL_HISTORY_PLOT)
        fig, ax = plt.subplots(figsize=(NEURAL_SCREEN_WIDTH_IN, NEURAL_SCREEN_HEIGHT_IN),
                               num='Neural Data Visualization (first 20 channels)')
        ani = FuncAnimation(fig, lambda i: visualize_neural_data(ax, neural_history), interval=1000/FPS,
                            cache_frame_data=False)
        plt.show(block=False)  # non-blocking, continues with script execution

    target_position = unnormalize_pos(tuple(target_gen.generate_targets()))
    recording = False
    online = False
    start_hold_time = None

    trial = 1
    trial_times = []
    trial_start_time = 0

    # Buttons
    start_stop_button = Button(10, 10, 150, 30, "Start Recording", lambda: toggle_recording())
    online_button = Button(170, 10, 150, 30, "Go Online", lambda: toggle_online())
    if decoder is None:
        online_button.visible = False

    def toggle_recording():
        nonlocal recording
        recording = not recording
        if recording:
            print("started recording")
            start_stop_button.text = "Stop Recording"
            start_stop_button.color = "red"

        if not recording:
            start_stop_button.text = "Start Recording"
            start_stop_button.color = "green"
            recorder.save_to_file()

    def toggle_online():
        nonlocal online
        online = not online
        if online:
            online_button.text = "Go Offline"
            online_button.color = "red"
            decoder.set_position(normalize_pos(pygame.mouse.get_pos()))
        else:
            online_button.text = "Go Online"
            online_button.color = "green"

    while True:
        screen.fill((255, 255, 255))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.MOUSEBUTTONDOWN:
                if start_stop_button.rect.collidepoint(event.pos):
                    start_stop_button.click(event.pos)
                if online_button.rect.collidepoint(event.pos):
                    online_button.click(event.pos)

        # Get cursor position
        cursor_position = pygame.mouse.get_pos()
        if online:
            # run the decoder to get cursor position
            cursor_pos_in = np.array(normalize_pos(cursor_position))
            cursor_position = decoder.decode(cursor_pos_in)
            cursor_position = unnormalize_pos(cursor_position)
            neural_history.append(decoder.get_recent_neural())

        # if space bar is pressed, reset the position to the cursor (useful if the decoded position gets biased)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            cursor_position = pygame.mouse.get_pos()
            decoder.set_position(normalize_pos(cursor_position))

        # Check target acquisition
        distance_to_target = pygame.math.Vector2(target_position).distance_to(cursor_position)
        if distance_to_target <= TARGET_RADIUS:
            if start_hold_time is None:
                start_hold_time = pygame.time.get_ticks()
            elif pygame.time.get_ticks() - start_hold_time >= HOLD_DURATION:
                # Target acquired -> new trial
                target_position = unnormalize_pos(tuple(target_gen.generate_targets()))
                start_hold_time = None
                trial += 1
                cur_time = pygame.time.get_ticks()
                trial_times.append(cur_time - trial_start_time)
                trial_start_time = cur_time
        else:
            start_hold_time = None

        # Draw target and cursor
        pygame.draw.circle(screen, (255, 0, 0), target_position, TARGET_RADIUS)
        pygame.draw.circle(screen, (0, 0, 255), cursor_position, CURSOR_RADIUS)

        # Draw buttons
        start_stop_button.draw(screen)
        online_button.draw(screen)

        # Draw info text
        time = pygame.time.get_ticks() / 1000
        font = pygame.font.SysFont(None, font_size)
        text1 = font.render(f"{time:.1f}", True, colors["black"])
        text2 = font.render(f'Trial {trial}', True, colors["black"])
        if trial_times:
            text3 = font.render(f"Avg Time {np.mean(trial_times) / 1000:.2f}s", True, colors["black"])
        screen.blit(text1, (SCREEN_WIDTH - text1.get_width() - 20, 20))
        screen.blit(text2, (SCREEN_WIDTH - text2.get_width() - 20, 50))
        if trial_times:
            screen.blit(text3, (SCREEN_WIDTH - text3.get_width() - 20, 80))

        # Draw neural data
        if DO_PLOT_NEURAL and fig is not None:
            fig.canvas.draw()
            fig.canvas.flush_events()

        # Update screen & tick clock
        pygame.display.flip()
        clock.tick(FPS)

        # Record data if recording is active
        if recording:
            recorder.record(pygame.time.get_ticks(),
                            trial,
                            normalize_pos(cursor_position),
                            normalize_pos(target_position),
                            online)


