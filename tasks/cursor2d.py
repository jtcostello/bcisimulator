from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import numpy as np
import collections
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'


# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
# NEURAL_SCREEN_WIDTH = 1000
# NEURAL_SCREEN_HEIGHT = 300
FPS = 50
TARGET_RADIUS = 30
CURSOR_RADIUS = 8
HOLD_DURATION = 200  # in milliseconds
DO_PLOT_NEURAL = True
NUM_CHANS_TO_PLOT = 20
NUM_NEURAL_HISTORY_PLOT = 100

colors = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'white': (255, 255, 255),
    'black': (0, 0, 0)
}


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


def generate_target():
    # generate a random target position
    x = np.random.randint(TARGET_RADIUS, SCREEN_WIDTH - TARGET_RADIUS)
    y = np.random.randint(TARGET_RADIUS, SCREEN_HEIGHT - TARGET_RADIUS)
    return x, y


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
    ax.set_title('Neural Data Visualization')
    ax.set_position([0, 0, 1, 1])
    ax.axis('off')


def cursor_task(input_source, recorder, decoder=None):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Cursor Task")
    pygame.mouse.set_visible(False)
    clock = pygame.time.Clock()

    # setup mpl figure for neural data visualization
    # DO_PLOT_NEURAL = DO_PLOT_NEURAL if decoder is not None else False
    fig = None
    if DO_PLOT_NEURAL and decoder is not None:
        neural_history = collections.deque(maxlen=NUM_NEURAL_HISTORY_PLOT)
        fig, ax = plt.subplots(figsize=(10, 3), num='Neural Data Visualization')
        ani = FuncAnimation(fig, lambda i: visualize_neural_data(ax, neural_history), interval=1000/FPS)
        plt.show(block=False)  # non-blocking, continues with script execution

    target_position = generate_target()
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

        # Check target acquisition
        distance_to_target = pygame.math.Vector2(target_position).distance_to(cursor_position)
        if distance_to_target <= TARGET_RADIUS:
            if start_hold_time is None:
                start_hold_time = pygame.time.get_ticks()
            elif pygame.time.get_ticks() - start_hold_time >= HOLD_DURATION:
                target_position = generate_target()
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
        text1 = pygame.font.SysFont(None, 24).render(f"{time:.1f}", True, colors["black"])
        screen.blit(text1, (SCREEN_WIDTH - 100, 20, 100, 100))
        text2 = pygame.font.SysFont(None, 24).render(f'Trial {trial}', True, colors["black"])
        screen.blit(text2, (SCREEN_WIDTH - 100, 50, 100, 100))
        if trial_times:
            text3 = pygame.font.SysFont(None, 24).render(f"Avg Time {np.mean(trial_times)/1000:.2f}s", True, colors["black"])
            screen.blit(text3, (SCREEN_WIDTH - 120, 80, 100, 100))

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


