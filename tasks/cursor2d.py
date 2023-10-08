from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import numpy as np


# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800
FPS = 50
TARGET_RADIUS = 30
CURSOR_RADIUS = 8
HOLD_DURATION = 200  # in milliseconds

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


def cursor_task(input_source, recorder, decoder=None):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.mouse.set_visible(False)
    pygame.display.set_caption("Cursor Task")
    clock = pygame.time.Clock()

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

        pygame.display.flip()
        clock.tick(FPS)

        # Record data if recording is active
        if recording:
            recorder.record(pygame.time.get_ticks(),
                            trial,
                            normalize_pos(cursor_position),
                            normalize_pos(target_position),
                            online)


