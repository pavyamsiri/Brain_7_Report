# Standard modules
import os
import random
import sys
import time
from typing import Dict, List, NamedTuple, Optional, Tuple

# External modules
import numpy as np
import pygame
from pygame import Color
from pygame.math import Vector2
import yaml

# Internal modules
from aliases import EntityImage, FilePath, ResourceName
from colours import *
from entity import Enemy, Player
from spikerbox import Control, SpikerBox


# Width of the game physics
GAME_WIDTH: int = 74
# Height of the game physics
GAME_HEIGHT: int = 144

# Width of UI panel
UI_PANEL_WIDTH: int = 37
# Width of the screen
SCREEN_WIDTH: int = UI_PANEL_WIDTH + GAME_WIDTH + UI_PANEL_WIDTH
# Height of the screen
SCREEN_HEIGHT: int = 144

# Pixel offset between lanes for player
PLAYER_LANE_OFFSET: int = 25


class GameArgs(NamedTuple):
    player_hp: int
    player_max_hp: int
    enemy_hp: int
    enemy_spawn_time: float
    enemy_speed: float
    enemy_score_gain: float


class HealthBar:
    """Class that contains health bar sprites and health percentage to be used to draw the health bar UI element.

    Attributes
    ----------
    unfilled : pygame.Surface
        sprite of an unfilled health bar.
    filled : pygame.Surface
        sprite of a filled health bar.
    health : float
        the percentage of the max health of the player.
    """

    def __init__(
        self,
        unfilled: pygame.Surface,
        filled: pygame.Surface,
        health: float,
    ):
        """Initialises the health bar.

        Parameters
        ----------
        unfilled : pygame.Surface
            sprite of an unfilled health bar.
        filled : pygame.Surface
            sprite of a filled health bar.
        health : float
            the percentage of the max health of the player.
        """
        self._unfilled: pygame.Surface = unfilled
        self._filled: pygame.Surface = filled
        self._health: float = health

    @property
    def unfilled(self) -> pygame.Surface:
        """pygame.Surface: sprite of an unfilled health bar."""
        return self._unfilled

    @property
    def unfilled_width(self) -> int:
        """int: the width of the unfilled sprite."""
        return self._unfilled.get_size()[0]

    @property
    def unfilled_height(self) -> int:
        """int: the height of the unfilled sprite."""
        return self._unfilled.get_size()[1]

    @property
    def filled(self) -> pygame.Surface:
        """pygame.Surface: sprite of a filled health bar."""
        return self._filled

    @property
    def filled_width(self) -> int:
        """int: the height of the filled sprite."""
        return self._filled.get_size()[0]

    @property
    def filled_height(self) -> int:
        """int: the height of the filled sprite."""
        return self._filled.get_size()[1]

    @property
    def health(self) -> float:
        """float: the percentage of the max health of the player."""
        return self._health

    @health.setter
    def health(self, new_health) -> None:
        self._health = max(0, min(1, new_health))


class ResourceManager:
    """Game resources such as sprites are stored here to be requested for use.

    Attributes
    ----------
    sprites : Dict[ResourceName, Surface]
        the loaded game sprites.
    """

    def __init__(self, asset_paths: FilePath):
        """Initialises the resource manager using a asset path configuration file.

        Parameters
        ----------
        asset_paths : FilePath
            the file path to the asset path configuration file that stores the paths to all assets.
        """
        # Initialise asset storage
        self._sprites: Dict[ResourceName, pygame.Surface] = {}

        # Check that the file exists
        if not os.path.isfile(asset_paths):
            raise FileNotFoundError("File containing asset paths does not exist!")

        # Load sprites from paths defined in the configuration file
        with open(asset_paths, "r") as asset_file:
            paths = yaml.safe_load(asset_file)
            # Dynamically load assets
            for current_path in paths:
                # Load inner paths
                if isinstance(paths[current_path], dict):
                    for inner_path in paths[current_path]:
                        self._sprites[
                            f"{current_path.upper()}_{inner_path.upper()}"
                        ] = pygame.image.load(paths[current_path][inner_path])
                # Load paths
                elif isinstance(paths[current_path], str):
                    self._sprites[f"{current_path.upper()}"] = pygame.image.load(
                        paths[current_path]
                    )

    def get(self, resource: ResourceName) -> pygame.Surface:
        """Returns the request resource.

        Parameters
        ----------
        resource : ResourceName
            the name of the resource requested.

        Returns
        -------
        resource : pygame.Surface
            the requested sprite.
        """
        return self._sprites[resource.upper()]

    @property
    def player(self) -> pygame.Surface:
        """pygame.Surface: the player sprite."""
        return self._sprites["PLAYER"]

    @property
    def obstacle(self) -> pygame.Surface:
        """pygame.Surface: the obstacle sprite."""
        return self._sprites["OBSTACLE"]

    @property
    def energy(self) -> pygame.Surface:
        """pygame.Surface: the energy sprite."""
        return self._sprites["ENERGY"]

    @property
    def health_bar(self) -> Tuple[pygame.Surface, pygame.Surface]:
        """Tuple[pygame.Surface, pygame.Surface]: the health bar sprites."""
        return (
            self._sprites["HEALTH_BAR_UNFILLED"],
            self._sprites["HEALTH_BAR_FILLED"],
        )


class Renderer:
    """Class that handles rendering with pygame. It contains many methods to draw a variety of sprites."""

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        game_width: int,
        game_height: int,
        window_icon: pygame.Surface,
    ):
        """Main Renderer constructor.

        Parameters
        ----------
        screen_width : int
            the width of the screen.
        screen_height : int
            the height of the screen.
        game_width : int
            the width of the game.
        game_height : int
            the height of the game.
        window_icon : pygame.Surface
            the window icon.
        """
        # Initialise window
        pygame.init()
        pygame.display.set_caption("Space Run")
        pygame.display.set_icon(window_icon)

        # Game clock
        self._clock: pygame.time.Clock = pygame.time.Clock()
        # Scale resolution to larger window
        scale: int = max((pygame.display.Info().current_h // screen_height) - 1, 1)
        self._master_screen: pygame.Surface = pygame.display.set_mode(
            (scale * screen_width, scale * screen_height), pygame.SCALED
        )
        # UI layer
        self._window: pygame.Surface = pygame.Surface((screen_width, screen_height))
        # Game layer
        self._screen: pygame.Surface = pygame.Surface((game_width, game_height))

        # Fonts
        self._font: pygame.font.Font = pygame.font.SysFont("Arial", 14)

        # Buffered HD draws
        self._hd_draws: List[Tuple[pygame.Surface, pygame.Rect]] = []

    def prepare(self):
        """Prepares the frame to be drawn to."""
        # Clear background to black
        self._master_screen.fill(BLACK)
        # Clear UI layer
        self._window.fill(PANEL_GREY)
        # Clear game layer
        self._screen.fill(BLACK)

    def draw_entity(self, entity: EntityImage):
        """Draws an entity to the game layer.

        Parameters
        ----------
        entity : EntityImage
            the entity to draw. Contains a pygame.Surface and a desired positon.
        """
        self._screen.blit(entity[0], entity[1])

    def draw_entities(self, entities: List[EntityImage]):
        """Draws a list of entities to the game layer.

        Parameters
        ----------
        entities : List[EntityImage]
            the entities to draw. Each element contains a pygame.Surface and a desired positon.
        """
        self._screen.blits(entities)

    def draw_ui(
        self,
        element: pygame.Surface,
        position: Vector2,
        center_x: bool = False,
        center_y: bool = False,
    ):
        """Draws a UI element to the UI layer.

        Parameters
        ----------
        element : pygame.Surface
            the UI element to draw.
        position : Vector2
            the desired position.
        center_x : bool
            if `True` will centre the element along the x-axis. Default is `False`.
        center_y : bool
            if `True` will centre the element along the y-axis. Default is `False`.
        """
        image_rect: pygame.Rect = element.get_rect()
        # Centre the element
        if center_x:
            image_rect.centerx = position.x
        else:
            image_rect.left = position.x
        if center_y:
            image_rect.centery = position.y
        else:
            image_rect.top = position.y
        self._window.blit(element, image_rect)

    def draw_ui_hd(
        self,
        element: pygame.Surface,
        position: Vector2,
        center_x: bool = False,
        center_y: bool = False,
    ):
        """Draws a UI element to the master layer for the highest definition. The draw is buffered and not is rendered until
        the `flush` method is called.

        Parameters
        ----------
        element : pygame.Surface
            the UI element to draw.
        position : Vector2
            the desired position.
        center_x : bool
            if `True` will centre the element along the x-axis. Default is `False`.
        center_y : bool
            if `True` will centre the element along the y-axis. Default is `False`.
        """
        image_rect: pygame.Rect = element.get_rect()
        # Centre the element
        if center_x:
            image_rect.centerx = position.x
        else:
            image_rect.left = position.x
        if center_y:
            image_rect.centery = position.y
        else:
            image_rect.top = position.y
        # Buffer the draw
        self._hd_draws.append((element, image_rect))

    def draw_text(
        self,
        text: str,
        position: Vector2,
        colour: Color,
        center_x: bool = False,
        center_y: bool = False,
    ):
        """Writes text to the UI layer.

        Parameters
        ----------
        text : str
            the text to write.
        position : Vector2
            the desired position.
        colour : Color
            the colour of the text.
        center_x : bool
            if `True` will centre the element along the x-axis. Default is `False`.
        center_y : bool
            if `True` will centre the element along the y-axis. Default is `False`.
        """
        text_image: pygame.Surface = self._font.render(text, False, colour)
        text_rect: pygame.Rect = text_image.get_rect()
        # Centre the element
        if center_x:
            text_rect.centerx = position.x
        else:
            text_rect.left = position.x
        if center_y:
            text_rect.centery = position.y
        else:
            text_rect.top = position.y
        # Set alpha
        text_image.set_alpha(colour.a)
        self._window.blit(text_image, text_rect)

    def flush(self):
        """Flushes the draws to the screen."""
        # Draw the screen to the window
        self._window.blit(self._screen, (UI_PANEL_WIDTH, 0))
        # Draw the window to the screen
        self._master_screen.blit(
            pygame.transform.scale(self._window, self._master_screen.get_rect().size),
            (0, 0),
        )
        # Carry out the buffered draw calls
        if len(self._hd_draws) > 0:
            self._master_screen.blits(self._hd_draws)
            self._hd_draws.clear()
        # Update the display window
        pygame.display.update()

    @property
    def screen_width(self) -> int:
        """int: the width of the game layer."""
        return self._screen.get_size()[0]

    @property
    def screen_height(self) -> int:
        """int: the height of the game layer."""
        return self._screen.get_size()[1]

    @property
    def window_width(self) -> int:
        """int: the width of the UI layer."""
        return self._window.get_size()[0]

    @property
    def window_height(self) -> int:
        """int: the height of the UI layer."""
        return self._window.get_size()[1]

    @property
    def master_width(self) -> int:
        """int: the width of the actual window."""
        return self._master_screen.get_size()[0]

    @property
    def master_height(self) -> int:
        """int: the height of the actual window."""
        return self._master_screen.get_size()[1]

    def tick(self) -> int:
        """Return the time in milliseconds.

        Returns
        -------
        tick : int
            time in milliseconds.
        """
        return self._clock.tick()

    def get_fps(self) -> float:
        """Return the average FPS.

        Returns
        -------
        fps : float
            the average frames per second.
        """
        return self._clock.get_fps()


class SpaceRun:
    """Class that handles game logic, physics and rendering."""

    def __init__(
        self,
        game_args: GameArgs,
        asset_paths: FilePath,
        spikerbox: Optional[SpikerBox],
        keyboard: bool = True,
    ):
        """Initialises the game.

        Parameters
        ----------
        game_args : GameArgs
            the game parameters.
        asset_paths : FilePath
            the file path to the asset path configuration file.
        spikerbox : Optional[SpikerBox]
            the SpikerBox that handles stream input and event classification.
        keyboard : bool
            `True` if using keyboard controls else `False`. Default is `True`.
        """
        # Game parameters
        self._game_params: GameArgs = game_args
        # Resource manager
        self._resources: ResourceManager = ResourceManager(asset_paths)

        # Game variables
        # Player entity
        self._player: Player = Player(
            self._resources.player,
            self._game_params.player_hp,
            self._game_params.player_max_hp,
        )
        # Game score
        self._score: int = 0
        # List of enemy entities
        self._enemies: List[Enemy] = []
        # Clock used in spawning enemies
        self._spawn_timer: int = pygame.time.get_ticks()
        # Number of ticks since last frame
        self._tick: int = 0

        # Control parameters
        self._controls: Dict[Control, int] = {
            Control.LEFT: 0,
            Control.RIGHT: 0,
        }
        # Clock used in animating the control parameter
        self._colour_timer: Optional[int] = None
        # Colours of the displayed control parameters
        self._control_colours: Dict[Control, Color] = {
            Control.LEFT: WHITE,
            Control.RIGHT: WHITE,
        }

        # Renderer
        self._renderer: Renderer = Renderer(
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            GAME_WIDTH,
            GAME_HEIGHT,
            self._resources.get("ICON"),
        )
        # Health bar
        self._health_bar: HealthBar = HealthBar(
            *self._resources.health_bar,
            self._player.hp / self._player.max_hp,
        )

        # Diagnostic values
        # List of frame times
        self._ticks: List[int] = []
        # FPS / Frame time flag
        self._show_fps: bool = True
        # Snapshot of the control parameters of the control input
        self._last_controls: Dict[Control, int] = self._controls.copy()
        # The last control parameter that was on
        self._last_control: Optional[Control] = None

        # File path to snapshot of last classification
        self._graph_path: Optional[FilePath] = None
        # Loaded sprite of the graph snapshot
        self._graph_sprite: Optional[pygame.Surface] = None

        # Flags
        # Keyboard controls
        self._keyboard: bool = keyboard
        # Flip controls
        self._flip: bool = False
        # Can move
        self._move_flag: bool = True

        # SpikerBox
        self._spikerbox: Optional[SpikerBox] = spikerbox
        if self._spikerbox is not None:
            self._spikerbox.reset_clock()

    def update(self):
        """Updates the game by one frame."""
        # Get the time since last frame in milliseconds
        self._tick = self._renderer.tick()
        if len(self._ticks) > 1000:
            self._ticks.clear()
        self._ticks.append(self._tick)
        # Spawn enemy
        self.spawn_enemy()
        # Process input
        self.process_input()
        # Update positions
        self.move_entities(self._tick)
        # Handle collisions
        self.handle_collision()

    def draw(self):
        """Draws one frame of the game."""
        # Prepare frame
        self._renderer.prepare()

        # Draw player
        self._renderer.draw_entity(self._player.sprite_position)
        # Draw obstacles and energy cubes
        self._renderer.draw_entities([enemy.sprite_position for enemy in self._enemies])

        # UI
        # Unfilled health bar
        self._renderer.draw_ui(
            self._health_bar.unfilled,
            Vector2(
                UI_PANEL_WIDTH + self._renderer.screen_width,
                self._renderer.window_height - self._health_bar.unfilled_height,
            ),
        )
        # Filled bar
        self._renderer.draw_ui(
            pygame.transform.scale(
                self._health_bar.filled,
                (
                    int(self._health_bar.health * self._health_bar.filled_width),
                    self._health_bar.filled_height,
                ),
            ),
            Vector2(
                UI_PANEL_WIDTH + self._renderer.screen_width + 3,
                self._renderer.window_height - self._health_bar.filled_height,
            ),
        )

        # Display score
        right_panel_centre: int = (
            UI_PANEL_WIDTH + self._renderer.screen_width + UI_PANEL_WIDTH // 2
        )
        self._renderer.draw_text(
            "Score:", Vector2(right_panel_centre, 0), WHITE, center_x=True
        )
        if self._score > 10 ** 5:
            score_text: str = f"{self._score//1000}.{str(self._score % 1000)[0]}k"
        else:
            score_text: str = str(int(self._score))
        self._renderer.draw_text(
            score_text, Vector2(right_panel_centre, 15), WHITE, center_x=True
        )
        # Display FPS or average frame time
        frame_text: str = "FPS:" if self._show_fps else "MSPF:"
        self._renderer.draw_text(
            frame_text, Vector2(UI_PANEL_WIDTH // 2, 0), WHITE, center_x=True
        )
        frame_display = (
            str(int(self._renderer.get_fps()))
            if self._show_fps
            else str(int(np.mean(self._ticks)))
        )
        self._renderer.draw_text(
            frame_display, Vector2(UI_PANEL_WIDTH // 2, 15), WHITE, center_x=True
        )
        # Move sign
        sign_sprite: pygame.Surface = (
            self._resources.get("MOVE_SIGN_GO")
            if self._move_flag
            else self._resources.get("MOVE_SIGN_STOP")
        )
        self._renderer.draw_ui(
            sign_sprite,
            Vector2(
                UI_PANEL_WIDTH // 2,
                self._renderer.window_height - self._health_bar.unfilled_height // 2,
            ),
            center_x=True,
            center_y=True,
        )
        # Display control parameters: header
        self._renderer.draw_text(
            "Ctrl:",
            Vector2(UI_PANEL_WIDTH // 2, 4 * self._renderer.window_height // 10),
            WHITE,
            center_x=True,
            center_y=True,
        )

        if self._last_control is not None:
            for key in self._control_colours:
                self._control_colours[key] = WHITE
                if key == self._last_control and self._colour_timer is None:
                    self._control_colours[key] = RED
                    self._colour_timer = pygame.time.get_ticks()
                elif key == self._last_control and self._colour_timer is not None:
                    target_time: float = 1.5
                    elapsed: float = (
                        pygame.time.get_ticks() - self._colour_timer
                    ) / 1000
                    if elapsed > target_time:
                        elapsed = target_time
                        self._colour_timer = None
                        self._last_control = None
                    self._control_colours[key] = Color(
                        int(np.interp(elapsed, [0, target_time], [RED.r, WHITE.r])),
                        int(np.interp(elapsed, [0, target_time], [RED.g, WHITE.g])),
                        int(np.interp(elapsed, [0, target_time], [RED.b, WHITE.b])),
                        int(np.interp(elapsed, [0, target_time], [RED.a, WHITE.a])),
                    )

        self._renderer.draw_text(
            f"L: {self._last_controls[Control.LEFT]}",
            Vector2(UI_PANEL_WIDTH // 2, 5 * self._renderer.window_height // 10),
            self._control_colours[Control.LEFT],
            center_x=True,
            center_y=True,
        )
        self._renderer.draw_text(
            f"R: {self._last_controls[Control.RIGHT]}",
            Vector2(UI_PANEL_WIDTH // 2, 6 * self._renderer.window_height // 10),
            self._control_colours[Control.RIGHT],
            center_x=True,
            center_y=True,
        )

        # Display graph snapshot
        if self._graph_path is not None:
            self._graph_sprite: pygame.Surface = pygame.image.load(self._graph_path)

        if self._graph_sprite is not None:
            scale_factor: int = (
                self._renderer.master_width // 4
            ) / self._graph_sprite.get_size()[0]
            scaled_graph = pygame.transform.smoothscale(
                self._graph_sprite,
                (
                    int(scale_factor * self._graph_sprite.get_size()[0]),
                    int(scale_factor * self._graph_sprite.get_size()[1]),
                ),
            )
            graph_position: Vector2 = Vector2(
                7 * self._renderer.master_width // 8, self._renderer.master_height // 2
            )
            self._renderer.draw_ui_hd(
                scaled_graph, graph_position, center_x=True, center_y=True
            )

        # Flush
        self._renderer.flush()

    def spawn_enemy(self):
        """Spawns an enemy in a random lane given enough time has passed."""
        elapsed_time: float = (pygame.time.get_ticks() - self._spawn_timer) / 1000
        if elapsed_time > self._game_params.enemy_spawn_time:
            # Randomise lane
            lane_idx = random.randint(0, 2)
            # Choose obstacle or energy
            score_flag = random.randint(0, 1) == 1
            # Load corresponding sprite
            sprite = self._resources.energy if score_flag else self._resources.obstacle
            # Add new enemy
            self._enemies.append(
                Enemy(
                    sprite,
                    lane_idx,
                    self._game_params.enemy_hp,
                    self._game_params.enemy_hp,
                    self._game_params.enemy_speed,
                    self._game_params.enemy_score_gain,
                    score_flag,
                )
            )

            # Reset clock
            self._spawn_timer = pygame.time.get_ticks()

    def process_input(self):
        """Processes game input."""
        for event in pygame.event.get():
            # Quit
            if event.type == pygame.QUIT:
                sys.exit()
            # Key down events
            if event.type == pygame.KEYDOWN:
                # Escape
                if event.key == pygame.K_ESCAPE:
                    sys.exit()
                # Left and right controls
                if event.key == pygame.K_LEFT and self._keyboard:
                    self._controls[Control.LEFT] = 1
                if event.key == pygame.K_RIGHT and self._keyboard:
                    self._controls[Control.RIGHT] = 1
                if event.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()
                if event.key == pygame.K_F9:
                    self._show_fps = not self._show_fps
                if event.key == pygame.K_F8:
                    self._flip = not self._flip
                # Switch to SpikerBox input
                if event.key == pygame.K_F10:
                    self._keyboard = not self._keyboard
        # SpikerBox processing
        if not self._keyboard and self._spikerbox is not None:
            self._controls = self._spikerbox.process_input(self._controls, self._tick)
            self._graph_path = self._spikerbox.graph_path
            self._move_flag = self._spikerbox.can_move()

        # Flip controls
        if self._flip:
            self._controls[Control.LEFT], self._controls[Control.RIGHT] = (
                self._controls[Control.RIGHT],
                self._controls[Control.LEFT],
            )

        # Record last control parameters
        count: int = 0
        for key in self._controls:
            count += self._controls[key]
            if self._controls[key] > 0:
                if self._last_control is not None and key != self._last_control:
                    self._colour_timer = None
                self._last_control = key
        if count > 0:
            self._last_controls = self._controls.copy()

    def move_entities(self, delta_time: int):
        """Updates the position of entities.

        Paramaters
        ----------
        delta_time: int
            the time since the last frame in milliseconds.
        """
        # Move the player left/right
        direction = 0
        if self._controls[Control.LEFT] == 1:
            direction = -1
            self._controls[Control.LEFT] = 0
        if self._controls[Control.RIGHT] == 1:
            direction = 1
            self._controls[Control.RIGHT] = 0
        self._player.x += direction * PLAYER_LANE_OFFSET
        # Apply periodic boundary conditions
        self._player.x %= GAME_WIDTH + 1

        # Move enemies down
        for enemy in self._enemies:
            enemy.y += enemy.speed * delta_time
        # Deleting them if they go out of screen
        self._enemies[:] = [enemy for enemy in self._enemies if enemy.y <= GAME_HEIGHT]

    def handle_collision(self):
        """Detects and handles collision between entities."""
        for enemy in self._enemies:
            if enemy.hp > 0 and enemy.detect_collision(self._player):
                enemy.hp -= 1
                if enemy.is_enemy:
                    self._player.hp -= 1
                    self._health_bar.health = self._player.hp / self._player.max_hp
                    if self._player.hp <= 0:
                        self.game_over()
                else:
                    self._score += enemy.score

        # Filter out dead enemies and projectiles
        self._enemies[:] = [enemy for enemy in self._enemies if enemy.hp > 0]

    def game_over(self):
        """Game over."""
        print("Game over!")
        print(f"Your score is {self._score}")
        time.sleep(2)
        sys.exit()