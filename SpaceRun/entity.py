# Future imports
from __future__ import annotations

# Standard modules
from abc import ABCMeta
from typing import Tuple

# External modules
import pygame
from pygame.math import Vector2

# Internal modules
from aliases import EntityImage


class Entity(metaclass=ABCMeta):
    """Class that forms the basis of all moving game objects. Can be drawn to the screen, move and has health.

    Attributes
    ----------------
    position : Vector2
        the entity position.
    sprite : pygame.Surface
        the entity sprite.
    collision_box : Tuple[int]
        the width and height of the entity's collision box.
    hp : int
        the current health of the entity.
    max_hp : int
        the max health of the entity.
    """

    def __init__(
        self,
        position: Vector2,
        sprite: pygame.Surface,
        hp: int,
        max_hp: int,
    ):
        """Constructs an entity.

        Parameters
        ----------
        position : Vector2
            the entity position.
        sprite : pygame.Surface
            the entity sprite.
        hp : int
            the current health of the entity.
        max_hp : int
            the max health of the entity.
        """
        # Current position
        self._position: Vector2 = position

        # Entity sprite
        self._sprite: pygame.Surface = sprite
        # Collision box used in collision detection. Based off the entity's sprite.
        self._collision_box: Tuple[int] = tuple(self._sprite.get_rect()[2::])

        # Entity health
        self._hp: int = hp
        # Entity max health
        self._max_hp: int = max_hp

    # Access methods
    @property
    def position(self) -> Vector2:
        """Vector2: the position vector of the entity."""
        return self._position

    @property
    def x(self) -> float:
        """float: the x coordinate of the entity's position."""
        return self._position[0]

    @x.setter
    def x(self, new_value: float) -> None:
        self._position[0] = new_value

    @property
    def y(self) -> float:
        """float: the y coordinate of the entity's position."""
        return self._position[1]

    @y.setter
    def y(self, new_value: float):
        self._position[1] = new_value

    @property
    def width(self) -> int:
        """int: the width of the entity's sprite."""
        return self._collision_box[0]

    @property
    def height(self) -> int:
        """int: the height of the entity's sprite."""
        return self._collision_box[1]

    @property
    def hp(self) -> int:
        """int: the current health of the entity."""
        return self._hp

    @hp.setter
    def hp(self, value: int):
        self._hp = value if value >= 0 else self._hp

    @property
    def max_hp(self) -> int:
        """int: the max health of the entity."""
        return self._max_hp

    @max_hp.setter
    def max_hp(self, value: int):
        if self._hp > value:
            self._hp = value
        self._max_hp = value

    @property
    def sprite_position(self) -> EntityImage:
        """EntityImage: the sprite and current position of the entity."""
        return (self._sprite, self._position)

    def detect_collision(self, other: Entity) -> bool:
        """Detects collision between entities, returing `True` if a collision has occurred, otherwise `False`, assuming all
        collision boxes are rectangles.

        Parameters
        ----------
        other : Entity
            other entity involved the potential collision.

        Returns
        -------
        result : bool
            `True` if collision has occurred else `False`.
        """
        return (
            self.x > other.x
            and self.x < (other.x + other.width)
            and self.y > other.y
            and self.y < (other.y + other.height)
        )


class Player(Entity):
    """The player entity.

    Class Attributes
    ----------------
    LEFT_LANE_SPAWN : Vector2
        the left lane spawn location.
    MIDDLE_LANE_SPAWN : Vector2
        the middle lane spawn location.
    RIGHT_LANE_SPAWN : Vector2
        the right lane spawn location.

    Attributes
    ----------
    position : Vector2
        the entity position.
    sprite : pygame.Surface
        the entity sprite.
    collision_box : Tuple[int]
        the width and height of the entity's collision box.
    hp : int
        the current health of the entity.
    max_hp : int
        the max health of the entity.
    """

    LEFT_LANE_SPAWN: Vector2 = Vector2(6, 132)
    MIDDLE_LANE_SPAWN: Vector2 = Vector2(6 + 25, 132)
    RIGHT_LANE_SPAWN: Vector2 = Vector2(6 + 2 * 25, 132)

    def __init__(self, sprite: pygame.Surface, hp: int, max_hp: int):
        """Initialises the player entity.

        Parameters
        ----------
        sprite : pygame.Surface
            the player sprite.
        hp : int
            the current health of the player.
        max_hp : int
            the max health of the player.
        """
        super().__init__(Vector2(Player.MIDDLE_LANE_SPAWN), sprite, hp, max_hp)


class Enemy(Entity):
    """Obstacle and energy entities.

    Class Attributes
    ----------------
    LEFT_LANE_SPAWN : Vector2
        the left lane spawn location.
    MIDDLE_LANE_SPAWN : Vector2
        the middle lane spawn location.
    RIGHT_LANE_SPAWN : Vector2
        the right lane spawn location.
    SPAWN_POSITIONS : Tuple[Vector2, ...]
        the spawn positions of all lanes.

    Attributes
    ----------
    position : Vector2
        the entity position.
    sprite : pygame.Surface
        the entity sprite.
    collision_box : Tuple[int]
        the width and height of the entity's collision box.
    hp : int
        the current health of the entity.
    max_hp : int
        the max health of the entity.
    speed : float
        the movement speed of the entity.
    score : int
        the score gained if the entity collides with the player and if the entity is energy.
    score_flag : bool
        if `True` the entity is an energy entity otherwise it is an obstacle.
    """

    LEFT_LANE_SPAWN: Vector2 = Vector2(10, 4)
    MIDDLE_LANE_SPAWN: Vector2 = Vector2(35, 4)
    RIGHT_LANE_SPAWN: Vector2 = Vector2(60, 4)

    SPAWN_POSITIONS: Tuple[Vector2, ...] = (
        LEFT_LANE_SPAWN,
        MIDDLE_LANE_SPAWN,
        RIGHT_LANE_SPAWN,
    )

    def __init__(
        self,
        sprite: pygame.Surface,
        position_idx: int,
        hp: int,
        max_hp: int,
        speed: float,
        score: int,
        score_flag: bool,
    ):
        """Initialises obstacle and energy entities.

        Parameters
        ----------
        sprite : pygame.Surface
            the entity sprite.
        position_idx : int
            the lane index corresponding to SPAWN_POSITIONS.
        hp : int
            the current health of the entity.
        max_hp : int
            the max health of the entity.
        speed : float
            the movement speed of the entity.
        score : int
            the score gained if the entity collides with the player and if the entity is energy.
        score_flag : bool
            if `True` the entity is an energy entity otherwise it is an obstacle.
        """
        super().__init__(
            Vector2(Enemy.SPAWN_POSITIONS[position_idx]), sprite, hp, max_hp
        )

        self._speed: float = speed
        self._score: int = score
        self._score_flag: bool = score_flag

    @property
    def speed(self) -> float:
        """float: the movement speed of the entity."""
        return self._speed

    @property
    def score(self) -> int:
        """int: the score gained if the entity collides with the player and if the entity is energy."""
        return self._score

    @property
    def is_enemy(self) -> bool:
        """bool: `True` if the entity is an obstacle else `False`."""
        return not self._score_flag
