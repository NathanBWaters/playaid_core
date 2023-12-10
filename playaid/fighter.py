from PIL import ImageOps, Image
import numpy as np
import imutils

from playaid.constants import CHAR_LIST
from playaid.anim_ontology import (
    HEX_TO_ACTION,
    ONTOLOGY,
    STAGE_ENUM_TO_DATA,
    STATUS_ENUM_TO_STRING,
    FIGHTER_STATUS_ENUM_TO_STRING,
    FIGHTER_ENUM_TO_NAME,
)
from playaid.frame_data import FIGHTER_FRAME_DATA
from playaid.dataset_utils import (
    get_anim_for_string_and_status_kind,
)


def normalize_yolo_pixel(yolo_bbox, image_width, image_height):
    """ """
    center_x, center_y, crop_width, crop_height = yolo_bbox
    return (
        center_x / image_width,
        center_y / image_height,
        crop_width / image_width,
        crop_height / image_height,
    )


def calculate_focal_length(fov, image_width):
    """
    Calculates the focal length of a camera given its field of view (FOV) and the image width.

    Parameters:
    fov (float): The field of view of the camera in degrees.
    image_width (int): The width of the image.

    Returns:
    float: The calculated focal length.
    """
    # convert fov from degrees to radians
    fov_rad = np.deg2rad(fov)

    # calculate the focal length
    focal_length = image_width / (2 * np.tan(fov_rad / 2))

    return focal_length


def create_translation_matrix(translation_vector):
    """
    Create a 4x4 translation matrix from a 3-element translation vector.

    Parameters:
    translation_vector (list): A 3-element list representing the translation in x, y, and z.

    Returns:
    numpy.ndarray: A 4x4 translation matrix.
    """
    T = np.eye(4)
    T[:3, 3] = translation_vector
    return T


def calculate_intrinsic_matrix(fov, image_width, image_height):
    """
    Calculates the camera intrinsic matrix.

    Parameters:
    fov (float): The field of view of the camera in degrees.
    image_width (int): The width of the image.
    image_height (int): The height of the image.

    Returns:
    numpy.ndarray: The 3x3 intrinsic matrix.
    """
    # calculate the focal length
    f = calculate_focal_length(fov, image_width)

    # define the intrinsic matrix
    K = np.array([[f, 0, image_width / 2], [0, f, image_height / 2], [0, 0, 1]])

    return K


def calculate_lookat_matrix(camera_position, target_position):
    """
    Calculate a look-at transformation matrix given a camera position and a target position.

    Parameters:
    camera_position (array-like): The 3D position of the camera.
    target_position (array-like): The 3D position of the target.

    Returns:
    numpy.ndarray: A 4x4 look-at transformation matrix.
    """

    # Compute the forward vector from the target and camera positions
    forward = np.array(camera_position) - np.array(target_position)
    forward /= np.linalg.norm(forward)  # Normalize the forward vector

    # Define the up vector
    up = np.array([0, 1, 0])

    # Compute the right vector as the cross product of up and forward vectors
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)  # Normalize the right vector

    # Recompute the orthonormal up vector as the cross product of forward and right vectors
    up = np.cross(forward, right)

    # Create a 4x4 look-at transformation matrix
    lookat_matrix = np.eye(4)
    lookat_matrix[0, :3] = right
    lookat_matrix[1, :3] = up
    lookat_matrix[2, :3] = -forward
    lookat_matrix[:3, 3] = camera_position

    return lookat_matrix


def project_point_to_pixel(point_world, intrinsic_matrix, camera_pose, image_height=720):
    """
    Project a 3D point in world space onto 2D image space.

    Parameters:
    point_world (array-like): The 3D coordinates of the point in world space.
    intrinsic_matrix (numpy.ndarray): The 3x3 intrinsic matrix.
    camera_pose (numpy.ndarray): The 4x4 pose of the camera in world space.

    Returns:
    numpy.ndarray: The 2D coordinates of the point in image space.
    """

    # Homogeneous coordinates for the 3D point
    point_world_homogeneous = np.append(point_world, 1)

    # Invert the look-at matrix
    camera_pose_inverse = np.linalg.inv(camera_pose)

    # Transform the 3D point from world space to camera space
    point_camera_homogeneous = camera_pose_inverse @ point_world_homogeneous

    # Apply perspective division to transform the 3D point from camera space to normalized image
    # space
    point_image_normalized = point_camera_homogeneous[:3] / point_camera_homogeneous[2]

    # Transform the 2D point from normalized image space to pixel space
    point_image_pixel = intrinsic_matrix @ point_image_normalized

    point_image_pixel[1] = image_height - point_image_pixel[1]

    # Return the 2D point as integer pixel coordinates
    return np.round(point_image_pixel[:2]).astype(int)


class YoloCrop:
    def __init__(self, center_x, center_y, crop_width, crop_height, confidence=0, class_id=-1):
        """
        Comes in as 0 to 1 normalized Yolo data.
        """
        self.center_x = center_x
        self.center_y = center_y
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.confidence = confidence
        self.class_id = class_id

    @classmethod
    def from_pixel_coordinates(cls, image_width, image_height, x1, y1, x2, y2, x3, y3, x4, y4):
        """
        Alternative constructor that takes in image width, height, and 4 points in pixel space.
        Converts the pixel coordinates to normalized values.
        """
        # Calculate center point in pixel space
        center_x = (x1 + x2 + x3 + x4) / 4
        center_y = (y1 + y2 + y3 + y4) / 4

        # Calculate crop width and height in pixel space
        crop_width = max(x1, x2, x3, x4) - min(x1, x2, x3, x4)
        crop_height = max(y1, y2, y3, y4) - min(y1, y2, y3, y4)

        # Convert to normalized values
        center_x /= image_width
        center_y /= image_height
        crop_width /= image_width
        crop_height /= image_height

        return cls(center_x, center_y, crop_width, crop_height)

    @classmethod
    def from_pixel_yolo(cls, image_width, image_height, center_x, center_y, width, height):
        """
        Saves the crop in its normalized space given the pixel data.
        """
        norm_center_x = center_x / image_width
        norm_center_y = center_y / image_height
        norm_width = width / image_width
        norm_height = height / image_height

        return cls(norm_center_x, norm_center_y, norm_width, norm_height)

    @classmethod
    def from_string(cls, yolo_string):
        """
        Alternative constructor that takes in image width, height, and 4 points in pixel space.
        Converts the pixel coordinates to normalized values.
        """
        class_id, center_x, center_y, width, height, confidence = yolo_string.split(" ")
        return cls(
            float(center_x),
            float(center_y),
            float(width),
            float(height),
            confidence=float(confidence),
            class_id=int(class_id),
        )

    def interp(self, b, percent):
        x = self.center_x + (percent * (b.center_x - self.center_x))
        y = self.center_y + (percent * (b.center_y - self.center_y))
        crop_width = self.crop_width + (percent * (b.crop_width - self.crop_width))
        crop_height = self.crop_height + (percent * (b.crop_height - self.crop_height))
        confidence = self.confidence + (percent * (b.confidence - self.confidence))

        assert self.class_id == b.class_id, "Interpolating between two different class ids"

        return YoloCrop(
            x, y, crop_width, crop_height, confidence=confidence, class_id=self.class_id
        )

    def yolo_crop(self):
        """
        Returns yolo crop, all normalized floats. Not necessarily a square crop.
        """
        return (
            self.center_x,
            self.center_y,
            self.crop_width,
            self.crop_height,
        )

    def square_yolo_crop_pixels(self, input_frame):
        """
        Returns yolo styled square crop, all normalized floats.
        """
        height, width, _ = input_frame.shape
        center_x, center_y = self.center_pixels(input_frame.shape[1], input_frame.shape[0])
        crop_width_pixels = self.crop_width * width
        crop_height_pixels = self.crop_height * height

        crop_length = max(crop_width_pixels, crop_height_pixels)
        return (
            center_x,
            center_y,
            crop_length,
        )

    def square_yolo_crop(self, input_frame):
        """
        Returns yolo styled square crop, all normalized floats.
        """
        height, width, _ = input_frame.shape
        _, _, len_in_pixels = self.square_yolo_crop_pixels(input_frame)
        return (
            self.center_x,
            self.center_y,
            len_in_pixels / width,
            len_in_pixels / height,
        )

    def xyxy_norm(self):
        """
        Returns normalized top left point and bottom left point as (x1,y1,x2,y2), all floats
        """
        return (
            self.center_x - (self.crop_width / 2),
            self.center_y - (self.crop_height / 2),
            self.center_x + (self.crop_width / 2),
            self.center_y + (self.crop_height / 2),
        )

    def xyxy_pixels(self, image_width, image_height):
        """
        Returns top left point and bottom left point in pixel space as (x1,y1,x2,y2), all ints
        """
        (x1, y1, x2, y2) = self.xyxy_norm()
        return (
            max(0, int(x1 * image_width)),
            max(0, int(y1 * image_height)),
            min(image_width, int(x2 * image_width)),
            min(image_height, int(y2 * image_height)),
        )

    def center_pixels(self, image_width, image_height):
        """
        Returns the center point in pixel space as (x1,y1,x2,y2), all ints
        """
        return (
            int(self.center_x * image_width),
            int(self.center_y * image_height),
        )

    def yolo_pixels(self, image_width, image_height):
        """
        Returns the center point in pixel space as (x1,y1,x2,y2), all ints
        """
        return (
            int(self.center_x * image_width),
            int(self.center_y * image_height),
            int(self.crop_width * image_width),
            int(self.crop_height * image_height),
        )

    def crop_img(self, image):
        # Calculate pixel coordinates
        (x1, y1, x2, y2) = self.xyxy_pixels(image.shape[1], image.shape[0])

        # Extract the crop from the image
        return image[y1:y2, x1:x2]

    def square_crop(self, image, output_size=128, padding=0):
        """
        Crops out a square version of the yolo_crop. The output crop has to be
        128 x 128.
        @param image numpy array
        @param padding additional pading in pixels
        """
        (center_x, center_y, crop_width, crop_height) = self.yolo_pixels(
            image.shape[1], image.shape[0]
        )
        square_dim = max(crop_width, crop_height)
        square_half = int(square_dim / 2)

        raw_crop = image[
            max(center_y - square_half - padding, 0) : min(
                center_y + square_half + padding, image.shape[0]
            ),
            max(center_x - square_half - padding, 0) : min(
                center_x + square_half + padding, image.shape[1]
            ),
            :,
        ]

        if raw_crop.shape[0] != square_dim or raw_crop.shape[1] != square_dim:
            # Pad it to make it always square_dim x square_dim with a black letterbox.
            try:
                raw_crop = np.array(
                    ImageOps.pad(
                        Image.fromarray(raw_crop),
                        (square_dim, square_dim),
                        color="black",
                    )
                )
            except ValueError:
                return False, None

        # If this happens, the fighter must be entirely offscreen
        if raw_crop.shape[0] == 0 or raw_crop.shape[1] == 0:
            print("Bad crop")
            return False, None

        crop = imutils.resize(raw_crop, width=output_size)

        # No clue imutils doesn't resize a square image perfectly.
        # Exception: Bad output shape, expected (128, 128, 3) got (127, 128, 3) and had raw_crop
        # shape (196, 196, 3)
        if crop.shape[0] != output_size or crop.shape[1] != output_size:
            # Pad it to make it always output_size x output_size with a black letterbox.
            crop = np.array(
                ImageOps.pad(Image.fromarray(crop), (output_size, output_size), color="black")
            )

        expected_output_shape = (output_size, output_size, 3)
        if crop.shape != expected_output_shape:
            raise Exception(
                f"Bad output shape, expected {expected_output_shape} got {crop.shape} and had "
                f"raw_crop shape {raw_crop.shape}"
            )
        return True, crop

    def __str__(self):
        return (
            f"{self.class_id} {self.center_x} {self.center_y} {self.crop_width} "
            + f"{self.crop_height} {self.confidence}"
        )

    def __repr__(self):
        return str(self)


class Fighter:
    def __init__(
        self,
        frame_num: int,
        fighter_name: str = "",
        char_class_id: int = -1,
        crop=None,
        crop_confidence: float = -1.0,
        yolo_string: str = "",
        action: str = "",
        action_confidence: float = 0.0,
        advantage_state: str = "",
        fighter_id: int = -1,
        data=None,
    ):
        """
        @param data: dict containing all of either the ground truth information or AI predicted info
        """
        self.frame_num = frame_num
        self.char_class_id = char_class_id
        self.fighter_name = fighter_name
        self.fighter_id = fighter_id
        self.crop = crop
        self.crop_confidence = crop_confidence
        self.action = action
        self.action_confidence = action_confidence
        self.advantage_state = advantage_state
        self.damage = 0
        self.previous_damage = 0
        self.damage_delta = 0
        self.new_action = True
        self.num_frames_left = 25200
        self.previous_non_damaged_action = None
        self.frames_since_damaged = 0
        self.frames_since_hit = 0
        self.last_frame_in_tech_situation = -1
        self.last_frame_in_ledge_situation = -1
        self.hitstun_left = 0
        self.attack_connected = False
        self.status_kind = -1
        self.can_act = True
        self.previous_action = ""
        # Whenever we get a new move, it gets incremented by 1.
        self.move_counter = 0

        # The animation number that we get from Smash. These numbers are odd, and can start like
        # negative 40. Don't quite understand it.
        self.raw_animation_frame_num = 0.0

        # This is the animation frame num that we compute and update each frame.
        self.animation_frame_num = 1

        if yolo_string:
            class_id, x, y, crop_width, crop_height, conf = yolo_string.split(" ")
            self.char_class_id = int(class_id)
            self.fighter_name = CHAR_LIST[self.char_class_id]
            self.crop = YoloCrop(float(x), float(y), float(crop_width), float(crop_height))
            self.crop_confidence = float(conf)

        if data:
            self.set_from_json(data)

        assert self.crop, "No crop specified"
        assert self.fighter_name, "No fighter_name specified"

    def set_from_json(self, data):
        """ """
        # position in world space.
        self.position_in_world = [data["pos_x"], data["pos_y"], 0]
        self.damage = data["damage"]

        self.damage = data["damage"]
        self.facing = data["facing"]
        self.fighter_id = data["fighter_id"]
        self.motion_kind = data["motion_kind"]
        self.num_frames_left = data["num_frames_left"]
        self.pos_x = data["pos_x"]
        self.pos_y = data["pos_y"]
        self.shield_size = data["shield_size"]
        self.status_kind = data["status_kind"]
        self.stock_count = data["stock_count"]
        self.can_act = data.get("can_act", True)
        self.attack_connected = data["attack_connected"]
        self.raw_animation_frame_num = data.get("animation_frame_num", 0)
        self.stage_id = data["stage_id"]

        if data["stage_id"] not in STAGE_ENUM_TO_DATA:
            self.stage_id = 0

        self.stage = STAGE_ENUM_TO_DATA[self.stage_id]["name"]

        # TODO - determine fighter_name for fighter_id
        self.fighter_name = FIGHTER_ENUM_TO_NAME[data["fighter_name"]]

        # camera_fov = data["camera_fov"]
        # camera_fov = 30
        # Kalos is 50, other stages are 30. The game even returns 30 for Kalos, so I'm not trusting
        # the ground truth for this.
        camera_fov = STAGE_ENUM_TO_DATA[self.stage_id]["fov"]
        camera_position = data["camera_position"]
        target_position = data["camera_target_position"]
        self.extrinsics = calculate_lookat_matrix(
            list(camera_position.values()), list(target_position.values())
        )
        self.intrinsics = calculate_intrinsic_matrix(camera_fov, image_width=1280, image_height=720)
        self.point_in_pixel = project_point_to_pixel(
            self.position_in_world, self.intrinsics, self.extrinsics
        )

        # This only comes from AI prediction
        if "crop" in data:
            self.crop = YoloCrop.from_string(data["crop"])

        else:
            top_left = project_point_to_pixel(
                self.position_in_world + np.array([-10, 20, 0]),
                self.intrinsics,
                self.extrinsics,
            )
            top_right = project_point_to_pixel(
                self.position_in_world + np.array([10, 20, 0]),
                self.intrinsics,
                self.extrinsics,
            )
            bottom_left = project_point_to_pixel(
                self.position_in_world + np.array([-10, -3, 0]),
                self.intrinsics,
                self.extrinsics,
            )
            bottom_right = project_point_to_pixel(
                self.position_in_world + np.array([10, -3, 0]),
                self.intrinsics,
                self.extrinsics,
            )

            self.crop = YoloCrop.from_pixel_coordinates(
                1280,
                720,
                top_left[0],
                top_left[1],
                top_right[0],
                top_right[1],
                bottom_left[0],
                bottom_left[1],
                bottom_right[0],
                bottom_right[1],
            )

        # We need the padding to have it match up with params_labels.csv.
        padding = 12
        self.motion_hex = f"{self.motion_kind:#0{padding}x}"
        self.action_string = HEX_TO_ACTION.get(self.motion_hex, "")
        # if not self.action_string:
        #     print(f'No hex to action for hex {self.motion_hex}')
        self.action = get_anim_for_string_and_status_kind(self.action_string, self.status_kind)
        # self.action = get_animation_type_for_param_string(self.action_string)

        # Only in AI predicted data
        if "action" in data:
            self.action = data["action"]

        self.hitstun_left = data["hitstun_left"]
        # self.hit_status = data['hit_status']

    def update(self, frame_number: int, data):
        """
        Given a new frame's ground truth, update this instance of the Fighter. By having the
        previous frame's information and the current frame, we can determine changes in the
        fighter's state.

        The flow is:
        - first calculate the differences between the fighter's saved state and this new state.
        - update the fighter's saved state with this new state.
        """
        self.frame_num = frame_number
        # Copy previous frame's data over.
        self.previous_position_in_world = self.position_in_world
        self.previous_damage = self.damage
        self.previous_facing = self.facing
        self.previous_fighter_id = self.fighter_id
        self.previous_motion_kind = self.motion_kind
        self.previous_num_frames_left = self.num_frames_left
        self.previous_pos_x = self.pos_x
        self.previous_pos_y = self.pos_y
        self.previous_shield_size = self.shield_size
        self.previous_status_kind = self.status_kind
        self.previous_stock_count = self.stock_count
        self.previous_fighter_name = self.fighter_name
        self.previous_crop = self.crop
        self.previous_motion_hex = self.motion_hex
        self.previous_action_string = self.action_string
        self.previous_attack_connected = self.attack_connected
        self.previous_action = self.action

        # Load new frame's data.
        self.set_from_json(data)

        # Calculate differences. The max() is needed when the character dies and then comes
        # back with 0 damage. This causes the "Wait" animation to have a large negative value.
        self.damage_delta = max(self.damage - self.previous_damage, 0)
        self.new_action = self.previous_action != self.action
        if self.new_action:
            self.move_counter += 1

        self.animation_frame_num = 1 if self.new_action else self.animation_frame_num + 1

        self.frames_since_damaged = 0 if self.damage_delta else self.frames_since_damaged + 1
        self.frames_since_hit = 0 if self.damage_delta else self.frames_since_hit + 1

        # If a fighter is getting comboed, then we want the damage to be associated with the
        # fighter's last move before the combo started. Otherwise, the punish is associated with
        # "Damaged" action.
        if self.previous_action != "Damaged":
            self.previous_non_damaged_action = self.previous_action

        if self.in_tech_situation:
            self.last_frame_in_tech_situation = frame_number

        if self.in_ledge_situation:
            self.last_frame_in_ledge_situation = frame_number

    @property
    def time_remaining(self) -> str:
        total_seconds = self.num_frames_left / 60
        minutes, seconds = divmod(total_seconds, 60)
        seconds, milliseconds = divmod(seconds, 1)
        milliseconds = round(milliseconds * 100)
        return f"{int(minutes)}:{int(seconds):02d}.{milliseconds:02d}"

    def offset(self, other_fighter) -> str:
        """
        Returns where the current fighter is relative to the other fighter.
        So if the current fighter is to the right 10 units and above 20, it will be (10x, 20y)
        """
        return (self.pos_x - other_fighter.pos_x, self.pos_y - other_fighter.pos_y)

    def offset_str(self, other_fighter) -> str:
        """
        Returns where the current fighter is relative to the other fighter as a string.
        """
        offset = self.offset(other_fighter)
        return f"{offset[0]:.2f}x, {offset[1]:.2f}y"

    @property
    def anim_state(self) -> str:
        if self.fighter_name not in FIGHTER_FRAME_DATA:
            # return f'No frame data for {self.fighter_name}'
            return ""
        if self.action not in FIGHTER_FRAME_DATA[self.fighter_name]:
            # return f'No frame data for {self.action}'
            return ""

        move_frame_data = FIGHTER_FRAME_DATA[self.fighter_name][self.action]

        if not move_frame_data.startup or not move_frame_data.active_start:
            return ""

        if self.animation_frame_num < move_frame_data.startup:
            return "startup"

        if (
            self.animation_frame_num >= move_frame_data.active_start
            and self.animation_frame_num <= move_frame_data.active_end
        ):
            return "active"

        else:
            return "end lag"

    @property
    def status(self) -> str:
        if self.status_kind < 0:
            return "Undefined"

        if self.status_kind in STATUS_ENUM_TO_STRING:
            return STATUS_ENUM_TO_STRING[self.status_kind].replace("FIGHTER_STATUS_KIND_", "")

        if (
            self.fighter_name in FIGHTER_STATUS_ENUM_TO_STRING
            and self.status_kind in FIGHTER_STATUS_ENUM_TO_STRING[self.fighter_name]
        ):
            return FIGHTER_STATUS_ENUM_TO_STRING[self.fighter_name][self.status_kind]

        return f"Undefined ({self.status_kind})"

    def get_action_crop(self, image):
        """
        REMOVE THIS LATER

        Uses the Yolo crop to return the crop from the passed image. The output crop has to be
        128 x 128.
        @param image numpy array
        """
        (x, y) = self.crop.center_pixels(image.shape[1], image.shape[0])
        raw_crop = image[y - 64 : y + 64, x - 64 : x + 64, :]
        if raw_crop.shape[0] != 128 or raw_crop.shape[1] != 128:
            # Pad it to make it always 128 x 128 with a black letterbox.
            try:
                raw_crop = np.array(
                    ImageOps.pad(Image.fromarray(raw_crop), (128, 128), color="black")
                )
            except ValueError as e:
                print("Hit a bad ValueError ", e)

        return raw_crop

    def __str__(self):
        return (
            f"<{self.fighter_name}@{self.action} | {self.advantage_state} | "
            + f"{self.crop_confidence:.2f}%  {self.crop.center_x:.2f}x{self.crop.center_y:.2f}y />"
        )

    @property
    def in_tech_situation(self):
        """
        Whether or not the fighter is in a tech situation.
        """
        return ONTOLOGY["all"][self.action].get("option_group", "") == "tech"

    @property
    def in_ledge_situation(self):
        """
        Whether or not the fighter is in a tech situation.
        """
        return ONTOLOGY["all"][self.action].get("option_group", "") == "ledge"

    @property
    def using_damage_move(self):
        """
        Returns a bool for whether the move is a damaging move.
        """
        move_data = FIGHTER_FRAME_DATA[self.fighter_name][self.action]
        return move_data.base_damage and move_data.base_damage > 0

    def interp(self, b, percent, frame_num):
        crop_confidence = self.crop_confidence + (
            percent * (b.crop_confidence - self.crop_confidence)
        )
        crop = self.crop.interp(b.crop, percent)

        return Fighter(
            frame_num,
            fighter_name=self.fighter_name,
            char_class_id=self.char_class_id,
            crop=crop,
            crop_confidence=crop_confidence,
        )
