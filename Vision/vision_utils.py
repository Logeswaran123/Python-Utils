import os
import cv2
from collections import deque
import math
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import emoji

NaN = 0

NECK = 0
L_SHOULDER = 1
R_SHOULDER = 2
L_ELBOW = 3
R_ELBOW = 4
L_WRIST = 5
R_WRIST = 6
L_HIP = 7
R_HIP = 8
L_KNEE = 9
R_KNEE = 10
L_ANKLE = 11
R_ANKLE = 12

POSE_PAIRS = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [5, 11], [6, 12], [11, 13], [12, 14], [13, 15], [14, 16], [11, 12]]

KP_CONF_THRESH = 60
FALL_THRESHOLD = 8
WALKING_THRESHOLD = 4
WALKING_THRESHOLD2 = 10


class VideoReader:
    """ Wrapper class for cv2.VideoCapture """
    def __init__(self, filename):
        self.cap = cv2.VideoCapture(filename)
        self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._current_frame = 0

    def read_frame(self):
        """ Read a single frame from video """
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret == False or frame is None:
                return None
            self._current_frame += 1
        else:
            return None
        return frame

    def read_n_frames(self, n=1):
        """ Read n frames from video """
        frames_list = []
        for i in range(n):
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret == False or frame is None:
                    return None
                frames_list.append(frame)
                self._current_frame += 1
            else:
                return None
        return frames_list

    def is_opened(self):
        """ Wrapper for VideoCapture.isOpened method """
        return self.cap.isOpened()

    def get_frame_width(self):
        """ Getter method for frame width """
        return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    def get_frame_height(self):
        """ Getter method for frame frame height """
        return self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_video_fps(self):
        """ Getter method for video FPS """
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_current_frame(self):
        """ Getter method for current frame number """
        return self._current_frame

    def get_total_frames(self):
        """ Getter method for total frame count """
        return self._total_frames

    def release(self):
        """ Wrapper for VideoCapture.release """
        self.cap.release()

    def __del__(self):
        self.release()


class ActivityDetector(object):
    """
    ActivityDetector class to detect human activity from pose estimation and pose classification data
    """
    def __init__(self, window_size, frame_shape, is_adding_noise=False, fps=None, display_fps=True):

        # activity related parameters
        self._window_size = window_size
        self.fps = fps
        self.display_fps = display_fps
        self.fall_started = False
        self.fall_ended = False
        self.fall_count = 0
        self.fall_window_size = self.fps
        self.prev_fall_by_velocity = False
        self.prev_lying_detected = False
        self.last_height = 1.0
        self.cricital_velocity = (1+(0.1*(self.fps-10)/4))/self.fps
        self.displacement_threshold = self.cricital_velocity * 4
        self.frame_check = int(fps * 0.3)
        self.alert_timeout = int(fps * 1)
        self.values_full = deque(maxlen=self.frame_check)
        for i in range(self.frame_check):
            self.values_full.append(5)
        self.sitting_counter = 0
        self.walking_counter = 0
        self.handraise_counter = 0
        self.standing_counter = 0
        self.last_walking_counter = -1

        self.lying_counter = 0
        self.prev_action = 5
        self.walking_lst = deque(maxlen=5)
        self.walking_lst2 = deque(maxlen=5)
        self.falling_lst = deque(maxlen=4)
        self.falling_lst2 = deque(maxlen=4)
        for i in range(4):
            self.falling_lst.append(-1)
            self.falling_lst2.append(-1)

        self.d1 = [-1,-1]
        self.d2 = [-1,-1]

        self.last_fall_counter = -1
        self.last_handwave_counter = -1
        self.last_alert_text = ''

        self.last_right_dist = 0.0
        self.last_left_dist = 0.0

        self.remaining_counter = 0

        self.reset()

        # visualization related parameters
        self.width = frame_shape[0]
        self.height = frame_shape[1]

        self.font_size = int(self.height/24)
        self.font_size_large = int(self.height/12)
        self.heading_font = ImageFont.truetype(os.path.join('fonts', 'Roboto-Bold.ttf'), self.font_size)
        self.regular_font = ImageFont.truetype(os.path.join('fonts', 'Roboto-Medium.ttf'), self.font_size)
        self.highlight_font = ImageFont.truetype(os.path.join('fonts', 'Roboto-Bold.ttf'), self.font_size)
        self.highlight_font_large = ImageFont.truetype(os.path.join('fonts', 'Roboto-Medium.ttf'), self.font_size_large)
        self.emoji_font = ImageFont.truetype(os.path.join('fonts', 'Symbola.ttf'), self.font_size)
        self.emoji_font_large = ImageFont.truetype(os.path.join('fonts', 'Symbola.ttf'), self.font_size_large)
        self.audio_emoji = str(emoji.emojize(':speaker_high_volume:'))

        self.alert_color = (0,64,255)
        self.regular_color = (0,255,0)

        self.y0 = int(self.height/4)
        self.x0 = int(self.width/32)
        self.dy = int(self.height/14)

        self.e1 = int(self.width/8)
        self.e2 = int(self.width/6.5)
        self.e3 = int(self.height/200)

        self.activities = ["Standing", "Sitting", "Hand Raise", "Walking", "Falling Down", "Lying", "Emergency", "Call 911"]
        self.y_pos = {}
        for i, line in enumerate(self.activities):
            y = self.y0 + i*self.dy
            self.y_pos[line] = y


    def reset(self):
        """ Reset all the queues contain temporal data """
        self._x_deque = deque(maxlen=self._window_size)
        self._angles_deque = deque(maxlen=self._window_size)
        self._lens_deque = deque(maxlen=self._window_size)
        self._pre_x = None
        self._best_pre_x = None
        self._y_velocity = deque(maxlen=self._window_size)
        self.head = deque(maxlen=self._window_size)
        self.shoulder = deque(maxlen=self._window_size)
        self.hip = deque(maxlen=self._window_size)
        self.ankle = deque(maxlen=self._window_size)
        self.ratio = deque(maxlen=self._window_size)
        self.fall_by_velocity = deque(maxlen=self._window_size)
        self.pre_height = None


    def to_numpy(self, tensor):
        """ Unlity function to convert torch.Tensor to numpy array """
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def dist(self, A,B):
        """ Euclidean distance between two points A-B """
        diffx = (A[0] - B[0])**2
        diffy = (A[1] - B[1])**2
        return (diffx + diffy)**0.5

    def distx(self, A, B):
        diffx = B[0] - A[0]
        return diffx

    def disty(self, A, B):
        diffy = B[1] - A[1]
        return diffy

    def get_joint(self, x, idx):
        """ Getter for joint value """
        px = x[2*idx]
        py = x[2*idx+1]
        return px, py

    def check_joint(self, x, idx):
        """ Check whether the joint is valid or not """
        return x[2*idx] != NaN

    def has_nose(self, x):
        """ Check if a skeleton has a neck """
        return self.check_joint(x, NECK)
      
    def _compute_v_all_joints_2(self, l1, l2):
        """ Compute the velocity of joints """
        dxdy = l2 - l1
        vel = dxdy.tolist()
        return np.array(vel)

    def get_body_height(self, x):
      """ 
      Compute height of the body, which is defined as:
      the distance between `neck` and `thigh`.
      """
      x0, y0 = self.get_joint(x, NECK)

      # Get average thigh height
      x11, y11 = self.get_joint(x, L_ANKLE)
      x12, y12 = self.get_joint(x, R_ANKLE)
      if y11 == NaN and y12 == NaN:  # Invalid data
          return 1.0
      if y11 == NaN:
          x1, y1 = x12, y12
      elif y12 == NaN:
          x1, y1 = x11, y11
      else:
          x1, y1 = (x11 + x12) / 2, (y11 + y12) / 2

      # Get body height
      height = ((x0-x1)**2 + (y0-y1)**2)**(0.5)
      return height
    
    def angle_y(self,p1,p2):
        # TODO: NaN handling
        angle = math.atan2(p2[0] - p1[0], p2[1] - p1[1])
        angle = abs(angle * 180 / math.pi)
        angle = round(angle)

        return angle

    def angle_2(self, p1, p2):
        """ Calculate angle of a single line """
        xDiff = p2[0] - p1[0]
        yDiff = p2[1] - p1[1]
        return math.degrees(math.atan2(yDiff, xDiff))

    def angle(self,p1,p2,p3):
        """ Calculate angle between two lines """
        if(p1==(0,0) or p2==(0,0) or p3==(0,0)):
            return 0
        nume = p2[1]*(p1[0] - p3[0]) + p1[1]*(p3[0]-p2[0]) + p3[1] * (p2[0]-p1[0])
        deno = (p2[0]-p1[0])*(p1[0]-p3[0]) + (p2[1]-p1[1])*(p1[1]-p3[1])
        try:
            ang = math.atan(nume/deno)
            ang = ang * 180 / math.pi
            if ang < 0:
                ang = 180 + ang
            return ang
        except:
            return 90.0

    def detect_activity(self, key_pts, bbox_x1, counter=None):
        """ Detect the Human activity from Pose Estimation and Pose Classification data """

        Xdata = []
        Ydata = []

        diff = 0

        body_straight = False
        ratio_rule = False

        l_hip_knee_dist = NaN
        r_hip_knee_dist = NaN
        l_knee_ankle_dist = NaN
        r_knee_ankle_dist = NaN
        l_hip_knee_angle = NaN
        r_hip_knee_angle = NaN
        l_shoulder_hip_angle = NaN
        r_shoulder_hip_angle = NaN

        frame_check_handraise = int(self.fps * 0.1)
        len_keypoints = 17
        skeleton = key_pts[:,:2].tolist()

        for key in range(len(skeleton)):
            if key not in [1,2,3,4]:
                Xdata.append(skeleton[key][0]-bbox_x1)
                Ydata.append(skeleton[key][1])

        output_actions = [5, 0, 0]
        # output_actions[0] => 0-Sitting, 1-Standing, 2-Lying, 3-Walking, 4-Falling Down, 5-No Activity
        # output_actions[1] => 0-No Hand Raise, 1-Hand Raise
        # output_actions[2] => 0-No Fall, 1-Falling Down

        l_hip_knee_dist = self.dist((Xdata[L_HIP],Ydata[L_HIP]), (Xdata[L_KNEE],Ydata[L_KNEE]))
        r_hip_knee_dist = self.dist((Xdata[R_HIP],Ydata[R_HIP]), (Xdata[R_KNEE],Ydata[R_KNEE]))

        l_knee_ankle_dist = self.dist((Xdata[L_KNEE],Ydata[L_KNEE]), (Xdata[L_ANKLE],Ydata[L_ANKLE]))
        r_knee_ankle_dist = self.dist((Xdata[R_KNEE],Ydata[R_KNEE]), (Xdata[R_ANKLE],Ydata[R_ANKLE]))
        
        l_shoulder_hip_angle = self.angle_y((Xdata[L_SHOULDER],Ydata[L_SHOULDER]), (Xdata[L_HIP],Ydata[L_HIP]))
        r_shoulder_hip_angle = self.angle_y((Xdata[R_SHOULDER],Ydata[R_SHOULDER]), (Xdata[R_HIP],Ydata[R_HIP]))

        l_hip_knee_angle = self.angle_y((Xdata[L_HIP],Ydata[L_HIP]), (Xdata[L_KNEE],Ydata[L_KNEE]))
        r_hip_knee_angle = self.angle_y((Xdata[R_HIP],Ydata[R_HIP]), (Xdata[R_KNEE],Ydata[R_KNEE]))

        angle_lying_left = self.angle_2((Ydata[L_SHOULDER],Xdata[L_SHOULDER]),(Ydata[L_HIP],Xdata[L_HIP]))
        angle_lying_right = self.angle_2((Ydata[R_SHOULDER],Xdata[R_SHOULDER]),(Ydata[R_HIP],Xdata[R_HIP]))

        if l_knee_ankle_dist != 0.0:
            l_ratio = l_hip_knee_dist / l_knee_ankle_dist
        else:
            l_ratio = 0.0
        if r_knee_ankle_dist != 0.0:
            r_ratio = r_hip_knee_dist / r_knee_ankle_dist
        else:
            r_ratio = 0.0

        if l_ratio > 0.8 and r_ratio > 0.8:
            ratio_rule = True

        if l_shoulder_hip_angle <= 25.0 and r_shoulder_hip_angle <= 25.0 and l_hip_knee_angle <= 45.0 and r_hip_knee_angle <= 45.0:
            body_straight = True

        # Standing & Walking
        if body_straight: # and ratio_rule:
            self.standing_counter += 1
            self.sitting_counter = 0

            new_right_dist = self.dist((Xdata[R_HIP],Ydata[R_HIP]), (Xdata[R_SHOULDER],Ydata[R_SHOULDER]))
            new_left_dist = self.dist((Xdata[L_HIP],Ydata[L_HIP]), (Xdata[L_SHOULDER],Ydata[L_SHOULDER]))

            right_change = self.last_right_dist - new_right_dist
            left_change = self.last_left_dist - new_left_dist

            self.last_right_dist = new_right_dist
            self.last_left_dist = new_left_dist

            centerCoord = [(Xdata[L_HIP]+Xdata[R_HIP])/2, (Ydata[L_HIP]+Ydata[R_HIP])/2]
            ankleDist = [Xdata[L_ANKLE]-Xdata[R_ANKLE], Ydata[L_ANKLE]-Ydata[R_ANKLE]]

            self.walking_lst.append(centerCoord)
            self.walking_lst2.append(ankleDist)
            if len(self.walking_lst) == 5:
                diffx1 = self.distx(self.walking_lst[0], self.walking_lst[1])
                diffx2 = self.distx(self.walking_lst[1], self.walking_lst[2])
                diffx3 = self.distx(self.walking_lst[2], self.walking_lst[3])
                diffx4 = self.distx(self.walking_lst[3], self.walking_lst[4])
                diffy1 = self.disty(self.walking_lst[0], self.walking_lst[1])
                diffy2 = self.disty(self.walking_lst[1], self.walking_lst[2])
                diffy3 = self.disty(self.walking_lst[2], self.walking_lst[3])
                diffy4 = self.disty(self.walking_lst[3], self.walking_lst[4])

                dist1 = self.dist(self.walking_lst2[0], self.walking_lst2[1])
                dist2 = self.dist(self.walking_lst2[1], self.walking_lst2[2])
                dist3 = self.dist(self.walking_lst2[2], self.walking_lst2[3])
                dist4 = self.dist(self.walking_lst2[3], self.walking_lst2[4])

                diffx = (diffx1 + diffx2 + diffx3 + diffx4) / 3
                diffy = (diffy1 + diffy2 + diffy3 + diffy4) / 3
                dist = (dist1 + dist2 + dist3 + dist4) / 4
                if abs(diffx) > WALKING_THRESHOLD or abs(diffy) > WALKING_THRESHOLD or abs(dist) > WALKING_THRESHOLD2:
                    self.walking_counter += 1
                    if self.walking_counter >= self.frame_check:
                        output_actions[0] = 3
                        self.values_full.append(3)
                        self.last_walking_counter = counter
                    if self.standing_counter > 0:
                        self.standing_counter -= 1

                elif self.standing_counter >= self.frame_check:
                    if (counter - self.last_walking_counter < 10) and self.last_walking_counter > 0:
                        output_actions[0] = 3
                        self.values_full.append(3)
                    else:
                        output_actions[0] = 1
                        self.values_full.append(1)

            elif self.standing_counter >= self.frame_check:
                output_actions[0] = 1
                self.values_full.append(1)

        ratio_rule = False
        angle_rule = False

        if (l_ratio < 0.6 and l_ratio > 0.1) or (r_ratio < 0.6 and r_ratio > 0.1):
            ratio_rule = True

        if (l_hip_knee_angle > 35.0 and l_hip_knee_angle < 100.0) and (r_hip_knee_angle > 35.0 and r_hip_knee_angle < 100.0):
            angle_rule = True

        lying = False

        if not body_straight:
            if (abs(angle_lying_left) >= 20) and (abs(angle_lying_left) <= 90) or (abs(angle_lying_right) >= 20) and (abs(angle_lying_right) <= 90):
                self.lying_counter += 1
                if self.lying_counter >= self.frame_check * 3.33:
                    output_actions[0] = 2
                    self.values_full.append(2)
                    lying = True

        # Sitting & Lying
        if ratio_rule or angle_rule:
            self.sitting_counter += 1
            self.standing_counter = 0

            if self.sitting_counter >= self.frame_check:
                res = 0
                self.sitting_counter += 1
                output_actions[0] = 0
                self.values_full.append(0)
            
            if (abs(angle_lying_left) >= 20) and (abs(angle_lying_left) <= 90) or (abs(angle_lying_right) >= 20) and (abs(angle_lying_right) <= 90):
                if not lying:
                    self.lying_counter += 1
                    self.sitting_counter -= 1
                    if self.lying_counter >= self.frame_check * 3.33:
                        output_actions[0] = 2
                        self.values_full.append(2)

        # Handraise activity
        if(Ydata[L_WRIST] < Ydata[L_SHOULDER] and Ydata[L_WRIST] < Ydata[R_SHOULDER]) or (Ydata[R_WRIST]<Ydata[L_SHOULDER] and Ydata[R_WRIST] < Ydata[R_SHOULDER]):
            if output_actions[0] == 2: # condition in lying posture
                if Ydata[L_WRIST] < Ydata[R_WRIST]:
                    hand_angle = self.angle_2((Ydata[L_WRIST],Xdata[L_WRIST]),(Ydata[L_ELBOW],Xdata[L_ELBOW]))
                else:
                    hand_angle = self.angle_2((Ydata[R_WRIST],Xdata[R_WRIST]),(Ydata[R_ELBOW],Xdata[R_ELBOW]))
                if hand_angle < 25.0 and hand_angle > -25.0:
                    self.handraise_counter += 1
                else:
                    self.handraise_counter = 0
            else:
                self.handraise_counter += 1
            if self.handraise_counter >= self.frame_check:
                output_actions[1] = 1
        else:
            self.handraise_counter = 0

        if Ydata[L_HIP] != -1 and Ydata[R_HIP] != -1:
            self.falling_lst.append((Ydata[L_HIP]+Ydata[R_HIP])/2)
        elif Ydata[L_HIP] != -1:
            self.falling_lst.append(Ydata[L_HIP])
        elif Ydata[R_HIP] != -1:
            self.falling_lst.append(Ydata[R_HIP])

        if Ydata[0] != -1:
            self.falling_lst2.append(Ydata[0])

        diff_fall_1a = 0
        diff_fall_1b = 0
        diff_fall_1c = 0
        diff_fall_2a = 0
        diff_fall_2b = 0
        diff_fall_2c = 0

        check_fall = False

        if self.falling_lst[0] != -1.0 and self.falling_lst[1] != -1.0:
            diff_fall_1a = self.falling_lst[1] - self.falling_lst[0]
            check_fall = True
        if self.falling_lst[1] != -1.0 and self.falling_lst[2] != -1.0:
            diff_fall_1b = self.falling_lst[2] - self.falling_lst[1]
            check_fall = True
        if self.falling_lst[2] != -1.0 and self.falling_lst[3] != -1.0:
            diff_fall_1c = self.falling_lst[3] - self.falling_lst[2]
            check_fall = True

        if self.falling_lst2[0] != -1.0 and self.falling_lst2[1] != -1.0:
            diff_fall_2a = self.falling_lst2[1] - self.falling_lst2[0]
            check_fall = True
        if self.falling_lst2[1] != -1.0 and self.falling_lst2[2] != -1.0:
            diff_fall_2b = self.falling_lst2[2] - self.falling_lst2[1]
            check_fall = True
        if self.falling_lst2[2] != -1.0 and self.falling_lst2[3] != -1.0:
            diff_fall_2c = self.falling_lst2[3] - self.falling_lst2[2]
            check_fall = True

        if check_fall:
            if (diff_fall_1a > FALL_THRESHOLD/2 and diff_fall_2a > FALL_THRESHOLD) \
                or (diff_fall_1b > FALL_THRESHOLD/2 and diff_fall_2b > FALL_THRESHOLD) \
                    or (diff_fall_1c > FALL_THRESHOLD/2 and diff_fall_2c > FALL_THRESHOLD):
                if (diff_fall_1b - diff_fall_1a > 0 and diff_fall_1c - diff_fall_1b > 0) or \
                    (diff_fall_2b - diff_fall_2a < 0 and diff_fall_2c - diff_fall_2b < 0):
                    output_actions[0] = 4
                    output_actions[2] = 1
                    self.last_fall_counter = counter
                else:
                    output_actions[2] = 0
            else:
                output_actions[2] = 0
        else:
            output_actions[2] = 0

        self.values_full.append(output_actions[0])

        check_equal = False
        if self.values_full[-1] == self.values_full[-2] == self.values_full[-3]:
            check_equal = True

        if check_equal:
            if self.prev_action != self.values_full[-1] and output_actions[0] != 2:
                self.lying_counter = 0
            self.prev_action = self.values_full[-1]
        else:
            output_actions[0] = self.prev_action

        return output_actions


    def draw_text(self, draw, text, alert, counter, show_emoji=False, emoji_offset=(0,0), fps_value=None, skip_default=False):

        if not skip_default:
            # headers
            draw.text((int(self.x0/2),self.y0-3*self.dy),"Frame: {:}".format(counter), (255,255,153), font=self.heading_font)
            if self.display_fps and fps_value is not None:
                draw.text((int(self.x0/2),self.y0-2*self.dy),"FPS: {:.1f}".format(fps_value), (252,220,3), font=self.heading_font)
            draw.text((int(self.x0/2),self.y0-self.dy),"Activities", (255,255,153), font=self.heading_font)

            # non-detected activities
            for line in self.activities:
                if line != text:
                    draw.text((self.x0,self.y_pos[line]), line, (150,150,150), font=self.regular_font)

        # detected activity
        if text is not None:
            color = self.alert_color if alert else self.regular_color
            draw.text((self.x0,self.y_pos[text]), text, color, font=self.highlight_font)
            if show_emoji:
                draw.text(emoji_offset, self.audio_emoji, self.alert_color, font=self.emoji_font)


    def draw_alert_text(self, draw, text, text_offset=(0,0), emoji=None, emoji_offset=(0,0)):

        emoji_width, emoji_height = 0, 0
        text_width, text_height = draw.textsize(text, font=self.highlight_font_large)
        if emoji is not None:
            emoji_width, emoji_height = draw.textsize(emoji, font=self.emoji_font_large)
        width = text_width + emoji_width
        text_offset = (self.width//10 + ((self.width - width) // 2), (self.height//8 - text_height)//2)
        emoji_offset = (self.width//10 + (self.width//2 + text_width//2), (self.height//8 - emoji_height)//2)
        draw.text(text_offset, text, self.alert_color, font=self.highlight_font_large)
        draw.text(emoji_offset, emoji, self.alert_color, font=self.emoji_font_large)


    def plot_output(self, out_frame, key_pts, activity, counter, skeleton=False, fps_value=None):

        # Visualization
        overlay = out_frame.copy()

        alert_text = False
        if activity[3] != 0:
            if (counter - self.last_fall_counter < self.alert_timeout * 5) and self.last_fall_counter > 0:
                alert_text = True
                if self.remaining_counter == 0 or activity[3] != self.last_alert_text:
                    self.remaining_counter = self.alert_timeout * 2
                    if activity[3] == 1:
                        self.last_alert_text = 'Call 911'
                    elif activity[3] == 2:
                        self.last_alert_text = 'Emergency'

        if self.remaining_counter > 0:
            alert_text = True
            self.remaining_counter -= 1
        else:
            self.remaining_counter = 0
            self.last_alert_text = ''

        alpha = 0.5
        cv2.rectangle(overlay, (0,0), (self.width//5,self.height) , (25,25,25) , -1)
        if alert_text:
            cv2.rectangle(overlay, (self.width//5,0), (self.width,self.height//8), (25,25,25) , -1)
        out_frame = cv2.addWeighted(overlay, alpha, out_frame, 1 - alpha, 0)

        # Draw Key Points to Frame
        for _, mat in enumerate(key_pts):
            x_coord, y_coord, conf = int(mat[0]), int(mat[1]), int(mat[2])
            if conf >= KP_CONF_THRESH:
                cv2.circle(out_frame, (x_coord, y_coord), 4, (252, 173, 3), -1)

        # Draw Human Skeleton to Frame
        if skeleton:
            for pair in POSE_PAIRS:
                idx1 = pair[0]
                idx2 = pair[1]
                if key_pts[idx1][2] >= KP_CONF_THRESH and key_pts[idx2][2] >= KP_CONF_THRESH:
                    cv2.line(out_frame, tuple(key_pts[idx1][:2]), tuple(key_pts[idx2][:2]), (252,194,3), 3)

        p_img = Image.fromarray(out_frame)
        draw = ImageDraw.Draw(p_img)

        text = None
        alert = False
        show_emoji = False
        emoji_offset = 0
        falling = False

        if activity[2] == 1 or (self.last_fall_counter > 0 and counter - self.last_fall_counter < self.alert_timeout):
            text = 'Falling Down'
            alert = True
            falling = True
        elif activity[1] == 1:
            text = 'Hand Raise'
            alert = True

        self.draw_text(draw, text, alert, counter, fps_value=fps_value)

        text = None
        alert = False
        if not falling:
            if activity[0] == 0:
                text = 'Sitting'
            elif activity[0] == 1:
                text = 'Standing'
            elif activity[0] == 2:
                text = 'Lying'
            elif activity[0] == 3:
                text = 'Walking'

        self.draw_text(draw, text, alert, counter, skip_default=True)

        text = None
        if activity[3] == 1:
            text = 'Call 911'
            alert = True
            show_emoji = True
            emoji_offset = (self.e1,self.y_pos[text]+self.e3)
        elif activity[3] == 2:
            text = 'Emergency'
            alert = True
            show_emoji = True
            emoji_offset = (self.e2,self.y_pos[text]+self.e3)

        if text is not None:
            self.draw_text(draw, text, alert, counter, show_emoji, emoji_offset, skip_default=True)

        if alert_text is True and self.last_alert_text != '':
            self.draw_alert_text(draw, '{} + {}'.format('Falling', self.last_alert_text), emoji=self.audio_emoji)

        res_frame = np.array(p_img)

        return res_frame


    def toggle_fps(self):
        if self.display_fps is True:
            self.display_fps = False
        else:
            self.display_fps = True
