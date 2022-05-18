##################################################
# Author       : Leonardo Cencetti
# Date         : 16.04.21
# Maintainer   : Leonardo Cencetti
# Email        : cencetti.leonardo@gmail.com
##################################################
import rospy
from enum import Enum
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import ExtendedState, State
from mavros_msgs.srv import CommandBool, SetMode


class FlightMode(Enum):
    Manual = 'MANUAL'
    Acrobatic = 'ACRO'
    Altitude = 'ALTCTL'
    Position = 'POSCTL'
    Offboard = 'OFFBOARD'
    Stabilized = 'STABILIZED'
    RAttitude = 'RATTITUDE'
    AutoMission = 'AUTO.MISSION'
    AutoLoiter = 'AUTO.LOITER'
    AutoRTL = 'AUTO.RTL'
    AutoLand = 'AUTO.LAND'
    AutoRTGS = 'AUTO.RTGS'
    AutoReady = 'AUTO.READY'
    AutoTakeoff = 'AUTO.TAKEOFF'

    def __str__(self):
        return self.value


class Mission:
    def __init__(self, task_list):  # type: (list) -> None
        if not isinstance(task_list, list):
            raise TypeError('task_list must be a list of tasks')
        self._task_list = task_list
        self.active = False
        self.done = False

    def run(self, drone):  # type: (Drone) -> None
        if self.done:
            rospy.logerr('Mission already completed, aborting.')
            return

        self.active = True
        for task in self._task_list:
            task.run(drone)
        self.done = True
        self.active = False


class Drone:
    RATE = 30  # Hz

    def __init__(self, namespace='', mode_control=False, arming_control=False,
                 pose_src='local'):  # type: (str, bool, bool, str) -> None
        self.namespace = namespace

        self._extended_state = ExtendedState()
        self._prev_state = State()
        self._current_state = State()
        self._current_pose = PoseStamped()
        self._home_pose = None

        self._mode_ctl = mode_control
        self._arming_ctl = arming_control
        self._pose_src = pose_src

        #self._init_ros_interfaces()
        rospy.loginfo(
            'Drone initialized with:\n\tnamespace: {}\n\tmode-ctl: {}\n\tarm-ctl: {}'.format(namespace, mode_control,
                                                                                             arming_control))
        self._init_ros_interfaces()
        #self._startup_sleep()
        self._flight_ready = False

    def _init_ros_interfaces(self):
        # Publishers
        self._target_pos_pub = rospy.Publisher('{}/mavros/setpoint_position/local'.format(self.namespace),
                                               PoseStamped, queue_size=10)
        
        # Subscribers
        position_src = dict(local='local_position', mocap='vision_pose')[self._pose_src]

        self._state_sub = rospy.Subscriber('{}/mavros/state'.format(self.namespace), State, self._state_cb)
        self._pos_sub = rospy.Subscriber('{}/mavros/{}/pose'.format(self.namespace, position_src), PoseStamped,
                                         self._pos_cb)
        self._ext_state_sub = rospy.Subscriber('{}/mavros/extended_state'.format(self.namespace), ExtendedState,
                                               self._extended_state_cb)

        # Services
        self._arming_client = rospy.ServiceProxy('{}/mavros/cmd/arming'.format(self.namespace), CommandBool)
        self._set_mode_client = rospy.ServiceProxy('{}/mavros/set_mode'.format(self.namespace), SetMode)

        #rospy.init_node('drone_interface', anonymous=True)
        self._rate = rospy.Rate(5)

    def _startup_sleep(self):
        for _ in range(100):
            self._rate.sleep()

    def set_target_pose(self, pose):  # type: (PoseStamped) -> None

        if not isinstance(pose, PoseStamped):
            raise TypeError('pose must be a PoseStamped instance')
        self._target_pos_pub.publish(pose)

    def get_local_pose(self):  # type: () -> PoseStamped
        return self._current_pose

    def get_state(self):  # type: () -> State
        return self._current_state

    def run_mission(self, mission):
        if not isinstance(mission, Mission):
            raise TypeError('mission must be a Mission instance')
        mission.run(self)

    @property
    def alive(self):  # type: () -> bool
        print("drone: {}".format(self._current_state.connected))
        return not rospy.is_shutdown() and self._current_state.connected

    @property
    def landed(self):  # type: () -> bool
        return self._extended_state.landed_state == 1

    @property
    def armed(self):  # type: () -> bool
        return self._current_state.armed

    @property
    def mode(self):  # type: () -> str
        return self._current_state.mode

    @property
    def home(self):  # type: () -> PoseStamped
        return self._home_pose

    def set_home(self, pose=None):  # type: (PoseStamped) -> None
        if pose is None:
            self._home_pose = self.get_local_pose()
            return
        if not isinstance(pose, PoseStamped):
            raise TypeError('pose must be a PoseStamped instance')
        self._home_pose = pose

    def sleep(self):
        self._rate.sleep()

    def set_mode(self, mode, blocking=False):  # type: (FlightMode, bool) -> None
        if not isinstance(mode, FlightMode):
            raise TypeError('mode must be a FlightMode instance')
        if mode.value == self._current_state.mode:
            rospy.logdebug_throttle_identical(2, 'FCU already in {} mode.'.format(mode))
            return
        if not self._mode_ctl:
            rospy.loginfo_throttle_identical(2, 'PLEASE SELECT {} MODE.'.format(str(mode).upper()))
            return
        rospy.loginfo_throttle_identical(2, 'Setting FCU to mode {} from {}'.format(mode, self._current_state.mode))
        while mode.value != self._current_state.mode:
            self._set_mode_client(base_mode=0, custom_mode=mode.value)
            if not blocking:
                break
        if mode.value == self._current_state.mode:
            rospy.loginfo_throttle_identical(2, 'FCU set to mode {}'.format(mode))

    def arm(self):
        self._handle_arming(True)

    def disarm(self):
        self._handle_arming(False)

    def connect(self):
        if self._current_state.connected:
            rospy.logdebug_throttle(2, 'FCU already connected.')
            return
        # wait for FCU connection
        rospy.loginfo('Waiting for FCU connection')
        while not self._current_state.connected and not rospy.is_shutdown():
            self._rate.sleep()
        rospy.loginfo('FCU connected')

    def _handle_arming(self, arm):  # type: (bool) -> None
        if not self._arming_ctl:
            rospy.loginfo_throttle_identical(2, 'PLEASE {} THE DRONE.'.format('ARM' if arm else 'DISARM'))
            return

        label = ['Disarm', 'Arm'][arm]
        if self._current_state.armed is arm:
            rospy.logdebug_throttle_identical(2, 'FCU already {}ed.'.format(label.lower()))
            return

        rospy.loginfo_throttle_identical(2, '{}ing FCU...'.format(label))
        while self._current_state.armed != arm and not rospy.is_shutdown():
            self._arming_client(arm)
            self._rate.sleep()
        rospy.loginfo_throttle_identical(2, 'FCU {}ed.'.format(label.lower()))

    def _state_cb(self, state):  # type: (State) -> None
        self._current_state = state

    def _pos_cb(self, pose):  # type: (PoseStamped) -> None
        self._current_pose = pose

    def _extended_state_cb(self, data):  # type: (ExtendedState) -> None
        self._extended_state = data
