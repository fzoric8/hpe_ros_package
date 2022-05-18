##################################################
# Author       : Leonardo Cencetti
# Date         : 16.04.21
# Maintainer   : Leonardo Cencetti
# Email        : cencetti.leonardo@gmail.com
##################################################
import sys

import rospy

from drone import Drone, FlightMode
from .trajectories import Line, Static, Lissajous3DEight


class _Task:
    NAME = None

    def __init__(self):
        self.active = False
        self.done = False

    def run(self, drone):  # type: (Drone) -> None
        self.active = True
        self._notify_start()
        self._activity(drone)
        self.done = True
        self._notify_finish()
        self.active = False

    def _activity(self, drone):  # type: (Drone) -> None
        raise NotImplementedError

    def _notify_start(self):
        rospy.loginfo('Starting >>> {}'.format(self.NAME))

    def _notify_finish(self):
        rospy.loginfo('Finished <<< {}'.format(self.NAME))


class PreFlight(_Task):
    NAME = 'PreFlight'

    def __init__(self, timeout):  # type: (float)->None
        _Task.__init__(self)
        self._timeout = timeout

    def _activity(self, drone):  # type: (Drone) -> None
        drone.set_home()
        t0 = rospy.get_time()
        while drone.alive:
            drone.set_target_pose(drone.home)
            drone.set_mode(FlightMode.Offboard)
            drone.arm()
            drone.sleep()

            if drone.armed and drone.mode == FlightMode.Offboard.value:
                break
            if rospy.get_time() - t0 >= self._timeout:
                rospy.logerr('Preflight timeout, quitting...')
                if drone.armed:
                    drone.disarm()
                rospy.signal_shutdown('Preflight timeout')
                sys.exit(1)



