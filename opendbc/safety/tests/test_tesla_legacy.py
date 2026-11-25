#!/usr/bin/env python3
# Safety tests for the legacy Tesla Model S/X mode (teslaLegacy): Autopilot HW1, HW2, HW3 and the
# external-panda (powertrain bus) setup. The HW1 class runs the full common car safety suite.
import unittest
import numpy as np

from opendbc.car.lateral import get_max_angle_delta_vm, get_max_angle_vm
from opendbc.car.structs import CarParams
from opendbc.car.tesla.values import CarControllerParams, TeslaSafetyFlags
from opendbc.car.vehicle_model import VehicleModel
import opendbc.safety.tests.common as common
from opendbc.safety.tests.common import CANPackerSafety, away_round, round_speed

MSG_DAS_steeringControl = 0x488
MSG_APS_eacMonitor = 0x27d
MSG_DAS_control_HW1 = 0x2b9
MSG_DAS_control_PT = 0x2bf

STEER_CONTROL_NONE = 0
STEER_CONTROL_ANGLE = 1
STEER_CONTROL_LKA = 2

CRUISE_OFF = 0
CRUISE_ENABLED = 2
CRUISE_STANDSTILL = 3

BRAKE_NOT_APPLIED = 1
BRAKE_APPLIED = 2


def legacy_safety_VM():
  # tesla_legacy.h steering limits are based on TESLA_MODEL_S_HW3, same as the legacy CarController
  from opendbc.car.tesla.interface import CarInterface
  return VehicleModel(CarInterface.get_non_essential_params("TESLA_MODEL_S_HW3"))


def round_angle(apply_angle, can_offset=0):
  apply_angle_can = (apply_angle + 1638.35) / 0.1 + can_offset
  # 0.49999_ == 0.5
  rnd_offset = 1e-5 if apply_angle >= 0 else -1e-5
  return away_round(apply_angle_can + rnd_offset) * 0.1 - 1638.35


class TestTeslaLegacyHW1Safety(common.CarSafetyTest, common.AngleSteeringSafetyTest, common.LongitudinalAccelSafetyTest):
  DBC = "tesla_can"
  SAFETY_MODEL = CarParams.SafetyModel.teslaLegacy
  SAFETY_PARAM = TeslaSafetyFlags.FLAG_HW1

  TX_MSGS = [[MSG_DAS_steeringControl, 0], [MSG_DAS_control_HW1, 0]]
  RELAY_MALFUNCTION_ADDRS = {0: (MSG_DAS_steeringControl, MSG_DAS_control_HW1)}
  FWD_BLACKLISTED_ADDRS = {2: [MSG_DAS_steeringControl, MSG_DAS_control_HW1]}

  STANDSTILL_THRESHOLD = 0
  GAS_PRESSED_THRESHOLD = 0

  # Angle control limits
  STEER_ANGLE_MAX = 360  # deg
  DEG_TO_CAN = 10
  # Tesla uses get_max_angle_delta_vm and get_max_angle_vm; lateral limits are tested below
  ANGLE_RATE_BP = None
  ANGLE_RATE_UP = None
  ANGLE_RATE_DOWN = None
  LATERAL_FREQUENCY = 50  # Hz

  # Long control limits
  MAX_ACCEL = 2.0
  MIN_ACCEL = -3.48
  INACTIVE_ACCEL = 0.0

  cnt_angle_cmd = 0

  def setUp(self):
    super().setUp()
    self.VM = legacy_safety_VM()

  def _get_steer_cmd_angle_max(self, speed):
    return get_max_angle_vm(max(speed, 1), self.VM, CarControllerParams)

  def _angle_cmd_msg(self, angle: float, state: bool | int, increment_timer: bool = True, bus: int = 0):
    values = {"DAS_steeringAngleRequest": angle, "DAS_steeringControlType": int(state)}
    if increment_timer:
      self.safety.set_timer(self.cnt_angle_cmd * int(1e6 / self.LATERAL_FREQUENCY))
      self.__class__.cnt_angle_cmd += 1
    return self.packer.make_can_msg_safety("DAS_steeringControl", bus, values)

  def _angle_meas_msg(self, angle: float, hands_on_level: int = 0, eac_status: int = 1, eac_error_code: int = 0):
    values = {"EPAS_internalSAS": angle, "EPAS_handsOnLevel": hands_on_level,
              "EPAS_eacStatus": eac_status, "EPAS_eacErrorCode": eac_error_code}
    return self.packer.make_can_msg_safety("EPAS_sysStatus", 0, values)

  def _user_brake_msg(self, brake):
    return self.packer.make_can_msg_safety("BrakeMessage", 0, {"driverBrakeStatus": BRAKE_APPLIED if brake else BRAKE_NOT_APPLIED})

  def _speed_msg(self, speed):
    return self.packer.make_can_msg_safety("ESP_B", 0, {"ESP_vehicleSpeed": speed * 3.6})

  def _speed_msg_2(self, speed):
    return self._speed_msg(speed)

  def _vehicle_moving_msg(self, speed: float):
    # legacy safety takes vehicle_moving from DI_state's cruise STANDSTILL state
    state = CRUISE_STANDSTILL if speed <= self.STANDSTILL_THRESHOLD else CRUISE_ENABLED
    return self.packer.make_can_msg_safety("DI_state", 0, {"DI_cruiseState": state})

  def _user_gas_msg(self, gas):
    return self.packer.make_can_msg_safety("DI_torque1", 0, {"DI_pedalPos": gas})

  def _pcm_status_msg(self, enable):
    return self.packer.make_can_msg_safety("DI_state", 0, {"DI_cruiseState": CRUISE_ENABLED if enable else CRUISE_OFF})

  def _long_control_msg(self, set_speed, accel_limits=(0, 0), aeb_event=0, bus=0):
    values = {
      "DAS_setSpeed": set_speed,
      "DAS_aebEvent": aeb_event,
      "DAS_accelMin": accel_limits[0],
      "DAS_accelMax": accel_limits[1],
    }
    return self.packer.make_can_msg_safety("DAS_control", bus, values)

  def _accel_msg(self, accel: float):
    # For common.LongitudinalAccelSafetyTest
    return self._long_control_msg(10, accel_limits=(accel, max(accel, 0)))

  # ***** lateral *****

  def test_steering_wheel_disengage(self):
    # Disengage on hands-on level >= 3, or on a high angle rate fault (eacStatus 0, error 9)
    for hands_on_level in range(4):
      for eac_status in range(8):
        for eac_error_code in range(16):
          self.safety.set_controls_allowed(True)
          should_disengage = hands_on_level >= 3 or (eac_status == 0 and eac_error_code == 9)
          self.assertTrue(self._rx(self._angle_meas_msg(0, hands_on_level=hands_on_level, eac_status=eac_status,
                                                        eac_error_code=eac_error_code)))
          self.assertNotEqual(should_disengage, self.safety.get_controls_allowed())

          # Should not recover
          self.assertTrue(self._rx(self._angle_meas_msg(0)))
          self.assertNotEqual(should_disengage, self.safety.get_controls_allowed())

  def test_steering_control_type(self):
    # Only NONE and ANGLE_CONTROL may be sent (no LANE_KEEP_ASSIST or above)
    self.safety.set_controls_allowed(True)
    for steer_control_type in range(4):
      should_tx = steer_control_type in (STEER_CONTROL_NONE, STEER_CONTROL_ANGLE)
      self.assertEqual(should_tx, self._tx(self._angle_cmd_msg(0, state=steer_control_type)))

  def test_stock_lkas_passthrough(self):
    no_lkas_msg = self._angle_cmd_msg(0, state=False)
    no_lkas_msg_cam = self._angle_cmd_msg(0, state=True, bus=2)
    lkas_msg_cam = self._angle_cmd_msg(0, state=STEER_CONTROL_LKA, bus=2)

    # stock system sends no LKAS -> no forwarding, and OP is allowed to TX
    self.assertTrue(self._rx(no_lkas_msg_cam))
    self.assertEqual(-1, self.safety.safety_fwd_hook(2, no_lkas_msg_cam.addr))
    self.assertTrue(self._tx(no_lkas_msg))

    # stock system sends LKAS -> forwarding, and OP is not allowed to TX
    self.assertTrue(self._rx(lkas_msg_cam))
    self.assertEqual(0, self.safety.safety_fwd_hook(2, lkas_msg_cam.addr))
    self.assertFalse(self._tx(no_lkas_msg))

    # LKAS that starts while openpilot is engaged is not respected
    self.assertTrue(self._rx(no_lkas_msg_cam))
    self.safety.set_controls_allowed(True)
    self.assertTrue(self._rx(lkas_msg_cam))
    self.assertEqual(-1, self.safety.safety_fwd_hook(2, lkas_msg_cam.addr))
    self.assertTrue(self._tx(no_lkas_msg))

  def test_angle_cmd_when_enabled(self):
    # lateral acceleration and jerk are tested below
    pass

  def test_rx_hook_speed_mismatch(self):
    # legacy S/X safety has a single vehicle speed source (ESP_B), nothing to cross-check
    pass

  def _round_speed(self, speed):
    # match ESP_vehicleSpeed rounding on CAN (0.01 kph)
    return round_speed(away_round(speed / 0.01 * 3.6) * 0.01 / 3.6)

  def test_lateral_accel_limit(self):
    for speed in np.linspace(0, 40, 100):
      speed = self._round_speed(max(speed, 1))
      for sign in (-1, 1):
        self.safety.set_controls_allowed(True)
        self._reset_speed_measurement(speed + 1)  # safety fudges the speed

        # angle signal can't represent 0, so it biases one unit down
        angle_unit_offset = -1 if sign == -1 else 0

        # at limit (safety tolerance adds 1)
        max_angle = round_angle(get_max_angle_vm(speed, self.VM, CarControllerParams), angle_unit_offset + 1) * sign
        max_angle = np.clip(max_angle, -self.STEER_ANGLE_MAX, self.STEER_ANGLE_MAX)
        self.safety.set_desired_angle_last(round(max_angle * self.DEG_TO_CAN))
        self.assertTrue(self._tx(self._angle_cmd_msg(max_angle, True)))

        # 1 unit above limit
        max_angle_raw = round_angle(get_max_angle_vm(speed, self.VM, CarControllerParams), angle_unit_offset + 2) * sign
        max_angle = np.clip(max_angle_raw, -self.STEER_ANGLE_MAX, self.STEER_ANGLE_MAX)
        self._tx(self._angle_cmd_msg(max_angle, True))

        # at low speeds max angle is above 360, so adding 1 has no effect
        should_tx = abs(max_angle_raw) >= self.STEER_ANGLE_MAX
        self.assertEqual(should_tx, self._tx(self._angle_cmd_msg(max_angle, True)))

  def test_lateral_jerk_limit(self):
    for speed in np.linspace(0, 40, 100):
      speed = self._round_speed(max(speed, 1))
      for sign in (-1, 1):
        self.safety.set_controls_allowed(True)
        self._reset_speed_measurement(speed + 1)  # safety fudges the speed
        self._tx(self._angle_cmd_msg(0, True))

        # angle signal can't represent 0, so it biases one unit down
        angle_unit_offset = 1 if sign == -1 else 0

        # Stay within limits: up, hold, down
        max_angle_delta = round_angle(get_max_angle_delta_vm(speed, self.VM, CarControllerParams), angle_unit_offset) * sign
        self.assertTrue(self._tx(self._angle_cmd_msg(max_angle_delta, True)))
        self.assertTrue(self._tx(self._angle_cmd_msg(max_angle_delta, True)))
        self.assertTrue(self._tx(self._angle_cmd_msg(0, True)))

        # Too high rates: up, hold, down, recover
        max_angle_delta = round_angle(get_max_angle_delta_vm(speed, self.VM, CarControllerParams), angle_unit_offset + 1) * sign
        self.assertFalse(self._tx(self._angle_cmd_msg(max_angle_delta, True)))
        self.safety.set_desired_angle_last(round(max_angle_delta * self.DEG_TO_CAN))
        self.assertTrue(self._tx(self._angle_cmd_msg(max_angle_delta, True)))
        self.assertFalse(self._tx(self._angle_cmd_msg(0, True)))
        self.assertTrue(self._tx(self._angle_cmd_msg(0, True)))

  # ***** longitudinal *****

  def test_no_aeb(self):
    self.safety.set_controls_allowed(True)
    for aeb_event in range(4):
      self.assertEqual(aeb_event == 0, self._tx(self._long_control_msg(10, aeb_event=aeb_event)))

  def test_stock_aeb_passthrough(self):
    no_aeb_msg = self._long_control_msg(10, aeb_event=0)
    no_aeb_msg_cam = self._long_control_msg(10, aeb_event=0, bus=2)
    aeb_msg_cam = self._long_control_msg(10, aeb_event=1, bus=2)

    # stock system sends no AEB -> no forwarding, and OP is allowed to TX
    self.assertTrue(self._rx(no_aeb_msg_cam))
    self.assertEqual(-1, self.safety.safety_fwd_hook(2, no_aeb_msg_cam.addr))
    self.assertTrue(self._tx(no_aeb_msg))

    # stock system sends AEB -> forwarding, and OP is not allowed to TX
    self.assertTrue(self._rx(aeb_msg_cam))
    self.assertEqual(0, self.safety.safety_fwd_hook(2, aeb_msg_cam.addr))
    self.assertFalse(self._tx(no_aeb_msg))

    # stock system stops AEB -> no forwarding, and OP is allowed to TX again
    self.assertTrue(self._rx(no_aeb_msg_cam))
    self.assertEqual(-1, self.safety.safety_fwd_hook(2, no_aeb_msg_cam.addr))
    self.assertTrue(self._tx(no_aeb_msg))

  def test_prevent_reverse(self):
    # Both accel limits negative could make the car reverse after standstill
    self.safety.set_controls_allowed(True)
    self.assertTrue(self._tx(self._long_control_msg(10, accel_limits=(-1.1, 0.8))))
    self.assertFalse(self._tx(self._long_control_msg(10, accel_limits=(-1.1, -0.6))))
    self.assertFalse(self._tx(self._long_control_msg(0, accel_limits=(-0.6, -1.1))))
    self.assertFalse(self._tx(self._long_control_msg(0, accel_limits=(-0.1, -0.1))))


class TestTeslaLegacyHW2Safety(common.SafetyTestBase):
  # HW2: steering only, APS_eacMonitor is blocked from the camera side
  DBC = "tesla_can"
  SAFETY_MODEL = CarParams.SafetyModel.teslaLegacy
  SAFETY_PARAM = TeslaSafetyFlags.FLAG_HW2
  TX_MSGS = [[MSG_DAS_steeringControl, 0], [MSG_APS_eacMonitor, 0]]
  CHASSIS_BUS = 0

  def _speed_msg(self, speed):
    return self.packer.make_can_msg_safety("ESP_B", self.CHASSIS_BUS, {"ESP_vehicleSpeed": speed * 3.6})

  def _pcm_status_msg(self, enable):
    return self.packer.make_can_msg_safety("DI_state", self.CHASSIS_BUS,
                                           {"DI_cruiseState": CRUISE_ENABLED if enable else CRUISE_OFF})

  def _user_brake_msg(self, brake):
    return self.packer.make_can_msg_safety("BrakeMessage", self.CHASSIS_BUS,
                                           {"driverBrakeStatus": BRAKE_APPLIED if brake else BRAKE_NOT_APPLIED})

  def test_fwd_hook(self):
    # -1 = blocked, otherwise the bus it is forwarded to
    self.assertEqual(-1, self.safety.safety_fwd_hook(2, MSG_APS_eacMonitor))
    self.assertEqual(-1, self.safety.safety_fwd_hook(2, MSG_DAS_steeringControl))
    self.assertEqual(0, self.safety.safety_fwd_hook(2, MSG_DAS_control_HW1))
    self.assertEqual(0, self.safety.safety_fwd_hook(2, MSG_DAS_control_PT))
    self.assertEqual(2, self.safety.safety_fwd_hook(0, MSG_APS_eacMonitor))

  def test_chassis_messages(self):
    self.assertTrue(self._rx(self._speed_msg(10)))
    self.assertTrue(self._rx(self._pcm_status_msg(True)))
    self.assertTrue(self.safety.get_controls_allowed())
    self.assertTrue(self._rx(self._user_brake_msg(True)))
    self.assertTrue(self.safety.get_brake_pressed_prev())
    self.assertTrue(self._rx(self._pcm_status_msg(False)))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_no_long_control(self):
    # HW2 has no DAS_control in its TX list
    self.safety.set_controls_allowed(True)
    msg = CANPackerSafety("tesla_can").make_can_msg_safety("DAS_control", 0, {"DAS_setSpeed": 10})
    self.assertFalse(self._tx(msg))


class TestTeslaLegacyHW3Safety(TestTeslaLegacyHW2Safety):
  # HW3: chassis messages come on bus 1
  SAFETY_PARAM = TeslaSafetyFlags.FLAG_HW3
  CHASSIS_BUS = 1

  def test_chassis_bus_0_ignored(self):
    msg = self.packer.make_can_msg_safety("DI_state", 0, {"DI_cruiseState": CRUISE_ENABLED})
    self._rx(msg)
    self.assertFalse(self.safety.get_controls_allowed())


class TestTeslaLegacyExternalPandaSafety(common.SafetyTestBase):
  # Second panda on the powertrain bus of an HW2/HW3 car: longitudinal only
  DBC = "tesla_powertrain"
  SAFETY_MODEL = CarParams.SafetyModel.teslaLegacy
  SAFETY_PARAM = TeslaSafetyFlags.FLAG_HW2 | TeslaSafetyFlags.FLAG_EXTERNAL_PANDA
  TX_MSGS = [[MSG_DAS_control_PT, 0]]

  def _long_control_msg(self, set_speed, accel_limits=(0, 0), aeb_event=0, bus=0):
    values = {"DAS_setSpeed": set_speed, "DAS_aebEvent": aeb_event,
              "DAS_accelMin": accel_limits[0], "DAS_accelMax": accel_limits[1]}
    return self.packer.make_can_msg_safety("DAS_control", bus, values)

  def test_rx_messages(self):
    self.assertTrue(self._rx(self.packer.make_can_msg_safety("DI_state", 0, {"DI_cruiseState": CRUISE_ENABLED})))
    self.assertTrue(self.safety.get_controls_allowed())
    self.assertTrue(self._rx(self.packer.make_can_msg_safety("DI_torque1", 0, {"DI_pedalPos": 20})))
    self.assertTrue(self.safety.get_gas_pressed_prev())
    self.assertTrue(self._rx(self.packer.make_can_msg_safety("BrakeMessage", 0, {"driverBrakeStatus": BRAKE_APPLIED})))
    self.assertTrue(self.safety.get_brake_pressed_prev())

  def test_long_control(self):
    self.safety.set_controls_allowed(True)
    self.assertTrue(self._tx(self._long_control_msg(10, accel_limits=(-1.0, 1.0))))
    self.assertFalse(self._tx(self._long_control_msg(10, accel_limits=(-1.0, 1.0), aeb_event=1)))
    self.assertFalse(self._tx(self._long_control_msg(10, accel_limits=(-1.0, -0.5))))
    self.assertFalse(self._tx(self._long_control_msg(10, accel_limits=(-1.0, 3.0))))

    # stock AEB on the camera side is forwarded and blocks openpilot
    self.assertEqual(-1, self.safety.safety_fwd_hook(2, MSG_DAS_control_PT))
    self.assertTrue(self._rx(self._long_control_msg(10, aeb_event=1, bus=2)))
    self.assertEqual(0, self.safety.safety_fwd_hook(2, MSG_DAS_control_PT))
    self.assertFalse(self._tx(self._long_control_msg(10, accel_limits=(-1.0, 1.0))))

  def test_no_steering(self):
    # the external panda never steers
    self.safety.set_controls_allowed(True)
    msg = CANPackerSafety("tesla_can").make_can_msg_safety("DAS_steeringControl", 0, {"DAS_steeringControlType": 1})
    self.assertFalse(self._tx(msg))
    self.assertEqual(2, self.safety.safety_fwd_hook(0, MSG_DAS_steeringControl))
    self.assertEqual(0, self.safety.safety_fwd_hook(2, MSG_DAS_steeringControl))


class TestTeslaLegacyIgnition(common.SafetyTestBase):
  DBC = "tesla_can"
  SAFETY_MODEL = None
  TX_MSGS: list = []

  def _gtw_status(self, counter, drive_rail, bus=0):
    return self.packer.make_can_msg_safety("GTW_status", bus, {"GTW_statusCounter": counter, "GTW_driveRailReq": drive_rail})

  def test_gtw_status_counter_gated(self):
    for bus in (0, 1):
      for i in range(16):
        self.safety.init_tests()
        self.safety.ignition_can_hook(self._gtw_status(i, 1, bus))
        self.assertFalse(self.safety.get_ignition_can())
        self.safety.ignition_can_hook(self._gtw_status((i + 1) % 16, 1, bus))
        self.assertTrue(self.safety.get_ignition_can())
        self.safety.ignition_can_hook(self._gtw_status((i + 2) % 16, 0, bus))
        self.assertFalse(self.safety.get_ignition_can())

  def test_gtw_status_wrong_counter(self):
    self.safety.init_tests()
    self.safety.ignition_can_hook(self._gtw_status(0, 1))
    self.safety.ignition_can_hook(self._gtw_status(5, 1))
    self.assertFalse(self.safety.get_ignition_can())

  def test_sdm1(self):
    # SDM1 (0x201) on bus 0 or 2 is treated as ignition on (see docs/TESLA_HW1_FORK_BUILD.md)
    for bus in (0, 2):
      self.safety.init_tests()
      self.safety.ignition_can_hook(self.packer.make_can_msg_safety("SDM1", bus, {}))
      self.assertTrue(self.safety.get_ignition_can())


if __name__ == "__main__":
  unittest.main()
