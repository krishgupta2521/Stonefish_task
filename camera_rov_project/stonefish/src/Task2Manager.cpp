#include "Task2Manager.h"
#include "kyubic_model.hpp"

#include <chrono>

#include <entities/statics/Obstacle.h>
#include <entities/solids/Box.h>
#include <entities/solids/Cylinder.h>
#include <entities/solids/Sphere.h>
#include <entities/solids/Compound.h>
#include <entities/solids/Polyhedron.h>
#include <entities/forcefields/Uniform.h>
#include <actuators/ActuatorDynamics.h>
#include <actuators/Thruster.h>
#include <sensors/scalar/Odometry.h>
#include <sensors/vision/ColorCamera.h>
#include <utils/SystemUtil.hpp>
#include <graphics/OpenGLTrackball.h>

#include <algorithm>
#include <ctime>
#include <fstream>
#include <random>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

// ---------------- Size of the simulation ----------------
// The robot is kyubic::SCALE x the real KYUBIC (1 = real size, with the real
// 0.40 m buoy). The controller and the scenarios were tuned on KYUBIC enlarged
// TUNED_SCALE x in a world TUNED_SCALE x larger. That robot is a Froude-scaled
// copy of the real one (mass and thrust x S^3, rotor lag x sqrt(S)), so every
// tuned value below carries its scaling law and the same code runs either the
// real-size KYUBIC in a TUNED_SCALE x smaller world or the enlarged demo:
//   lengths x L, times and speeds x sqrt(L), rates (rad/s, 1/s, Hz) x 1/sqrt(L),
//   accelerations and thrust commands x 1, masses x L^3, inertias x L^5,
// with L = kyubic::SCALE / TUNED_SCALE (0.234 at real size). The camera frame
// rate scales too, so only the detector's processing delay does not.
namespace Sim {
constexpr double TUNED_SCALE = 4.275;
constexpr double Sqrt(double x)
{
    double r = x > 1.0 ? x : 1.0;
    for (int i = 0; i < 60; ++i)
        r = 0.5 * (r + x / r);
    return r;
}
constexpr double L = kyubic::SCALE / TUNED_SCALE;   // length
constexpr double T = Sqrt(L);                       // time
constexpr double V = Sqrt(L);                       // speed
constexpr double W = 1.0 / Sqrt(L);                 // rate
constexpr double M = L * L * L;                     // mass
constexpr double I = M * L * L;                     // rotational inertia

// Target buoy: the team's buoy model (models/yellow_sensor_model.obj: yellow
// bulb on a thin dark grey pole, white mounts), scaled uniformly. Real size:
// bulb 16 cm tall and 9.8 cm wide (scale 0.16 / 1.54); with the enlarged
// robot the 1.71 m demo buoy (x 0.5). In model units the bulb is 1.54 tall,
// 0.94 x 0.72 across, its centre 0.18 above the origin; the whole buoy runs
// from 1.40 above to 2.02 below the origin.
constexpr bool REAL_SIZE = kyubic::SCALE < 1.5;     // the real-size KYUBIC
constexpr bool REAL_BUOY = REAL_SIZE;
constexpr double BUOY_SCALE = REAL_BUOY ? 0.16 / 1.54 : 0.5;
constexpr double BULB_ABOVE_ORIGIN = 0.18 * BUOY_SCALE;         // bulb centre above the target origin
constexpr double BUOY_TOP = 1.40 * BUOY_SCALE;                  // top above the origin
constexpr double BUOY_BOTTOM = 2.02 * BUOY_SCALE;               // bottom below the origin
constexpr double SEABED_DEPTH = 5.0 * L;                        // top of the seabed

// Target origin for a demo-buoy origin position of the tuned world: the BULB
// (what YOLO sees) lands on the scaled spot, so the camera sees the same view.
inline sf::Vector3 TargetFromTuned(double x, double y, double z)
{
    return sf::Vector3(x * L, y * L, (z - 0.09) * L + BULB_ABOVE_ORIGIN);
}
}

// Select the case at start-up, no rebuild needed:
//   ./Task2Baseline 3        or   ./Task2Baseline --case 3   or   TASK2_CASE=3
// 1=front, 2=above, 3=below, 4=behind, 5=behind+below,
// 6=moving target driven by /object/position, 7=NO target (search, then surface),
// 8 = ROV MODE ("./Task2Baseline rov"): piloted with the game controller or the
//     keyboard, no autonomy; the target follows the real camera like case 6.
// NOTE: in cases 4 and 5 the target starts BEHIND the robot, so the robot
// must search (turn around) before it can see it. That is expected.
namespace ScenarioConfig {
constexpr int DEFAULT_CASE = 6;
constexpr int ROV_CASE = 8;
constexpr int NUM_CASES = 8;

inline bool HasTarget(int activeCase) { return activeCase != 7; }

// $TASK2_TARGET_POS="x,y,depth" places the target anywhere (any case with a
// target), e.g. for random-placement experiments.
inline bool TargetPositionOverride(sf::Vector3& out) {
    const char* env = std::getenv("TASK2_TARGET_POS");
    double x, y, z;
    if (env == nullptr || std::sscanf(env, "%lf,%lf,%lf", &x, &y, &z) != 3)
        return false;
    out = sf::Vector3(x, y, z);
    return true;
}

inline sf::Vector3 TargetPosition(int activeCase) {
    sf::Vector3 overridePos;
    if (TargetPositionOverride(overridePos))
        return overridePos;
    switch (activeCase) {
        // Positions of the tuned (enlarged) world, scaled by Sim::TargetFromTuned.
        case 1: return Sim::TargetFromTuned(8.0, 0.0, 1.5);
        // NED: smaller Z = higher. Shallow target: the robot centres on the
        // bulb, and its axis must stay >= MIN_DEPTH so the floaters stay under water.
        case 2: return Sim::TargetFromTuned(8.0, 0.0, 1.45);
        case 3: return Sim::TargetFromTuned(8.0, 0.0, 3.2);
        case 4: return Sim::TargetFromTuned(-8.0, 0.0, 1.5);
        case 5: return Sim::TargetFromTuned(-8.0, 0.0, 3.2);
        default: return Sim::TargetFromTuned(8.0, 0.0, 1.5);
    }
}
}

namespace
{
    bool g_haveSmoothedCentroid = false;
    double g_smoothedPixelX = 320.0;
    double g_smoothedPixelY = 240.0;

    // ---------------- Vehicle / sensor configuration (KYUBIC V2) ----------------
    // BlueRobotics T200 at 16 V (datasheet, approximate): 51.5 N forward,
    // 40.2 N reverse at about 3700 rpm; ESC dead band about +-6 % throttle.
    constexpr double T200_MAX_OMEGA = 387.0;                     // rad/s
    constexpr double T200_KT_FORWARD = 51.5 / (387.0 * 387.0);  // N / (rad/s)^2
    constexpr double T200_KT_REVERSE = 40.2 / (387.0 * 387.0);
    constexpr double T200_DEADBAND_W2 = (0.06 * 387.0) * (0.06 * 387.0);
    constexpr double T200_TIME_CONSTANT = 0.10;                 // s, spin-up lag
    constexpr double T200_DIAMETER = 0.076;                      // m
    constexpr double THRUST_LIMIT = 0.60;                        // never above 60 % thrust
    constexpr double THRUST_SLEW_RATE = 1.5 * Sim::W;                     // thrust fraction per second
    constexpr double ODOMETRY_RATE_HZ = 100.0;

    // Detection camera: ZED Mini left lens, VGA stream (672 x 376, 90 x 58 deg).
    // (The OAK-1 is not used: the claw hides the lower ~40 % of its view.)
    constexpr int    CAMERA_WIDTH_PX = 672;
    constexpr int    CAMERA_HEIGHT_PX = 376;
    constexpr double CAMERA_WIDTH = CAMERA_WIDTH_PX;
    constexpr double CAMERA_HEIGHT = CAMERA_HEIGHT_PX;
    constexpr double CAMERA_HFOV_DEG = 90.0;
    // 10 Hz on the tuned (enlarged) robot; a smaller robot moves faster
    // (Froude), so its camera runs faster in proportion (~21 Hz at real size).
    constexpr double CAMERA_RATE_HZ = 10.0 * Sim::W;

    // Robot size: kyubic::SCALE x the real KYUBIC (the hull, mass and added mass
    // come scaled from kyubic_model.hpp). Thrust is scaled by SCALE^3 like the
    // mass, so every axis accelerates as the real robot per unit command;
    // rotor lag by sqrt(SCALE) (Froude time scale).
    constexpr double ROBOT_SCALE = kyubic::SCALE;
    constexpr double THRUST_SCALE = ROBOT_SCALE * ROBOT_SCALE * ROBOT_SCALE;

    // The REAL USB camera (case 6 moves the simulated target from its pixels)
    constexpr double REAL_CAMERA_WIDTH = 640.0;
    constexpr double REAL_CAMERA_HEIGHT = 480.0;

    // Command axes (normalised, -1..1): surge +forward, sway +right, heave +down,
    // roll +right side down, yaw +turn right (NED).
    enum Axis { SURGE = 0, SWAY = 1, HEAVE = 2, ROLL = 3, YAW = 4 };

    // ---------------- Self-checks ----------------
    // 1) At startup the robot pulses surge, yaw and heave through the thrust
    //    allocation and measures, from odometry, which way it actually moves.
    //    A wrong sign is the usual reason a robot drives AWAY from the target.
    constexpr bool   ENABLE_ACTUATOR_CALIBRATION = true;
    // Gentle pulses: with the camera depth estimate (no depth sensor) a fast
    // surge pulse lifts the hull more than the camera can measure without the
    // target in view (0.30 lost ~0.5 m of depth). 0.15 still gives 25x the
    // detection threshold.
    constexpr double CALIB_COMMAND = 0.15;
    constexpr double CALIB_MIN_DELTA = 0.004 * Sim::V;       // m/s, below this = inconclusive
    // Heave is measured by the camera (flow), which is weak and noisy: a strong
    // pulse, and the sign known from the CAD geometry is only flipped on clear
    // evidence. (A noise-driven flip inverted the depth loop in a test: the
    // robot pushed itself against the surface.)
    constexpr double CALIB_HEAVE_COMMAND = 0.30;
    constexpr double CALIB_HEAVE_FLIP_MIN = 0.03 * Sim::V;   // m/s, camera-measured response
    // Vertical alignment that cannot converge means vertical control is not
    // trustworthy: vertical thrusters off, float up on buoyancy alone.
    constexpr double VERTICAL_ALIGN_TIMEOUT_SEC = 60.0 * Sim::T;
    // 2) While the robot is aligning (never while moving forward) the
    //    controller checks that the image moves the way physics says it must
    //    (turn right -> target moves LEFT; go deeper -> target moves UP).
    //    If not, it flips / swaps camera axes. Disabled for case 6 (moving target).
    constexpr bool ENABLE_ONLINE_CAMERA_CHECK = true;

    // ---------------- Perception -> guidance ----------------
    // Detector confidence gates. The real-size world's YOLO model (sim-trained,
    // 960 px) scores small far bulbs ~0.45 and empty frames at most 0.23:
    // 0.30 found 100 % of held-out buoy frames with no false alarm. The
    // enlarged world's older model used 0.5 / 0.8.
    constexpr double MIN_CONFIDENCE = Sim::REAL_BUOY ? 0.30 : 0.5;
    // Perception timeouts stay in real seconds at every size: YOLO's processing
    // delay and its occasional slow frame do not shrink with the world (scaled
    // to 0.39 s, one late frame "lost" the buoy 7 cm from it).
    constexpr double TARGET_TIMEOUT_SEC = 0.8;      // sim seconds without a confident detection
    constexpr double DETECTOR_TIMEOUT_SEC = 2.0;    // no detector message at all -> perception fault
    constexpr double PERCEPTION_ABORT_SEC = 30.0 * Sim::T;   // fault this long -> surface
    constexpr double TARGET_FILTER_ALPHA = 0.5;
    constexpr double OUTLIER_JUMP_RAD = 0.30;
    constexpr int    OUTLIER_CONFIRM_FRAMES = 3;
    constexpr double DEPTH_PER_RAD = 2.0 * Sim::L;           // lever (m/rad) used before range is known
    constexpr double MAX_DEPTH_STEP = 0.5 * Sim::L;
    // Hull-axis depth limits: floaters 0.95 m above the axis stay >= 0.35 m
    // under water, the keel 0.97 m below stays >= 0.3 m above the seabed (5 m).
    constexpr double MIN_DEPTH = 1.3 * Sim::L;
    constexpr double MAX_DEPTH = 3.7 * Sim::L;

    // ---------------- Range from bounding box ----------------
    // distance to the target AXIS = f * size / pixels, minus the target radius
    // = distance to its FRONT face. Sizes were fitted to ground truth in the
    // simulator for KYUBIC's ZED Mini + the BULB-ONLY YOLO model at 640 px
    // (runs/stonefish/buoy_yolo26n_300_epochs, 1.5-7.5 m): boxes are
    // 0.38-0.40 m wide and 0.74-0.80 m tall in effect (geometric bulb height
    // 0.77 m). Rounded down so errors make the robot stop early.
    // Re-fit them for a different camera, detector or target.
    // (Fitted on the demo buoy, x0.5; proportional to the buoy's scale.)
    constexpr double TARGET_WIDTH_M = 0.76 * Sim::BUOY_SCALE;
    constexpr double TARGET_HEIGHT_M = 1.48 * Sim::BUOY_SCALE;
    constexpr double TARGET_RADIUS_M = 0.47 * Sim::BUOY_SCALE;
    constexpr double RANGE_FILTER_ALPHA = 0.3;
    // Closer than this (camera to the bulb's front face), the box can only show
    // part of the bulb and look too small = too far. Then only "closer"
    // readings are accepted; odometry carries the estimate the rest of the way.
    // The bulb box touches the image edge below ~0.9 m camera distance.
    constexpr double RANGE_TRUST_FAR = 0.7 * Sim::L;
    constexpr double MIN_RANGE = 0.05 * Sim::L;
    constexpr double MAX_RANGE = 15.0 * Sim::L;
    constexpr double BBOX_FRESH_SEC = 0.3;
    // A sudden large drop in range (e.g. a false blob) must repeat this many
    // frames in a row before it is believed.
    constexpr double RANGE_JUMP_RATIO = 0.6;
    constexpr int    RANGE_JUMP_CONFIRM_FRAMES = 3;

    // ---------------- Vehicle geometry (simulation hull) ----------------
    // ZED Mini left lens, carried on the robot's centreline (on the real
    // KYUBIC it is 3.9 cm left of it): with the real offset the robot centred
    // its hull on the buoy (within 4 mm) but, 8 cm from it, the camera showed
    // the buoy at x = +0.19, right of the cross. On the centreline, centred
    // hull = buoy on the cross (as the user wants, at both sizes).
    constexpr double CAMERA_MOUNT[3] = {kyubic::CAMERA_POS[0], 0.0, kyubic::CAMERA_POS[2]};
    constexpr double CAMERA_TO_FRONT = kyubic::FRONT_X - CAMERA_MOUNT[0];   // camera to recovery hooks

    // ---------------- Depth from the camera (no depth sensor) ----------------
    // The controller never reads the simulator's depth (that is evaluation
    // ground truth only). visual_heave_estimator.py measures how fast the scene
    // slides up or down in the image; with the camera's focal length f, the
    // effective distance of the tracked scene and the gyro pitch rate q that is
    //     vertical speed vz = -(b - VISION_PITCH_COEFF * f * q) * D / f,
    // integrated from the depth the robot is released at. Fitted on the
    // simulator (case 3, 0.9-3.4 m): D = 1.49 m, pitch coefficient 1.2; speed
    // error median 0.009 m/s, integrated depth drift ~0.4 m per mission.
    // Alignment with the target does not suffer from the drift: the target's
    // depth is measured in the same (drifting) frame, from the image.
    constexpr double LAUNCH_DEPTH = 1.5 * Sim::L;            // m, depth the robot is released at
    constexpr double VISION_FLOW_DISTANCE_M = Sim::REAL_SIZE ? 0.80 : 1.49 * Sim::L;
    // Real size: the floating particles do not shrink with the world, so the
    // tracked scene is relatively farther: the camera read 0.3-0.6 x the true
    // vertical speed with 0.35 m (heave identification runs); 0.80 m fits.
    constexpr double VISION_PITCH_COEFF = 1.2;
    constexpr double VISION_VZ_FILTER_SEC = 0.3 * Sim::T;    // smooths the 10 Hz estimate
    constexpr double VISION_FLOW_TIMEOUT_SEC = 1.0 * Sim::T; // no flow this long: depth unknown
    // The scene's image motion has no metric scale (near particles and the far
    // target move very differently: the flow estimate saturated at ~0.08 m/s
    // while KYUBIC descended at 0.37 m/s), and differentiating the target's
    // image position lags ~1 s, which made the depth loop oscillate.
    // So depth comes from an OBSERVER (Kalman filter), state [depth, vz, bias]:
    //  - predicted from the robot's own thrust command (no sensor): heave plant
    //    identified on KYUBIC in the simulator (cases 1-3 + ROV runs, below
    //    1.05 m depth, R^2 = 0.913):
    //        dvz/dt = F_heave / HEAVE_MASS_KG + bias - HEAVE_DRAG * vz|vz|
    //                 + HEAVE_SURGE_LIFT * vx^2   (the moving hull lifts, either way)
    //  - corrected by the target's height in the image (metric: range x angle),
    //    in a frame anchored to the target, so drift relative to it vanishes;
    //  - corrected weakly by the camera flow speed when the target is not seen.
    constexpr double HEAVE_MASS_KG = Sim::REAL_SIZE ? 89.34 : 7628.0 * Sim::M;        // incl. added mass (enlarged KYUBIC)
    constexpr double HEAVE_DRAG = Sim::REAL_SIZE ? 0.0 : 1.83 / Sim::L;             // 1/m
    // Real size (identified 2026-10-23 on the real-size KYUBIC, ROV stick
    // schedules, R^2 0.94-0.98): drag is mostly LINEAR at this size, which
    // the Froude-scaled models (quadratic only) could not represent: their
    // open-loop speed was 1.9x the truth. Linear drag terms [1/s]:
    constexpr double HEAVE_DRAG_LIN = Sim::REAL_SIZE ? 1.082 : 0.0;
    constexpr double HEAVE_BIAS_INIT = Sim::REAL_SIZE ? -0.0896 : -0.081;      // m/s^2 (net buoyancy, + = down)
    constexpr double HEAVE_SURGE_LIFT = Sim::REAL_SIZE ? 0.0 : -0.058 / Sim::L;     // (m/s^2) per (m/s)^2 of forward speed, up
    constexpr double KF_ACCEL_NOISE = 0.02 * Sim::Sqrt(Sim::V);         // m/s^2 per sqrt(s): model error
    // Real size: the identified heave model knows the buoyancy; the camera
    // flow under-reads small vertical speeds (reads ~0 at 0.005 m/s), and a
    // freer bias learned that and cancelled half of the model's (correct) rise.
    constexpr double KF_BIAS_NOISE = Sim::REAL_SIZE ? 0.0003 : 0.003 / Sim::Sqrt(Sim::V);   // m/s^2 per sqrt(s)
    constexpr double KF_TARGET_NOISE_M = 0.02 * Sim::L;      // target height: 2 cm + 0.6 % of range
    constexpr double KF_TARGET_NOISE_PER_M = 0.006;
    constexpr double KF_FLOW_NOISE = 0.03 * Sim::V;          // flow speed: 0.03 m/s + 50 % (scale unknown)
    constexpr double KF_FLOW_NOISE_PER_SPEED = 0.5; // + this x forward speed
    constexpr double KF_GATE_SIGMA = 3.0;           // reject measurements beyond 3 sigma
    constexpr int    KF_REANCHOR_REJECTS = 8;       // this many rejections in a row: re-anchor
    constexpr double TARGET_ANCHOR_TIMEOUT_SEC = 3.0 * Sim::T;
    constexpr double TERMINAL_VERTICAL_MIN_GAP = 0.15 * Sim::L;  // terminal: target height used above this gap
    // The depth reference moves at most this fast: descents stay gentle even
    // when the speed estimate is poor.
    constexpr double VERTICAL_REF_RATE = 0.12 * Sim::V;      // m/s
    // Surfacing without a depth sensor: rise until the camera shows the robot
    // has stopped rising while it is still pushed up, i.e. it is at the surface.
    constexpr double SURFACE_SLOW_ZONE = 0.6 * Sim::L;       // m above SURFACE_DEPTH (estimate) -> final rate
    constexpr double SURFACE_STILL_VZ = 0.01 * Sim::V;       // m/s, "not rising"
    constexpr double SURFACE_STILL_SEC = 4.0 * Sim::T;
    constexpr double SURFACE_PUSH_LEAD = 0.10 * Sim::L;      // m, reference this far above = pushed up
    constexpr double SURFACE_SETTLE_SEC = 6.0 * Sim::T;      // thrusters off and still this long = surfaced
    // At the surface (and when floating up without a depth estimate) a small
    // constant upward thrust lifts the floaters clear of the water for
    // recovery: on buoyancy alone they barely break the surface (~2 mm).
    constexpr double SURFACE_HOLD_PUSH = 0.08;      // heave command, upward

    // ---------------- Navigation WITHOUT IMU and DVL ----------------
    // The project uses no IMU and no DVL: heading, turn rate, forward speed and
    // position come from observers (Kalman filters) that predict from the
    // robot's own thrust commands with models identified on KYUBIC in the
    // simulator (case 4: full search turn + approach), corrected by the camera:
    //   turn:  dr/dt = N / YAW_INERTIA + bias - YAW_DRAG r|r|          (R^2 0.81)
    //          measured: the whole image shifts sideways by
    //          VISION_YAW_FLOW_GAIN px/s per rad/s of turn (corr -0.90)
    //   surge: dv/dt = Fx / SURGE_MASS + bias - SURGE_DRAG_COEF v|v|   (R^2 0.87)
    //          measured: the target's range (box size) shrinks as the robot
    //          moves toward it (frame anchored to the target)
    // Roll and pitch are not measured: KYUBIC is passively stable (righting
    // lever), so there is no active roll control.
    constexpr double YAW_INERTIA = Sim::REAL_SIZE ? 6.43 : 10068.0 * Sim::I;         // kg m^2, incl. added inertia (enlarged KYUBIC)
    constexpr double YAW_DRAG = Sim::REAL_SIZE ? 0.0 : 5.773;
    constexpr double YAW_DRAG_LIN = Sim::REAL_SIZE ? 1.913 : 0.0;
    constexpr double SURGE_MASS = Sim::REAL_SIZE ? 92.24 : 8077.0 * Sim::M;           // kg, incl. added mass
    constexpr double SURGE_DRAG_COEF = Sim::REAL_SIZE ? 0.490 : 1.246 / Sim::L;
    constexpr double SURGE_DRAG_LIN = Sim::REAL_SIZE ? 1.125 : 0.0;       // 1/m
    constexpr double VISION_YAW_FLOW_GAIN = Sim::REAL_SIZE ? 500.0 : 550.0;  // px/s of sideways image shift per rad/s of turn
    constexpr double KF_YAW_ACCEL_NOISE = 0.004 * Sim::W * Sim::Sqrt(Sim::W);    // rad/s^2 per sqrt(s)
    constexpr double KF_YAW_BIAS_NOISE = 0.0005 * Sim::W * Sim::W * Sim::Sqrt(Sim::W);
    constexpr double KF_YAW_RATE_NOISE = 0.008 * Sim::W;     // camera turn rate: 0.008 rad/s + 30 %
    // Sliding sideways shifts the image like a turn does (by sway speed /
    // scene distance): a sideways test turned the robot 0.15 rad/s while the
    // heading hold "held" the image still. Sway speed from the thrust model
    // (identified on the enlarged KYUBIC): Fy / SWAY_MASS - SWAY_DRAG v|v|.
    // Identified with sideways runs at 0.5 / 0.7 / 1.0 stick: 1 cm/s rms.
    constexpr double SWAY_MASS = Sim::REAL_SIZE ? 86.46 : 8318.0 * Sim::M;            // kg incl. added mass
    constexpr double SWAY_DRAG = Sim::REAL_SIZE ? 1.245 : 0.672 / Sim::L;
    constexpr double SWAY_DRAG_LIN = Sim::REAL_SIZE ? 0.717 : 0.0;             // 1/m
    // Pure sway (heading held by a gyro) read +0.03..+0.06 rad/s of "turn"
    // at 0.34..0.48 m/s; at the capped camera-only 0.22 m/s it acts like a
    // scene ~5.5 m away (scene dependent, hence the extra uncertainty).
    // Real size (yaw / sway identification runs): the camera read 0.86-1.0
    // of the true turn rate (gain 500), a slide as if the scene were 1.47 m away.
    constexpr double SWAY_FLOW_DISTANCE_M = Sim::REAL_SIZE ? 1.47 : 5.5 * Sim::L;
    constexpr double SWAY_FLOW_DISTANCE_MIN = Sim::REAL_SIZE ? 0.94 : 3.5 * Sim::L;  // sets the uncertainty of that correction
    constexpr double KF_SURGE_ACCEL_NOISE = 0.01 * Sim::Sqrt(Sim::V);   // m/s^2 per sqrt(s)
    constexpr double KF_SURGE_BIAS_NOISE = 0.001 / Sim::Sqrt(Sim::V);
    constexpr double KF_RANGE_NOISE = 0.03 * Sim::L;         // target range: 3 cm + 3 % (box size)
    constexpr double KF_RANGE_NOISE_PER_M = 0.03;
    // Range from box size: a box-width error of e px gives a range error of
    // R^2 e / (f W), so far away it grows with R^2 (at 6.5 m the box is ~20 px
    // and 1-2 px of YOLO jitter is +-0.5 m). Trusting it like 3 % made a
    // 0.8 m jump read as forward motion; the hold then reversed the robot to
    // 0.35 m/s unseen (case 4). Beyond ~2.5 m this term is the larger.
    constexpr double KF_RANGE_BOX_JITTER_PX = 1.5;
    // Safety net: the speed the thrust history implies (surge model, no
    // camera corrections, so bad measurements cannot fool it). Camera only,
    // thrust that would push it past the limit is faded out.
    constexpr double MODEL_SPEED_FADE_START = 0.24 * Sim::V;  // m/s
    constexpr double MODEL_SPEED_LIMIT = 0.30 * Sim::V;
    // Steady pitch from thrust (camera only, real size; 0 = not identified).
    constexpr double PITCH_TRIM_DEG = Sim::REAL_SIZE ? 0.72 : 0.0;
    constexpr double PITCH_PER_FX_DEG = Sim::REAL_SIZE ? 0.0281 : 0.0;   // per N forward
    constexpr double PITCH_PER_FZ_DEG = Sim::REAL_SIZE ? 0.0244 : 0.0;   // per N down
    constexpr double PITCH_PER_V2_DEG = Sim::REAL_SIZE ? 94.0 : 0.0;     // per (m/s)^2 forward
    constexpr double FLOW_DIVERGENCE_TO_SPEED = 1.5 * Sim::L;   // m: sign check only (calibration)
    constexpr double RANGE_RATE_WINDOW_SEC = 1.0 * Sim::T;   // closing speed: fit over this
    constexpr double RANGE_RATE_MAX_RANGE = 3.0 * Sim::L;    // m, and only this close
    // Pitch is not measured: KYUBIC rocks (period ~3.6 s, +-2-4 deg) and that
    // moves the target in the image. The rocking averages out, the target's
    // height does not: without a gyro the target depth is averaged over
    // TARGET_DEPTH_TAU_NO_GYRO and single heights are trusted less.
    constexpr double TARGET_DEPTH_TAU_NO_GYRO = 4.0 * Sim::T;   // s
    constexpr double PITCH_UNCERTAINTY_RAD = 0.04;     // ~2.3 deg
    constexpr double FLOW_PITCH_NOISE_NO_GYRO = 0.12 * Sim::V;  // m/s, rocking seen as heave by the camera
    // Turning makes the vertical image shift unreliable (rocking, close
    // drifting particles in open water): case 5's calibration turn read 0.27 m/s
    // of rise while the robot sank, and the depth loop dove 1.1 m. Trust the
    // flow less in proportion to the turn rate (the thrust model carries on).
    constexpr double FLOW_TURN_NOISE_PER_RATE = 2.0 * Sim::L;  // m/s per rad/s
    // Likewise while sliding sideways (a camera-only ROV slide drifted the
    // depth estimate 0.2 m the wrong way at real size).
    constexpr double FLOW_SWAY_NOISE_PER_SPEED = 1.0;          // (m/s) per (m/s)
    // Real size: the camera's vertical speed carries steady offsets (+0.003
    // m/s still, +0.0075 moving vertically, +0.015 near the surface; rms
    // 0.02-0.04) that ~20 readings a second turn into 0.2-0.9 m/min of drift,
    // while the identified heave model follows the truth within ~1 mm/s. So
    // the flow only trims the model slowly.
    constexpr double FLOW_NOISE_SCALE = Sim::REAL_SIZE ? 4.0 : 1.0;
    constexpr double SURFACE_STILL_VZ_NO_GYRO = 0.02 * Sim::V;  // m/s, on the rocking-averaged speed
    // Camera only: the ascent is open-loop upward thrust, so once the surface
    // stops the robot the thrust model keeps "rising"; an estimate this far
    // above the floating depth means it is at the surface. (At real size the
    // animated surface puts a steady +-0.015-0.02 m/s into the flow there, so
    // "no more rise" from the flow alone never came.)
    constexpr double SURFACE_EST_MARGIN = 0.4 * Sim::L;
    constexpr double HEAVE_TRIM_TAU = 5.0 * Sim::T;          // s, learning the hover thrust
    constexpr double SURFACE_ASCENT_THRUST = 0.06;  // camera only: ascend at trim - this (~0.09 m/s)
    constexpr double SURFACE_MIN_ASCENT_SEC = 10.0 * Sim::T; // camera only: ascend at least this long

    // ---------------- ROV mode (piloted) ----------------
    // The sticks command SPEEDS, not thrust (raw full thrust drove the 3 t
    // enlarged KYUBIC to 1.3 m/s into the target in a test):
    //  - forward/reverse = surge speed through the mission's speed loop
    //    (gentle ramps), at most ROV_MAX_SPEED / ROV_MAX_REVERSE;
    //  - turn = yaw rate; released: heading hold (gyro/IMU);
    //  - up/down = vertical thrust; released: depth hold on the CAMERA depth
    //    estimate (KYUBIC is positively buoyant and would float up);
    //  - roll is always held level.
    // Collision guard (Stonefish camera, bulb box size): forward speed limited
    // to the braking-aware profile of the mission, TERMINAL_MAX_SPEED inside
    // 1 m, zero at ROV_STOP_GAP. The gap is remembered (and advanced with the
    // robot's own motion) when the target leaves the view up close.
    constexpr double ROV_STICK_DEADZONE = 0.10;
    constexpr double ROV_KEY_INPUT = 0.65;          // keyboard = 65 % stick
    constexpr double ROV_MAX_SPEED = 0.30 * Sim::V;          // m/s forward
    constexpr double ROV_MAX_REVERSE = 0.20 * Sim::V;        // m/s reverse
    constexpr double ROV_MAX_SWAY = 0.30;           // sway thrust command
    // Camera only there is no sideways speed measurement: cap the thrust so
    // full stick slides at ~0.22 m/s (0.30 slid at 0.48 m/s, over the 0.30 limit).
    constexpr double ROV_MAX_SWAY_NO_DVL = 0.06;
    constexpr double ROV_MAX_HEAVE = 0.30;          // vertical thrust command
    constexpr double ROV_MAX_YAW_RATE = 0.15 * Sim::W;       // rad/s
    constexpr double ROV_YAW_RATE_GAIN = 4.0 * Sim::T;       // command per rad/s of rate error
    constexpr double ROV_TURN_MAX = 0.75;           // yaw command limit
    constexpr double ROV_SLOW_GAP = 1.5 * Sim::L;            // m, guard zone
    constexpr double ROV_STOP_GAP = 0.40 * Sim::L;           // m, front gap: forward speed zero (stops ~0.3 m)
    constexpr double ROV_GUARD_BBOX_SEC = 0.3;      // box older than this: use the remembered gap
    constexpr double ROV_GUARD_FORGET_GAP = 3.0 * Sim::L;    // remembered gap beyond this: forget it
    constexpr int    ROV_GAP_CONFIRM_BOXES = 5;     // consistent farther whole boxes to believe them
    constexpr double ROV_SPEED_ACCEL = 0.08;        // m/s^2, pilot speed ramp up (mission: 0.03)
    constexpr double ROV_SPEED_DECEL = 0.10;        // m/s^2, ramp down
    constexpr double ROV_SURGE_CMD_FORWARD_NO_DVL = 0.10;   // camera only: thrust caps
    constexpr double ROV_SURGE_CMD_REVERSE_NO_DVL = 0.08;
    constexpr double ROV_TURN_MAX_NO_GYRO = 0.08;   // camera only: ~0.15-0.2 rad/s
    // Real camera -> sim target in ROV mode (image position matched).
    constexpr double ROV_TARGET_BOX_DZ = -0.112 * Sim::BUOY_SCALE;    // m, YOLO box centre above the target origin
    constexpr double ROV_TARGET_BOX_W = 0.78 * Sim::BUOY_SCALE;       // m, effective bulb box width (fitted)
    constexpr double ROV_TARGET_BOX_H = 1.56 * Sim::BUOY_SCALE;       // m, effective bulb box height (fitted)
    constexpr double ROV_TARGET_MIN_DIST = 2.0 * Sim::L;     // m from the camera (clear of the hull)
    constexpr double ROV_TARGET_MAX_DIST = 10.0 * Sim::L;
    constexpr double ROV_TARGET_MIN_Z = Sim::BUOY_TOP + 0.05 * Sim::L;       // origin depth: target top stays under water
    constexpr double ROV_TARGET_MAX_Z = Sim::SEABED_DEPTH - Sim::BUOY_BOTTOM - 0.04 * Sim::L;       // target bottom stays above the seabed
    constexpr double ROV_VIEW_MATCH_GAIN = 1.5 * Sim::W;     // 1/s, view-matching correction rate
    constexpr double ROV_TARGET_MOVE_DEADBAND = 0.04;   // real image change that counts as "moved"
    constexpr double ROV_TARGET_STILL_TOL = 0.02;   // real object "still" within this
    constexpr double ROV_TARGET_SETTLE_SEC = 1.0 * Sim::T;   // still this long (and matched) -> freeze

    // ---------------- Approach / mission ----------------
    // Gap = distance from the robot's front face to the target's front face.
    constexpr double STOP_GAP = 0.20 * Sim::L;               // stop this far from the target
    constexpr double REACH_TOL = 0.03 * Sim::L;
    constexpr double MIN_SAFE_GAP = 0.10 * Sim::L;           // closer than this -> gently move away
    // Terminal approach: closer than this the heading is locked and the rest
    // is driven on odometry; detections can then only shorten the range (and,
    // with the camera depth estimate, correct the depth while the box is whole).
    // Bulb-only YOLO at 640 px, aligned: 100 % detection down to 0.7 m camera
    // distance (front gap ~0), box whole above ~0.9 m. (The earlier full-stick
    // model was unreliable below 1.75 m and needed TERMINAL_GAP = 1.8.)
    constexpr double TERMINAL_GAP = 0.50 * Sim::L;
    // Within this gap the approach is held to TERMINAL_MAX_SPEED whatever the
    // detections do (as slow near the target as with the long blind stretch).
    constexpr double SLOW_ZONE_GAP = 1.80 * Sim::L;
    constexpr double TERMINAL_MAX_SEC = 60.0 * Sim::T;
    // The terminal lock only engages from a good track: this long of
    // uninterrupted confident detections. Otherwise stop, then retreat.
    // (A buoy 1-2 m away scores 0.45-0.5: 0.6 shut out the buoy-height
    // depth correction, and the depth estimate drifted 0.1 m in a ROV slide.)
    constexpr double GOOD_TRACK_CONFIDENCE = Sim::REAL_BUOY ? 0.4 : 0.8;
    constexpr double TERMINAL_MIN_TRACK_SEC = 2.0 * Sim::T;
    constexpr double TERMINAL_WAIT_SEC = 4.0 * Sim::T;
    // Inside this gap detections are not trusted for (re)acquisition: a target
    // lost here is never searched for on the spot. The robot backs away to
    // RETREAT_GAP, where detection is reliable, and looks again.
    constexpr double RELIABLE_GAP = 0.8 * Sim::L;
    constexpr double RETREAT_GAP = 1.2 * Sim::L;
    constexpr double RETREAT_TIMEOUT_SEC = 40.0 * Sim::T;
    // A moving target (case 6) can swing into the hull: never drive blind,
    // stop where the camera still sees it reliably.
    constexpr double STOP_GAP_MOVING_TARGET = 1.0 * Sim::L;
    // Speed profile: cruise far away, then v = gain * remaining distance, a
    // deceleration the thrusters can always deliver, then creep.
    constexpr double APPROACH_CRUISE_SPEED = 0.25 * Sim::V;  // m/s, far from the target
    constexpr double APPROACH_CRUISE_SPEED_NO_DVL = 0.20 * Sim::V;  // camera only: speed known less well
    constexpr double APPROACH_CREEP_SPEED = 0.03 * Sim::V;   // m/s, final centimetres
    constexpr double APPROACH_SPEED_GAIN = 0.12 * Sim::W;    // (m/s) per m of remaining gap
    constexpr double TERMINAL_MAX_SPEED = 0.08 * Sim::V;     // m/s while driving on odometry only
    constexpr double TERMINAL_MAX_SPEED_NO_DVL = 0.065 * Sim::V;   // camera only
    // Braking, measured on KYUBIC in the simulator: thrust gives 0.39 m/s^2 per
    // unit command, so SURGE_MAX brakes at ~0.14 m/s^2 (+ drag). The speed
    // profile plans with far less than that; the stopping guard below brakes
    // at once if the robot's ACTUAL speed could not stop in time.
    constexpr double BRAKE_DECEL_PLAN = 0.02;       // m/s^2, v <= sqrt(2 a d)
    constexpr double BRAKE_DECEL_GUARD = 0.035;     // m/s^2, stopping-distance guard
    constexpr double BRAKE_GUARD_MARGIN = 0.05 * Sim::L;     // m
    constexpr double REACHED_SPEED = 0.03 * Sim::V;          // m/s, must be this slow to count as reached
    constexpr double LOST_REACHED_MARGIN = 0.25 * Sim::L;    // lost this close to STOP_GAP -> reached
    constexpr double REACHED_HOLD_SEC = 3.0 * Sim::T;
    constexpr double BACKOFF_DISTANCE = 0.50 * Sim::L;       // back away before ascending (0 = off)
    // Camera only (no DVL): nothing holds position during the ascent, so back
    // off further first, and drift slowly away from the target while rising.
    constexpr double BACKOFF_DISTANCE_NO_DVL = 1.0 * Sim::L;
    constexpr double SURFACING_DRIFT_SPEED = 0.02 * Sim::V;  // m/s, backwards
    constexpr double BACKOFF_SPEED = 0.08 * Sim::V;          // m/s max, backing away (BACK_OFF / RETREAT): the
                                                    // enlarged robot overshoots faster references near the target
    constexpr double BACKOFF_TIMEOUT_SEC = 30.0 * Sim::T;
    // Hull-axis depth when surfaced: the floater tops (0.223 m above the axis
    // on the real robot, x ROBOT_SCALE) are about 3 cm x ROBOT_SCALE out of the
    // water. Higher, KYUBIC loses more buoyancy than the heave limit can hold
    // (it is trimmed only 0.5 kg x ROBOT_SCALE^3 positive).
    constexpr double SURFACE_DEPTH = 0.19 * ROBOT_SCALE;
    constexpr double ASCENT_RATE = 0.10 * Sim::V;            // m/s depth reference ramp
    constexpr double ASCENT_RATE_FINAL = 0.03 * Sim::V;      // m/s close to the surface
    constexpr double ASCENT_SLOW_GAIN = 0.25 * Sim::W;       // (m/s) per m above SURFACE_DEPTH
    constexpr double ASCENT_MAX_LEAD = 0.15 * Sim::L;        // m, reference ahead of the robot
    constexpr double SURFACED_TOL = 0.02 * ROBOT_SCALE;   // m: close to SURFACE_DEPTH
    // Shallowest hull-axis depth KYUBIC can reach (full upward thrust at the
    // surface: 0.164 m at real size).
    constexpr double MIN_PHYSICAL_DEPTH = 0.85 * SURFACE_DEPTH;
    // Contact / obstruction: thrusting forward but not moving -> stop at once.
    constexpr double STALL_MIN_SPEED_REF = 0.03 * Sim::V;
    constexpr double STALL_SPEED_RATIO = 0.3;
    constexpr double STALL_SEC = 1.0 * Sim::T;

    // ---------------- State-machine thresholds ----------------
    constexpr double H_ALIGN_TOL = 0.07;            // rad (~4 deg, ~32 px)
    constexpr double V_ALIGN_TOL = 0.10 * Sim::L;            // m depth error
    constexpr double LEAVE_FACTOR = 1.8;            // hysteresis while aligning
    constexpr double APPROACH_LEAVE_FACTOR = 2.5;   // looser while approaching
    constexpr double YAW_SETTLED_RATE = 0.03 * Sim::W;       // rad/s
    constexpr double HEAVE_SETTLED_VEL = 0.03 * Sim::V;      // m/s
    constexpr double SETTLE_HOLD_SEC = 0.5 * Sim::T;
    constexpr double VERIFY_HOLD_SEC = 0.8 * Sim::T;
    constexpr double LOST_HOLD_SEC = 2.0 * Sim::T;
    constexpr double SEARCH_YAW_RATE = 0.12 * Sim::W;
    constexpr double SEARCH_MAX_LEAD = 0.35;
    constexpr double SEARCH_MAX_TURN = 2.0 * M_PI + 0.35;  // one full turn, then give up
    // Camera only, no single source of the turn angle is reliable: the camera
    // over-reads slow turns in open water (x1.7: 576 deg counted, 331 true)
    // and under-reads fast ones (x0.66), the thrust model over-reads slow
    // turns (x1.1-1.3). The search ends only when BOTH say 1.4 turns.
    constexpr double SEARCH_MAX_TURN_NO_GYRO = 1.4 * 2.0 * M_PI;
    constexpr double SEARCH_MAX_SEC = 180.0 * Sim::T;        // backstop if the turn is obstructed

    // ---------------- Inner loops (fractions of max thrust) ----------------
    constexpr double YAW_KP = 0.80;
    // Rotational inertia grows as ROBOT_SCALE^5 but thrust x lever only as
    // ROBOT_SCALE^4: the loop is sqrt(ROBOT_SCALE) slower, damping x sqrt(SCALE)
    // keeps it as well damped as on the real robot (2.00 = real size).
    constexpr double YAW_KD = 2.00 * Sim::Sqrt(ROBOT_SCALE);
    constexpr double YAW_KI = 0.00;
    constexpr double YAW_MAX = 0.30;
    constexpr double YAW_MAX_NO_GYRO = 0.10;        // camera only (~0.2 rad/s at most)

    // Tuned with working vertical thrusters (~3-4 t effective heave mass):
    // damping ratio ~0.7, small steady error under the lift of forward motion.
    constexpr double DEPTH_KP = 1.20 / Sim::L;
    constexpr double DEPTH_KD = 4.00 / Sim::V;
    constexpr double DEPTH_KI = 0.10 / (Sim::L * Sim::T);
    constexpr double DEPTH_I_LIMIT = 0.25;
    constexpr double HEAVE_MAX = 0.40;

    // Roll hold with the two middle thrusters (pitch is passive: no thruster
    // can pitch KYUBIC, the 5 cm righting lever keeps it level).
    constexpr double ROLL_KP = 0.50;
    constexpr double ROLL_KD = 0.30 * Sim::Sqrt(ROBOT_SCALE);
    constexpr double ROLL_MAX = 0.20;

    // Surge: speed loop on odometry body velocity. KYUBIC's measured surge plant
    // (simulator, full mission, real size): dv/dt = 0.391 * command - 1.80 * v|v|.
    // Drag per unit mass falls as 1 / ROBOT_SCALE.
    constexpr double SURGE_KDRAG = Sim::REAL_SIZE ? 0.490 / 1.103 : 1.80 / ROBOT_SCALE / 0.391;   // drag feed-forward: command per (m/s)^2
    constexpr double SURGE_KA = Sim::REAL_SIZE ? 1.0 / 1.103 : 1.0 / 0.391;
    // Real size: 101.7 N per unit surge command = 1.103 m/s^2 (measured).
    constexpr double SURGE_KDRAG_LIN = Sim::REAL_SIZE ? 1.125 / 1.103 : 0.0;   // command per m/s        // acceleration feed-forward: command per m/s^2
    constexpr double SURGE_ACCEL_FILTER_SEC = 0.3 * Sim::T;  // smooths the feed-forward (no chatter)
    constexpr double SURGE_KP = 2.50 / Sim::V;
    constexpr double SURGE_KI = 0.30 / Sim::L;
    constexpr double SURGE_I_LIMIT = 0.10;          // thrust fraction
    constexpr double SURGE_MAX = 0.35;
    // Reference ramps the thrusters can actually follow (so the robot never
    // lags far behind its own speed reference).
    constexpr double SPEED_ACCEL = 0.03;            // m/s^2, speed reference ramp up
    constexpr double SPEED_DECEL = 0.035;           // m/s^2, speed reference ramp down
    // Position hold along the heading (used whenever the robot is not travelling).
    constexpr double HOLD_KP = 0.30 * Sim::W;                // (m/s) per m
    constexpr double HOLD_MAX_SPEED = 0.08 * Sim::V;
    constexpr double TRAVEL_HOLD_KP = 0.60 * Sim::W;         // (m/s) per m, near a BACK_OFF/RETREAT goal
    constexpr double HOLD_CAPTURE_SPEED = 0.02 * Sim::V;     // capture the hold point once this slow

    double WrapAngle(double a) { return std::remainder(a, 2.0 * M_PI); }
    double SignOf(double v) { return (v > 0.0) ? 1.0 : ((v < 0.0) ? -1.0 : 0.0); }

    double FocalLengthPx()
    {
        return (CAMERA_WIDTH * 0.5) / std::tan(CAMERA_HFOV_DEG * M_PI / 360.0);
    }

    // T200 thrust ~ k (omega^2 - deadband): choose omega so thrust is the
    // requested fraction of the maximum in that direction (setpoint = omega / omega_max).
    double ThrustFractionToSetpoint(double fraction)
    {
        fraction = std::clamp(fraction, -1.0, 1.0);
        if (std::fabs(fraction) < 1e-4)
            return 0.0;
        const double wMax2 = T200_MAX_OMEGA * T200_MAX_OMEGA;
        // A T200 pushes only 78 % as hard in reverse (40.2 vs 51.5 N): spin it
        // faster so the fraction means the SAME force both ways. Otherwise the
        // allocation's cancellations fail (sideways moves pushed up/down and
        // turned the robot). Within THRUST_LIMIT this stays below max speed.
        const double scale = (fraction < 0.0) ? T200_KT_FORWARD / T200_KT_REVERSE : 1.0;
        const double w2 = std::min(wMax2, T200_DEADBAND_W2 + std::fabs(fraction) * scale * (wMax2 - T200_DEADBAND_W2));
        return std::copysign(std::sqrt(w2 / wMax2), fraction);
    }

    const char* StateName(int s)
    {
        static const char* names[] = {
            "SEARCH", "YAW_ALIGN", "YAW_SETTLE", "VERTICAL_ALIGN", "VERIFY_CENTER",
            "APPROACH", "TARGET_LOST_HOLD", "CALIBRATE", "STOPPED",
            "TARGET_REACHED", "BACK_OFF", "SURFACING", "SURFACED", "RETREAT", "ROV_MANUAL"};
        return (s >= 0 && s <= 14) ? names[s] : "?";
    }

    // ---------------- Added mass of KYUBIC's solids ----------------
    // Stonefish fits a slender ellipsoid to every solid, which is wrong for flat
    // plates: it gave the 6.8 mm bottom plate 314 kg of heave added mass. These
    // standard estimates (DNV-RP-H103 strip form) replace it.
    constexpr double PERFORATED_PLATE_FACTOR = 0.5;   // frame plates have large cut-outs

    // Wall-clock time [epoch s], the same clock as ROS header stamps.
    double WallNow()
    {
        return std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }

    // Flat face p x q (p <= q) moving along its normal.
    double FaceAddedMass(double a, double b, double rho)
    {
        const double p = std::min(a, b), q = std::max(a, b);
        if (p <= 0.0)
            return 0.0;
        return rho * (1.0 - 0.42 * p / q) * M_PI / 4.0 * p * p * q;
    }

    template <class Base>
    class ExplicitAddedMass : public Base
    {
    public:
        template <class... Args>
        ExplicitAddedMass(const sf::Vector3& addedMass, const sf::Vector3& addedInertia, Args&&... args)
            : Base(std::forward<Args>(args)...)
        {
            this->aMass = addedMass;
            this->aI = addedInertia;
        }
    };

    int ResolveScenarioCase(int requested)
    {
        int c = requested;
        if (c == 0)
        {
            const char* env = std::getenv("TASK2_CASE");
            if (env != nullptr && std::string(env) == "rov")
                c = ScenarioConfig::ROV_CASE;
            else
                c = (env != nullptr) ? std::atoi(env) : ScenarioConfig::DEFAULT_CASE;
        }
        if (c < 1 || c > ScenarioConfig::NUM_CASES)
        {
            std::cerr << "[SCENARIO] invalid case " << c << ", using case "
                      << ScenarioConfig::DEFAULT_CASE << std::endl;
            c = ScenarioConfig::DEFAULT_CASE;
        }
        return c;
    }
}

/* =========================================================
 * CONSTRUCTOR / DESTRUCTOR
 * ========================================================= */
Task2Manager::Task2Manager(sf::Scalar stepsPerSecond, int scenarioCase)
    : sf::SimulationManager(stepsPerSecond, sf::Solver::SI, sf::CollisionFilter::EXCLUSIVE),
      activeCase(ResolveScenarioCase(scenarioCase))
{
    if (const char* env = std::getenv("TASK2_AUTOTEST_MAX_TIME"))
        autotestMaxTime = std::atof(env);
}

Task2Manager::~Task2Manager()
{
    if (controller != nullptr)
    {
        SDL_GameControllerClose(controller);
        controller = nullptr;
    }
}

/* =========================================================
 * BUILD SCENARIO
 * ========================================================= */
void Task2Manager::BuildScenario()
{
    /* ---------------- MATERIALS ---------------- */
    // KYUBIC part groups use effective densities (estimated mass / displaced
    // volume) generated from the CAD by tools/kyubic_step_export.py.
    std::vector<std::string> materials = {"PropellerMaterial", "TargetMaterial", "Rock", "Lead"};
    CreateMaterial("PropellerMaterial", 1000.0, 0.3);
    CreateMaterial("TargetMaterial", 1000.0, 0.3);
    CreateMaterial("Rock", 2700.0, 0.3);
    CreateMaterial("Lead", 11340.0, 0.3);
    CreateMaterial("KYUBIC_visual", 1.0, 0.3);      // graphics carriers: ~1 mg each
    materials.push_back("KYUBIC_visual");
    for (const auto& group : kyubic::GROUPS)
    {
        CreateMaterial(std::string("KYUBIC_") + group.name, group.density, 0.3);
        materials.push_back(std::string("KYUBIC_") + group.name);
    }
    for (size_t i = 0; i < materials.size(); ++i)
        for (size_t j = i; j < materials.size(); ++j)
            SetMaterialsInteraction(materials[i], materials[j], 0.5, 0.2);

    /* ---------------- LOOKS ---------------- */
    // Yellow like the real target. A red target is invisible beyond ~4 m here:
    // water absorbs red first and the target turns into a dark silhouette.
    CreateLook("tracked_object_yellow", sf::Color::RGB(1.0f, 0.85f, 0.0f), 0.6f, 0.0f, 0.0f);
    CreateLook("buoy_pole", sf::Color::RGB(0.18f, 0.20f, 0.22f), 0.4f, 0.2f, 0.0f);     // dark grey pole
    CreateLook("buoy_white", sf::Color::RGB(0.88f, 0.88f, 0.84f), 0.4f, 0.0f, 0.0f);    // mounts
    CreateLook("buoy_metal", sf::Color::RGB(0.48f, 0.50f, 0.52f), 0.3f, 0.8f, 0.0f);    // bracket
    CreateLook("kyubic_frame", sf::Color::RGB(0.74f, 0.76f, 0.79f), 0.35f, 0.8f, 0.0f);   // aluminium
    CreateLook("kyubic_hull", sf::Color::RGB(0.47f, 0.63f, 0.78f), 0.2f, 0.1f, 0.1f);    // acrylic tube
    CreateLook("kyubic_dark", sf::Color::RGB(0.12f, 0.12f, 0.14f), 0.4f, 0.3f, 0.0f);    // battery tubes, DVL
    CreateLook("kyubic_float", sf::Color::RGB(0.94f, 0.47f, 0.12f), 0.6f, 0.0f, 0.0f);   // floaters
    CreateLook("kyubic_thruster", sf::Color::RGB(0.03f, 0.03f, 0.03f), 0.3f, 0.1f, 0.0f);
    CreateLook("propeller", sf::Color::RGB(0.05f, 0.05f, 0.05f), 0.3f, 0.0f, 0.0f);
    CreateLook("TargetRed", sf::Color::RGB(1.0f, 0.05f, 0.05f), 0.5f, 0.0f, 0.0f);
    CreateLook("seabed", sf::Color::RGB(0.45f, 0.35f, 0.20f), 0.9f, 0.0f, 0.0f);

    /* ---------------- OCEAN ---------------- */
    EnableOcean(0.0);
    // TASK2_WATER_TYPE / TASK2_SUN_ELEV vary the look (detector datasets).
    const char* waterEnv = std::getenv("TASK2_WATER_TYPE");
    getOcean()->setWaterType(waterEnv ? std::atof(waterEnv) : 0.2);
    getOcean()->AddVelocityField(new sf::Uniform(sf::Vector3(0.0, 0.0, 0.0)));
    getOcean()->EnableCurrents();
    const char* sunEnv = std::getenv("TASK2_SUN_ELEV");
    getAtmosphere()->SetSunPosition(0.0, sunEnv ? std::atof(sunEnv) : 60.0);

    /* ---------------- SEABED (top at Sim::SEABED_DEPTH) ---------------- */
    // Its depth scales with the world; it stays 30 m wide (plain colour, so
    // no visible scale cue, and no floor edge in view).
    sf::Obstacle* seabed = new sf::Obstacle(
        "Seabed", sf::Vector3(30.0, 30.0, 0.5 * Sim::L), sf::I4(), "Rock", "seabed");
    AddStaticEntity(seabed, sf::Transform(sf::IQ(), sf::Vector3(0.0, 0.0, Sim::SEABED_DEPTH + 0.25 * Sim::L)));

    /* ---------------- TRACKED OBJECT ---------------- */
    // The team's buoy in its real colours: one object per part
    // (models/buoy/, tools/split_buoy.py), all scaled by Sim::BUOY_SCALE and
    // moved together. trackedObject = the yellow bulb (its origin is the buoy's).
    const std::string buoyDir = std::string(BUOY_MODEL_DIR);
    static const char* const buoyParts[][2] = {
        {"yellow", "tracked_object_yellow"}, {"dark_gray", "buoy_pole"},
        {"white", "buoy_white"}, {"metal", "buoy_metal"}};

    if (ScenarioConfig::HasTarget(activeCase))
    {
        const sf::Vector3 targetPos = ScenarioConfig::TargetPosition(activeCase);
        for (const auto& part : buoyParts)
        {
            const std::string path = buoyDir + part[0] + ".obj";
            sf::Obstacle* obstacle = new sf::Obstacle(
                std::string("TrackedObject_") + part[0], path, Sim::BUOY_SCALE, sf::I4(),
                path, Sim::BUOY_SCALE, sf::I4(), true, "TargetMaterial", part[1]);
            AddStaticEntity(obstacle, sf::Transform(sf::Quaternion(0.0, M_PI, 0.0), targetPos));
            targetParts.push_back(obstacle);
        }
        trackedObject = targetParts.front();
        std::cout << "[SCENARIO] target at x=" << targetPos.getX() << " y=" << targetPos.getY()
                  << " depth=" << targetPos.getZ() << std::endl;
    }

    /* ---------------- KYUBIC V2 HULL ---------------- */
    // Physics: every plate, extrusion, tube and floater is its own box /
    // cylinder / sphere with the CAD's volume and its group's density, so each
    // gets its own buoyancy, drag and added mass. They are external parts (the
    // fluid acts on them) and hidden. Graphics: the real CAD meshes, carried by
    // internal parts with negligible mass, displayed instead of the physics.
    // Generated from "KYUBIC V2 2025 2026Iwakuni.step" by tools/kyubic_step_export.py.
    sf::PhysicsSettings hullPhysics;
    hullPhysics.mode = sf::PhysicsMode::SUBMERGED;
    hullPhysics.collisions = true;
    hullPhysics.buoyancy = true;

    sf::Compound* hull = nullptr;
    double addedMassBody[3] = {0.0, 0.0, 0.0};
    for (unsigned i = 0; i < kyubic::PRIMITIVE_COUNT; ++i)
    {
        const auto& prim = kyubic::PRIMITIVES[i];
        const std::string name = std::string("KYUBIC_") + prim.group + "_" + std::to_string(i);
        const std::string material = std::string("KYUBIC_") + prim.group;
        const double rho = getOcean()->getLiquid().density;
        sf::SolidEntity* solid = nullptr;
        sf::Quaternion axisRotation = sf::IQ();      // Stonefish cylinders run along local z
        if (prim.type == 'b')
        {
            const double lx = prim.size[0], ly = prim.size[1], lz = prim.size[2];
            double m[3] = {FaceAddedMass(ly, lz, rho), FaceAddedMass(lx, lz, rho), FaceAddedMass(lx, ly, rho)};
            const double dims[3] = {lx, ly, lz};
            for (int k = 0; k < 3; ++k)      // thin plate: its broadside is perforated
                if (dims[k] < 0.012 * ROBOT_SCALE && dims[(k + 1) % 3] > 0.1 * ROBOT_SCALE
                    && dims[(k + 2) % 3] > 0.1 * ROBOT_SCALE)
                    m[k] *= PERFORATED_PLATE_FACTOR;
            const sf::Vector3 am(m[0], m[1], m[2]);
            const sf::Vector3 ai((m[1] * lz * lz + m[2] * ly * ly) / 12.0,
                                 (m[0] * lz * lz + m[2] * lx * lx) / 12.0,
                                 (m[0] * ly * ly + m[1] * lx * lx) / 12.0);
            solid = new ExplicitAddedMass<sf::Box>(am, ai, name, hullPhysics, sf::Vector3(lx, ly, lz),
                                                  sf::I4(), material, "kyubic_dark");
        }
        else if (prim.type == 'c')
        {
            const double r = prim.size[0], len = prim.size[1];
            const double lateral = rho * M_PI * r * r * len;            // Ca = 1 across the axis
            const sf::Vector3 am(lateral, lateral, 0.1 * lateral);      // local z = axis
            const sf::Vector3 ai(lateral * len * len / 12.0, lateral * len * len / 12.0, 0.0);
            solid = new ExplicitAddedMass<sf::Cylinder>(am, ai, name, hullPhysics, r, len, sf::I4(),
                                                       material, "kyubic_dark");
            if (prim.axis == 'x')
                axisRotation = sf::Quaternion(0.0, M_PI_2, 0.0);
            else if (prim.axis == 'y')
                axisRotation = sf::Quaternion(0.0, 0.0, -M_PI_2);
        }
        else
        {
            const double r = prim.size[0];
            const double half = 0.5 * rho * 4.0 / 3.0 * M_PI * r * r * r;
            solid = new ExplicitAddedMass<sf::Sphere>(sf::Vector3(half, half, half), sf::Vector3(0, 0, 0),
                                                     name, hullPhysics, r, sf::I4(), material, "kyubic_dark");
        }

        const sf::Transform pose(axisRotation, sf::Vector3(prim.center[0], prim.center[1], prim.center[2]));
        // added mass of this solid, rotated into the body frame
        const sf::Vector3 am = solid->getAddedMass();
        if (std::getenv("KYUBIC_DEBUG_ADDED_MASS"))
            std::cout << "[KYUBIC] " << name << " " << prim.type << " added mass local " << am.getX() << " "
                      << am.getY() << " " << am.getZ() << std::endl;
        const sf::Matrix3 rot(axisRotation);
        for (int k = 0; k < 3; ++k)
            addedMassBody[k] += std::fabs(rot[k][0]) * am.getX() + std::fabs(rot[k][1]) * am.getY()
                              + std::fabs(rot[k][2]) * am.getZ();
        if (hull == nullptr)
            hull = new sf::Compound("Hull", hullPhysics, solid, pose);
        else
            hull->AddExternalPart(solid, pose);
    }

    const std::string kyubicDir = std::string(KYUBIC_MODEL_DIR);
    struct PartLook { const char* group; const char* look; };
    const PartLook partLooks[] = {
        {"control_hull", "kyubic_hull"}, {"frame", "kyubic_frame"}, {"floaters", "kyubic_float"},
        {"battery_hulls", "kyubic_dark"}, {"dvl", "kyubic_dark"}, {"thrusters", "kyubic_thruster"},
        {"claw", "kyubic_frame"}};
    for (const auto& part : partLooks)
        hull->AddInternalPart(
            new sf::Polyhedron(std::string("KYUBIC_visual_") + part.group, hullPhysics,
                               kyubicDir + part.group + "_visual.obj", 1.0, sf::I4(),
                               kyubicDir + "tiny_physics.obj", 1.0, sf::I4(),
                               "KYUBIC_visual", part.look),
            sf::I4());
    hull->setDisplayInternalParts(true);

    // Trim ballast (internal: mass only) so KYUBIC floats level, slightly positive.
    const double ballastVolume = kyubic::BALLAST_KG / 11340.0;
    const double ballastSide = std::cbrt(ballastVolume / 0.2);       // a 1 x 1 x 0.2 slab
    hull->AddInternalPart(
        new sf::Box("TrimBallast", hullPhysics,
                    sf::Vector3(ballastSide, ballastSide, 0.2 * ballastSide), sf::I4(), "Lead", "kyubic_dark"),
        sf::Transform(sf::IQ(), sf::Vector3(kyubic::BALLAST_POS[0], kyubic::BALLAST_POS[1],
                                            kyubic::BALLAST_POS[2])));

    /* ---------------- THRUSTERS: 6 x BlueRobotics T200 ---------------- */
    sf::PhysicsSettings propellerPhysics;
    propellerPhysics.mode = sf::PhysicsMode::SUBMERGED;
    propellerPhysics.collisions = false;
    propellerPhysics.buoyancy = false;

    for (size_t i = 0; i < kyubic::THRUSTER_COUNT; ++i)
    {
        const auto& mount = kyubic::THRUSTERS[i];
        // Stonefish's propeller mesh (0.178 m) scaled to the T200's 76 mm propeller
        auto propeller = std::make_shared<sf::Polyhedron>(
            std::string(mount.name) + "Propeller", propellerPhysics, sf::GetDataPath() + "propeller.obj",
            T200_DIAMETER * ROBOT_SCALE / 0.178, sf::I4(), "PropellerMaterial", "propeller");
        auto rotor = std::make_shared<sf::FirstOrder>(T200_TIME_CONSTANT * std::sqrt(ROBOT_SCALE));
        // Asymmetric quadratic thrust with the ESC dead band (T200 at 16 V), x THRUST_SCALE
        auto thrustModel = std::make_shared<sf::DeadbandThrust>(
            T200_KT_REVERSE * THRUST_SCALE, T200_KT_FORWARD * THRUST_SCALE,
            -T200_DEADBAND_W2, T200_DEADBAND_W2);
        thrusters[i] = new sf::Thruster(mount.name, propeller, rotor, thrustModel,
                                        T200_DIAMETER * ROBOT_SCALE, true, T200_MAX_OMEGA, false, true);
    }

    /* ---------------- SENSORS ---------------- */
    odometry = new sf::Odometry("Odometry", ODOMETRY_RATE_HZ);

    // ZED Mini left lens (the detection camera) behind the front dome
    sf::ColorCamera* camera = new sf::ColorCamera(
        "FrontCamera", CAMERA_WIDTH_PX, CAMERA_HEIGHT_PX, CAMERA_HFOV_DEG, CAMERA_RATE_HZ);
    camera->InstallNewDataHandler([this](sf::ColorCamera* cam) { PublishCameraImage(cam); });

    /* ---------------- ROBOT ---------------- */
    sf::FeatherstoneRobot* boat = new sf::FeatherstoneRobot("AUV");
    boat->DefineLinks(hull);
    robotHull = hull;
    boat->BuildKinematicStructure();

    for (size_t i = 0; i < kyubic::THRUSTER_COUNT; ++i)
    {
        const auto& mount = kyubic::THRUSTERS[i];
        // The thruster pushes along its local +x: rotate +x onto the mount direction.
        const double yawAngle = std::atan2(mount.dir[1], mount.dir[0]);
        const double pitchAngle = -std::asin(std::clamp(mount.dir[2], -1.0, 1.0));
        boat->AddLinkActuator(thrusters[i], "Hull",
            sf::Transform(sf::Quaternion(yawAngle, pitchAngle, 0.0),
                          sf::Vector3(mount.pos[0], mount.pos[1], mount.pos[2])));
    }

    boat->AddLinkSensor(odometry, "Hull", sf::I4());
    boat->AddVisionSensor(camera, "Hull",
        sf::Transform(sf::Quaternion(1.57, 0.0, 1.57),
                      sf::Vector3(CAMERA_MOUNT[0], CAMERA_MOUNT[1], CAMERA_MOUNT[2])));

    AddRobot(boat, sf::Transform(sf::IQ(), sf::Vector3(0.0, 0.0, LAUNCH_DEPTH)));
    // Detector dataset mode: the buoy is placed anywhere near the camera, also
    // where it overlaps the hull (close-up views); a contact there made the
    // physics explode, so the buoy passes through the robot in that mode.
    if (std::getenv("TASK2_DATASET") != nullptr && activeCase == ScenarioConfig::ROV_CASE)
        for (sf::Obstacle* part : targetParts)
            DisableCollision(hull, part);

    BuildAllocation();
    std::cout << "[KYUBIC] added mass (sum of solids) surge/sway/heave: " << addedMassBody[0] << " / "
              << addedMassBody[1] << " / " << addedMassBody[2] << " kg" << std::endl;
    std::cout << "[KYUBIC] mass " << hull->getMass() << " kg, displaced "
              << hull->getVolume() * getOcean()->getLiquid().density << " kg ("
              << kyubic::THRUSTER_COUNT << " x T200)" << std::endl;

    /* ---------------- ROS 2 ---------------- */
    rosNode = rclcpp::Node::make_shared("task2_baseline");

    odomPublisher = rosNode->create_publisher<nav_msgs::msg::Odometry>("/usv/odom", 10);
    imagePublisher = rosNode->create_publisher<sensor_msgs::msg::Image>(
        "Stonefish/camera/image_raw", 10);
    controlDebugPublisher = rosNode->create_publisher<std_msgs::msg::Float64MultiArray>(
        "/usv/control_debug", 10);
    // Detector dataset mode (ROV case, TASK2_DATASET=1): the buoy jumps to
    // random places in view and every camera frame gets its exact label
    // (tools/dataset_saver.py writes the YOLO dataset).
    datasetMode = std::getenv("TASK2_DATASET") != nullptr && activeCase == ScenarioConfig::ROV_CASE;
    if (datasetMode)
    {
        datasetLabelPublisher = rosNode->create_publisher<std_msgs::msg::Float64MultiArray>("/dataset/label", 50);
        std::ifstream obj(std::string(BUOY_MODEL_DIR) + "yellow.obj");
        std::string line;
        while (std::getline(obj, line))
        {
            double x, y, z;
            if (line.rfind("v ", 0) == 0 && std::sscanf(line.c_str() + 2, "%lf %lf %lf", &x, &y, &z) == 3)
                bulbVertices.push_back({x * Sim::BUOY_SCALE, y * Sim::BUOY_SCALE, z * Sim::BUOY_SCALE});
        }
        std::cout << "[DATASET] mode on: " << bulbVertices.size() << " bulb vertices for the labels" << std::endl;
    }

    objectPositionSubscriber = rosNode->create_subscription<geometry_msgs::msg::PointStamped>(
        "/object/position", 10,
        std::bind(&Task2Manager::ObjectPositionCallback, this, std::placeholders::_1));
    targetDetectionSubscriber = rosNode->create_subscription<geometry_msgs::msg::PointStamped>(
        "/stonefish/target/detection", 10,
        std::bind(&Task2Manager::TargetDetectionCallback, this, std::placeholders::_1));
    targetBoxSubscriber = rosNode->create_subscription<std_msgs::msg::Float64MultiArray>(
        "/stonefish/target/bbox", 10,
        std::bind(&Task2Manager::TargetBoxCallback, this, std::placeholders::_1));
    visionFlowSubscriber = rosNode->create_subscription<std_msgs::msg::Float64MultiArray>(
        "/vision/heave_flow", 10,
        std::bind(&Task2Manager::VisionFlowCallback, this, std::placeholders::_1));

    if (const char* src = std::getenv("TASK2_DEPTH_SOURCE"))
        useVisionDepth = std::string(src) != "sensor";
    if (const char* src = std::getenv("TASK2_NAV_SOURCE"))
        useSensorNav = std::string(src) == "sensor";
    std::cout << "[NAV] heading / speed / position: " << (useSensorNav
        ? "simulator IMU + odometry (TASK2_NAV_SOURCE=sensor, comparison only)"
        : "camera + thrust-model observers (no IMU, no DVL)") << std::endl;
    visionDepth = LAUNCH_DEPTH;
    depthRef = filteredTargetDepth = LAUNCH_DEPTH;
    kfX[0] = LAUNCH_DEPTH;
    kfX[1] = 0.0;
    kfX[2] = HEAVE_BIAS_INIT;
    kfP[0][0] = 0.01 * Sim::L * Sim::L;   // released at a known depth
    kfP[1][1] = 0.0025 * Sim::L;          // (m/s)^2 = x V^2
    kfP[2][2] = 0.0004;
    std::cout << "[DEPTH] source: " << (useVisionDepth
        ? "CAMERA (visual_heave_estimator.py on /vision/heave_flow), released at "
          + std::to_string(LAUNCH_DEPTH) + " m"
        : std::string("simulator truth (TASK2_DEPTH_SOURCE=sensor, comparison only)")) << std::endl;

    InitializeController();

    // Starting hover thrust from the identified net buoyancy (observer bias
    // prior), refined while holding depth. Starting at 0 made a robot whose
    // turn stick was held from the start float up 0.65 m (positively buoyant).
    if (heaveForcePerCmd > 1e-6)
    {
        heaveTrim = rovHeaveTrim = hoverTrimPrior =
            std::clamp(-HEAVE_BIAS_INIT * HEAVE_MASS_KG / heaveForcePerCmd, -HEAVE_MAX, HEAVE_MAX);
        std::cout << "[DEPTH] hover trim prior " << heaveTrim << " (heave force " << heaveForcePerCmd
                  << " N per unit command)" << std::endl;
    }

    rovMode = (activeCase == ScenarioConfig::ROV_CASE);
    autonomousMode = !rovMode;
    ResetAutonomousControl();
    if (ENABLE_ACTUATOR_CALIBRATION && !rovMode)
        EnterState(MotionState::CALIBRATE);
    if (rovMode)
    {
        std::cout
            << "\n========================================\n"
            << "       KYUBIC ROV MODE (piloted)\n"
            << "========================================\n"
            << "  No autonomy. The target follows the real camera (/object/position).\n"
            << "  Controller: left stick  = forward/reverse + turn\n"
            << "              right stick = sideways + up/down\n"
            << "              B           = stop all thrusters\n"
            << "  Keyboard:   W/S forward/reverse  A/D turn  Q/E sideways\n"
            << "              R/F ascend/descend   SPACE stop\n"
            << "  Assists:    roll hold, heading hold when not turning (IMU)\n"
            << "  Guard:      forward thrust reduced from " << ROV_SLOW_GAP << " m and cut at "
            << ROV_STOP_GAP << " m\n"
            << "              from the buoy (needs the Stonefish YOLO detector running)\n"
            << "========================================\n" << std::endl;
        return;
    }

    std::cout
        << "\n========================================\n"
        << "       UNDERWATER AUV SIMULATION\n"
        << "========================================\n"
        << "  Test case " << activeCase
        << (ScenarioConfig::HasTarget(activeCase) ? "" : " (NO target in the world)") << "\n"
        << "  Autonomous visual servoing ENABLED\n"
        << "  Needs a detector on /stonefish/target/detection (+ /stonefish/target/bbox):\n"
        << "    kyushu_project/start_pipeline.sh starts the YOLO detector for you\n"
        << "  Mission: align -> approach -> stop at target -> back off -> surface\n"
        << "  Startup: ~20 s thruster sign calibration\n"
        << "  M     : toggle autonomous / manual\n"
        << "  KYUBIC V2 (CAD model), 6 x T200, ZED Mini camera\n"
        << "  W/S   : forward / reverse   A/D : turn left / right\n"
        << "  Q/E   : sideways left / right\n"
        << "  R/F   : ascend / descend    SPACE / B : stop\n"
        << "========================================\n" << std::endl;
}

/* =========================================================
 * XBOX CONTROLLER
 * ========================================================= */
bool Task2Manager::InitializeController()
{
    if (SDL_WasInit(SDL_INIT_GAMECONTROLLER) == 0)
    {
        if (SDL_InitSubSystem(SDL_INIT_GAMECONTROLLER | SDL_INIT_JOYSTICK) != 0)
        {
            std::cerr << "SDL controller initialization failed: " << SDL_GetError() << std::endl;
            return false;
        }
    }

    const int controllerCount = SDL_NumJoysticks();
    std::cout << "SDL controllers detected: " << controllerCount << std::endl;

    for (int i = 0; i < controllerCount; ++i)
    {
        if (SDL_IsGameController(i))
        {
            controller = SDL_GameControllerOpen(i);
            if (controller != nullptr)
            {
                controllerConnected = true;
                std::cout << "Controller connected: " << SDL_GameControllerName(controller) << std::endl;
                return true;
            }
        }
    }

    std::cout << "No SDL Game Controller found. Keyboard control will be used." << std::endl;
    return false;
}

/* =========================================================
 * CONTROLLER UPDATE (manual + mode switching)
 * ========================================================= */
void Task2Manager::UpdateController()
{
    if (controllerConnected && controller != nullptr)
    {
        SDL_GameControllerUpdate();
        if (SDL_GameControllerGetButton(controller, SDL_CONTROLLER_BUTTON_B))
        {
            ResetAutonomousControl();
            rovHeadingHold = false;
            SetAxisCommands(0.0, 0.0, 0.0, 0.0, 0.0);
            return;
        }
    }

    SDL_PumpEvents();
    const Uint8* keys = SDL_GetKeyboardState(nullptr);

    static bool previousM = false;
    const bool mPressed = keys[SDL_SCANCODE_M] != 0;
    if (mPressed && !previousM && !rovMode)
    {
        autonomousMode = !autonomousMode;
        ResetAutonomousControl();
        std::cout << "[MODE] " << (autonomousMode ? "AUTONOMOUS" : "MANUAL") << std::endl;
    }
    previousM = mPressed;

    if (keys[SDL_SCANCODE_SPACE])
    {
        ResetAutonomousControl();
        rovHeadingHold = false;
        SetAxisCommands(0.0, 0.0, 0.0, 0.0, 0.0);
        return;
    }

    if (autonomousMode)
    {
        UpdateAutonomousControl(lastStepDt);
        return;
    }

    if (rovMode)
    {
        UpdateRovControl(keys);
        return;
    }

    // ---------------- MANUAL (uses calibrated signs too) ----------------
    if (controllerConnected && controller != nullptr)
    {
        float leftX  = SDL_GameControllerGetAxis(controller, SDL_CONTROLLER_AXIS_LEFTX) / 32767.0f;
        float leftY  = SDL_GameControllerGetAxis(controller, SDL_CONTROLLER_AXIS_LEFTY) / 32767.0f;
        float rightX = SDL_GameControllerGetAxis(controller, SDL_CONTROLLER_AXIS_RIGHTX) / 32767.0f;
        float rightY = SDL_GameControllerGetAxis(controller, SDL_CONTROLLER_AXIS_RIGHTY) / 32767.0f;

        const float DEADZONE = 0.10f;
        if (std::fabs(leftX) < DEADZONE)  leftX = 0.0f;
        if (std::fabs(leftY) < DEADZONE)  leftY = 0.0f;
        if (std::fabs(rightX) < DEADZONE) rightX = 0.0f;
        if (std::fabs(rightY) < DEADZONE) rightY = 0.0f;

        // left stick: forward / turn, right stick: sideways / up-down
        SetAxisCommands(-leftY, rightX, rightY, 0.0, leftX);
        return;
    }

    double forward = 0.0, sideways = 0.0, yaw = 0.0, vertical = 0.0;
    if (keys[SDL_SCANCODE_W]) forward += 0.65;
    if (keys[SDL_SCANCODE_S]) forward -= 0.65;
    if (keys[SDL_SCANCODE_Q]) sideways -= 0.40;
    if (keys[SDL_SCANCODE_E]) sideways += 0.40;
    if (keys[SDL_SCANCODE_A]) yaw -= 0.40;
    if (keys[SDL_SCANCODE_D]) yaw += 0.40;
    if (keys[SDL_SCANCODE_R]) vertical -= 0.40;
    if (keys[SDL_SCANCODE_F]) vertical += 0.40;

    SetAxisCommands(forward, sideways, vertical, 0.0, yaw);
}

/* =========================================================
 * ROV MODE: pilot input + assists + collision guard
 * ========================================================= */
void Task2Manager::UpdateRovControl(const Uint8* keys)
{
    controlWallTime = WallNow();
    const double dt = std::clamp<double>(lastStepDt, 1e-4, 0.05);

    // ---- pilot input (normalised -1..1) ----
    double forward = 0.0, sideways = 0.0, vertical = 0.0, turn = 0.0;
    if (controllerConnected && controller != nullptr)
    {
        auto axis = [this](SDL_GameControllerAxis a)
        {
            const double v = SDL_GameControllerGetAxis(controller, a) / 32767.0;
            return std::fabs(v) < ROV_STICK_DEADZONE ? 0.0 : std::clamp(v, -1.0, 1.0);
        };
        forward = -axis(SDL_CONTROLLER_AXIS_LEFTY);
        turn = axis(SDL_CONTROLLER_AXIS_LEFTX);
        sideways = axis(SDL_CONTROLLER_AXIS_RIGHTX);
        vertical = axis(SDL_CONTROLLER_AXIS_RIGHTY);     // stick down = descend
    }
    if (keys[SDL_SCANCODE_W]) forward += ROV_KEY_INPUT;
    if (keys[SDL_SCANCODE_S]) forward -= ROV_KEY_INPUT;
    if (keys[SDL_SCANCODE_A]) turn -= ROV_KEY_INPUT;
    if (keys[SDL_SCANCODE_D]) turn += ROV_KEY_INPUT;
    if (keys[SDL_SCANCODE_Q]) sideways -= ROV_KEY_INPUT;
    if (keys[SDL_SCANCODE_E]) sideways += ROV_KEY_INPUT;
    if (keys[SDL_SCANCODE_R]) vertical -= ROV_KEY_INPUT;
    if (keys[SDL_SCANCODE_F]) vertical += ROV_KEY_INPUT;
    // Test hooks (no pilot at the controls): hold sticks at fixed values.
    //   TASK2_ROV_TEST="forward,sideways,vertical,turn"   (each -1..1)
    //   TASK2_ROV_TEST_FORWARD=x                          (forward only)
    static double testInput[4] = {0.0, 0.0, 0.0, 0.0};
    static bool testInputRead = false;
    if (!testInputRead)
    {
        testInputRead = true;
        if (const char* env = std::getenv("TASK2_ROV_TEST"))
            std::sscanf(env, "%lf,%lf,%lf,%lf", &testInput[0], &testInput[1], &testInput[2], &testInput[3]);
        if (const char* env = std::getenv("TASK2_ROV_TEST_FORWARD"))
            testInput[0] = std::atof(env);
    }
    //   TASK2_ROV_SEQ="sec:f,s,v,t;sec:f,s,v,t;..."     (stick schedule, repeats;
    //   used to identify the thrust models on the robot)
    static std::vector<std::array<double, 5>> testSequence;
    static bool testSequenceRead = false;
    if (!testSequenceRead)
    {
        testSequenceRead = true;
        if (const char* env = std::getenv("TASK2_ROV_SEQ"))
        {
            std::string spec(env);
            size_t start = 0;
            while (start < spec.size())
            {
                const size_t end = std::min(spec.find(';', start), spec.size());
                std::array<double, 5> step{};
                if (std::sscanf(spec.substr(start, end - start).c_str(), "%lf:%lf,%lf,%lf,%lf",
                                &step[0], &step[1], &step[2], &step[3], &step[4]) == 5 && step[0] > 0.0)
                    testSequence.push_back(step);
                start = end + 1;
            }
            std::cout << "[ROV] test schedule: " << testSequence.size() << " steps" << std::endl;
        }
    }
    if (!testSequence.empty())
    {
        double period = 0.0;
        for (const auto& step : testSequence)
            period += step[0];
        double tIn = std::fmod(simTime, period);
        for (const auto& step : testSequence)
        {
            if (tIn < step[0])
            {
                for (int k = 0; k < 4; ++k)
                    testInput[k] = step[k + 1];
                break;
            }
            tIn -= step[0];
        }
    }
    if (datasetMode)
        testInput[3] = 0.35;   // dataset: keep turning slowly (the depth sweeps below)
    forward = std::clamp(forward + testInput[0], -1.0, 1.0);
    sideways += testInput[1];
    vertical += testInput[2];
    turn += testInput[3];
    pilotInput[0] = forward;
    pilotInput[1] = std::clamp(sideways, -1.0, 1.0);
    pilotInput[2] = std::clamp(vertical, -1.0, 1.0);
    pilotInput[3] = std::clamp(turn, -1.0, 1.0);
    turn = std::clamp(turn, -1.0, 1.0);
    sideways = std::clamp(sideways, -1.0, 1.0);
    vertical = std::clamp(vertical, -1.0, 1.0);

    // ---- IMU (attitude, rates) and speed through the water (DVL on the real
    //      robot). Position and depth truth below are telemetry only. ----
    double yaw = 0.0, roll = 0.0, pitch = 0.0, yawRate = 0.0, rollRate = 0.0, pitchRate = 0.0;
    double bodyVx = 0.0, posX = 0.0, posY = 0.0, depthTruth = 0.0, heaveVelTruth = 0.0;
    if (odometry != nullptr)
    {
        const double qx = odometry->getLastValue(6), qy = odometry->getLastValue(7);
        const double qz = odometry->getLastValue(8), qw = odometry->getLastValue(9);
        yaw = std::atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
        roll = std::atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy));
        pitch = std::asin(std::clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0));   // + = nose up
        rollRate = odometry->getLastValue(10);
        pitchRate = odometry->getLastValue(11);
        yawRate = odometry->getLastValue(12);
        bodyVx = odometry->getLastValue(3);
        posX = odometry->getLastValue(0);
        posY = odometry->getLastValue(1);
        depthTruth = odometry->getLastValue(2);
        heaveVelTruth = odometry->getLastValue(5);
    }
    // Truth kept for the telemetry only; the controls use the observers
    // (camera + thrust models, no IMU / DVL).
    const double yawTruth = yaw, yawRateTruth = yawRate, bodyVxTruth = bodyVx;
    UpdateNavigation(dt);
    yaw = navYaw;
    yawRate = navYawRate;
    roll = navRoll;
    pitch = navPitch;
    rollRate = navRollRate;
    pitchRate = navPitchRate;
    bodyVx = navVx;
    UpdateDepthObserver(dt, pitchRate, bodyVx, depthTruth, heaveVelTruth);
    // Real size, camera only: the ROV depth estimate is the thrust model from
    // the launch depth, and the model does not know that the surface stops a
    // rise. A full-up stick to the surface had it "rise" 0.4 m above the water,
    // and on release the hold dived the robot 0.4 m. KYUBIC cannot be shallower
    // than this (floaters out of the water even at full upward thrust).
    if (useVisionDepth && Sim::REAL_SIZE && kfX[0] < MIN_PHYSICAL_DEPTH)
    {
        kfX[0] = MIN_PHYSICAL_DEPTH;
        kfX[1] = std::max(kfX[1], 0.0);
        visionDepth = kfX[0];
        visionVz = kfX[1];
    }
    // Depth hold on the camera estimate; TASK2_DEPTH_SOURCE=sensor (comparison,
    // detector datasets) holds on the simulator's depth as in the mission.
    const double holdDepth = useVisionDepth ? visionDepth : depthTruth;
    const double holdVz = useVisionDepth ? visionVz : heaveVelTruth;

    // ---- roll: always held level ----
    const double rollCommand = std::clamp(-ROLL_KP * roll - ROLL_KD * rollRate, -ROLL_MAX, ROLL_MAX);

    // ---- turn: yaw-rate command; released -> heading hold ----
    double yawCommand = 0.0;
    if (turn != 0.0)
    {
        rovHeadingHold = false;
        const double turnMax = useSensorNav ? ROV_TURN_MAX : ROV_TURN_MAX_NO_GYRO;
        yawCommand = std::clamp(ROV_YAW_RATE_GAIN * (turn * ROV_MAX_YAW_RATE - yawRate),
                                -turnMax, turnMax);
    }
    else
    {
        if (!rovHeadingHold && std::fabs(yawRate) < 0.02 * Sim::W)
        {
            rovHeadingHold = true;
            rovHeadingRef = yaw;
        }
        yawCommand = rovHeadingHold
            ? std::clamp(YAW_KP * WrapAngle(rovHeadingRef - yaw) - YAW_KD * yawRate, -YAW_MAX, YAW_MAX)
            : std::clamp(-YAW_KD * yawRate, -YAW_MAX, YAW_MAX);   // damp the turn out first
    }

    // ---- vertical: thrust while pressed; released -> depth hold (camera estimate) ----
    double heaveCommand = 0.0;
    // Real size: the ROV depth estimate is the thrust model, available from the
    // start (waiting for the camera flow let the buoyant robot float up 7 cm).
    const bool rovDepthAvailable = !useVisionDepth || visionDepthOk || Sim::REAL_SIZE;
    if (vertical != 0.0 || !rovDepthAvailable)
    {
        rovDepthHold = false;
        depthIntegral = 0.0;
        // Relative to the hover thrust (KYUBIC floats): half stick down only
        // cancelled the buoyancy before, and full up was 3x full down.
        heaveCommand = std::clamp((rovDepthAvailable ? hoverTrimPrior : 0.0) + vertical * ROV_MAX_HEAVE,
                                  -HEAVE_MAX, HEAVE_MAX);
    }
    else
    {
        if (!rovDepthHold)
        {
            rovDepthHold = true;
            rovDepthRef = holdDepth;
        }
        // Enlarged world only: there the camera flow fed the depth estimate and
        // a turn corrupted it, so the learned trim was held while turning. At
        // real size the ROV depth estimate is the thrust model alone (turning
        // cannot corrupt it) and the hold stays on: the learned trim had
        // absorbed the hold's push that stopped an ascent (0.23 vs 0.15) and a
        // turn right after sank the robot 0.5 m to the floor.
        if (!useSensorNav && turn != 0.0 && !Sim::REAL_SIZE)
        {
            // Camera only: while turning the camera depth estimate is not
            // reliable (the view sweeps; a spin dived 2.6 m): hold the trim.
            heaveCommand = rovHeaveTrim;
            rovDepthRef = holdDepth;   // resume from here afterwards
        }
        else
        {
            if (datasetMode)   // dataset: sweep the camera between 1.4 and 3.2 m (tuned world)
                rovDepthRef = (2.3 + 0.9 * std::sin(2.0 * M_PI * simTime / (90.0 * Sim::T))) * Sim::L;
            const double err = rovDepthRef - holdDepth;
            const double unsat = DEPTH_KP * err - DEPTH_KD * holdVz + DEPTH_KI * depthIntegral;
            if (DEPTH_KI > 0.0 && (std::fabs(unsat) < HEAVE_MAX || unsat * err < 0.0))
                depthIntegral = std::clamp(depthIntegral + err * dt, -DEPTH_I_LIMIT / DEPTH_KI, DEPTH_I_LIMIT / DEPTH_KI);
            heaveCommand = std::clamp(unsat, -HEAVE_MAX, HEAVE_MAX);
            rovHeaveTrim += std::min(1.0, dt / HEAVE_TRIM_TAU) * (heaveCommand - rovHeaveTrim);
        }
    }

    // ---- collision guard (Stonefish camera: bulb box size) ----
    const bool bboxFresh = lastBboxTime >= 0.0 && simTime - lastBboxTime <= ROV_GUARD_BBOX_SEC;
    if (bboxFresh && bboxWidth > 1.0 && targetConfidence >= MIN_CONFIDENCE)
    {
        const double f = FocalLengthPx();
        double axisRange = f * TARGET_WIDTH_M / bboxWidth;
        if (!bboxClippedV && bboxHeight > 1.0)
            axisRange = std::min(axisRange, f * TARGET_HEIGHT_M / bboxHeight);
        const double measured = axisRange - TARGET_RADIUS_M - CAMERA_TO_FRONT;
        const double predicted = (rovFrontGap >= 0.0) ? rovFrontGap - bodyVx * dt : -1.0;
        // Close in (or a box cut off at the image edge), a partial box looks too
        // small = too far: only "closer" readings are believed; moving away
        // still opens the gap through the robot's own motion.
        // In ROV mode the target itself moves (real camera): a farther reading
        // from WHOLE boxes, repeated for ROV_GAP_CONFIRM_BOXES new boxes, is
        // believed (a single partial box is not).
        const bool newBox = lastBboxTime != rovLastGapBoxTime;
        rovLastGapBoxTime = lastBboxTime;
        if (newBox)
            rovGapFartherCount = (!bboxClippedH && !bboxClippedV && measured > predicted + 0.2 * Sim::L)
                                     ? rovGapFartherCount + 1 : 0;
        // Forward speed without a DVL: range to the target while it is a fixed
        // reference (not moved by the real camera) and its box is whole.
        if (newBox && !bboxClippedH && !bboxClippedV && !objectDetected && measured > 0.3 * Sim::L)
            ObserveTargetRange(measured + CAMERA_TO_FRONT);
        if (predicted >= 0.0 && (predicted < ROV_SLOW_GAP || bboxClippedH) &&
            rovGapFartherCount < ROV_GAP_CONFIRM_BOXES)
            rovFrontGap = std::min(predicted, measured);
        else
            rovFrontGap = measured;
    }
    else if (rovFrontGap >= 0.0)
    {
        rovFrontGap -= bodyVx * dt;                       // out of view: advance with own motion
        if (rovFrontGap > ROV_GUARD_FORGET_GAP)
            rovFrontGap = -1.0;
    }
    // Depth correction from the buoy's height in the image, while the buoy is
    // a FIXED reference (not being moved by the real camera).
    const bool buoyStatic = !objectDetected;
    if (newDetectionPending)
    {
        newDetectionPending = false;
        // Real size: not used. The pilot sets the depth, the identified heave
        // model and the camera flow track it within ~1 cm/s, while the buoy's
        // height (pitch / range errors that change as the gap closes) made
        // the hold sink the robot 0.08-0.15 m on the way to the buoy.
        if (useVisionDepth && !Sim::REAL_SIZE && buoyStatic && bboxFresh && !bboxClippedV && rovFrontGap > 0.0 &&
            targetConfidence >= GOOD_TRACK_CONFIDENCE)
        {
            const double f = FocalLengthPx();
            const double vRaw = camHorizontalIsX ? targetErrorY : targetErrorX;
            const double elevation = std::atan2(camVSign * vRaw, f) - pitch;
            const double lever = rovFrontGap + CAMERA_TO_FRONT + TARGET_RADIUS_M;
            ObserveTargetHeight(lever * std::tan(elevation), lever);
        }
    }

    double speedLimit = ROV_MAX_SPEED;
    if (rovFrontGap >= 0.0)
    {
        const double room = std::max(0.0, rovFrontGap - ROV_STOP_GAP);
        speedLimit = std::min(speedLimit, std::sqrt(2.0 * BRAKE_DECEL_PLAN * room));
        // as slow near the target as the autonomous approach: TERMINAL_MAX_SPEED
        // inside SLOW_ZONE_GAP, reached smoothly before it
        const double slowSpeed = useSensorNav ? TERMINAL_MAX_SPEED : TERMINAL_MAX_SPEED_NO_DVL;
        if (rovFrontGap < SLOW_ZONE_GAP)
            speedLimit = std::min(speedLimit, slowSpeed);
        else
            speedLimit = std::min(speedLimit, std::sqrt(slowSpeed * slowSpeed +
                                  2.0 * BRAKE_DECEL_PLAN * (rovFrontGap - SLOW_ZONE_GAP)));
        if (room <= 0.0)
            speedLimit = 0.0;
    }
    double speedTarget = (forward >= 0.0) ? forward * ROV_MAX_SPEED : forward * ROV_MAX_REVERSE;
    const bool limited = speedTarget > speedLimit;
    speedTarget = std::min(speedTarget, speedLimit);       // reversing is never limited
    // Camera only (no DVL) the robot's speed is only known from the model, so
    // the guard is closed on the camera-measured GAP: inside the slow zone the
    // speed follows the gap to the stop distance, and the robot backs away by
    // itself if it has crept closer (a model "zero speed" crept into contact).
    if (!useSensorNav && rovFrontGap >= 0.0 && rovFrontGap < SLOW_ZONE_GAP && forward >= 0.0)
        speedTarget = std::min(speedTarget, HOLD_KP * (rovFrontGap - ROV_STOP_GAP));
    if (limited && simTime - rovLastGuardPrint > 2.0)
    {
        rovLastGuardPrint = simTime;
        std::cout << "[ROV GUARD] target ~" << rovFrontGap << " m ahead: forward speed limited to "
                  << speedLimit << " m/s" << std::endl;
    }
    double surgeCommand = SurgeSpeedControl(speedTarget, bodyVx, dt, ROV_SPEED_ACCEL, ROV_SPEED_DECEL);
    // Camera only: cap the THRUST so the real speed stays safe even when the
    // model's speed is wrong (reversing while turning reached 0.55 m/s while
    // the model said 0.16 m/s). These caps give ~0.29 m/s forward, ~0.2 m/s reverse.
    if (!useSensorNav)
        surgeCommand = std::clamp(surgeCommand, -ROV_SURGE_CMD_REVERSE_NO_DVL, ROV_SURGE_CMD_FORWARD_NO_DVL);

    SetAxisCommands(surgeCommand, sideways * (useSensorNav ? ROV_MAX_SWAY : ROV_MAX_SWAY_NO_DVL),
                    heaveCommand, rollCommand, yawCommand);

    // ---- telemetry (same layout as the mission; state = ROV_MANUAL) ----
    if (controlDebugPublisher)
    {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        sf::Vector3 truth(nan, nan, nan);
        if (trackedObject != nullptr)
            truth = trackedObject->getTransform().getOrigin();
        std_msgs::msg::Float64MultiArray msg;
        msg.data.assign(42, nan);
        msg.data[0] = static_cast<double>(static_cast<int>(MotionState::ROV_MANUAL));
        msg.data[1] = simTime;
        msg.data[2] = posX;
        msg.data[3] = posY;
        msg.data[4] = depthTruth;
        msg.data[5] = yawTruth;
        msg.data[6] = rovHeadingHold ? rovHeadingRef : nan;
        msg.data[7] = yawRateTruth;
        msg.data[9] = heaveVelTruth;
        msg.data[10] = yawCommand;
        msg.data[8] = rovDepthHold ? rovDepthRef : nan;
        msg.data[11] = heaveCommand;
        msg.data[12] = surgeCommand;
        msg.data[32] = speedTarget;
        msg.data[14] = targetConfidence;
        msg.data[15] = targetErrorX;
        msg.data[16] = targetErrorY;
        msg.data[30] = rovFrontGap;
        msg.data[31] = bboxWidth;
        msg.data[33] = bodyVxTruth;
        msg.data[37] = truth.getX();
        msg.data[38] = truth.getY();
        msg.data[39] = truth.getZ();
        msg.data[40] = bboxHeight;
        msg.data.insert(msg.data.end(), {controlWallTime, thrusterWallTime, detectionRxWallTime,
                                         detectionImageStamp, objectRxWallTime, objectImageStamp});
        AppendThrustTelemetry(msg.data, visionDepth, visionVz);
        controlDebugPublisher->publish(msg);
    }
}

/* =========================================================
 * STATE HELPERS
 * ========================================================= */
void Task2Manager::EnterState(MotionState next)
{
    if (motionState == next)
        return;
    // Leaving the approach: the hold point is captured once the robot has stopped.
    // (BACK_OFF's goal stays the hold point while surfacing.)
    if (motionState == MotionState::APPROACH)
        holdValid = false;
    motionState = next;
    stateTime = 0.0;
    settledTime = 0.0;
    stallTime = 0.0;
    approachMoving = false;
    terminalLock = false;
    terminalWaitTime = 0.0;
    retreatGoalSet = false;
    if (next == MotionState::SURFACING)
    {
        // Start of the ascent: the thrust model's own rise is counted from here.
        surfaceStartDepth = visionDepth;
        surfaceModelVz = visionVz;
        surfaceModelRise = 0.0;
    }
    std::cout << "[MOTION STATE] " << StateName(static_cast<int>(next))
              << " (t=" << simTime << " s)" << std::endl;
}

void Task2Manager::ResetAutonomousControl()
{
    // Calibration results and camera mapping are kept on purpose.
    motionState = MotionState::TARGET_LOST_HOLD;
    stateTime = 0.0;
    settledTime = 0.0;
    searchTime = 0.0;
    searchTurned = searchTurnedModel = 0.0;
    refsInitialized = false;
    haveTargetEstimate = false;
    haveRange = false;
    rangeJumpCount = 0;
    lostGap = 1e9;
    outlierCount = 0;
    holdValid = false;
    speedRef = 0.0;
    surgeIntegral = 0.0;
    yawIntegral = 0.0;
    depthIntegral = 0.0;
    surgeCommand = 0.0;
    missionAborted = false;
    acquisitionStartTime = -1.0;
    lostSinceTime = simTime;
}

void Task2Manager::BeginSurfacing(const char* reason, bool aborted)
{
    missionAborted = aborted;
    std::cout << (aborted ? "[SAFETY] " : "[MISSION] ") << reason
              << " -> surfacing slowly." << std::endl;
    EnterState(MotionState::SURFACING);
}

/* =========================================================
 * STARTUP ACTUATOR CALIBRATION
 *
 * Pulse surge, yaw and heave through the thrust allocation and measure the
 * CHANGE in body velocity from odometry. Convention required by the controller:
 *   surge > 0 -> forward (+vx)   yaw > 0 -> turn right (+r)   heave > 0 -> down (+w)
 * ========================================================= */
bool Task2Manager::RunActuatorCalibration(double dt, double bodyVx, double heaveVel, double yawRate,
                                          double depthHold)
{
    struct Step { double surge, yaw, heave, duration; int measure; };
    static const Step steps[] = {
        {0.0, 0.0, 0.0, 2.0 * Sim::T, 0},                       // settle
        {CALIB_COMMAND, 0.0, 0.0, 2.0 * Sim::T, 1},             // surge
        {-CALIB_COMMAND, 0.0, 0.0, 2.0 * Sim::T, 0},            // undo the motion
        {0.0, 0.0, 0.0, 2.0 * Sim::T, 0},
        {0.0, CALIB_COMMAND, 0.0, 2.0 * Sim::T, 2},             // yaw
        {0.0, -CALIB_COMMAND, 0.0, 2.0 * Sim::T, 0},
        {0.0, 0.0, 0.0, 2.0 * Sim::T, 0},
        {0.0, 0.0, CALIB_HEAVE_COMMAND, 2.0 * Sim::T, 3},       // heave (down)
        {0.0, 0.0, -CALIB_HEAVE_COMMAND, 2.0 * Sim::T, 0},      // undo the depth change
        {0.0, 0.0, 0.0, 2.0 * Sim::T, 0}};
    constexpr int N = sizeof(steps) / sizeof(steps[0]);

    if (calibStep >= N)
        return true;

    const Step& s = steps[calibStep];
    if (!calibStepStarted)
    {
        calibStartVx = bodyVx;
        calibStartHeave = heaveVel;
        calibStartYawRate = yawRate;
        calibStepTime = 0.0;
        calibStepStarted = true;
    }

    // depth is held by the depth loop except while heave itself is measured
    const bool heaveStep = (s.measure == 3) || (s.heave != 0.0);
    // Camera only: the test turn corrupts the camera depth estimate and the
    // depth loop then dove 1.1 m (case 5). Hold the hover trim instead.
    const double hold = useSensorNav ? depthHold * heaveSign : heaveTrim;
    const double raw[5] = {s.surge, 0.0, heaveStep ? s.heave : hold, 0.0, s.yaw};
    ApplyAxisCommands(raw);   // RAW: no sign correction
    calibStepTime += dt;

    if (calibStepTime < s.duration)
        return false;

    const bool cameraOnly = !useSensorNav;
    auto decide = [cameraOnly](double delta, const char* name) -> double
    {
        if (cameraOnly)
        {
            // No IMU / DVL: the camera's response measurement is too weak and
            // noisy to reverse a thruster on (it flipped yaw in a test): keep the
            // sign from the CAD thruster geometry and only warn.
            std::cout << "[CALIB] " << name << ": camera response " << delta << " -> keeping +1"
                      << (delta < -CALIB_MIN_DELTA ? " (WARNING: response looked reversed - check wiring)" : "")
                      << std::endl;
            return 1.0;
        }
        if (std::fabs(delta) < CALIB_MIN_DELTA)
        {
            std::cout << "[CALIB] " << name << ": response too small (" << delta
                      << "), keeping +1. Increase CALIB_COMMAND if this repeats." << std::endl;
            return 1.0;
        }
        const double sign = (delta > 0.0) ? 1.0 : -1.0;
        std::cout << "[CALIB] " << name << ": delta=" << delta << " -> sign "
                  << (sign > 0 ? "+1" : "-1 (REVERSED, corrected)") << std::endl;
        return sign;
    };

    if (s.measure == 1)
        surgeSign = decide(bodyVx - calibStartVx, "surge (forward)");
    else if (s.measure == 2)
        yawSign = decide(yawRate - calibStartYawRate, "yaw (turn right)");
    else if (s.measure == 3)
    {
        const double delta = heaveVel - calibStartHeave;
        // Without a gyro the camera cannot separate pitching from heaving:
        // keep the sign from the thruster geometry.
        if (useVisionDepth && delta < 0.0 && (delta > -CALIB_HEAVE_FLIP_MIN || !useSensorNav))
        {
            std::cout << "[CALIB] heave (down): delta=" << delta << " (camera) is too weak to"
                      << " trust a reversal: keeping +1 from the thruster geometry." << std::endl;
            heaveSign = 1.0;
        }
        else
        {
            heaveSign = decide(delta, "heave (down)");
        }
    }

    ++calibStep;
    calibStepStarted = false;

    if (calibStep >= N)
    {
        std::cout << "[CALIB] done: surgeSign=" << surgeSign << " yawSign=" << yawSign
                  << " heaveSign=" << heaveSign << std::endl;
        return true;
    }
    return false;
}

/* =========================================================
 * ONLINE CAMERA-AXIS CHECK
 *
 * Physics for a static target:
 *   robot yaws right by dpsi  -> horizontal pixel error changes by ~ -f*dpsi
 *   robot goes deeper by dz   -> vertical pixel error DEcreases (target moves up)
 * Evidence is accumulated from consecutive detections; once it is clear,
 * the mapping (axis swap and signs) is fixed and reported.
 * ========================================================= */
void Task2Manager::UpdateCameraAxisCheck()
{
    const double f = FocalLengthPx();

    if (!camHVerified && evPsiAbs > 0.15)
    {
        const bool xDominant = std::fabs(evSx) >= std::fabs(evSy);
        const double best = xDominant ? evSx : evSy;
        const double other = xDominant ? evSy : evSx;

        if (std::fabs(best) > 0.3 * f * evPsiSq && std::fabs(best) > 2.0 * std::fabs(other))
        {
            const bool newIsX = xDominant;
            const double newSign = (best < 0.0) ? 1.0 : -1.0;
            if (newIsX != camHorizontalIsX || newSign != camHSign)
            {
                std::cout << "[CAMERA CHECK] Horizontal mapping corrected: axis="
                          << (newIsX ? "e_x" : "e_y (image rotated)")
                          << " sign=" << newSign << std::endl;
                haveTargetEstimate = false;   // old estimate used the wrong mapping
                if (newIsX != camHorizontalIsX)
                {
                    evVz = evZSq = evZAbs = 0.0;
                    camVVerified = false;
                }
            }
            else
            {
                std::cout << "[CAMERA CHECK] Horizontal mapping confirmed." << std::endl;
            }
            camHorizontalIsX = newIsX;
            camHSign = newSign;
            camHVerified = true;
        }
        else if (evPsiAbs > 1.0)
        {
            std::cout << "[CAMERA CHECK] Horizontal evidence inconclusive, restarting." << std::endl;
            evSx = evSy = evPsiSq = evPsiAbs = 0.0;
        }
    }

    if (!camVVerified && evZAbs > 0.25 * Sim::L)
    {
        if (std::fabs(evVz) > 0.02 * f * evZSq)
        {
            const double newSign = (evVz < 0.0) ? 1.0 : -1.0;
            std::cout << "[CAMERA CHECK] Vertical mapping "
                      << (newSign != camVSign ? "corrected" : "confirmed")
                      << ": sign=" << newSign << std::endl;
            if (newSign != camVSign)
                haveTargetEstimate = false;
            camVSign = newSign;
            camVVerified = true;
        }
        else if (evZAbs > 1.5 * Sim::L)
        {
            evVz = evZSq = evZAbs = 0.0;
        }
    }
}

/* =========================================================
 * PERCEPTION -> TARGET MEMORY (world frame)
 * ========================================================= */
void Task2Manager::ProcessDetection(double yaw, double depth, double pitch)
{
    const double f = FocalLengthPx();

    // 1) Use consecutive detections to verify camera axes. Only while the
    //    robot is aligning in place: forward motion and a moving target also
    //    shift the image and would give false evidence.
    const bool aligning =
        motionState == MotionState::YAW_ALIGN || motionState == MotionState::YAW_SETTLE ||
        motionState == MotionState::VERTICAL_ALIGN || motionState == MotionState::VERIFY_CENTER;
    if (ENABLE_ONLINE_CAMERA_CHECK && activeCase != 6 && aligning && havePrevDetection)
    {
        const double dtDet = simTime - prevDetTime;
        if (dtDet > 0.0 && dtDet < 0.4 * Sim::T)
        {
            const double dPsi = WrapAngle(yaw - prevDetYaw);
            const double dZ = depth - prevDetDepth;
            const double dU = targetErrorX - prevErrX;
            const double dV = targetErrorY - prevErrY;

            if (std::fabs(dPsi) > 0.002)
            {
                evSx += dU * dPsi;
                evSy += dV * dPsi;
                evPsiSq += dPsi * dPsi;
                evPsiAbs += std::fabs(dPsi);
            }
            // Vertical axis: only against an INDEPENDENT depth. With the camera
            // depth estimate the check would compare the image with itself (it
            // wrongly flipped the axis in tests); the mounting is known.
            if (!useVisionDepth && std::fabs(dZ) > 0.002 * Sim::L && std::fabs(dPsi) < 0.01)
            {
                const double dVert = camHorizontalIsX ? dV : dU;
                evVz += dVert * dZ;
                evZSq += dZ * dZ;
                evZAbs += std::fabs(dZ);
            }
            UpdateCameraAxisCheck();
        }
    }
    prevDetTime = simTime;
    prevDetYaw = yaw;
    prevDetDepth = depth;
    prevErrX = targetErrorX;
    prevErrY = targetErrorY;
    havePrevDetection = true;

    // 2) Range from the bounding box. A box touching the image border is cut
    //    off, so it can only say "the target is at most this far".
    //    The smaller of the width/height estimates is used (conservative:
    //    a partly segmented target looks smaller = farther, never closer).
    const bool bboxFresh = lastBboxTime >= 0.0 && simTime - lastBboxTime <= BBOX_FRESH_SEC;
    const bool clipH = bboxFresh && bboxClippedH;
    const bool clipV = bboxFresh && bboxClippedV;
    if (bboxFresh && bboxWidth > 1.0)
    {
        double axisRange = f * TARGET_WIDTH_M / bboxWidth;
        if (!clipV && bboxHeight > 1.0)
            axisRange = std::min(axisRange, f * TARGET_HEIGHT_M / bboxHeight);
        const double measRange = std::clamp(axisRange - TARGET_RADIUS_M, MIN_RANGE, MAX_RANGE);
        // Forward speed without a DVL: the range to a static target, from whole
        // boxes (a partial box looks too far).
        if (!clipH && !clipV && activeCase != 6 && !rovMode && measRange > 0.3 * Sim::L)
        {
            if (std::fabs(std::atan2(targetErrorX, f)) < 0.09 && std::fabs(navYawRate) < 0.02 * Sim::W)
                ObserveTargetRange(measRange);
            // Closing speed measured by the camera alone: slope of the range
            // over the last second (box-size noise is too large frame to
            // frame), only close in where the range is accurate.
            if (!rangeHistory.empty() && simTime - rangeHistory.back().first > 0.5 * Sim::T)
                rangeHistory.clear();
            rangeHistory.emplace_back(simTime, measRange);
            while (simTime - rangeHistory.front().first > RANGE_RATE_WINDOW_SEC)
                rangeHistory.pop_front();
            if (measRange < RANGE_RATE_MAX_RANGE && rangeHistory.size() >= 5 &&
                rangeHistory.back().first - rangeHistory.front().first >= 0.6 * RANGE_RATE_WINDOW_SEC)
            {
                double mt = 0.0, mr = 0.0;
                for (const auto& [ts, r] : rangeHistory) { mt += ts; mr += r; }
                mt /= rangeHistory.size();
                mr /= rangeHistory.size();
                double num = 0.0, den = 0.0;
                for (const auto& [ts, r] : rangeHistory) { num += (ts - mt) * (r - mr); den += (ts - mt) * (ts - mt); }
                if (den > 1e-9)
                {
                    rangeRateSpeed = std::clamp(-num / den, -0.3 * Sim::V, 0.4 * Sim::V);
                    rangeRateTime = simTime;
                }
            }
        }

        const bool suddenDrop = haveRange && measRange < RANGE_JUMP_RATIO * filteredRange;
        rangeJumpCount = suddenDrop ? rangeJumpCount + 1 : 0;

        if (suddenDrop && rangeJumpCount < RANGE_JUMP_CONFIRM_FRAMES)
        {
            // wait for confirmation
        }
        else if (clipH)
        {
            if (haveRange)
                filteredRange = std::min(filteredRange, measRange);
        }
        else if (!haveRange)
        {
            filteredRange = measRange;
            haveRange = true;
        }
        else if (measRange < filteredRange || (filteredRange > RANGE_TRUST_FAR && !terminalLock))
        {
            filteredRange += RANGE_FILTER_ALPHA * (measRange - filteredRange);
        }
        else if (!useSensorNav && !clipH && !clipV && !terminalLock)
        {
            // Camera only: a WHOLE box close in may also correct the range
            // outward (slowly); otherwise the prediction drifts with no DVL.
            filteredRange += 0.1 * (measRange - filteredRange);
        }
    }

    // In the terminal approach the heading stays locked. With the camera depth
    // estimate the target's height keeps correcting the depth while its box is
    // whole (not touching the top or bottom of the image); otherwise only the
    // range may change (and only shorter).
    const bool verticalOnly = terminalLock;
    if (terminalLock && (!useVisionDepth || clipV || !bboxFresh ||
                         filteredRange - CAMERA_TO_FRONT <= TERMINAL_VERTICAL_MIN_GAP ||
                         targetConfidence < GOOD_TRACK_CONFIDENCE))
        return;

    // 3) Pixel error -> angles (pinhole model).
    const double hRaw = camHorizontalIsX ? targetErrorX : targetErrorY;
    const double vRaw = camHorizontalIsX ? targetErrorY : targetErrorX;
    const double bearing   = std::atan2(camHSign * hRaw, f);   // + = target right
    // + = target below. A nose-up pitch makes the target look lower than it is.
    const double elevation = std::atan2(camVSign * vRaw, f) - pitch;

    // 4) Store as WORLD heading/depth so robot rotation does not corrupt memory.
    //    Depth offset = range * tan(elevation) once the range is known.
    const double lever = haveRange ? std::clamp(filteredRange, 0.5 * Sim::L, 8.0 * Sim::L) : DEPTH_PER_RAD;
    // Steer the hull centreline, not the camera, at the target (they differ
    // when the lens sits off the centreline). Needs the range; without it, aim
    // the camera.
    double hullBearing = bearing;
    if (haveRange)
    {
        const double toCentre = filteredRange + TARGET_RADIUS_M;
        hullBearing = std::atan2(CAMERA_MOUNT[1] + toCentre * std::sin(bearing),
                                 CAMERA_MOUNT[0] + toCentre * std::cos(bearing));
    }
    const double measYaw = WrapAngle(yaw + hullBearing);
    const double targetDz = lever * std::tan(elevation);   // + = target below the camera
    const double measDepth = depth + std::clamp(targetDz, -MAX_DEPTH_STEP, MAX_DEPTH_STEP);

    // Depth measurement from the target's height in the image (metric: needs
    // the range and a whole box; static targets only). The observer works in
    // a frame anchored to the target: depth = anchor - height offset.
    const double frontGapNow = filteredRange - CAMERA_TO_FRONT;
    const bool heightUsable = useVisionDepth && activeCase != 6 &&
        activeCase != ScenarioConfig::ROV_CASE && haveRange && bboxFresh && !clipV &&
        (!terminalLock || (frontGapNow > TERMINAL_VERTICAL_MIN_GAP &&
                           targetConfidence >= GOOD_TRACK_CONFIDENCE));
    const bool heightAccepted = heightUsable && ObserveTargetHeight(targetDz, lever);

    if (verticalOnly)
    {
        if (!heightAccepted)
            return;   // rejected by the observer (e.g. a partial box at close range)
        filteredTargetDepth += 0.5 * TARGET_FILTER_ALPHA * (measDepth - filteredTargetDepth);
        if (!useVisionDepth)   // camera depth: absolute depth is not observable
            filteredTargetDepth = std::clamp(filteredTargetDepth, MIN_DEPTH, MAX_DEPTH);
        return;
    }

    if (!haveTargetEstimate)
    {
        filteredTargetYaw = measYaw;
        filteredTargetDepth = measDepth;
        haveTargetEstimate = true;
        outlierCount = 0;
    }
    else
    {
        // A clipped centroid is biased toward the visible part: keep that axis.
        const bool jump = std::fabs(WrapAngle(measYaw - filteredTargetYaw)) > OUTLIER_JUMP_RAD;
        if (jump && outlierCount + 1 < OUTLIER_CONFIRM_FRAMES)
        {
            ++outlierCount;
        }
        else
        {
            if (jump)
            {
                filteredTargetYaw = measYaw;
                if (!clipV)
                    filteredTargetDepth = measDepth;
            }
            else
            {
                if (!clipH)
                    filteredTargetYaw = WrapAngle(filteredTargetYaw +
                        TARGET_FILTER_ALPHA * WrapAngle(measYaw - filteredTargetYaw));
                if (!clipV)
                {
                    // No gyro: average over the rocking period (10 Hz detections).
                    const double alpha = useSensorNav ? TARGET_FILTER_ALPHA
                                                      : std::min(1.0, 0.1 / TARGET_DEPTH_TAU_NO_GYRO);
                    filteredTargetDepth += alpha * (measDepth - filteredTargetDepth);
                }
            }
            outlierCount = 0;
        }
    }
    // With the camera depth estimate the absolute depth is NOT observable (it
    // drifts ~0.75 m during a blind search): clamping the target there held the
    // robot 0.6 m below the target in a test. Alignment is relative to the target.
    if (!useVisionDepth)
        filteredTargetDepth = std::clamp(filteredTargetDepth, MIN_DEPTH, MAX_DEPTH);

    if (std::fabs(bearing) > 0.02)
        searchDirection = (bearing > 0.0) ? 1.0 : -1.0;
}

/* =========================================================
 * SURGE: speed loop (thrust fraction from desired body speed)
 * ========================================================= */
double Task2Manager::SurgeSpeedControl(double speedTarget, double bodyVx, double dt,
                                       double accel, double decel)
{
    // Smooth speed reference: no thrust steps, gentle on the hull and thrusters.
    if (accel <= 0.0) accel = SPEED_ACCEL;
    if (decel <= 0.0) decel = SPEED_DECEL;
    const double rate = (std::fabs(speedTarget) > std::fabs(speedRef)) ? accel : decel;
    const double refStep = std::clamp(speedTarget - speedRef, -rate * dt, rate * dt);
    speedRef += refStep;
    // Filtered rate of the reference: the ramp can flip direction from one step
    // to the next, and an unfiltered feed-forward would then chatter.
    const double alpha = dt / (SURGE_ACCEL_FILTER_SEC + dt);
    surgeRefAccel += alpha * (refStep / dt - surgeRefAccel);

    const double err = speedRef - bodyVx;
    const double unsat = SURGE_KDRAG_LIN * speedRef + SURGE_KDRAG * speedRef * std::fabs(speedRef) + SURGE_KA * surgeRefAccel +
                         SURGE_KP * err + SURGE_KI * surgeIntegral;
    if (std::fabs(unsat) < SURGE_MAX || unsat * err < 0.0)
        surgeIntegral = std::clamp(surgeIntegral + err * dt,
                                   -SURGE_I_LIMIT / SURGE_KI, SURGE_I_LIMIT / SURGE_KI);
    return std::clamp(unsat, -SURGE_MAX, SURGE_MAX);
}

/* =========================================================
 * AUTONOMOUS CONTROL
 *
 * Guidance (state machine) decides WHICH setpoints follow the target.
 * Control (always on, 100 Hz odometry) holds yaw, depth and position with
 * PD(+I) loops, so the robot never drifts in one axis while another is
 * being corrected.
 *
 * Mission:  CALIBRATE -> SEARCH -> YAW_ALIGN -> YAW_SETTLE -> VERTICAL_ALIGN
 *           -> VERIFY_CENTER -> APPROACH -> TARGET_REACHED -> BACK_OFF
 *           -> SURFACING -> SURFACED
 * No target after one full search turn, or no detector: -> SURFACING.
 * ========================================================= */
void Task2Manager::UpdateAutonomousControl(double dt)
{
    controlWallTime = WallNow();
    if (odometry == nullptr)
    {
        SetAxisCommands(0.0, 0.0, 0.0, 0.0, 0.0);
        return;
    }
    dt = std::clamp(dt, 1e-4, 0.05);

    // Simulator pose: evaluation ground truth for the telemetry only.
    const double posXTruth     = odometry->getLastValue(0);
    const double posYTruth     = odometry->getLastValue(1);
    const double bodyVxTruth   = odometry->getLastValue(3);
    const double depthTruth    = odometry->getLastValue(2);
    const double heaveVelTruth = odometry->getLastValue(5);
    const double yawRateTruth  = odometry->getLastValue(12);
    double yawTruth = 0.0;
    {
        const double qx = odometry->getLastValue(6), qy = odometry->getLastValue(7);
        const double qz = odometry->getLastValue(8), qw = odometry->getLastValue(9);
        yawTruth = std::atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
    }

    // What the controller uses: camera + thrust-model observers (no IMU, no DVL).
    UpdateNavigation(dt);
    const double posX = navX, posY = navY, bodyVx = navVx;
    const double yaw = navYaw, yawRate = navYawRate;
    const double pitch = navPitch, roll = navRoll, rollRate = navRollRate, pitchRate = navPitchRate;

    // ---------------- Depth from the camera (observer) ----------------
    UpdateDepthObserver(dt, pitchRate, bodyVx, depthTruth, heaveVelTruth);

    const double depth    = useVisionDepth ? visionDepth : depthTruth;
    const double heaveVel = useVisionDepth ? visionVz : heaveVelTruth;
    // Without a depth estimate the depth loop cannot work: thrusters stay off
    // vertically and KYUBIC (positively buoyant) floats up to the surface.
    const bool depthUnknown = useVisionDepth && !visionDepthOk;
    posXNow = posX;
    posYNow = posY;

    // ---------------- CALIBRATE ----------------
    if (motionState == MotionState::CALIBRATE)
    {
        // Hold the starting depth while surge and yaw are pulsed: KYUBIC is
        // slightly positively buoyant and would otherwise float up.
        if (calibDepthRef < 0.0)
            calibDepthRef = depth;
        // Track the target already: its image gives the metric vertical speed.
        if (newDetectionPending)
        {
            newDetectionPending = false;
            if (targetDetected)
                ProcessDetection(yaw, depth, pitch);
        }
        const double depthHold = std::clamp(
            DEPTH_KP * (calibDepthRef - depth) - DEPTH_KD * heaveVel, -HEAVE_MAX, HEAVE_MAX);
        // The heave sign must come from a measurement, not from the observer
        // (which predicts from the command itself): camera flow speed.
        // Signs must come from measurements, not from observers that predict
        // from the command itself: camera flow (spread, sideways shift, rise).
        if (RunActuatorCalibration(dt, useSensorNav ? bodyVx : flowSurgeSign,
                                   useVisionDepth ? flowVz : heaveVel,
                                   useSensorNav ? yawRate : flowYawRate, depthHold))
        {
            refsInitialized = false;   // capture the pose reached after calibration
            lostSinceTime = simTime;
            EnterState(MotionState::TARGET_LOST_HOLD);
        }
        stateTime += dt;
        return;
    }

    if (!refsInitialized)
    {
        yawRef = yaw;
        depthRef = std::clamp(depth, MIN_DEPTH, MAX_DEPTH);
        depthRefSlewed = depthRef;
        holdValid = false;
        speedRef = 0.0;
        refsInitialized = true;
    }
    if (autonomyStartTime < 0.0)
        autonomyStartTime = simTime;

    // ---------------- Perception ----------------
    // Range prediction between camera frames: the target gets closer by
    // exactly what the robot travels toward it.
    // Without a DVL: the larger of the model speed and the camera-measured
    // closing speed, so errors make the robot think it is closer (stops early).
    const bool rangeRateFresh = rangeRateTime >= 0.0 && simTime - rangeRateTime < 0.5 * Sim::T;
    // Camera only, also the thrust-model speed (no measurement can bias it).
    const double guardSpeed = useSensorNav ? bodyVx
        : std::max({bodyVx, surgeVelModel, rangeRateFresh ? rangeRateSpeed : bodyVx});
    if (haveRange)
        filteredRange = std::max(0.0, filteredRange - guardSpeed * dt);

    if (newDetectionPending)
    {
        newDetectionPending = false;
        if (targetDetected)
            ProcessDetection(yaw, depth, pitch);
    }

    const bool detectorAlive =
        lastDetectorMsgTime >= 0.0 && simTime - lastDetectorMsgTime <= DETECTOR_TIMEOUT_SEC;
    const double detectorSilentFor =
        simTime - std::max(lastDetectorMsgTime, autonomyStartTime);
    if (!detectorAlive && simTime - lastDetectorWarnTime >= 5.0)
    {
        lastDetectorWarnTime = simTime;
        std::cout << "[PERCEPTION] No messages on /stonefish/target/detection for "
                  << detectorSilentFor << " s. Start the detector, e.g. with\n"
                  << "             kyushu_project/start_pipeline.sh" << std::endl;
    }

    // Seen within the timeout: single missed frames do not drop the track.
    const bool targetFresh =
        lastDetectionTime >= 0.0 &&
        simTime - lastDetectionTime <= TARGET_TIMEOUT_SEC && haveTargetEstimate;

    const double frontGap = haveRange ? filteredRange - CAMERA_TO_FRONT : 1e9;
    const bool movingTarget = (activeCase == 6 || activeCase == ScenarioConfig::ROV_CASE);
    const double stopGap = movingTarget ? STOP_GAP_MOVING_TARGET : STOP_GAP;
    const bool tooCloseToAcquire = haveRange && frontGap < RELIABLE_GAP;
    const double goodTrackTime = (goodTrackSince >= 0.0) ? simTime - goodTrackSince : 0.0;

    const bool tracking =
        motionState == MotionState::YAW_ALIGN || motionState == MotionState::YAW_SETTLE ||
        motionState == MotionState::VERTICAL_ALIGN || motionState == MotionState::VERIFY_CENTER ||
        motionState == MotionState::APPROACH;
    const bool finishing =
        motionState == MotionState::TARGET_REACHED || motionState == MotionState::BACK_OFF ||
        motionState == MotionState::SURFACING || motionState == MotionState::SURFACED;
    const bool retreating = motionState == MotionState::RETREAT;

    // Alignment errors between robot and REMEMBERED target.
    const double hErr = WrapAngle(filteredTargetYaw - yaw);
    const double vErr = filteredTargetDepth - depth;

    // ---------------- Guidance: perception fault ----------------
    if (!finishing && depthUnknown && simTime - autonomyStartTime > VISION_FLOW_TIMEOUT_SEC)
        BeginSurfacing("No camera depth estimate (/vision/heave_flow silent): floating up", true);
    else if (!finishing && !retreating && !detectorAlive && detectorSilentFor >= PERCEPTION_ABORT_SEC)
        BeginSurfacing("Perception fault: detector silent too long", true);

    // ---------------- Guidance: target lost / reacquired ----------------
    else if (tracking && !targetFresh && !terminalLock)
    {
        lostSinceTime = simTime;
        lostGap = frontGap;
        EnterState(MotionState::TARGET_LOST_HOLD);   // stop using stale image data
    }
    else if (!tracking && !finishing && !retreating && targetFresh && !tooCloseToAcquire)
    {
        if (lostSinceTime >= 0.0)
            std::cout << "[METRIC] reacquisition time = " << (simTime - lostSinceTime) << " s" << std::endl;
        lostSinceTime = -1.0;
        acquisitionStartTime = simTime;
        initialErrorSign = SignOf(hErr);
        lastErrorSign = initialErrorSign;
        maxOvershoot = 0.0;
        correctiveTurns = 0;
        EnterState(MotionState::YAW_ALIGN);
    }
    stateTime += dt;

    // Overshoot / corrective-turn metrics during alignment.
    if (tracking && motionState != MotionState::APPROACH && acquisitionStartTime >= 0.0)
    {
        const double s = SignOf(hErr);
        if (s != 0.0 && lastErrorSign != 0.0 && s != lastErrorSign && std::fabs(hErr) > 0.01)
        {
            ++correctiveTurns;
            lastErrorSign = s;
        }
        if (s != 0.0 && s != initialErrorSign)
            maxOvershoot = std::max(maxOvershoot, std::fabs(hErr));
    }

    // ---------------- Guidance: state machine ----------------
    double speedTarget = 0.0;
    bool travelling = false;          // false -> hold position along the heading
    double holdSpeedLimit = HOLD_MAX_SPEED;
    switch (motionState)
    {
        case MotionState::YAW_ALIGN:
            yawRef = filteredTargetYaw;              // depthRef held (not updated)
            if (std::fabs(hErr) < H_ALIGN_TOL)
                EnterState(MotionState::YAW_SETTLE);
            break;

        case MotionState::YAW_SETTLE:
            yawRef = filteredTargetYaw;
            if (std::fabs(hErr) > LEAVE_FACTOR * H_ALIGN_TOL)
            {
                EnterState(MotionState::YAW_ALIGN);
                break;
            }
            settledTime = (std::fabs(yawRate) < YAW_SETTLED_RATE) ? settledTime + dt : 0.0;
            if (settledTime >= SETTLE_HOLD_SEC)
                EnterState(std::fabs(vErr) < V_ALIGN_TOL
                    ? MotionState::VERIFY_CENTER : MotionState::VERTICAL_ALIGN);
            break;

        case MotionState::VERTICAL_ALIGN:
            if (useVisionDepth && stateTime >= VERTICAL_ALIGN_TIMEOUT_SEC && !verticalFault)
            {
                verticalFault = true;
                BeginSurfacing("Vertical alignment does not converge: vertical control not"
                               " trusted, thrusters off, floating up on buoyancy", true);
                break;
            }
            yawRef = filteredTargetYaw;              // yaw HELD, not re-aligned
            depthRef = filteredTargetDepth;
            if (std::fabs(hErr) > LEAVE_FACTOR * H_ALIGN_TOL)
                EnterState(MotionState::YAW_ALIGN);
            else if (std::fabs(vErr) < V_ALIGN_TOL && std::fabs(heaveVel) < HEAVE_SETTLED_VEL)
                EnterState(MotionState::VERIFY_CENTER);
            break;

        case MotionState::VERIFY_CENTER:
            yawRef = filteredTargetYaw;
            depthRef = filteredTargetDepth;
            if (std::fabs(hErr) > LEAVE_FACTOR * H_ALIGN_TOL)
                EnterState(MotionState::YAW_ALIGN);
            else if (std::fabs(vErr) > LEAVE_FACTOR * V_ALIGN_TOL)
                EnterState(MotionState::VERTICAL_ALIGN);
            else if (stateTime >= VERIFY_HOLD_SEC && tooCloseToAcquire && !movingTarget)
            {
                std::cout << "[SAFETY] Aligned too close to the target to trust it:"
                          << " retreating to look again." << std::endl;
                EnterState(MotionState::RETREAT);
            }
            else if (stateTime >= VERIFY_HOLD_SEC)
            {
                lastAlignmentTime = simTime - acquisitionStartTime;
                std::cout << "[METRIC] alignment time = " << lastAlignmentTime << " s"
                          << ", yaw overshoot = " << maxOvershoot * 180.0 / M_PI << " deg"
                          << ", corrective turns = " << correctiveTurns << std::endl;
                EnterState(MotionState::APPROACH);
            }
            break;

        case MotionState::APPROACH:
        {
            // Without a DVL nothing measures the blind stretch: the approach
            // stays on the camera range to the end (the bulb is detected down
            // to ~0.7 m camera distance, i.e. past the stop point).
            if (useSensorNav && !movingTarget && !terminalLock && haveRange && frontGap < TERMINAL_GAP)
            {
                if (goodTrackTime < TERMINAL_MIN_TRACK_SEC)
                {
                    // Not a clean lock: wait stopped, never creep in on a poor track.
                    terminalWaitTime += dt;
                    if (terminalWaitTime >= TERMINAL_WAIT_SEC)
                    {
                        std::cout << "[SAFETY] Track too poor to start the terminal approach:"
                                  << " retreating to look again." << std::endl;
                        EnterState(MotionState::RETREAT);
                        break;
                    }
                    travelling = true;
                    break;
                }
                terminalLock = true;
                terminalStartTime = simTime;
                std::cout << "[MISSION] Terminal approach: heading and depth locked at gap ~"
                          << frontGap << " m, final " << frontGap - STOP_GAP
                          << " m on odometry." << std::endl;
            }
            if (terminalLock && useVisionDepth)
                depthRef = filteredTargetDepth;      // updated only while the box is whole
            if (!terminalLock)
            {
                yawRef = filteredTargetYaw;          // continuous small corrections
                depthRef = filteredTargetDepth;
                if (std::fabs(hErr) > APPROACH_LEAVE_FACTOR * H_ALIGN_TOL)
                {
                    EnterState(MotionState::YAW_ALIGN);
                    break;
                }
                if (std::fabs(vErr) > APPROACH_LEAVE_FACTOR * V_ALIGN_TOL)
                {
                    EnterState(MotionState::VERTICAL_ALIGN);
                    break;
                }
            }
            else if (simTime - terminalStartTime > TERMINAL_MAX_SEC)
            {
                std::cout << "[SAFETY] Terminal approach took too long: stopping here." << std::endl;
                EnterState(MotionState::TARGET_REACHED);
                break;
            }
            if (!haveRange)
                break;                               // never drive without a range estimate

            const double remaining = frontGap - stopGap;
            if (remaining <= REACH_TOL)
            {
                // Arrived: brake to zero, count as reached once nearly still.
                if (std::fabs(bodyVx) < REACHED_SPEED)
                {
                    std::cout << "[MISSION] Target reached: gap ~" << frontGap << " m"
                              << ", centre error e_x=" << targetErrorX << " px e_y=" << targetErrorY
                              << " px (t=" << simTime << " s)" << std::endl;
                    EnterState(MotionState::TARGET_REACHED);
                }
                travelling = true;
                break;
            }

            // Contact / obstruction guard: pushing but not moving. A stop the
            // controller asked for (speed reference ~0) is not a stall.
            if (speedRef < STALL_MIN_SPEED_REF)
                approachMoving = false;
            else if (std::fabs(bodyVx) > 0.05 * Sim::V)
                approachMoving = true;
            stallTime = (approachMoving && speedRef > STALL_MIN_SPEED_REF &&
                         bodyVx < STALL_SPEED_RATIO * speedRef) ? stallTime + dt : 0.0;
            if (stallTime >= STALL_SEC)
            {
                std::cout << "[SAFETY] No forward progress while approaching (contact or obstruction):"
                          << " stopping. Estimated gap was " << frontGap << " m." << std::endl;
                speedRef = 0.0;
                surgeIntegral = 0.0;
                EnterState(MotionState::TARGET_REACHED);
                break;
            }

            // Fast while far, then slow down so the robot can always stop in
            // time (planned braking), never below creep speed.
            double profile = std::min({useSensorNav ? APPROACH_CRUISE_SPEED : APPROACH_CRUISE_SPEED_NO_DVL,
                                       std::sqrt(2.0 * BRAKE_DECEL_PLAN * remaining),
                                       APPROACH_CREEP_SPEED + APPROACH_SPEED_GAIN * remaining});
            // Camera only the speed is known less well (real speed overshoots
            // the estimate by ~0.02 m/s): slower in the slow zone.
            const double slowSpeed = useSensorNav ? TERMINAL_MAX_SPEED : TERMINAL_MAX_SPEED_NO_DVL;
            if (terminalLock || frontGap < SLOW_ZONE_GAP)
                profile = std::min(profile, slowSpeed);
            else if (!movingTarget)
                // Arrive at the slow zone already at its speed (no speed cliff).
                profile = std::min(profile, std::sqrt(
                    slowSpeed * slowSpeed +
                    2.0 * BRAKE_DECEL_PLAN * std::max(0.0, frontGap - SLOW_ZONE_GAP)));
            profile = std::max(profile, APPROACH_CREEP_SPEED);
            const double hGate = std::clamp(
                1.0 - std::fabs(hErr) / (APPROACH_LEAVE_FACTOR * H_ALIGN_TOL), 0.0, 1.0);
            const double vGate = std::clamp(
                1.0 - std::fabs(vErr) / (APPROACH_LEAVE_FACTOR * V_ALIGN_TOL), 0.0, 1.0);
            const double rGate = std::clamp(1.0 - std::fabs(yawRate) / (0.15 * Sim::W), 0.0, 1.0);
            speedTarget = profile * hGate * vGate * rGate;
            travelling = true;
            break;
        }

        case MotionState::TARGET_LOST_HOLD:
            // Hold last heading, depth and position.
            if (stateTime >= LOST_HOLD_SEC)
            {
                if (haveRange && lostGap <= stopGap + LOST_REACHED_MARGIN)
                {
                    std::cout << "[MISSION] Target left the view at gap ~" << lostGap
                              << " m (too close to see): treating it as reached." << std::endl;
                    EnterState(MotionState::TARGET_REACHED);
                }
                else if (tooCloseToAcquire)
                {
                    std::cout << "[SAFETY] Target lost " << frontGap << " m away: too close to"
                              << " search here, retreating to look again." << std::endl;
                    EnterState(MotionState::RETREAT);
                }
                else if (detectorAlive)
                {
                    haveTargetEstimate = false;
                    haveRange = false;
                    searchTime = 0.0;
                    searchTurned = searchTurnedModel = 0.0;
                    prevSearchYaw = yaw;
                    EnterState(MotionState::SEARCH);
                }
                // else: wait here for the detector (perception-fault timer is running).
            }
            break;

        case MotionState::SEARCH:
            if (!detectorAlive)
            {
                yawRef = yaw;
                EnterState(MotionState::TARGET_LOST_HOLD);   // don't spin blind
                break;
            }
            // Bounded, smooth yaw sweep toward where the target was last seen.
            if (std::fabs(WrapAngle(yawRef - yaw)) < SEARCH_MAX_LEAD)
                yawRef = WrapAngle(yawRef + searchDirection * SEARCH_YAW_RATE * dt);
            searchTurned += std::fabs(WrapAngle(yaw - prevSearchYaw));
            searchTurnedModel += std::fabs(yawRateModel) * dt;
            prevSearchYaw = yaw;
            searchTime += dt;
            // Camera only: both the camera heading and the thrust model must
            // count the turn, so the TRUE turn is at least a full circle.
            if ((useSensorNav ? searchTurned : std::min(searchTurned, searchTurnedModel))
                    >= (useSensorNav ? SEARCH_MAX_TURN : SEARCH_MAX_TURN_NO_GYRO) ||
                searchTime >= SEARCH_MAX_SEC)
            {
                yawRef = yaw;
                std::cout << "[SEARCH] No target after " << searchTurned * 180.0 / M_PI
                          << " deg of search (thrust model: " << searchTurnedModel * 180.0 / M_PI
                          << " deg, " << searchTime << " s)." << std::endl;
                BeginSurfacing("Target absent", true);
            }
            break;

        case MotionState::RETREAT:
        {
            // Straight back along the heading (no turning near the target),
            // then look again from where detection is reliable.
            if (!retreatGoalSet && (holdValid || std::fabs(bodyVx) < HOLD_CAPTURE_SPEED))
            {
                const double dist = haveRange
                    ? std::clamp(RETREAT_GAP - frontGap, 0.3 * Sim::L, 2.0 * Sim::L) : 1.0;
                holdX = posX - dist * std::cos(yaw);
                holdY = posY - dist * std::sin(yaw);
                holdValid = true;
                retreatGoalSet = true;
                rangeGoalGap = haveRange ? frontGap + dist : -1.0;
            }
            holdSpeedLimit = BACKOFF_SPEED;
            double along = (holdX - posX) * std::cos(yaw) + (holdY - posY) * std::sin(yaw);
            if (!useSensorNav && rangeGoalGap > 0.0 && haveRange && targetFresh)
                along = frontGap - rangeGoalGap;   // no DVL: the camera range decides
            if ((retreatGoalSet && std::fabs(along) < 0.05 * Sim::L && std::fabs(bodyVx) < REACHED_SPEED) ||
                stateTime >= RETREAT_TIMEOUT_SEC)
            {
                lostGap = 1e9;
                lostSinceTime = simTime;
                EnterState(MotionState::TARGET_LOST_HOLD);
            }
            break;
        }

        case MotionState::TARGET_REACHED:
            // Stop and hold. Never creep closer (frontGap guard below).
            if (stateTime >= REACHED_HOLD_SEC && holdValid)
            {
                const double backoff = useSensorNav ? BACKOFF_DISTANCE : BACKOFF_DISTANCE_NO_DVL;
                holdX = posX - backoff * std::cos(yaw);
                holdY = posY - backoff * std::sin(yaw);
                EnterState(MotionState::BACK_OFF);
                holdValid = true;                    // hold point = BACK_OFF goal
                rangeGoalGap = haveRange ? frontGap + backoff : -1.0;
            }
            break;

        case MotionState::BACK_OFF:
        {
            holdSpeedLimit = BACKOFF_SPEED;
            double along = (holdX - posX) * std::cos(yaw) + (holdY - posY) * std::sin(yaw);
            if (!useSensorNav && rangeGoalGap > 0.0 && haveRange && targetFresh)
                along = frontGap - rangeGoalGap;   // no DVL: the camera range decides
            if ((std::fabs(along) < 0.05 * Sim::L && std::fabs(bodyVx) < REACHED_SPEED) ||
                stateTime >= BACKOFF_TIMEOUT_SEC)
                BeginSurfacing("Clear of the target", false);
            break;
        }

        case MotionState::SURFACING:
        {
            // Heading and position held; depth reference rises, slower near the surface.
            // The reference never leads the robot by more than ASCENT_MAX_LEAD,
            // so the heavy hull cannot build up speed and overshoot the surface.
            bool surfaced = false;
            if (useVisionDepth)
            {
                // The depth estimate drifts, so the surface is not a number here:
                // slow down early (estimate within SURFACE_SLOW_ZONE), keep rising,
                // and stop when the camera shows no more rise while pushed up.
                const double rate = (depth - SURFACE_DEPTH < SURFACE_SLOW_ZONE)
                                        ? ASCENT_RATE_FINAL : ASCENT_RATE;
                depthRef = std::max(depthRef, depth - ASCENT_MAX_LEAD) - rate * dt;
                // Camera only: the ascent is open loop (see the heave command
                // below), so "pushed up" = ascending for a while already.
                const bool pushedUp = useSensorNav ? depth - depthRef >= SURFACE_PUSH_LEAD
                                                   : stateTime >= SURFACE_MIN_ASCENT_SEC;
                // Measured (flow), not the observer: the model would "see" a
                // rise that the surface stops.
                // No gyro: judge on the rocking-averaged camera speed.
                // Camera only: "no longer RISING" (one-sided). At the surface the
                // flow reads a steady +0.015 m/s (sinking) at real size, which a
                // two-sided test never accepted; the surface can only stop a rise.
                // The thrust model's rise since the ascent began, integrated on
                // its own (camera corrections - e.g. the buoy's height, which kept
                // the estimate at 0.25 m while the robot sat at the surface in
                // case 4 - cannot hold it back). It keeps "rising" once the
                // surface stops the robot, so beyond depth-at-start + margin it
                // must be at the surface.
                surfaceModelVz += (heaveForceCmd / HEAVE_MASS_KG + HEAVE_BIAS_INIT - HEAVE_DRAG_LIN * surfaceModelVz
                                   - HEAVE_DRAG * surfaceModelVz * std::fabs(surfaceModelVz)) * dt;
                surfaceModelRise -= surfaceModelVz * dt;
                const bool risenPastSurface = surfaceStartDepth - surfaceModelRise < SURFACE_DEPTH - SURFACE_EST_MARGIN;
                const bool still = useSensorNav ? std::fabs(flowVz) < SURFACE_STILL_VZ
                                                : (flowVzSlow > -SURFACE_STILL_VZ_NO_GYRO || risenPastSurface ||
                                                   depth < SURFACE_DEPTH - SURFACE_EST_MARGIN);
                if (!surfacePassive && !depthUnknown)
                {
                    // 1) rise under control until no more rise while pushed up
                    surfaceStillTime = (pushedUp && still) ? surfaceStillTime + dt : 0.0;
                    if (surfaceStillTime >= SURFACE_STILL_SEC)
                    {
                        surfacePassive = true;
                        surfaceStillTime = 0.0;
                        std::cout << "[MISSION] No more rise: vertical thrusters off, checking"
                                  << " that buoyancy holds KYUBIC at the surface." << std::endl;
                    }
                }
                else if (surfacePassive)
                {
                    // 2) thrusters off: if not yet at the surface, buoyancy lifts it
                    //    (visible); still for SURFACE_SETTLE_SEC = floating at the surface.
                    surfaceStillTime = still ? surfaceStillTime + dt : 0.0;
                    surfaced = surfaceStillTime >= SURFACE_SETTLE_SEC;
                }
                // (No estimate at all: thrusters are off and buoyancy lifts it;
                // there is no way to tell when it arrives, so stay in SURFACING.)
                surfaced = surfaced && !depthUnknown && !verticalFault;
                // Camera only: floating at the surface is the one absolute depth
                // reference: reset the observer there (it had run away while the
                // surface blocked the rise the thrust model predicted).
                if (surfaced && useVisionDepth)
                {
                    kfX[0] = SURFACE_DEPTH;
                    kfX[1] = 0.0;
                    kfP[0][0] = 0.0001 * Sim::L * Sim::L;
                    kfP[0][1] = kfP[1][0] = kfP[0][2] = kfP[2][0] = 0.0;
                    visionDepth = SURFACE_DEPTH;
                    visionVz = 0.0;
                }
            }
            else
            {
                const double rate = std::clamp(ASCENT_SLOW_GAIN * (depth - SURFACE_DEPTH),
                                               ASCENT_RATE_FINAL, ASCENT_RATE);
                depthRef = std::max(SURFACE_DEPTH,
                                    std::max(depthRef, depth - ASCENT_MAX_LEAD) - rate * dt);
                surfaced = std::fabs(depth - SURFACE_DEPTH) <= SURFACED_TOL &&
                           std::fabs(heaveVel) < HEAVE_SETTLED_VEL;
            }
            if (surfaced)
            {
                std::cout << "[MISSION] Surfaced (t=" << simTime << " s)"
                          << (missionAborted ? " - mission ABORTED, target not reached." : " - mission complete.")
                          << std::endl;
                EnterState(MotionState::SURFACED);
            }
            break;
        }

        case MotionState::SURFACED:
            // With the camera estimate: vertical thrusters off, buoyancy holds
            // KYUBIC at the surface (no depth number needed). The observer is
            // pinned there (the hold-up push would otherwise read as a rise).
            if (useVisionDepth)
            {
                kfX[0] = SURFACE_DEPTH;
                kfX[1] = 0.0;
                visionDepth = SURFACE_DEPTH;
                visionVz = 0.0;
            }
            depthRef = useVisionDepth ? SURFACE_DEPTH : SURFACE_DEPTH;
            break;

        case MotionState::STOPPED:
        default:
            break;
    }

    // ---------------- Control: surge (position hold or travel) ----------------
    if (!travelling)
    {
        if (!holdValid && std::fabs(bodyVx) < HOLD_CAPTURE_SPEED)
        {
            holdX = posX;
            holdY = posY;
            holdValid = true;
            rangeHoldValid = haveRange;
            rangeHoldGap = frontGap;
        }
        if (holdValid)
        {
            double along = (holdX - posX) * std::cos(yaw) + (holdY - posY) * std::sin(yaw);
            // Without a DVL the dead-reckoned position drifts (it let the robot
            // creep into the target in a test): a stationary hold keeps the
            // camera-measured RANGE instead, and applies no forward thrust when
            // the target is not seen (nothing to hold against).
            const bool movingAway = motionState == MotionState::BACK_OFF || motionState == MotionState::RETREAT;
            if (!useSensorNav && movingAway)
            {
                // BACK_OFF / RETREAT: to the goal on the camera range; buoy not
                // seen: keep the dead-reckoned goal (moving away is the safe way).
                if (rangeGoalGap > 0.0 && haveRange && targetFresh)
                    along = frontGap - rangeGoalGap;
            }
            else if (!useSensorNav && finishing)
            {
                // Surfacing: partial views of the target while rising read "far",
                // so no range hold here; drift slowly AWAY from the target while
                // rising (only if a target was ever seen), and stop once surfaced.
                along = (lastBboxTime >= 0.0 && motionState != MotionState::SURFACED)
                            ? -SURFACING_DRIFT_SPEED / HOLD_KP : 0.0;
            }
            else if (!useSensorNav && holdSpeedLimit <= HOLD_MAX_SPEED)
                along = (rangeHoldValid && haveRange && targetFresh) ? frontGap - rangeHoldGap : 0.0;
            if (holdSpeedLimit > HOLD_MAX_SPEED)
            {
                // Travelling to a goal (BACK_OFF / RETREAT): fast, but slow
                // enough to stop on it.
                const double v = std::min({holdSpeedLimit,
                                           std::sqrt(2.0 * BRAKE_DECEL_PLAN * std::fabs(along)),
                                           TRAVEL_HOLD_KP * std::fabs(along)});
                speedTarget = std::copysign(v, along);
            }
            else
            {
                speedTarget = std::clamp(HOLD_KP * along, -holdSpeedLimit, holdSpeedLimit);
            }
        }
    }
    // Safety: stopping-distance guard. If the robot, at its actual speed and
    // a braking rate it can really achieve, would not stop before the stop
    // point, brake now - whatever the profile asked for. (At creep speed the
    // robot stops within ~1 cm, so the final creep is not blocked.)
    if (haveRange && !finishing && guardSpeed > 1.5 * APPROACH_CREEP_SPEED)
    {
        const double stoppingDistance = guardSpeed * guardSpeed / (2.0 * BRAKE_DECEL_GUARD);
        if (stoppingDistance + BRAKE_GUARD_MARGIN >= frontGap - stopGap)
        {
            // Routine at creep speed in the last centimetres; only a real
            // over-speed is reported.
            if (speedTarget > 0.0 && !brakeGuardActive && bodyVx > TERMINAL_MAX_SPEED)
                std::cout << "[SAFETY] Braking: at " << bodyVx << " m/s the robot needs "
                          << stoppingDistance << " m to stop, " << frontGap - stopGap
                          << " m left." << std::endl;
            brakeGuardActive = speedTarget > 0.0;
            speedTarget = std::min(speedTarget, 0.0);
        }
        else
        {
            brakeGuardActive = false;
        }
    }
    else
    {
        brakeGuardActive = false;
    }
    // Safety: never move toward a target that is already this close.
    if (haveRange && !finishing && frontGap < stopGap)
        speedTarget = std::min(speedTarget, 0.0);
    if (haveRange && motionState != MotionState::SURFACING &&
        motionState != MotionState::SURFACED && frontGap < MIN_SAFE_GAP)
        speedTarget = std::min(speedTarget, -APPROACH_CREEP_SPEED);
    // Camera only near the target the speed estimate cannot be trusted alone:
    // at real size it drifted to -0.02 m/s while the robot crept on at
    // +0.015 m/s; the hold then pushed the robot onto the buoy, and once it
    // was blocked the estimate "reversed" at the back-off speed while the
    // robot pressed forward through the whole back-off and ascent.
    // After reaching it (and retreating) the speed loop runs on the
    // thrust-model speed, which measurements cannot bias, so backing off
    // really reverses, and there is no forward thrust at all.
    const bool afterReach = finishing || retreating;
    const double speedFeedback = (!useSensorNav && afterReach) ? surgeVelModel : bodyVx;
    surgeCommand = SurgeSpeedControl(speedTarget, speedFeedback, dt);
    if (!useSensorNav && afterReach)
        surgeCommand = std::min(surgeCommand, 0.0);
    // Thrust that would push the thrust-model speed past a limit is faded
    // out: the overall limit, and inside the slow zone 1.3 x the slow-zone
    // speed (the estimate under-read the final approach by ~35 %).
    if (!useSensorNav && surgeCommand * surgeVelModel > 0.0)
    {
        double limit = MODEL_SPEED_LIMIT, start = MODEL_SPEED_FADE_START;
        if (surgeVelModel > 0.0 && haveRange && frontGap < SLOW_ZONE_GAP)
        {
            limit = std::min(limit, 1.3 * TERMINAL_MAX_SPEED_NO_DVL);
            start = std::min(start, TERMINAL_MAX_SPEED_NO_DVL);
        }
        surgeCommand *= std::clamp((limit - std::fabs(surgeVelModel)) / (limit - start), 0.0, 1.0);
    }

    // ---------------- Control: yaw PD(+I) ----------------
    const double yawError = WrapAngle(yawRef - yaw);
    if (YAW_KI > 0.0)
        yawIntegral = std::clamp(yawIntegral + yawError * dt, -0.1 / YAW_KI, 0.1 / YAW_KI);
    // Camera only: turns kept slow enough (<~0.2 rad/s) for the camera to
    // measure them (it reads ~0.08 rad/s at a true 0.47 rad/s spin).
    const double yawMax = useSensorNav ? YAW_MAX : YAW_MAX_NO_GYRO;
    const double yawCommand = std::clamp(
        YAW_KP * yawError - YAW_KD * yawRate + YAW_KI * yawIntegral, -yawMax, yawMax);

    // ---------------- Control: depth PID ----------------
    // The reference moves toward depthRef at most VERTICAL_REF_RATE (SURFACING
    // has its own ramp).
    if (motionState == MotionState::SURFACING || motionState == MotionState::SURFACED)
        depthRefSlewed = depthRef;
    else
        depthRefSlewed += std::clamp(depthRef - depthRefSlewed,
                                     -VERTICAL_REF_RATE * dt, VERTICAL_REF_RATE * dt);
    const double depthError = depthRefSlewed - depth;
    const double heaveUnsat =
        DEPTH_KP * depthError - DEPTH_KD * heaveVel + DEPTH_KI * depthIntegral;
    if (DEPTH_KI > 0.0 && (std::fabs(heaveUnsat) < HEAVE_MAX || heaveUnsat * depthError < 0.0))
        depthIntegral = std::clamp(depthIntegral + depthError * dt,
                                   -DEPTH_I_LIMIT / DEPTH_KI, DEPTH_I_LIMIT / DEPTH_KI);
    // Surfacing: no downward trim from the integrator, so "pushed up and not
    // rising" really means the surface.
    if (motionState == MotionState::SURFACING)
        depthIntegral = std::min(depthIntegral, 0.0);
    const bool settleCheck = useVisionDepth && motionState == MotionState::SURFACING && surfacePassive;
    const bool holdAtSurface = useVisionDepth &&
        (motionState == MotionState::SURFACED || (depthUnknown && motionState == MotionState::SURFACING));
    const bool openLoopHeave = depthUnknown || settleCheck || holdAtSurface;
    if (openLoopHeave)
        depthIntegral = 0.0;
    double heaveCommand = std::clamp(heaveUnsat, -HEAVE_MAX, HEAVE_MAX);
    // Trim: the vertical thrust that holds the robot level, learned while
    // hovering. Camera only, the ascent runs open loop at trim - margin: near
    // the surface the depth estimate is unreliable (floaters leave the water,
    // flow sees the rocking) and a depth loop there oscillated +-0.25 m.
    const bool hovering = motionState == MotionState::TARGET_REACHED || motionState == MotionState::BACK_OFF ||
                          motionState == MotionState::VERIFY_CENTER || motionState == MotionState::TARGET_LOST_HOLD;
    if (hovering && !openLoopHeave)
        heaveTrim += std::min(1.0, dt / HEAVE_TRIM_TAU) * (heaveCommand - heaveTrim);
    if (!useSensorNav && useVisionDepth && motionState == MotionState::SURFACING && !surfacePassive && !depthUnknown)
    {
        depthIntegral = 0.0;
        heaveCommand = std::clamp(heaveTrim - SURFACE_ASCENT_THRUST, -HEAVE_MAX, HEAVE_MAX);
    }
    if (verticalFault)
        heaveCommand = 0.0;                  // never trust the vertical sign again
    else if (holdAtSurface)
        heaveCommand = -SURFACE_HOLD_PUSH;   // up (heave > 0 is down)
    else if (openLoopHeave)
        heaveCommand = 0.0;                  // thrusters off: buoyancy only

    // ---------------- Control: roll hold (middle thrusters) ----------------
    const double rollCommand = std::clamp(-ROLL_KP * roll - ROLL_KD * rollRate, -ROLL_MAX, ROLL_MAX);

    // Thrust allocation over the six T200s (sway is not used by the mission).
    // Camera only, surfaced: float. Nothing measures position or heading
    // there, so holding them only crept (2 cm/s backwards and ~1 deg of turn
    // in a test); the horizontal thrusters stay off, the hold-up push stays.
    const bool floating = !useSensorNav && motionState == MotionState::SURFACED;
    SetAxisCommands(floating ? 0.0 : surgeCommand, 0.0, heaveCommand, rollCommand,
                    floating ? 0.0 : yawCommand);   // (roll: not measured, already 0)

    // ---------------- Telemetry ----------------
    // [0] state [1] simTime [2] x [3] y [4] depth [5] yaw [6] yawRef [7] yawRate
    // [8] depthRef [9] heaveVel [10] yawCmd [11] heaveCmd [12] surgeCmd
    // [13] fresh [14] confidence [15] e_x [16] e_y [17] hErr [18] vErr
    // [19] camHorizontalIsX [20] camHSign [21] camVSign [22] camHVerified [23] camVVerified
    // [24] surgeSign [25] yawSign [26] heaveSign [27] maxOvershoot [28] lastAlignmentTime
    // [29] range [30] frontGap [31] bboxWidth [32] speedRef [33] bodyVx
    // [34] missionAborted [35] detectorAlive [36] haveRange
    // [37..39] TRUE target position (simulation ground truth, for evaluation only; NaN = none)
    // [40] bboxHeight [41] terminalLock
    // System-log timing, wall clock [epoch s]: [42] control step start
    // [43] thruster setpoints applied [44] Stonefish detection received
    // [45] its image stamp [46] real-camera /object/position received [47] its image stamp
    // [48..53] thrust of each T200 [N] (kyubic::THRUSTERS order, actual rotor state)
    // [54..56] total thrust Fx (surge), Fy (sway), Fz (heave) [N] in the body frame
    // [57] yaw torque [N m] about the centre of gravity
    // [58] depth the controller uses [m] (camera estimate, or truth with
    //      TASK2_DEPTH_SOURCE=sensor) [59] its vertical speed [m/s]
    // [60] tracked image features [61] camera depth estimate running [62] camera depth in use
    // NOTE: [4] depth and [9] heaveVel are simulator TRUTH (evaluation only).
    if (controlDebugPublisher)
    {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        sf::Vector3 truth(nan, nan, nan);
        if (trackedObject != nullptr)
            truth = trackedObject->getTransform().getOrigin();
        std_msgs::msg::Float64MultiArray msg;
        msg.data = {
            static_cast<double>(static_cast<int>(motionState)), simTime,
            posXTruth, posYTruth, depthTruth, yawTruth, yawRef, yawRateTruth,
            depthRef, heaveVelTruth, yawCommand, heaveCommand, surgeCommand,
            targetFresh ? 1.0 : 0.0, targetConfidence, targetErrorX, targetErrorY,
            hErr, vErr,
            camHorizontalIsX ? 1.0 : 0.0, camHSign, camVSign,
            camHVerified ? 1.0 : 0.0, camVVerified ? 1.0 : 0.0,
            surgeSign, yawSign, heaveSign, maxOvershoot, lastAlignmentTime,
            haveRange ? filteredRange : -1.0, haveRange ? frontGap : -1.0, bboxWidth,
            speedRef, bodyVxTruth,
            missionAborted ? 1.0 : 0.0, detectorAlive ? 1.0 : 0.0, haveRange ? 1.0 : 0.0,
            truth.getX(), truth.getY(), truth.getZ(), bboxHeight, terminalLock ? 1.0 : 0.0,
            controlWallTime, thrusterWallTime, detectionRxWallTime, detectionImageStamp,
            objectRxWallTime, objectImageStamp};
        AppendThrustTelemetry(msg.data, depth, heaveVel);
        controlDebugPublisher->publish(msg);
    }
}

/* =========================================================
 * THRUSTER OUTPUT
 * ========================================================= */
void Task2Manager::SetAxisCommands(double surge, double sway, double heave, double roll, double yaw)
{
    const double cmd[5] = {surge * surgeSign, sway, heave * heaveSign, roll, yaw * yawSign};
    ApplyAxisCommands(cmd);
}

/* =========================================================
 * THRUST ALLOCATION
 *
 * Column i of B = what thruster i produces per unit thrust:
 *   [Fx, Fy, Fz, roll moment, yaw moment] about the centre of gravity.
 * Commands map to thrusts through the pseudo-inverse B+ = B^T (B B^T)^-1.
 * Each axis is scaled so a command of 1 alone just saturates one thruster.
 * ========================================================= */
void Task2Manager::BuildAllocation()
{
    constexpr int A = 5, T = static_cast<int>(kyubic::THRUSTER_COUNT);
    double B[A][T];
    for (int i = 0; i < T; ++i)
    {
        const auto& m = kyubic::THRUSTERS[i];
        const double r[3] = {m.pos[0] - kyubic::CENTRE_OF_GRAVITY[0],
                             m.pos[1] - kyubic::CENTRE_OF_GRAVITY[1],
                             m.pos[2] - kyubic::CENTRE_OF_GRAVITY[2]};
        B[SURGE][i] = m.dir[0];
        B[SWAY][i] = m.dir[1];
        B[HEAVE][i] = m.dir[2];
        B[ROLL][i] = r[1] * m.dir[2] - r[2] * m.dir[1];     // (r x d).x
        B[YAW][i] = r[0] * m.dir[1] - r[1] * m.dir[0];      // (r x d).z
    }

    // M = B B^T, then invert with Gauss-Jordan
    double M[A][2 * A] = {};
    for (int a = 0; a < A; ++a)
    {
        for (int b = 0; b < A; ++b)
            for (int i = 0; i < T; ++i)
                M[a][b] += B[a][i] * B[b][i];
        M[a][A + a] = 1.0;
    }
    for (int c = 0; c < A; ++c)
    {
        int pivot = c;
        for (int r = c + 1; r < A; ++r)
            if (std::fabs(M[r][c]) > std::fabs(M[pivot][c]))
                pivot = r;
        for (int k = 0; k < 2 * A; ++k)
            std::swap(M[c][k], M[pivot][k]);
        const double d = M[c][c];
        for (int k = 0; k < 2 * A; ++k)
            M[c][k] /= d;
        for (int r = 0; r < A; ++r)
            if (r != c)
            {
                const double f = M[r][c];
                for (int k = 0; k < 2 * A; ++k)
                    M[r][k] -= f * M[c][k];
            }
    }
    for (int i = 0; i < T; ++i)
        for (int a = 0; a < A; ++a)
        {
            allocation[i][a] = 0.0;
            for (int b = 0; b < A; ++b)
                allocation[i][a] += B[b][i] * M[b][A + a];
        }

    for (int a = 0; a < A; ++a)
    {
        double peak = 0.0;
        for (int i = 0; i < T; ++i)
            peak = std::max(peak, std::fabs(allocation[i][a]));
        axisScale[a] = (peak > 1e-9) ? 1.0 / peak : 0.0;
    }
    {
        double fz = 0.0;   // vertical force for a heave command of 1 (through the allocation)
        for (int i = 0; i < T; ++i)
            fz += allocation[i][HEAVE] * axisScale[HEAVE] * THRUST_LIMIT * 51.5 * THRUST_SCALE * kyubic::THRUSTERS[i].dir[2];
        heaveForcePerCmd = fz;
    }

    static const char* axisNames[5] = {"surge", "sway", "heave", "roll", "yaw"};
    std::cout << "[KYUBIC] full-command authority (before the "
              << THRUST_LIMIT * 100 << " % thrust limit, in thruster units):";
    for (int a = 0; a < A; ++a)
        std::cout << " " << axisNames[a] << "=" << axisScale[a];
    std::cout << std::endl;
}

void Task2Manager::ApplyAxisCommands(const double cmd[5])
{
    constexpr int T = static_cast<int>(kyubic::THRUSTER_COUNT);
    double u[T];
    double peak = 0.0;
    for (int i = 0; i < T; ++i)
    {
        u[i] = 0.0;
        for (int a = 0; a < 5; ++a)
            u[i] += allocation[i][a] * axisScale[a] * std::clamp(cmd[a], -1.0, 1.0);
        peak = std::max(peak, std::fabs(u[i]));
    }
    // Saturation keeps the direction of the wrench: scale every thruster together.
    const double scale = (peak > 1.0) ? 1.0 / peak : 1.0;

    const double dt = std::clamp<double>(lastStepDt, 0.0, 0.05);
    const double maxDelta = THRUST_SLEW_RATE * dt;
    for (int i = 0; i < T; ++i)
    {
        const double desired = u[i] * scale * THRUST_LIMIT;
        appliedThrust[i] += std::clamp(desired - appliedThrust[i], -maxDelta, maxDelta);
        if (thrusters[i] != nullptr)
            thrusters[i]->setSetpoint(ThrustFractionToSetpoint(appliedThrust[i]));
    }
    thrusterWallTime = WallNow();
}

/* =========================================================
 * PUBLISH ODOMETRY
 * ========================================================= */
void Task2Manager::PublishOdometry()
{
    if (!odometry || !odomPublisher)
        return;

    nav_msgs::msg::Odometry msg;
    msg.header.stamp = rosNode->get_clock()->now();
    msg.header.frame_id = "world";
    msg.child_frame_id = "auv";

    msg.pose.pose.position.x = odometry->getLastValue(0);
    msg.pose.pose.position.y = odometry->getLastValue(1);
    msg.pose.pose.position.z = odometry->getLastValue(2);
    msg.pose.pose.orientation.x = odometry->getLastValue(6);
    msg.pose.pose.orientation.y = odometry->getLastValue(7);
    msg.pose.pose.orientation.z = odometry->getLastValue(8);
    msg.pose.pose.orientation.w = odometry->getLastValue(9);

    // Body frame (as reported by Stonefish)
    msg.twist.twist.linear.x = odometry->getLastValue(3);
    msg.twist.twist.linear.y = odometry->getLastValue(4);
    msg.twist.twist.linear.z = odometry->getLastValue(5);
    msg.twist.twist.angular.x = odometry->getLastValue(10);
    msg.twist.twist.angular.y = odometry->getLastValue(11);
    msg.twist.twist.angular.z = odometry->getLastValue(12);

    odomPublisher->publish(msg);
}

/* =========================================================
 * PUBLISH CAMERA IMAGE
 * ========================================================= */
void Task2Manager::PublishCameraImage(sf::ColorCamera* camera)
{
    if (!camera || !imagePublisher)
        return;

    void* dataPointer = camera->getImageDataPointer();
    if (dataPointer == nullptr)
        return;

    const unsigned int width = CAMERA_WIDTH_PX;
    const unsigned int height = CAMERA_HEIGHT_PX;
    const unsigned int channels = 3;

    sensor_msgs::msg::Image msg;
    msg.header.stamp = rosNode->get_clock()->now();
    msg.header.frame_id = "front_camera";
    msg.height = height;
    msg.width = width;
    msg.encoding = "rgb8";
    msg.is_bigendian = false;
    msg.step = width * channels;

    const size_t dataSize = static_cast<size_t>(width) * height * channels;
    msg.data.resize(dataSize);
    std::memcpy(msg.data.data(), dataPointer, dataSize);

    imagePublisher->publish(msg);
    if (datasetMode)
        PublishDatasetLabel(msg.header.stamp);
}

/* =========================================================
 * 3D VIEW: from behind the robot, looking forward
 * ========================================================= */
// Stonefish's default view looks at the scene from the SIDE (from 1 m above
// the water): the robot's left/right then points into the screen and shows
// as up/down, and its forward/back as left/right. Behind the robot, screen
// left/right and up/down match the camera. The view follows the robot (the
// mouse still rotates and zooms it).
void Task2Manager::SetupChaseView()
{
    sf::OpenGLTrackball* view = getTrackball();
    if (view == nullptr || robotHull == nullptr)
        return;
    const glm::vec3 look = glm::normalize(glm::vec3(1.f, 0.f, 0.15f));       // forward, slightly down (NED)
    const glm::vec3 worldUp(0.f, 0.f, -1.f);
    const glm::vec3 up = glm::normalize(worldUp - glm::dot(worldUp, look) * look);
    // Trackball convention: local -y = looking direction, local +z = up.
    const glm::mat3 localToWorld(glm::cross(-look, up), -look, up);
    const glm::quat wanted = glm::quat_cast(glm::transpose(localToWorld));
    const glm::quat base = glm::rotation(worldUp, glm::vec3(0.f, 0.f, 1.f));   // Rotate() pre-multiplies this
    view->Rotate(glm::inverse(base) * wanted);
    view->GlueToMoving(robotHull);
    // Orbit radius 5 m -> 1.6 m x scale: the robot and the buoy ahead in view.
    const float radius = static_cast<float>(1.6 * kyubic::SCALE);
    view->MouseScroll(15.f * (radius / 5.f - 1.f));
}

/* =========================================================
 * DETECTOR DATASET (TASK2_DATASET=1, ROV case)
 * ========================================================= */
// Bounding box of the bulb in this camera frame: every vertex of the bulb mesh
// projected with the true camera pose. [stamp, usable, u0, v0, u1, v1];
// usable = 0: the buoy just moved (the frame may show the old place).
void Task2Manager::PublishDatasetLabel(const builtin_interfaces::msg::Time& stamp)
{
    if (!datasetLabelPublisher || !trackedObject || !odometry)
        return;
    const double qx = odometry->getLastValue(6), qy = odometry->getLastValue(7);
    const double qz = odometry->getLastValue(8), qw = odometry->getLastValue(9);
    const double R[3][3] = {
        {1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)},
        {2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)},
        {2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)}};
    const double pos[3] = {odometry->getLastValue(0), odometry->getLastValue(1), odometry->getLastValue(2)};
    double cam[3];
    for (int r = 0; r < 3; ++r)
        cam[r] = pos[r] + R[r][0] * CAMERA_MOUNT[0] + R[r][1] * CAMERA_MOUNT[1] + R[r][2] * CAMERA_MOUNT[2];

    const sf::Transform pose = trackedObject->getTransform();
    const double f = FocalLengthPx();
    double u0 = 1e9, v0 = 1e9, u1 = -1e9, v1 = -1e9;
    bool behind = false;
    for (const auto& vtx : bulbVertices)
    {
        const sf::Vector3 w = pose * sf::Vector3(vtx[0], vtx[1], vtx[2]);
        const double d[3] = {w.getX() - cam[0], w.getY() - cam[1], w.getZ() - cam[2]};
        // world -> body (R transposed): x forward, y = image right, z = image down
        const double bx = R[0][0] * d[0] + R[1][0] * d[1] + R[2][0] * d[2];
        const double by = R[0][1] * d[0] + R[1][1] * d[1] + R[2][1] * d[2];
        const double bz = R[0][2] * d[0] + R[1][2] * d[1] + R[2][2] * d[2];
        if (bx < 0.02)
        {
            behind = true;
            break;
        }
        const double u = CAMERA_WIDTH * 0.5 + f * by / bx;
        const double v = CAMERA_HEIGHT * 0.5 + f * bz / bx;
        u0 = std::min(u0, u); u1 = std::max(u1, u);
        v0 = std::min(v0, v); v1 = std::max(v1, v);
    }
    // Usable: the buoy has stood still for 3 frames, and the robot is sane
    // (level within 25 deg, near the start), so a physics glitch never
    // poisons the dataset.
    const bool sane = R[2][2] > std::cos(25.0 * M_PI / 180.0) && std::hypot(pos[0], pos[1]) < 20.0 * Sim::L;
    const bool settled = sane && simTime - datasetMoveTime >= 3.0 / CAMERA_RATE_HZ;
    std_msgs::msg::Float64MultiArray label;
    label.data = {stamp.sec + stamp.nanosec * 1e-9, settled ? 1.0 : 0.0,
                  behind ? -1.0 : u0, behind ? -1.0 : v0, behind ? -1.0 : u1, behind ? -1.0 : v1};
    datasetLabelPublisher->publish(label);
}

// Every DATASET_PERIOD the buoy jumps to a random place relative to the
// camera (0.5-13 x L from it, inside or just outside the view, any rotation);
// 10 % of the time out of view (frames with no buoy).
void Task2Manager::UpdateDatasetTarget()
{
    constexpr double DATASET_PERIOD = 0.3;   // s
    if (simTime - datasetMoveTime < DATASET_PERIOD)
        return;
    static std::mt19937 rng(12345 + static_cast<unsigned>(std::time(nullptr)));
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    const double qx = odometry->getLastValue(6), qy = odometry->getLastValue(7);
    const double qz = odometry->getLastValue(8), qw = odometry->getLastValue(9);
    const double R[3][3] = {
        {1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)},
        {2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)},
        {2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)}};
    const double pos[3] = {odometry->getLastValue(0), odometry->getLastValue(1), odometry->getLastValue(2)};
    double cam[3];
    for (int r = 0; r < 3; ++r)
        cam[r] = pos[r] + R[r][0] * CAMERA_MOUNT[0] + R[r][1] * CAMERA_MOUNT[1] + R[r][2] * CAMERA_MOUNT[2];

    double dist, az, el;
    if (uni(rng) < 0.10)
    {
        dist = 3.0 * Sim::L; az = M_PI; el = 0.0;            // behind the camera
    }
    else
    {
        dist = 0.5 * Sim::L * std::exp(uni(rng) * std::log(26.0));   // log-uniform 0.5-13 x L
        az = (uni(rng) * 2.0 - 1.0) * 50.0 * M_PI / 180.0;
        el = (uni(rng) * 2.0 - 1.0) * 30.0 * M_PI / 180.0;
    }
    const double b[3] = {std::cos(el) * std::cos(az), std::cos(el) * std::sin(az), std::sin(el)};
    double bulb[3];
    for (int r = 0; r < 3; ++r)
        bulb[r] = cam[r] + dist * (R[r][0] * b[0] + R[r][1] * b[1] + R[r][2] * b[2]);
    const double z = std::clamp(bulb[2] + Sim::BULB_ABOVE_ORIGIN, Sim::BUOY_TOP + 0.01 * Sim::L,
                                Sim::SEABED_DEPTH - Sim::BUOY_BOTTOM);
    const double spin = uni(rng) * 2.0 * M_PI;
    MoveTarget(sf::Transform(sf::Quaternion(sf::Vector3(0.0, 0.0, 1.0), spin) * sf::Quaternion(0.0, M_PI, 0.0),
                             sf::Vector3(bulb[0], bulb[1], z)));
    datasetMoveTime = simTime;
}

/* =========================================================
 * PRINT ODOMETRY
 * ========================================================= */
void Task2Manager::PrintOdometry()
{
    if (!odometry)
        return;

    const double x = odometry->getLastValue(0);
    const double y = odometry->getLastValue(1);
    const double z = odometry->getLastValue(2);
    const double qx = odometry->getLastValue(6);
    const double qy = odometry->getLastValue(7);
    const double qz = odometry->getLastValue(8);
    const double qw = odometry->getLastValue(9);

    const double yaw = std::atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
    const double roll = std::atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy));
    const double pitch = std::asin(std::clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0));

    std::cout << "[AUV ODOM] X=" << x << " Y=" << y << " Depth=" << z
              << "  Roll=" << roll * 180.0 / M_PI
              << " Pitch=" << pitch * 180.0 / M_PI
              << " Yaw=" << yaw * 180.0 / M_PI << " deg"
              << "  State=" << StateName(static_cast<int>(motionState)) << std::endl;
}

/* =========================================================
 * SIMULATION STEP
 * ========================================================= */
void Task2Manager::SimulationStepCompleted(sf::Scalar timeStep)
{
    lastStepDt = timeStep;
    simTime += timeStep;
    if (!chaseViewSet)
    {
        chaseViewSet = true;
        SetupChaseView();
    }

    if (rclcpp::ok())
        rclcpp::spin_some(rosNode);

    UpdateController();
    UpdateTrackedObject();

    odometryTimer += timeStep;
    if (odometryTimer >= 0.5)
    {
        odometryTimer = 0.0;
        PrintOdometry();
        PublishOdometry();
    }

    CheckAutotestExit();
}

/* =========================================================
 * AUTOTEST EXIT ($TASK2_AUTOTEST_MAX_TIME = sim seconds)
 * Closes the app once the mission is finished or the time is up.
 * ========================================================= */
void Task2Manager::CheckAutotestExit()
{
    if (autotestMaxTime <= 0.0 || autotestQuitSent)
        return;

    // With the camera depth estimate, also watch the hull settle at the surface.
    const bool done = motionState == MotionState::SURFACED &&
                      stateTime >= (useVisionDepth ? 10.0 : 5.0) * Sim::T;
    if (!done && simTime < autotestMaxTime)
        return;

    std::cout << "[AUTOTEST] " << (done ? "mission finished" : "time limit reached")
              << ", exiting (t=" << simTime << " s)" << std::endl;
    SDL_Event quitEvent;
    quitEvent.type = SDL_QUIT;
    SDL_PushEvent(&quitEvent);
    autotestQuitSent = true;
}

/* =========================================================
 * REAL-CAMERA OBJECT CALLBACK (moves the sim target in case 6)
 * ========================================================= */
void Task2Manager::ObjectPositionCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg)
{
    if (!msg)
        return;

    constexpr double MIN_OBJECT_CONFIDENCE = 0.40;
    constexpr double SMOOTHING_ALPHA = 0.20;

    objectRxWallTime = WallNow();
    objectImageStamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

    const double rawX = std::clamp(msg->point.x, 0.0, REAL_CAMERA_WIDTH);
    const double rawY = std::clamp(msg->point.y, 0.0, REAL_CAMERA_HEIGHT);
    objectConfidence = msg->point.z;
    if (objectConfidence < MIN_OBJECT_CONFIDENCE)
        return;

    if (!g_haveSmoothedCentroid)
    {
        g_smoothedPixelX = rawX;
        g_smoothedPixelY = rawY;
        g_haveSmoothedCentroid = true;
    }
    else
    {
        g_smoothedPixelX = (1.0 - SMOOTHING_ALPHA) * g_smoothedPixelX + SMOOTHING_ALPHA * rawX;
        g_smoothedPixelY = (1.0 - SMOOTHING_ALPHA) * g_smoothedPixelY + SMOOTHING_ALPHA * rawY;
    }

    objectPixelX = g_smoothedPixelX;
    objectPixelY = g_smoothedPixelY;
    objectDetected = true;
}

/* =========================================================
 * STONEFISH-CAMERA DETECTION CALLBACK
 * Expected message: x = e_x = u_t - 320, y = e_y = v_t - 240, z = confidence
 * ========================================================= */
void Task2Manager::TargetDetectionCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg)
{
    if (!msg)
        return;

    // Callbacks run inside spin_some() in SimulationStepCompleted, so simTime
    // is the current simulation time (timeouts stay correct at any sim speed).
    lastDetectorMsgTime = simTime;
    detectionRxWallTime = WallNow();
    detectionImageStamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
    targetConfidence = msg->point.z;
    targetDetected = (targetConfidence >= MIN_CONFIDENCE);

    if (targetDetected)
    {
        targetErrorX = msg->point.x;
        targetErrorY = msg->point.y;
        lastDetectionTime = simTime;
        newDetectionPending = true;
    }

    // Uninterrupted run of confident detections (gates the terminal approach).
    if (targetConfidence >= GOOD_TRACK_CONFIDENCE)
    {
        if (goodTrackSince < 0.0)
            goodTrackSince = simTime;
    }
    else
    {
        goodTrackSince = -1.0;
    }
}

/* =========================================================
 * OBSERVER: depth measurement from the target's height in the image.
 * Frame anchored to the target: depth = anchor - height offset.
 * targetDz = range * tan(elevation), + = target below the camera.
 * ========================================================= */
bool Task2Manager::ObserveTargetHeight(double targetDz, double lever)
{
    bool accepted = false;
    if (!targetAnchorValid)
    {
        targetAnchor = kfX[0] + targetDz;
        targetAnchorValid = true;
        anchorRejects = 0;
        accepted = true;
    }
    else
    {
        // + the unmeasured pitch (rocking) without a gyro
        const double sigma = KF_TARGET_NOISE_M + KF_TARGET_NOISE_PER_M * lever +
                             (useSensorNav ? 0.0 : PITCH_UNCERTAINTY_RAD * lever);
        accepted = ObserverUpdate(0, targetAnchor - targetDz, sigma * sigma);
        if (!accepted && ++anchorRejects >= KF_REANCHOR_REJECTS)
        {
            targetAnchor = kfX[0] + targetDz;   // persistent disagreement: re-anchor
            anchorRejects = 0;
        }
        else if (accepted)
        {
            anchorRejects = 0;
        }
    }
    if (accepted)
        lastTargetUpdateTime = simTime;
    return accepted;
}

/* =========================================================
 * DEPTH OBSERVER STEP: model prediction + camera flow (both modes)
 * ========================================================= */
void Task2Manager::UpdateDepthObserver(double dt, double pitchRate, double bodyVx,
                                       double depthTruth, double heaveVelTruth)
{
    ObserverPredict(dt, bodyVx);
    if (targetAnchorValid && simTime - lastTargetUpdateTime > TARGET_ANCHOR_TIMEOUT_SEC)
        targetAnchorValid = false;   // target not measured for a while: re-anchor when seen again
    if (visionFlowNew)
    {
        visionFlowNew = false;
        const double f = FocalLengthPx();
        const double vzMeas = -(visionFlowB - VISION_PITCH_COEFF * f * pitchRate)
                              * VISION_FLOW_DISTANCE_M / f;
        flowVzRaw = vzMeas;
        flowVz += std::min(1.0, 1.0 / (CAMERA_RATE_HZ * VISION_VZ_FILTER_SEC)) * (vzMeas - flowVz);
        // Averaged over the rocking period (camera only: pitch is not measured).
        flowVzSlow += std::min(1.0, 1.0 / (CAMERA_RATE_HZ * TARGET_DEPTH_TAU_NO_GYRO)) * (vzMeas - flowVzSlow);
        // Only when the target is not measuring the depth: its scale is uncertain.
        // Real-size ROV: not at all. With no absolute depth reference its
        // offsets (+0.004-0.009 m/s moving, under-reading slow motion) reached
        // the depth estimate and made the hold climb 0.1-0.15 m; the identified
        // thrust model alone stayed within ~3 cm over 20 s (the pilot sees the rest).
        if ((lastTargetUpdateTime < 0.0 || simTime - lastTargetUpdateTime > 0.5 * Sim::T) &&
            !(rovMode && Sim::REAL_SIZE))
        {
            // Forward motion spreads the scene out from the image centre and
            // biases the vertical shift: trust the flow less while moving.
            // Without a gyro the rocking (~3 deg at 3.6 s) cannot be removed
            // from the vertical shift: it looks like +-0.13 m/s of heave.
            const double sigma = FLOW_NOISE_SCALE * (
                KF_FLOW_NOISE + 0.5 * std::fabs(vzMeas) + KF_FLOW_NOISE_PER_SPEED * std::fabs(bodyVx) +
                (useSensorNav ? 0.0 : FLOW_PITCH_NOISE_NO_GYRO
                                      + FLOW_TURN_NOISE_PER_RATE * std::fabs(navYawRate)
                                      + FLOW_SWAY_NOISE_PER_SPEED * std::fabs(swayVelModel)));
            // Never gated: without the target it is the only correction, and
            // rejecting it once the model has drifted lets the model run away.
            ObserverUpdate(1, vzMeas, sigma * sigma, false);
        }
    }
    const bool flowFresh = visionFlowTime >= 0.0 && simTime - visionFlowTime <= VISION_FLOW_TIMEOUT_SEC;
    if (flowFresh != visionDepthOk && useVisionDepth)
        std::cout << (flowFresh ? "[DEPTH] camera depth estimate running"
                                : "[SAFETY] camera depth estimate lost (no /vision/heave_flow)")
                  << " (t=" << simTime << " s)" << std::endl;
    visionDepthOk = flowFresh;
    if (!visionDepthOk)
        flowVz = 0.0;
    visionDepth = kfX[0];
    visionVz = kfX[1];
    if (std::getenv("KYUBIC_DEBUG_DEPTH"))
    {
        static double lastPrint = -1.0;
        if (simTime - lastPrint >= 0.5)
        {
            lastPrint = simTime;
            std::cout << "[DEPTH-DBG] t=" << simTime << " truth " << depthTruth << " vz " << heaveVelTruth
                      << " | est " << kfX[0] << " vz " << kfX[1] << " bias " << kfX[2]
                      << " flowVz " << flowVz << " Fz " << heaveForceCmd << std::endl;
        }
    }

}

/* =========================================================
 * TELEMETRY [48..62]: thrust of each T200, body force/torque, depth estimate
 * ========================================================= */
void Task2Manager::AppendThrustTelemetry(std::vector<double>& data, double depth, double heaveVel)
{
    double force[3] = {0.0, 0.0, 0.0};
    double yawTorque = 0.0;
    for (size_t i = 0; i < kyubic::THRUSTER_COUNT; ++i)
    {
        const double thrust = thrusters[i] != nullptr ? thrusters[i]->getThrust() : 0.0;
        data.push_back(thrust);
        const auto& m = kyubic::THRUSTERS[i];
        for (int k = 0; k < 3; ++k)
            force[k] += thrust * m.dir[k];
        yawTorque += thrust * ((m.pos[0] - kyubic::CENTRE_OF_GRAVITY[0]) * m.dir[1]
                               - (m.pos[1] - kyubic::CENTRE_OF_GRAVITY[1]) * m.dir[0]);
    }
    data.insert(data.end(), {force[0], force[1], force[2], yawTorque,
                             depth, heaveVel, visionFlowFeatures,
                             visionDepthOk ? 1.0 : 0.0, useVisionDepth ? 1.0 : 0.0});
    // [63] mode (0 AUV, 1 ROV) [64..67] pilot input forward, sideways,
    // vertical, turn (-1..1; NaN in AUV) [68] wall time this step read the
    // pilot input [69] wall time the real-camera detection moved the sim target
    // [70] that detection's image stamp
    const double nan = std::numeric_limits<double>::quiet_NaN();
    data.insert(data.end(), {rovMode ? 1.0 : 0.0,
                             rovMode ? pilotInput[0] : nan, rovMode ? pilotInput[1] : nan,
                             rovMode ? pilotInput[2] : nan, rovMode ? pilotInput[3] : nan,
                             rovMode ? controlWallTime : nan,
                             targetMovedWallTime, targetMovedImageStamp});
    // [71] heading [72] turn rate [73] forward speed the controller used
    // (observers; truth is in [5], [7], [33]) [74] 1 = no IMU/DVL
    data.insert(data.end(), {navYaw, navYawRate, navVx, useSensorNav ? 0.0 : 1.0});
    // [75] camera turn rate (raw image shift) [76] sideways speed (thrust model)
    // [77] camera vertical speed (raw, one frame) [78] forward speed (thrust model)
    data.insert(data.end(), {flowYawRate, swayVelModel, flowVzRaw, surgeVelModel});
}

/* =========================================================
 * NAVIGATION WITHOUT IMU / DVL
 * ========================================================= */
void Task2Manager::Kf3::Predict(double dt, double accel, double dAccelDv, double qAccel, double qBias)
{
    x[0] += x[1] * dt;
    x[1] += (accel + x[2]) * dt;
    const double F[3][3] = {{1.0, dt, 0.0}, {0.0, 1.0 + dAccelDv * dt, dt}, {0.0, 0.0, 1.0}};
    double FP[3][3] = {}, Pn[3][3] = {};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            for (int k = 0; k < 3; ++k)
                FP[r][c] += F[r][k] * P[k][c];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            for (int k = 0; k < 3; ++k)
                Pn[r][c] += FP[r][k] * F[c][k];
    Pn[0][0] += 1e-6 * dt;
    Pn[1][1] += qAccel * qAccel * dt;
    Pn[2][2] += qBias * qBias * dt;
    std::copy(&Pn[0][0], &Pn[0][0] + 9, &P[0][0]);
}

bool Task2Manager::Kf3::Update(int i, double z, double var, bool gate)
{
    const double innovation = z - x[i];
    const double S = P[i][i] + var;
    if (gate && innovation * innovation > 9.0 * S)
        return false;
    double K[3];
    for (int r = 0; r < 3; ++r)
        K[r] = P[r][i] / S;
    for (int r = 0; r < 3; ++r)
        x[r] += K[r] * innovation;
    double Pn[3][3];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            Pn[r][c] = P[r][c] - K[r] * P[i][c];
    std::copy(&Pn[0][0], &Pn[0][0] + 9, &P[0][0]);
    return true;
}

void Task2Manager::UpdateNavigation(double dt)
{
    // Commanded forward force and yaw torque: each T200's thrust fraction
    // through its thrust curve, along its axis, with the rotor lag.
    double fx = 0.0, fy = 0.0, nz = 0.0;
    for (size_t i = 0; i < kyubic::THRUSTER_COUNT; ++i)
    {
        const double frac = appliedThrust[i];
        const double thrust = frac * (51.5) * THRUST_SCALE;
        const auto& m = kyubic::THRUSTERS[i];
        fx += thrust * m.dir[0];
        fy += thrust * m.dir[1];
        nz += thrust * ((m.pos[0] - kyubic::CENTRE_OF_GRAVITY[0]) * m.dir[1]
                        - (m.pos[1] - kyubic::CENTRE_OF_GRAVITY[1]) * m.dir[0]);
    }
    const double lag = std::min(1.0, dt / (T200_TIME_CONSTANT * std::sqrt(ROBOT_SCALE)));
    surgeForceCmd += (fx - surgeForceCmd) * lag;
    swayForceCmd += (fy - swayForceCmd) * lag;
    yawTorqueCmd += (nz - yawTorqueCmd) * lag;
    // Sideways speed: model only (nothing measures it camera only).
    swayVelModel += (swayForceCmd / SWAY_MASS - SWAY_DRAG_LIN * swayVelModel
                     - SWAY_DRAG * swayVelModel * std::fabs(swayVelModel)) * dt;
    surgeVelModel += (surgeForceCmd / SURGE_MASS - SURGE_DRAG_LIN * surgeVelModel
                      - SURGE_DRAG_COEF * surgeVelModel * std::fabs(surgeVelModel)) * dt;
    yawRateModel += (yawTorqueCmd / YAW_INERTIA - YAW_DRAG_LIN * yawRateModel
                     - YAW_DRAG * yawRateModel * std::fabs(yawRateModel)) * dt;

    // Camera-only measurements (also used by the start-up calibration).
    const double flowFresh = visionFlowTime >= 0.0 && simTime - visionFlowTime <= VISION_FLOW_TIMEOUT_SEC;
    if (visionFlowHorizNew)
    {
        flowYawRate = -visionFlowC / VISION_YAW_FLOW_GAIN;
        flowSurgeSign = visionFlowD * FLOW_DIVERGENCE_TO_SPEED;
    }
    if (!flowFresh)
        flowYawRate = flowSurgeSign = 0.0;

    if (useSensorNav)
    {
        // Simulator IMU / odometry (comparison only).
        const double qx = odometry->getLastValue(6), qy = odometry->getLastValue(7);
        const double qz = odometry->getLastValue(8), qw = odometry->getLastValue(9);
        navYaw = std::atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
        navPitch = std::asin(std::clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0));
        navRoll = std::atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy));
        navRollRate = odometry->getLastValue(10);
        navPitchRate = odometry->getLastValue(11);
        navYawRate = odometry->getLastValue(12);
        navVx = odometry->getLastValue(3);
        navX = odometry->getLastValue(0);
        navY = odometry->getLastValue(1);
        visionFlowHorizNew = false;
        return;
    }

    // Heading observer: turn model + camera turn rate.
    const double r = yawKf.x[1];
    yawKf.Predict(dt, yawTorqueCmd / YAW_INERTIA - YAW_DRAG_LIN * r - YAW_DRAG * r * std::fabs(r),
                  -YAW_DRAG_LIN - 2.0 * YAW_DRAG * std::fabs(r), KF_YAW_ACCEL_NOISE, KF_YAW_BIAS_NOISE);
    if (visionFlowHorizNew && flowFresh)
    {
        // Remove the image shift the sideways slide causes; the unknown scene
        // distance makes that correction uncertain, so trust the camera less.
        const double swayShift = swayVelModel / SWAY_FLOW_DISTANCE_M;
        const double rate = flowYawRate - swayShift;
        const double sigma = KF_YAW_RATE_NOISE + 0.3 * std::fabs(rate)
                             + std::fabs(swayVelModel) / SWAY_FLOW_DISTANCE_MIN;
        yawKf.Update(1, rate, sigma * sigma, false);
    }
    visionFlowHorizNew = false;

    // Surge observer: thrust model (+ target range in ObserveTargetRange).
    const double v = surgeKf.x[1];
    surgeKf.Predict(dt, surgeForceCmd / SURGE_MASS - SURGE_DRAG_LIN * v - SURGE_DRAG_COEF * v * std::fabs(v),
                    -SURGE_DRAG_LIN - 2.0 * SURGE_DRAG_COEF * std::fabs(v), KF_SURGE_ACCEL_NOISE, KF_SURGE_BIAS_NOISE);
    if (rangeAnchorValid && simTime - lastRangeUpdateTime > TARGET_ANCHOR_TIMEOUT_SEC)
        rangeAnchorValid = false;

    navYaw = WrapAngle(yawKf.x[0]);
    navYawRate = yawKf.x[1];
    navVx = surgeKf.x[1];
    navX += navVx * std::cos(navYaw) * dt;   // dead reckoning in the observer frame
    navY += navVx * std::sin(navYaw) * dt;
    navRoll = navRollRate = navPitchRate = 0.0;   // not measured (passively stable)
    // Steady pitch from the thrust (no IMU): the hull's trim plus the moments
    // of the commanded forces and of the drag. Thrusting forward tilted the
    // real-size KYUBIC ~3 deg nose up, which put a first buoy sighting's
    // height 6.5 cm off. Identified at real size (2 s averages, R^2 0.87,
    // 0.22 deg rms); the rocking around it is averaged out elsewhere.
    navPitch = (PITCH_TRIM_DEG + PITCH_PER_FX_DEG * surgeForceCmd + PITCH_PER_FZ_DEG * heaveForceCmd
                + PITCH_PER_V2_DEG * navVx * std::fabs(navVx)) * M_PI / 180.0;
}

void Task2Manager::ObserveTargetRange(double range)
{
    // Frame anchored to the target: distance travelled = anchor - range.
    if (useSensorNav)
        return;
    if (!rangeAnchorValid)
    {
        rangeAnchor = surgeKf.x[0] + range;
        rangeAnchorValid = true;
        rangeAnchorRejects = 0;
    }
    else
    {
        // The range is the only measurement of forward motion: never gated
        // and never re-anchored (that hid a real drift into the target).
        const double sigmaBox = range * range * KF_RANGE_BOX_JITTER_PX / (FocalLengthPx() * TARGET_WIDTH_M);
        const double sigma = KF_RANGE_NOISE + std::max(KF_RANGE_NOISE_PER_M * range, sigmaBox);
        surgeKf.Update(0, rangeAnchor - range, sigma * sigma, false);
    }
    lastRangeUpdateTime = simTime;
}

/* =========================================================
 * DEPTH OBSERVER (Kalman filter)
 * State [depth, vz, bias]; input = the commanded vertical thrust.
 * ========================================================= */
void Task2Manager::ObserverPredict(double dt, double bodyVx)
{
    // Commanded vertical thrust: each T200's thrust fraction through its
    // thrust curve, along its axis, with the rotor lag.
    double fz = 0.0;
    for (size_t i = 0; i < kyubic::THRUSTER_COUNT; ++i)
    {
        const double frac = appliedThrust[i];
        const double maxThrust = (51.5) * THRUST_SCALE;
        fz += frac * maxThrust * kyubic::THRUSTERS[i].dir[2];
    }
    const double tau = T200_TIME_CONSTANT * std::sqrt(ROBOT_SCALE);
    heaveForceCmd += (fz - heaveForceCmd) * std::min(1.0, dt / tau);

    double& d = kfX[0];
    double& v = kfX[1];
    const double b = kfX[2];
    // bodyVx = speed through the water (DVL on the real robot)
    const double accel = heaveForceCmd / HEAVE_MASS_KG + b - HEAVE_DRAG_LIN * v - HEAVE_DRAG * v * std::fabs(v)
                         + HEAVE_SURGE_LIFT * bodyVx * bodyVx;   // lifts forward AND reverse
    d += v * dt;
    v += accel * dt;

    const double F[3][3] = {{1.0, dt, 0.0},
                            {0.0, 1.0 - (HEAVE_DRAG_LIN + 2.0 * HEAVE_DRAG * std::fabs(v)) * dt, dt},
                            {0.0, 0.0, 1.0}};
    double FP[3][3] = {};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            for (int k = 0; k < 3; ++k)
                FP[r][c] += F[r][k] * kfP[k][c];
    double P[3][3] = {};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            for (int k = 0; k < 3; ++k)
                P[r][c] += FP[r][k] * F[c][k];
    P[1][1] += KF_ACCEL_NOISE * KF_ACCEL_NOISE * dt;
    P[2][2] += KF_BIAS_NOISE * KF_BIAS_NOISE * dt;
    P[0][0] += 1e-6 * dt;
    std::copy(&P[0][0], &P[0][0] + 9, &kfP[0][0]);
}

bool Task2Manager::ObserverUpdate(int index, double measurement, double variance, bool gate)
{
    const double innovation = measurement - kfX[index];
    const double S = kfP[index][index] + variance;
    if (gate && innovation * innovation > KF_GATE_SIGMA * KF_GATE_SIGMA * S)
        return false;   // outlier
    double K[3];
    for (int r = 0; r < 3; ++r)
        K[r] = kfP[r][index] / S;
    for (int r = 0; r < 3; ++r)
        kfX[r] += K[r] * innovation;
    double P[3][3];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            P[r][c] = kfP[r][c] - K[r] * kfP[index][c];
    std::copy(&P[0][0], &P[0][0] + 9, &kfP[0][0]);
    return true;
}

/* =========================================================
 * CAMERA VERTICAL MOTION (visual_heave_estimator.py)
 * [0] image stamp [1] vertical image shift [px/s] [2] divergence [3] horizontal
 * shift [4] inlier features [5] residual [6] frame interval
 * ========================================================= */
void Task2Manager::VisionFlowCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
    if (!msg || msg->data.size() < 5)
        return;
    visionFlowTime = simTime;              // the node is alive
    visionFlowFeatures = msg->data[4];
    if (!std::isfinite(msg->data[1]))
        return;                            // too few features this frame: no measurement
    visionFlowB = msg->data[1];
    visionFlowNew = true;
    if (std::isfinite(msg->data[2]) && std::isfinite(msg->data[3]))
    {
        visionFlowD = msg->data[2];
        visionFlowC = msg->data[3];
        visionFlowHorizNew = true;
    }
}

/* =========================================================
 * STONEFISH-CAMERA BOUNDING BOX CALLBACK
 * Expected message: [x_min, y_min, x_max, y_max, area, image_width, image_height]
 * ========================================================= */
void Task2Manager::TargetBoxCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
    if (!msg || msg->data.size() < 4)
        return;

    const double xMin = msg->data[0];
    const double yMin = msg->data[1];
    const double xMax = msg->data[2];
    const double yMax = msg->data[3];
    const double width  = (msg->data.size() >= 7) ? msg->data[5] : CAMERA_WIDTH;
    const double height = (msg->data.size() >= 7) ? msg->data[6] : CAMERA_HEIGHT;
    constexpr double BORDER_PX = 2.0;

    bboxWidth = xMax - xMin;
    bboxHeight = yMax - yMin;
    bboxClippedH = xMin <= BORDER_PX || xMax >= width - BORDER_PX;
    bboxClippedV = yMin <= BORDER_PX || yMax >= height - BORDER_PX;
    lastBboxTime = simTime;
}

/* =========================================================
 * ROV MODE: put the sim target where the real camera sees the object
 *
 * The real detector gives the object's position in its image, -1..+1 over the
 * range the box can move (x right, y up). The target is placed on the line of
 * sight of the Stonefish camera that shows its box at the SAME -1..+1
 * position, at its current distance from the camera, so the two camera views
 * match frame by frame. (Case 6 keeps a world-fixed mapping instead: an
 * autonomous robot could never reach a target locked to its own view.)
 * ========================================================= */
void Task2Manager::PlaceTargetInView()
{
    if (odometry == nullptr)
        return;
    // Real image position, -1..+1 (the detector publishes it as a pixel of
    // the full 640x480 frame).
    const double nx = std::clamp((objectPixelX - REAL_CAMERA_WIDTH * 0.5) / (REAL_CAMERA_WIDTH * 0.5), -1.0, 1.0);
    const double ny = std::clamp(-(objectPixelY - REAL_CAMERA_HEIGHT * 0.5) / (REAL_CAMERA_HEIGHT * 0.5), -1.0, 1.0);

    const bool seen = lastBboxTime >= 0.0 && simTime - lastBboxTime <= BBOX_FRESH_SEC && bboxWidth > 1.0 &&
                      targetConfidence >= MIN_CONFIDENCE;
    const double dt = std::clamp<double>(lastStepDt, 0.0, 0.05);
    // Where the Stonefish camera sees the target now (same -1..+1 scale).
    const double seenX = seen ? std::clamp(targetErrorX / std::max(CAMERA_WIDTH * 0.5 - bboxWidth * 0.5, 1.0), -1.0, 1.0) : 0.0;
    const double seenY = seen ? std::clamp(-targetErrorY / std::max(CAMERA_HEIGHT * 0.5 - bboxHeight * 0.5, 1.0), -1.0, 1.0) : 0.0;

    // The target stays FIXED in the world while the real object is still
    // (the pilot can fly around it). When the real object moves, the target
    // moves by the same amount in the Stonefish view, starting from where it
    // is seen now; when the real object is still again, it freezes again.
    if (rovTargetFrozen)
    {
        if (std::max(std::fabs(nx - rovFrozenX), std::fabs(ny - rovFrozenY)) < ROV_TARGET_MOVE_DEADBAND)
            return;
        rovTargetFrozen = false;
        const bool havePrevious = rovFrozenX < 1e8;   // first detection: match the view directly
        rovViewOffsetX = (seen && havePrevious) ? seenX - rovFrozenX : 0.0;   // current view position = base
        rovViewOffsetY = (seen && havePrevious) ? seenY - rovFrozenY : 0.0;
        rovViewCorrX = rovViewCorrY = 0.0;
        rovTargetDist = -1.0;   // re-measure the distance where it is now
        rovStillRefX = nx;
        rovStillRefY = ny;
        rovStillTime = 0.0;
    }
    if (std::max(std::fabs(nx - rovStillRefX), std::fabs(ny - rovStillRefY)) > ROV_TARGET_STILL_TOL)
    {
        rovStillRefX = nx;
        rovStillRefY = ny;
        rovStillTime = 0.0;
    }
    else
    {
        rovStillTime += dt;
    }
    const double wantX = std::clamp(nx + rovViewOffsetX, -1.0, 1.0);
    const double wantY = std::clamp(ny + rovViewOffsetY, -1.0, 1.0);
    const bool matched = seen && std::fabs(wantX - seenX) < 0.03 && std::fabs(wantY - seenY) < 0.03;
    if (rovStillTime >= ROV_TARGET_SETTLE_SEC && (matched || rovStillTime >= 3.0 * ROV_TARGET_SETTLE_SEC))
    {
        rovTargetFrozen = true;
        rovFrozenX = nx;
        rovFrozenY = ny;
        return;
    }

    // Closed loop: where the Stonefish camera actually sees the target is
    // compared with where it should be, and the placement is nudged until they
    // agree (removes box-size / model errors).
    if (seen)
    {
        // A box touching the image edge reads +-1 (it is at least that far
        // out): the correction then still pulls the target back inward.
        const double sx = std::clamp(targetErrorX / std::max(CAMERA_WIDTH * 0.5 - bboxWidth * 0.5, 1.0), -1.0, 1.0);
        const double sy = std::clamp(-targetErrorY / std::max(CAMERA_HEIGHT * 0.5 - bboxHeight * 0.5, 1.0), -1.0, 1.0);
        rovViewCorrX = std::clamp(rovViewCorrX + ROV_VIEW_MATCH_GAIN * dt * (wantX - sx), -0.5, 0.5);
        if (!rovTargetDepthLimited)
            rovViewCorrY = std::clamp(rovViewCorrY + ROV_VIEW_MATCH_GAIN * dt * (wantY - sy), -0.5, 0.5);
    }
    const double px = std::clamp(wantX + rovViewCorrX, -1.05, 1.05);
    const double py = std::clamp(wantY + rovViewCorrY, -1.05, 1.05);

    // Camera pose in the world (body -> world rotation from the quaternion).
    const double qx = odometry->getLastValue(6), qy = odometry->getLastValue(7);
    const double qz = odometry->getLastValue(8), qw = odometry->getLastValue(9);
    const double R[3][3] = {
        {1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)},
        {2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)},
        {2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)}};
    const double pos[3] = {odometry->getLastValue(0), odometry->getLastValue(1), odometry->getLastValue(2)};
    double cam[3];
    for (int r = 0; r < 3; ++r)
        cam[r] = pos[r] + R[r][0] * CAMERA_MOUNT[0] + R[r][1] * CAMERA_MOUNT[1] + R[r][2] * CAMERA_MOUNT[2];

    // Keep the target's current distance ALONG the camera axis (a plane in
    // front of the camera): image left/right moves it straight sideways and
    // image up/down straight up/down. (Keeping the straight-line distance
    // moved it on an arc, which looked like the wrong axis in the 3D view.)
    const sf::Vector3 origin = trackedObject->getTransform().getOrigin();
    const double box[3] = {origin.getX(), origin.getY(), origin.getZ() + ROV_TARGET_BOX_DZ};
    // The distance is measured once, when the target starts moving, and kept:
    // read back every step, a target held at the surface or seabed (off its
    // line of sight) crept 1 m toward / away from the robot through the
    // camera's ~1 deg pitch.
    if (rovTargetDist < 0.0)
    {
        const double rel[3] = {box[0] - cam[0], box[1] - cam[1], box[2] - cam[2]};
        rovTargetDist = std::clamp(R[0][0] * rel[0] + R[1][0] * rel[1] + R[2][0] * rel[2],
                                   ROV_TARGET_MIN_DIST, ROV_TARGET_MAX_DIST);   // along the camera axis
    }
    const double dist = rovTargetDist;

    // Pixel in the Stonefish image with the same -1..+1 position (box size
    // at this distance from the fitted effective bulb size).
    const double f = FocalLengthPx();
    const double boxW = f * ROV_TARGET_BOX_W / dist, boxH = f * ROV_TARGET_BOX_H / dist;
    const double u = CAMERA_WIDTH * 0.5 + px * std::max(CAMERA_WIDTH * 0.5 - boxW * 0.5, 1.0);
    const double v = CAMERA_HEIGHT * 0.5 - py * std::max(CAMERA_HEIGHT * 0.5 - boxH * 0.5, 1.0);

    // Line of sight: body x forward, image right = body +y, image down = body +z
    // (per unit of distance along the camera axis).
    const double d[3] = {1.0, (u - CAMERA_WIDTH * 0.5) / f, (v - CAMERA_HEIGHT * 0.5) / f};
    double dir[3];
    for (int r = 0; r < 3; ++r)
        dir[r] = R[r][0] * d[0] + R[r][1] * d[1] + R[r][2] * d[2];
    // Too high (above the surface) or too deep (into the seabed) at that
    // distance: bring it closer along the SAME line of sight, so the Stonefish
    // camera still shows it exactly where the real camera does (the real-size
    // pool has only ~0.2 m of water above the camera); it goes back out to its
    // distance when it comes down again.
    double distNow = dist;
    const double boxMinZ = ROV_TARGET_MIN_Z + ROV_TARGET_BOX_DZ;
    const double boxMaxZ = ROV_TARGET_MAX_Z + ROV_TARGET_BOX_DZ;
    if (dir[2] < -1e-6 && cam[2] + dist * dir[2] < boxMinZ)
        distNow = std::max(ROV_TARGET_MIN_DIST, (boxMinZ - cam[2]) / dir[2]);
    else if (dir[2] > 1e-6 && cam[2] + dist * dir[2] > boxMaxZ)
        distNow = std::max(ROV_TARGET_MIN_DIST, (boxMaxZ - cam[2]) / dir[2]);
    double target[3];
    for (int r = 0; r < 3; ++r)
        target[r] = cam[r] + distNow * dir[r];

    // Still beyond the surface / seabed at the closest distance: it stops there.
    const double zWanted = target[2] - ROV_TARGET_BOX_DZ;
    const double z = std::clamp(zWanted, ROV_TARGET_MIN_Z, ROV_TARGET_MAX_Z);
    rovTargetDepthLimited = std::fabs(z - zWanted) > 1e-6;   // no vertical correction then
    if (std::getenv("KYUBIC_DEBUG_ROV"))
    {
        static double lastPrint = -1.0;
        if (simTime - lastPrint >= 0.5)
        {
            lastPrint = simTime;
            std::printf("[ROV-DBG] t=%.1f real %+.2f %+.2f | seen %+.2f %+.2f | off %+.2f %+.2f | corr %+.2f %+.2f | "
                        "p %+.2f %+.2f | dist %.2f | roll %+.1f pitch %+.1f deg | z %.3f%s | origin %.3f %.3f %.3f cam %.3f %.3f %.3f\n",
                        simTime, nx, ny, seenX, seenY, rovViewOffsetX, rovViewOffsetY, rovViewCorrX, rovViewCorrY,
                        px, py, dist, std::atan2(R[2][1], R[2][2]) * 180.0 / M_PI, -std::asin(R[2][0]) * 180.0 / M_PI,
                        z, rovTargetDepthLimited ? " LIMITED" : "", origin.getX(), origin.getY(), origin.getZ(),
                        cam[0], cam[1], cam[2]);
        }
    }

    if (objectImageStamp != targetMovedImageStamp)
    {
        targetMovedImageStamp = objectImageStamp;
        targetMovedWallTime = WallNow();
    }
    MoveTarget(sf::Transform(sf::Quaternion(0.0, M_PI, 0.0), sf::Vector3(target[0], target[1], z)));
}

void Task2Manager::MoveTarget(const sf::Transform& pose)
{
    for (sf::Obstacle* part : targetParts)
        part->setTransform(pose);
}

/* =========================================================
 * MOVE SIM TARGET FROM REAL CAMERA (case 6 only)
 * ========================================================= */
void Task2Manager::UpdateTrackedObject()
{
    if (activeCase != 6 && activeCase != ScenarioConfig::ROV_CASE)
        return;
    if (datasetMode && trackedObject != nullptr)
    {
        UpdateDatasetTarget();
        return;
    }
    if (trackedObject == nullptr || !objectDetected)
        return;
    if (rovMode)
    {
        PlaceTargetInView();
        return;
    }

    // In tuned-world metres (scaled by Sim::TargetFromTuned below).
    constexpr double targetAnchorX = 8.0;
    constexpr double targetAnchorY = 0.0;
    constexpr double targetAnchorZ = 1.5;
    // Real camera image -> simulated target: image left/right moves the target
    // left/right (+Y = robot's right), image up/down moves it up/down
    // (+Z = deeper, NED). Ranges follow the camera's 4:3 shape.
    constexpr double targetMotionY = 1.5;    // m at the image edge
    constexpr double targetMotionZ = 1.1;

    const double pixelX = std::clamp(objectPixelX, 0.0, REAL_CAMERA_WIDTH);
    const double pixelY = std::clamp(objectPixelY, 0.0, REAL_CAMERA_HEIGHT);
    const double normalizedHorizontal = (pixelX - REAL_CAMERA_WIDTH * 0.5) / (REAL_CAMERA_WIDTH * 0.5);
    const double normalizedVertical   = (pixelY - REAL_CAMERA_HEIGHT * 0.5) / (REAL_CAMERA_HEIGHT * 0.5);

    // Keep the target between the case-2 and case-3 depths: never above the
    // surface or inside the seabed.
    constexpr double targetMinZ = 1.45;   // shallowest the robot can centre on
    constexpr double targetMaxZ = 3.2;

    if (objectImageStamp != targetMovedImageStamp)
    {
        // first step that applies this real-camera detection to the sim target
        targetMovedImageStamp = objectImageStamp;
        targetMovedWallTime = WallNow();
    }
    MoveTarget(sf::Transform(
        sf::Quaternion(0.0, M_PI, 0.0),
        Sim::TargetFromTuned(
            targetAnchorX,
            targetAnchorY + normalizedHorizontal * targetMotionY,
            std::clamp(targetAnchorZ + normalizedVertical * targetMotionZ,
                       targetMinZ, targetMaxZ))));
}