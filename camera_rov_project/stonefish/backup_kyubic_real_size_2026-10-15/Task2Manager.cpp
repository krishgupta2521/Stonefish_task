#include "Task2Manager.h"
#include "kyubic_model.hpp"

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

#include <algorithm>
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

// Select the case at start-up, no rebuild needed:
//   ./Task2Baseline 3        or   ./Task2Baseline --case 3   or   TASK2_CASE=3
// 1=front, 2=above, 3=below, 4=behind, 5=behind+below,
// 6=moving target driven by /object/position, 7=NO target (search, then surface).
// NOTE: in cases 4 and 5 the target starts BEHIND the robot, so the robot
// must search (turn around) before it can see it. That is expected.
namespace ScenarioConfig {
constexpr int DEFAULT_CASE = 6;
constexpr int NUM_CASES = 7;

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
        case 1: return sf::Vector3(8.0, 0.0, 1.5);
        case 2: return sf::Vector3(8.0, 0.0, 0.65); // NED: smaller Z = higher
        case 3: return sf::Vector3(8.0, 0.0, 3.2);
        case 4: return sf::Vector3(-8.0, 0.0, 1.5);
        case 5: return sf::Vector3(-8.0, 0.0, 3.2);
        default: return sf::Vector3(8.0, 0.0, 1.5);
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
    constexpr double THRUST_SLEW_RATE = 1.5;                     // thrust fraction per second
    constexpr double ODOMETRY_RATE_HZ = 100.0;

    // Detection camera: ZED Mini left lens, VGA stream (672 x 376, 90 x 58 deg).
    // (The OAK-1 is not used: the claw hides the lower ~40 % of its view.)
    constexpr int    CAMERA_WIDTH_PX = 672;
    constexpr int    CAMERA_HEIGHT_PX = 376;
    constexpr double CAMERA_WIDTH = CAMERA_WIDTH_PX;
    constexpr double CAMERA_HEIGHT = CAMERA_HEIGHT_PX;
    constexpr double CAMERA_HFOV_DEG = 90.0;

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
    constexpr double CALIB_COMMAND = 0.30;
    constexpr double CALIB_MIN_DELTA = 0.004;       // m/s, below this = inconclusive
    // 2) While the robot is aligning (never while moving forward) the
    //    controller checks that the image moves the way physics says it must
    //    (turn right -> target moves LEFT; go deeper -> target moves UP).
    //    If not, it flips / swaps camera axes. Disabled for case 6 (moving target).
    constexpr bool ENABLE_ONLINE_CAMERA_CHECK = true;

    // ---------------- Perception -> guidance ----------------
    constexpr double MIN_CONFIDENCE = 0.5;
    constexpr double TARGET_TIMEOUT_SEC = 0.8;      // sim seconds without a confident detection
    constexpr double DETECTOR_TIMEOUT_SEC = 2.0;    // no detector message at all -> perception fault
    constexpr double PERCEPTION_ABORT_SEC = 30.0;   // fault this long -> surface
    constexpr double TARGET_FILTER_ALPHA = 0.5;
    constexpr double OUTLIER_JUMP_RAD = 0.30;
    constexpr int    OUTLIER_CONFIRM_FRAMES = 3;
    constexpr double DEPTH_PER_RAD = 2.0;           // lever (m/rad) used before range is known
    constexpr double MAX_DEPTH_STEP = 0.5;
    constexpr double MIN_DEPTH = 0.7;
    constexpr double MAX_DEPTH = 4.2;

    // ---------------- Range from bounding box ----------------
    // distance to the target AXIS = f * size / pixels, minus the target radius
    // = distance to its FRONT face. Sizes were fitted to ground truth in the
    // simulator for KYUBIC's ZED Mini + YOLO at 960 px (yellow_sensor_model,
    // 1.9-7 m): boxes are 0.413 m wide and 1.80 m tall in effect (geometric
    // height 1.71 m). Rounded down so errors make the robot stop early.
    // Re-fit them for a different camera, detector or target.
    constexpr double TARGET_WIDTH_M = 0.41;
    constexpr double TARGET_HEIGHT_M = 1.76;
    constexpr double TARGET_RADIUS_M = 0.235;
    constexpr double RANGE_FILTER_ALPHA = 0.3;
    // Closer than this, the box only shows part of the target (e.g. its thin
    // stem) and looks too small = too far. Then only "closer" readings are
    // accepted; odometry carries the estimate the rest of the way.
    constexpr double RANGE_TRUST_FAR = 1.2;
    constexpr double MIN_RANGE = 0.05;
    constexpr double MAX_RANGE = 15.0;
    constexpr double BBOX_FRESH_SEC = 0.3;
    // A sudden large drop in range (e.g. a false blob) must repeat this many
    // frames in a row before it is believed.
    constexpr double RANGE_JUMP_RATIO = 0.6;
    constexpr int    RANGE_JUMP_CONFIRM_FRAMES = 3;

    // ---------------- Vehicle geometry (simulation hull) ----------------
    constexpr double CAMERA_TO_FRONT = kyubic::FRONT_X - kyubic::CAMERA_POS[0];   // OAK-1 to recovery hooks

    // ---------------- Approach / mission ----------------
    // Gap = distance from the robot's front face to the target's front face.
    constexpr double STOP_GAP = 0.20;               // stop this far from the target
    constexpr double REACH_TOL = 0.03;
    constexpr double MIN_SAFE_GAP = 0.10;           // closer than this -> gently move away
    // Terminal approach: closer than this the target fills the camera view and
    // detections become unreliable (YOLO below ~1 m: ~30% dropouts, box
    // centre jumps 30-60 px; solid above 1 m). Heading and depth are locked and the rest is driven on
    // odometry; detections can then only shorten the range estimate.
    // ZED Mini (90 deg) + YOLO at 960 px: 100 % detection down to 1.75 m camera
    // distance (gap ~1.4 m), 40 % below. The 1.71 m target fills the 58 deg
    // vertical view at 1.54 m, so the blind final stretch starts earlier than
    // with a narrow camera.
    constexpr double TERMINAL_GAP = 1.80;
    constexpr double TERMINAL_MAX_SEC = 60.0;
    // The terminal lock only engages from a good track: this long of
    // uninterrupted confident detections. Otherwise stop, then retreat.
    constexpr double GOOD_TRACK_CONFIDENCE = 0.8;
    constexpr double TERMINAL_MIN_TRACK_SEC = 2.0;
    constexpr double TERMINAL_WAIT_SEC = 4.0;
    // Inside this gap detections are not trusted for (re)acquisition: a target
    // lost here is never searched for on the spot. The robot backs away to
    // RETREAT_GAP, where detection is reliable, and looks again.
    constexpr double RELIABLE_GAP = 2.0;
    constexpr double RETREAT_GAP = 2.3;
    constexpr double RETREAT_TIMEOUT_SEC = 40.0;
    // A moving target (case 6) can swing into the hull: never drive blind,
    // stop where the camera still sees it reliably.
    constexpr double STOP_GAP_MOVING_TARGET = 1.0;
    // Speed profile: cruise far away, then v = gain * remaining distance, a
    // deceleration the thrusters can always deliver, then creep.
    constexpr double APPROACH_CRUISE_SPEED = 0.25;  // m/s, far from the target
    constexpr double APPROACH_CREEP_SPEED = 0.03;   // m/s, final centimetres
    constexpr double APPROACH_SPEED_GAIN = 0.12;    // (m/s) per m of remaining gap
    constexpr double TERMINAL_MAX_SPEED = 0.08;     // m/s while driving on odometry only
    // Braking, measured on KYUBIC in the simulator: thrust gives 0.39 m/s^2 per
    // unit command, so SURGE_MAX brakes at ~0.14 m/s^2 (+ drag). The speed
    // profile plans with far less than that; the stopping guard below brakes
    // at once if the robot's ACTUAL speed could not stop in time.
    constexpr double BRAKE_DECEL_PLAN = 0.02;       // m/s^2, v <= sqrt(2 a d)
    constexpr double BRAKE_DECEL_GUARD = 0.035;     // m/s^2, stopping-distance guard
    constexpr double BRAKE_GUARD_MARGIN = 0.05;     // m
    constexpr double REACHED_SPEED = 0.03;          // m/s, must be this slow to count as reached
    constexpr double LOST_REACHED_MARGIN = 0.25;    // lost this close to STOP_GAP -> reached
    constexpr double REACHED_HOLD_SEC = 3.0;
    constexpr double BACKOFF_DISTANCE = 0.50;       // back away before ascending (0 = off)
    constexpr double BACKOFF_SPEED = 0.20;          // m/s max, backing away (BACK_OFF / RETREAT)
    constexpr double BACKOFF_TIMEOUT_SEC = 30.0;
    // Hull-axis depth when surfaced: the floater tops (0.223 m above the axis)
    // are about 3 cm out of the water. Higher, KYUBIC loses more buoyancy
    // than the heave limit can hold (it is trimmed only 0.5 kg positive).
    constexpr double SURFACE_DEPTH = 0.19;
    constexpr double ASCENT_RATE = 0.10;            // m/s depth reference ramp
    constexpr double ASCENT_RATE_FINAL = 0.03;      // m/s close to the surface
    constexpr double ASCENT_SLOW_GAIN = 0.25;       // (m/s) per m above SURFACE_DEPTH
    constexpr double ASCENT_MAX_LEAD = 0.15;        // m, reference ahead of the robot
    constexpr double SURFACED_TOL = 0.02;           // m: within 2 cm of SURFACE_DEPTH
    // Contact / obstruction: thrusting forward but not moving -> stop at once.
    constexpr double STALL_MIN_SPEED_REF = 0.03;
    constexpr double STALL_SPEED_RATIO = 0.3;
    constexpr double STALL_SEC = 1.0;

    // ---------------- State-machine thresholds ----------------
    constexpr double H_ALIGN_TOL = 0.07;            // rad (~4 deg, ~32 px)
    constexpr double V_ALIGN_TOL = 0.10;            // m depth error
    constexpr double LEAVE_FACTOR = 1.8;            // hysteresis while aligning
    constexpr double APPROACH_LEAVE_FACTOR = 2.5;   // looser while approaching
    constexpr double YAW_SETTLED_RATE = 0.03;       // rad/s
    constexpr double HEAVE_SETTLED_VEL = 0.03;      // m/s
    constexpr double SETTLE_HOLD_SEC = 0.5;
    constexpr double VERIFY_HOLD_SEC = 0.8;
    constexpr double LOST_HOLD_SEC = 2.0;
    constexpr double SEARCH_YAW_RATE = 0.12;
    constexpr double SEARCH_MAX_LEAD = 0.35;
    constexpr double SEARCH_MAX_TURN = 2.0 * M_PI + 0.35;  // one full turn, then give up
    constexpr double SEARCH_MAX_SEC = 180.0;        // backstop if the turn is obstructed

    // ---------------- Inner loops (fractions of max thrust) ----------------
    constexpr double YAW_KP = 0.80;
    constexpr double YAW_KD = 2.00;
    constexpr double YAW_KI = 0.00;
    constexpr double YAW_MAX = 0.30;

    // Tuned with working vertical thrusters (~3-4 t effective heave mass):
    // damping ratio ~0.7, small steady error under the lift of forward motion.
    constexpr double DEPTH_KP = 1.20;
    constexpr double DEPTH_KD = 4.00;
    constexpr double DEPTH_KI = 0.10;
    constexpr double DEPTH_I_LIMIT = 0.25;
    constexpr double HEAVE_MAX = 0.40;

    // Roll hold with the two middle thrusters (pitch is passive: no thruster
    // can pitch KYUBIC, the 5 cm righting lever keeps it level).
    constexpr double ROLL_KP = 0.50;
    constexpr double ROLL_KD = 0.30;
    constexpr double ROLL_MAX = 0.20;

    // Surge: speed loop on odometry body velocity. KYUBIC's measured surge plant
    // (simulator, full mission): dv/dt = 0.391 * command - 1.80 * v|v|.
    constexpr double SURGE_KDRAG = 1.80 / 0.391;    // drag feed-forward: command per (m/s)^2
    constexpr double SURGE_KA = 1.0 / 0.391;        // acceleration feed-forward: command per m/s^2
    constexpr double SURGE_ACCEL_FILTER_SEC = 0.3;  // smooths the feed-forward (no chatter)
    constexpr double SURGE_KP = 2.50;
    constexpr double SURGE_KI = 0.30;
    constexpr double SURGE_I_LIMIT = 0.10;          // thrust fraction
    constexpr double SURGE_MAX = 0.35;
    // Reference ramps the thrusters can actually follow (so the robot never
    // lags far behind its own speed reference).
    constexpr double SPEED_ACCEL = 0.03;            // m/s^2, speed reference ramp up
    constexpr double SPEED_DECEL = 0.035;           // m/s^2, speed reference ramp down
    // Position hold along the heading (used whenever the robot is not travelling).
    constexpr double HOLD_KP = 0.30;                // (m/s) per m
    constexpr double HOLD_MAX_SPEED = 0.08;
    constexpr double TRAVEL_HOLD_KP = 0.60;         // (m/s) per m, near a BACK_OFF/RETREAT goal
    constexpr double HOLD_CAPTURE_SPEED = 0.02;     // capture the hold point once this slow

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
        const double w2 = T200_DEADBAND_W2 + std::fabs(fraction) * (wMax2 - T200_DEADBAND_W2);
        return std::copysign(std::sqrt(w2 / wMax2), fraction);
    }

    const char* StateName(int s)
    {
        static const char* names[] = {
            "SEARCH", "YAW_ALIGN", "YAW_SETTLE", "VERTICAL_ALIGN", "VERIFY_CENTER",
            "APPROACH", "TARGET_LOST_HOLD", "CALIBRATE", "STOPPED",
            "TARGET_REACHED", "BACK_OFF", "SURFACING", "SURFACED", "RETREAT"};
        return (s >= 0 && s <= 13) ? names[s] : "?";
    }

    // ---------------- Added mass of KYUBIC's solids ----------------
    // Stonefish fits a slender ellipsoid to every solid, which is wrong for flat
    // plates: it gave the 6.8 mm bottom plate 314 kg of heave added mass. These
    // standard estimates (DNV-RP-H103 strip form) replace it.
    constexpr double PERFORATED_PLATE_FACTOR = 0.5;   // frame plates have large cut-outs

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
    getOcean()->setWaterType(0.2);
    getOcean()->AddVelocityField(new sf::Uniform(sf::Vector3(0.0, 0.0, 0.0)));
    getOcean()->EnableCurrents();
    getAtmosphere()->SetSunPosition(0.0, 60.0);

    /* ---------------- SEABED (top at Z = 5.0) ---------------- */
    sf::Obstacle* seabed = new sf::Obstacle(
        "Seabed", sf::Vector3(30.0, 30.0, 0.5), sf::I4(), "Rock", "seabed");
    AddStaticEntity(seabed, sf::Transform(sf::IQ(), sf::Vector3(0.0, 0.0, 5.25)));

    /* ---------------- TRACKED OBJECT ---------------- */
    // Enlarged buoy (1.71 m tall) for demonstrations. The real-size one
    // (0.40 m, models/buoy_real.obj from tools/make_buoy.py) is only detected
    // by the current YOLO model out to ~3 m and needs a retrained model.
    std::string objectPath =
        "/home/krish/Documents/Development/kyushu_project/camera_rov_project/stonefish/models/yellow_sensor_model.obj";

    if (ScenarioConfig::HasTarget(activeCase))
    {
        trackedObject = new sf::Obstacle(
            "TrackedObject", objectPath, 0.5, sf::I4(), objectPath, 0.5, sf::I4(),
            true, "TargetMaterial", "tracked_object_yellow");
        const sf::Vector3 targetPos = ScenarioConfig::TargetPosition(activeCase);
        AddStaticEntity(trackedObject, sf::Transform(sf::Quaternion(0.0, M_PI, 0.0), targetPos));
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
                if (dims[k] < 0.012 && dims[(k + 1) % 3] > 0.1 && dims[(k + 2) % 3] > 0.1)
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
            T200_DIAMETER / 0.178, sf::I4(), "PropellerMaterial", "propeller");
        auto rotor = std::make_shared<sf::FirstOrder>(T200_TIME_CONSTANT);
        // Asymmetric quadratic thrust with the ESC dead band (T200 at 16 V)
        auto thrustModel = std::make_shared<sf::DeadbandThrust>(
            T200_KT_REVERSE, T200_KT_FORWARD, -T200_DEADBAND_W2, T200_DEADBAND_W2);
        thrusters[i] = new sf::Thruster(mount.name, propeller, rotor, thrustModel,
                                        T200_DIAMETER, true, T200_MAX_OMEGA, false, true);
    }

    /* ---------------- SENSORS ---------------- */
    odometry = new sf::Odometry("Odometry", ODOMETRY_RATE_HZ);

    // ZED Mini left lens (the detection camera) behind the front dome
    sf::ColorCamera* camera = new sf::ColorCamera(
        "FrontCamera", CAMERA_WIDTH_PX, CAMERA_HEIGHT_PX, CAMERA_HFOV_DEG, 10.0);
    camera->InstallNewDataHandler([this](sf::ColorCamera* cam) { PublishCameraImage(cam); });

    /* ---------------- ROBOT ---------------- */
    sf::FeatherstoneRobot* boat = new sf::FeatherstoneRobot("AUV");
    boat->DefineLinks(hull);
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
                      sf::Vector3(kyubic::CAMERA_POS[0], kyubic::CAMERA_POS[1], kyubic::CAMERA_POS[2])));

    AddRobot(boat, sf::Transform(sf::IQ(), sf::Vector3(0.0, 0.0, 1.5)));

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

    objectPositionSubscriber = rosNode->create_subscription<geometry_msgs::msg::PointStamped>(
        "/object/position", 10,
        std::bind(&Task2Manager::ObjectPositionCallback, this, std::placeholders::_1));
    targetDetectionSubscriber = rosNode->create_subscription<geometry_msgs::msg::PointStamped>(
        "/stonefish/target/detection", 10,
        std::bind(&Task2Manager::TargetDetectionCallback, this, std::placeholders::_1));
    targetBoxSubscriber = rosNode->create_subscription<std_msgs::msg::Float64MultiArray>(
        "/stonefish/target/bbox", 10,
        std::bind(&Task2Manager::TargetBoxCallback, this, std::placeholders::_1));

    InitializeController();

    autonomousMode = true;
    ResetAutonomousControl();
    if (ENABLE_ACTUATOR_CALIBRATION)
        EnterState(MotionState::CALIBRATE);

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
            SetAxisCommands(0.0, 0.0, 0.0, 0.0, 0.0);
            return;
        }
    }

    SDL_PumpEvents();
    const Uint8* keys = SDL_GetKeyboardState(nullptr);

    static bool previousM = false;
    const bool mPressed = keys[SDL_SCANCODE_M] != 0;
    if (mPressed && !previousM)
    {
        autonomousMode = !autonomousMode;
        ResetAutonomousControl();
        std::cout << "[MODE] " << (autonomousMode ? "AUTONOMOUS" : "MANUAL") << std::endl;
    }
    previousM = mPressed;

    if (keys[SDL_SCANCODE_SPACE])
    {
        ResetAutonomousControl();
        SetAxisCommands(0.0, 0.0, 0.0, 0.0, 0.0);
        return;
    }

    if (autonomousMode)
    {
        UpdateAutonomousControl(lastStepDt);
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
    searchTurned = 0.0;
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
        {0.0, 0.0, 0.0, 2.0, 0},                       // settle
        {CALIB_COMMAND, 0.0, 0.0, 2.0, 1},             // surge
        {-CALIB_COMMAND, 0.0, 0.0, 2.0, 0},            // undo the motion
        {0.0, 0.0, 0.0, 2.0, 0},
        {0.0, CALIB_COMMAND, 0.0, 2.0, 2},             // yaw
        {0.0, -CALIB_COMMAND, 0.0, 2.0, 0},
        {0.0, 0.0, 0.0, 2.0, 0},
        {0.0, 0.0, CALIB_COMMAND, 2.0, 3},             // heave (down)
        {0.0, 0.0, -CALIB_COMMAND, 2.0, 0},            // undo the depth change
        {0.0, 0.0, 0.0, 2.0, 0}};
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
    const double raw[5] = {s.surge, 0.0, heaveStep ? s.heave : depthHold * heaveSign, 0.0, s.yaw};
    ApplyAxisCommands(raw);   // RAW: no sign correction
    calibStepTime += dt;

    if (calibStepTime < s.duration)
        return false;

    auto decide = [](double delta, const char* name) -> double
    {
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
        heaveSign = decide(heaveVel - calibStartHeave, "heave (down)");

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

    if (!camVVerified && evZAbs > 0.25)
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
        else if (evZAbs > 1.5)
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
        if (dtDet > 0.0 && dtDet < 0.4)
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
            if (std::fabs(dZ) > 0.002 && std::fabs(dPsi) < 0.01)
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
    }

    // In the terminal approach only the range may change (and only shorter).
    if (terminalLock)
        return;

    // 3) Pixel error -> angles (pinhole model).
    const double hRaw = camHorizontalIsX ? targetErrorX : targetErrorY;
    const double vRaw = camHorizontalIsX ? targetErrorY : targetErrorX;
    const double bearing   = std::atan2(camHSign * hRaw, f);   // + = target right
    // + = target below. A nose-up pitch makes the target look lower than it is.
    const double elevation = std::atan2(camVSign * vRaw, f) - pitch;

    // 4) Store as WORLD heading/depth so robot rotation does not corrupt memory.
    //    Depth offset = range * tan(elevation) once the range is known.
    const double lever = haveRange ? std::clamp(filteredRange, 0.5, 8.0) : DEPTH_PER_RAD;
    const double measYaw = WrapAngle(yaw + bearing);
    const double measDepth = depth + std::clamp(
        lever * std::tan(elevation), -MAX_DEPTH_STEP, MAX_DEPTH_STEP);

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
                    filteredTargetDepth += TARGET_FILTER_ALPHA * (measDepth - filteredTargetDepth);
            }
            outlierCount = 0;
        }
    }
    filteredTargetDepth = std::clamp(filteredTargetDepth, MIN_DEPTH, MAX_DEPTH);

    if (std::fabs(bearing) > 0.02)
        searchDirection = (bearing > 0.0) ? 1.0 : -1.0;
}

/* =========================================================
 * SURGE: speed loop (thrust fraction from desired body speed)
 * ========================================================= */
double Task2Manager::SurgeSpeedControl(double speedTarget, double bodyVx, double dt)
{
    // Smooth speed reference: no thrust steps, gentle on the hull and thrusters.
    const double rate = (std::fabs(speedTarget) > std::fabs(speedRef)) ? SPEED_ACCEL : SPEED_DECEL;
    const double refStep = std::clamp(speedTarget - speedRef, -rate * dt, rate * dt);
    speedRef += refStep;
    // Filtered rate of the reference: the ramp can flip direction from one step
    // to the next, and an unfiltered feed-forward would then chatter.
    const double alpha = dt / (SURGE_ACCEL_FILTER_SEC + dt);
    surgeRefAccel += alpha * (refStep / dt - surgeRefAccel);

    const double err = speedRef - bodyVx;
    const double unsat = SURGE_KDRAG * speedRef * std::fabs(speedRef) + SURGE_KA * surgeRefAccel +
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
    if (odometry == nullptr)
    {
        SetAxisCommands(0.0, 0.0, 0.0, 0.0, 0.0);
        return;
    }
    dt = std::clamp(dt, 1e-4, 0.05);

    const double posX     = odometry->getLastValue(0);
    const double posY     = odometry->getLastValue(1);
    const double depth    = odometry->getLastValue(2);
    const double bodyVx   = odometry->getLastValue(3);
    const double heaveVel = odometry->getLastValue(5);
    const double qx = odometry->getLastValue(6);
    const double qy = odometry->getLastValue(7);
    const double qz = odometry->getLastValue(8);
    const double qw = odometry->getLastValue(9);
    const double yawRate  = odometry->getLastValue(12);
    const double yaw = std::atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
    const double pitch = std::asin(std::clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0));  // + = nose up
    const double roll = std::atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy));
    const double rollRate = odometry->getLastValue(10);
    posXNow = posX;
    posYNow = posY;

    // ---------------- CALIBRATE ----------------
    if (motionState == MotionState::CALIBRATE)
    {
        // Hold the starting depth while surge and yaw are pulsed: KYUBIC is
        // slightly positively buoyant and would otherwise float up.
        if (calibDepthRef < 0.0)
            calibDepthRef = depth;
        const double depthHold = std::clamp(
            DEPTH_KP * (calibDepthRef - depth) - DEPTH_KD * heaveVel, -HEAVE_MAX, HEAVE_MAX);
        if (RunActuatorCalibration(dt, bodyVx, heaveVel, yawRate, depthHold))
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
        holdValid = false;
        speedRef = 0.0;
        refsInitialized = true;
    }
    if (autonomyStartTime < 0.0)
        autonomyStartTime = simTime;

    // ---------------- Perception ----------------
    // Range prediction between camera frames: the target gets closer by
    // exactly what the robot travels toward it.
    if (haveRange)
        filteredRange = std::max(0.0, filteredRange - bodyVx * dt);

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
    const bool movingTarget = (activeCase == 6);
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
    if (!finishing && !retreating && !detectorAlive && detectorSilentFor >= PERCEPTION_ABORT_SEC)
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
            if (!movingTarget && !terminalLock && haveRange && frontGap < TERMINAL_GAP)
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
            else if (std::fabs(bodyVx) > 0.05)
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
            double profile = std::min({APPROACH_CRUISE_SPEED,
                                       std::sqrt(2.0 * BRAKE_DECEL_PLAN * remaining),
                                       APPROACH_CREEP_SPEED + APPROACH_SPEED_GAIN * remaining});
            if (terminalLock)
                profile = std::min(profile, TERMINAL_MAX_SPEED);
            else if (!movingTarget)
                // Arrive at the terminal point already at terminal speed
                // (no speed cliff there).
                profile = std::min(profile, std::sqrt(
                    TERMINAL_MAX_SPEED * TERMINAL_MAX_SPEED +
                    2.0 * BRAKE_DECEL_PLAN * std::max(0.0, frontGap - TERMINAL_GAP)));
            profile = std::max(profile, APPROACH_CREEP_SPEED);
            const double hGate = std::clamp(
                1.0 - std::fabs(hErr) / (APPROACH_LEAVE_FACTOR * H_ALIGN_TOL), 0.0, 1.0);
            const double vGate = std::clamp(
                1.0 - std::fabs(vErr) / (APPROACH_LEAVE_FACTOR * V_ALIGN_TOL), 0.0, 1.0);
            const double rGate = std::clamp(1.0 - std::fabs(yawRate) / 0.15, 0.0, 1.0);
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
                    searchTurned = 0.0;
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
            prevSearchYaw = yaw;
            searchTime += dt;
            if (searchTurned >= SEARCH_MAX_TURN || searchTime >= SEARCH_MAX_SEC)
            {
                yawRef = yaw;
                std::cout << "[SEARCH] No target after " << searchTurned * 180.0 / M_PI
                          << " deg of search." << std::endl;
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
                    ? std::clamp(RETREAT_GAP - frontGap, 0.3, 2.0) : 1.0;
                holdX = posX - dist * std::cos(yaw);
                holdY = posY - dist * std::sin(yaw);
                holdValid = true;
                retreatGoalSet = true;
            }
            holdSpeedLimit = BACKOFF_SPEED;
            const double along = (holdX - posX) * std::cos(yaw) + (holdY - posY) * std::sin(yaw);
            if ((retreatGoalSet && std::fabs(along) < 0.05 && std::fabs(bodyVx) < REACHED_SPEED) ||
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
                holdX = posX - BACKOFF_DISTANCE * std::cos(yaw);
                holdY = posY - BACKOFF_DISTANCE * std::sin(yaw);
                EnterState(MotionState::BACK_OFF);
                holdValid = true;                    // hold point = BACK_OFF goal
            }
            break;

        case MotionState::BACK_OFF:
        {
            holdSpeedLimit = BACKOFF_SPEED;
            const double along = (holdX - posX) * std::cos(yaw) + (holdY - posY) * std::sin(yaw);
            if ((std::fabs(along) < 0.05 && std::fabs(bodyVx) < REACHED_SPEED) ||
                stateTime >= BACKOFF_TIMEOUT_SEC)
                BeginSurfacing("Clear of the target", false);
            break;
        }

        case MotionState::SURFACING:
        {
            // Heading and position held; depth reference rises, slower near the surface.
            const double rate = std::clamp(ASCENT_SLOW_GAIN * (depth - SURFACE_DEPTH),
                                           ASCENT_RATE_FINAL, ASCENT_RATE);
            // The reference never leads the robot by more than ASCENT_MAX_LEAD,
            // so the heavy hull cannot build up speed and overshoot the surface.
            depthRef = std::max(SURFACE_DEPTH,
                                std::max(depthRef, depth - ASCENT_MAX_LEAD) - rate * dt);
            if (std::fabs(depth - SURFACE_DEPTH) <= SURFACED_TOL &&
                std::fabs(heaveVel) < HEAVE_SETTLED_VEL)
            {
                std::cout << "[MISSION] Surfaced (t=" << simTime << " s)"
                          << (missionAborted ? " - mission ABORTED, target not reached." : " - mission complete.")
                          << std::endl;
                EnterState(MotionState::SURFACED);
            }
            break;
        }

        case MotionState::SURFACED:
            depthRef = SURFACE_DEPTH;
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
        }
        if (holdValid)
        {
            const double along = (holdX - posX) * std::cos(yaw) + (holdY - posY) * std::sin(yaw);
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
    if (haveRange && !finishing && bodyVx > 1.5 * APPROACH_CREEP_SPEED)
    {
        const double stoppingDistance = bodyVx * bodyVx / (2.0 * BRAKE_DECEL_GUARD);
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
    surgeCommand = SurgeSpeedControl(speedTarget, bodyVx, dt);

    // ---------------- Control: yaw PD(+I) ----------------
    const double yawError = WrapAngle(yawRef - yaw);
    if (YAW_KI > 0.0)
        yawIntegral = std::clamp(yawIntegral + yawError * dt, -0.1 / YAW_KI, 0.1 / YAW_KI);
    const double yawCommand = std::clamp(
        YAW_KP * yawError - YAW_KD * yawRate + YAW_KI * yawIntegral, -YAW_MAX, YAW_MAX);

    // ---------------- Control: depth PID ----------------
    const double depthError = depthRef - depth;
    const double heaveUnsat =
        DEPTH_KP * depthError - DEPTH_KD * heaveVel + DEPTH_KI * depthIntegral;
    if (DEPTH_KI > 0.0 && (std::fabs(heaveUnsat) < HEAVE_MAX || heaveUnsat * depthError < 0.0))
        depthIntegral = std::clamp(depthIntegral + depthError * dt,
                                   -DEPTH_I_LIMIT / DEPTH_KI, DEPTH_I_LIMIT / DEPTH_KI);
    const double heaveCommand = std::clamp(heaveUnsat, -HEAVE_MAX, HEAVE_MAX);

    // ---------------- Control: roll hold (middle thrusters) ----------------
    const double rollCommand = std::clamp(-ROLL_KP * roll - ROLL_KD * rollRate, -ROLL_MAX, ROLL_MAX);

    // Thrust allocation over the six T200s (sway is not used by the mission).
    SetAxisCommands(surgeCommand, 0.0, heaveCommand, rollCommand, yawCommand);

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
    if (controlDebugPublisher)
    {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        sf::Vector3 truth(nan, nan, nan);
        if (trackedObject != nullptr)
            truth = trackedObject->getTransform().getOrigin();
        std_msgs::msg::Float64MultiArray msg;
        msg.data = {
            static_cast<double>(static_cast<int>(motionState)), simTime,
            posX, posY, depth, yaw, yawRef, yawRate,
            depthRef, heaveVel, yawCommand, heaveCommand, surgeCommand,
            targetFresh ? 1.0 : 0.0, targetConfidence, targetErrorX, targetErrorY,
            hErr, vErr,
            camHorizontalIsX ? 1.0 : 0.0, camHSign, camVSign,
            camHVerified ? 1.0 : 0.0, camVVerified ? 1.0 : 0.0,
            surgeSign, yawSign, heaveSign, maxOvershoot, lastAlignmentTime,
            haveRange ? filteredRange : -1.0, haveRange ? frontGap : -1.0, bboxWidth,
            speedRef, bodyVx,
            missionAborted ? 1.0 : 0.0, detectorAlive ? 1.0 : 0.0, haveRange ? 1.0 : 0.0,
            truth.getX(), truth.getY(), truth.getZ(), bboxHeight, terminalLock ? 1.0 : 0.0};
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

    const bool done = motionState == MotionState::SURFACED && stateTime >= 5.0;
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
 * MOVE SIM TARGET FROM REAL CAMERA (case 6 only)
 * ========================================================= */
void Task2Manager::UpdateTrackedObject()
{
    if (activeCase != 6)
        return;
    if (trackedObject == nullptr || !objectDetected)
        return;

    constexpr double targetAnchorX = 8.0;
    constexpr double targetAnchorY = 0.0;
    constexpr double targetAnchorZ = 1.5;
    constexpr double targetMotionY = 1.5;
    constexpr double targetMotionZ = 3.0;

    const double pixelX = std::clamp(objectPixelX, 0.0, REAL_CAMERA_WIDTH);
    const double pixelY = std::clamp(objectPixelY, 0.0, REAL_CAMERA_HEIGHT);
    const double normalizedHorizontal = (pixelX - REAL_CAMERA_WIDTH * 0.5) / (REAL_CAMERA_WIDTH * 0.5);
    const double normalizedVertical   = (pixelY - REAL_CAMERA_HEIGHT * 0.5) / (REAL_CAMERA_HEIGHT * 0.5);

    // Keep the target between the case-2 and case-3 depths: never above the
    // surface or inside the seabed.
    constexpr double targetMinZ = 0.65;
    constexpr double targetMaxZ = 3.2;

    trackedObject->setTransform(sf::Transform(
        sf::Quaternion(0.0, M_PI, 0.0),
        sf::Vector3(
            targetAnchorX,
            targetAnchorY + normalizedVertical * targetMotionY,
            std::clamp(targetAnchorZ + normalizedHorizontal * targetMotionZ,
                       targetMinZ, targetMaxZ))));
}