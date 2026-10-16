#include "Task2Manager.h"

#include <entities/statics/Obstacle.h>
#include <entities/solids/Box.h>
#include <entities/solids/Cylinder.h>
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

    sf::Thruster* boostLeftThruster = nullptr;
    sf::Thruster* boostRightThruster = nullptr;

    // ---------------- Vehicle / sensor configuration ----------------
    constexpr double THRUSTER_MAX_OMEGA = 150.0;    // setpoint 1.0 = 150 rad/s
    // Counter-rotating pairs cancel propeller torque. A left-handed propeller
    // pushes the other way for the same spin command, so its setpoint is
    // inverted (as an ESC would be configured) - otherwise the vertical pair,
    // which shares one command, cancels out and the robot cannot change depth.
    constexpr bool   COUNTER_ROTATING_PAIRS = true;
    constexpr double ODOMETRY_RATE_HZ = 100.0;

    constexpr double CAMERA_WIDTH = 640.0;
    constexpr double CAMERA_HEIGHT = 480.0;
    constexpr double CAMERA_HFOV_DEG = 70.0;

    // ---------------- Self-checks ----------------
    // 1) At startup the robot pulses each thruster group and measures, from
    //    odometry, which way it actually moves. Wrong thruster signs are the
    //    usual reason a robot drives AWAY from the target.
    constexpr bool   ENABLE_ACTUATOR_CALIBRATION = true;
    constexpr double CALIB_COMMAND = 0.35;
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
    // simulator (yellow_sensor_model, 1.2-6 m: YOLO boxes 0.406, HSV mask 0.402;
    // rounded down so errors make the robot stop early). Set for the real target.
    constexpr double TARGET_WIDTH_M = 0.40;
    constexpr double TARGET_HEIGHT_M = 1.71;
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
    constexpr double CAMERA_TO_FRONT = 0.275;       // camera x=1.05, front housing ends at x=1.325

    // ---------------- Approach / mission ----------------
    // Gap = distance from the robot's front face to the target's front face.
    constexpr double STOP_GAP = 0.20;               // stop this far from the target
    constexpr double REACH_TOL = 0.03;
    constexpr double MIN_SAFE_GAP = 0.10;           // closer than this -> gently move away
    // Terminal approach: closer than this the target fills the camera view and
    // detections become unreliable (YOLO below ~1 m: ~30% dropouts, box
    // centre jumps 30-60 px; solid above 1 m). Heading and depth are locked and the rest is driven on
    // odometry; detections can then only shorten the range estimate.
    constexpr double TERMINAL_GAP = 1.20;
    constexpr double TERMINAL_MAX_SEC = 60.0;
    // The terminal lock only engages from a good track: this long of
    // uninterrupted confident detections. Otherwise stop, then retreat.
    constexpr double GOOD_TRACK_CONFIDENCE = 0.8;
    constexpr double TERMINAL_MIN_TRACK_SEC = 2.0;
    constexpr double TERMINAL_WAIT_SEC = 4.0;
    // Inside this gap detections are not trusted for (re)acquisition: a target
    // lost here is never searched for on the spot. The robot backs away to
    // RETREAT_GAP, where detection is reliable, and looks again.
    constexpr double RELIABLE_GAP = 1.5;
    constexpr double RETREAT_GAP = 1.8;
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
    // Braking, measured in the simulator: thrust gives 0.111 m/s^2 per unit
    // command, so SURGE_MAX brakes at ~0.04 m/s^2 (+ drag ~0.007). The speed
    // profile plans with less than half of that; the stopping guard below
    // brakes at once if the robot's ACTUAL speed could not stop in time.
    constexpr double BRAKE_DECEL_PLAN = 0.02;       // m/s^2, v <= sqrt(2 a d)
    constexpr double BRAKE_DECEL_GUARD = 0.035;     // m/s^2, stopping-distance guard
    constexpr double BRAKE_GUARD_MARGIN = 0.05;     // m
    constexpr double REACHED_SPEED = 0.03;          // m/s, must be this slow to count as reached
    constexpr double LOST_REACHED_MARGIN = 0.25;    // lost this close to STOP_GAP -> reached
    constexpr double REACHED_HOLD_SEC = 3.0;
    constexpr double BACKOFF_DISTANCE = 0.50;       // back away before ascending (0 = off)
    constexpr double BACKOFF_SPEED = 0.20;          // m/s max, backing away (BACK_OFF / RETREAT)
    constexpr double BACKOFF_TIMEOUT_SEC = 30.0;
    // Hull-centre depth when surfaced: top sensor and mast are out of the water,
    // the main body stays submerged (if it breaks the surface it loses more
    // buoyancy than the vertical thrusters can hold and falls back).
    constexpr double SURFACE_DEPTH = 0.80;
    constexpr double ASCENT_RATE = 0.10;            // m/s depth reference ramp
    constexpr double ASCENT_RATE_FINAL = 0.03;      // m/s close to the surface
    constexpr double ASCENT_SLOW_GAIN = 0.25;       // (m/s) per m above SURFACE_DEPTH
    constexpr double ASCENT_MAX_LEAD = 0.15;        // m, reference ahead of the robot
    constexpr double SURFACED_TOL = 0.05;
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

    // Surge: speed loop on odometry body velocity (thrust fraction per m/s).
    constexpr double SURGE_KFF = 0.10;
    // Acceleration feed-forward = 1 / (0.111 m/s^2 per unit command): thrust
    // (or brake) exactly when the speed reference changes, instead of
    // waiting for the heavy hull to fall behind it.
    constexpr double SURGE_KA = 9.0;
    constexpr double SURGE_KP = 2.00;
    constexpr double SURGE_KI = 0.10;
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

    // FD thrust ~ omega^2, so sqrt makes the command proportional to thrust.
    double ThrustFractionToSetpoint(double fraction)
    {
        fraction = std::clamp(fraction, -1.0, 1.0);
        return std::copysign(std::sqrt(std::fabs(fraction)), fraction);
    }

    const char* StateName(int s)
    {
        static const char* names[] = {
            "SEARCH", "YAW_ALIGN", "YAW_SETTLE", "VERTICAL_ALIGN", "VERIFY_CENTER",
            "APPROACH", "TARGET_LOST_HOLD", "CALIBRATE", "STOPPED",
            "TARGET_REACHED", "BACK_OFF", "SURFACING", "SURFACED", "RETREAT"};
        return (s >= 0 && s <= 13) ? names[s] : "?";
    }

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
    CreateMaterial("AUVMaterial", 1000.0, 0.3);
    CreateMaterial("TargetMaterial", 1000.0, 0.3);
    CreateMaterial("Rock", 2700.0, 0.3);
    CreateMaterial("Ballast", 7800.0, 0.3);   // low  -> CG below CB (righting moment)
    CreateMaterial("Foam", 400.0, 0.3);       // high

    const std::vector<std::string> materials = {
        "AUVMaterial", "TargetMaterial", "Rock", "Ballast", "Foam"};
    for (size_t i = 0; i < materials.size(); ++i)
        for (size_t j = i; j < materials.size(); ++j)
            SetMaterialsInteraction(materials[i], materials[j], 0.5, 0.2);

    /* ---------------- LOOKS ---------------- */
    // Yellow like the real target. A red target is invisible beyond ~4 m here:
    // water absorbs red first and the target turns into a dark silhouette.
    CreateLook("tracked_object_yellow", sf::Color::RGB(1.0f, 0.85f, 0.0f), 0.6f, 0.0f, 0.0f);
    CreateLook("robot_yellow", sf::Color::RGB(1.0f, 0.85f, 0.0f), 0.6f, 0.0f, 0.0f);
    CreateLook("robot_black", sf::Color::RGB(0.03f, 0.03f, 0.03f), 0.3f, 0.1f, 0.0f);
    CreateLook("robot_grey", sf::Color::RGB(0.25f, 0.27f, 0.30f), 0.45f, 0.25f, 0.0f);
    CreateLook("camera_blue", sf::Color::RGB(0.05f, 0.20f, 0.65f), 0.15f, 0.7f, 0.0f);
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

    /* ---------------- HULL ---------------- */
    sf::PhysicsSettings hullPhysics;
    hullPhysics.mode = sf::PhysicsMode::SUBMERGED;
    hullPhysics.collisions = true;
    hullPhysics.buoyancy = true;

    sf::Box* mainBody = new sf::Box("MainBody", hullPhysics,
        sf::Vector3(2.4, 1.3, 0.7), sf::I4(), "AUVMaterial", "robot_yellow");
    sf::Box* frontHousing = new sf::Box("FrontHousing", hullPhysics,
        sf::Vector3(0.35, 0.80, 0.45), sf::I4(), "AUVMaterial", "robot_black");
    sf::Cylinder* sensorMast = new sf::Cylinder("SensorMast", hullPhysics,
        0.12, 0.45, sf::I4(), "Foam", "robot_black");
    sf::Cylinder* topSensor = new sf::Cylinder("TopSensor", hullPhysics,
        0.16, 0.12, sf::I4(), "Foam", "camera_blue");
    sf::Box* leftRail = new sf::Box("LeftRail", hullPhysics,
        sf::Vector3(1.9, 0.10, 0.25), sf::I4(), "AUVMaterial", "robot_black");
    sf::Box* rightRail = new sf::Box("RightRail", hullPhysics,
        sf::Vector3(1.9, 0.10, 0.25), sf::I4(), "AUVMaterial", "robot_black");
    sf::Box* bottomBar = new sf::Box("BottomBar", hullPhysics,
        sf::Vector3(1.8, 0.9, 0.12), sf::I4(), "Ballast", "robot_black");

    sf::Compound* hull = new sf::Compound("Hull", hullPhysics, mainBody, sf::I4());
    hull->AddExternalPart(frontHousing, sf::Transform(sf::IQ(), sf::Vector3(1.15, 0.0, 0.0)));
    hull->AddExternalPart(sensorMast,   sf::Transform(sf::IQ(), sf::Vector3(0.0, 0.0, -0.55)));
    hull->AddExternalPart(topSensor,    sf::Transform(sf::IQ(), sf::Vector3(0.0, 0.0, -0.85)));
    hull->AddExternalPart(leftRail,     sf::Transform(sf::IQ(), sf::Vector3(0.0, -0.72, 0.0)));
    hull->AddExternalPart(rightRail,    sf::Transform(sf::IQ(), sf::Vector3(0.0, 0.72, 0.0)));
    // Ballast slightly forward so the centre of gravity sits under the centre of
    // buoyancy (the front housing moves CB forward). At x = 0 the hull trimmed
    // 7.4 deg nose-up, which tilted the camera and biased the depth alignment.
    hull->AddExternalPart(bottomBar,    sf::Transform(sf::IQ(), sf::Vector3(0.059, 0.0, 0.43)));

    const sf::Scalar neutralMass = hull->getVolume() * getOcean()->getLiquid().density;
    hull->ScalePhysicalPropertiesToArbitraryMass(neutralMass);

    /* ---------------- THRUSTERS ---------------- */
    sf::PhysicsSettings propellerPhysics;
    propellerPhysics.mode = sf::PhysicsMode::SUBMERGED;
    propellerPhysics.collisions = true;
    propellerPhysics.buoyancy = false;

    auto makeThruster = [&](const std::string& name, bool rightHand) -> sf::Thruster*
    {
        auto propeller = std::make_shared<sf::Polyhedron>(
            name + "Propeller", propellerPhysics, sf::GetDataPath() + "propeller.obj",
            1.0, sf::I4(), "AUVMaterial", "propeller");
        auto rotor = std::make_shared<sf::MechanicalPI>(1.0, 10.0, 5.0, 5.0);
        auto thrustModel = std::make_shared<sf::FDThrust>(
            0.18, 0.80, 0.80, 0.05, rightHand, getOcean()->getLiquid().density);
        // normalized setpoint: setSetpoint() takes [-1, 1]; left-handed -> inverted
        return new sf::Thruster(name, propeller, rotor, thrustModel,
                                0.18, rightHand, THRUSTER_MAX_OMEGA, !rightHand, true);
    };

    const bool rightSideRH = !COUNTER_ROTATING_PAIRS;
    leftThruster          = makeThruster("LeftThruster", true);
    rightThruster         = makeThruster("RightThruster", rightSideRH);
    boostLeftThruster     = makeThruster("BoostLeftThruster", true);
    boostRightThruster    = makeThruster("BoostRightThruster", rightSideRH);
    verticalLeftThruster  = makeThruster("VerticalLeftThruster", true);
    verticalRightThruster = makeThruster("VerticalRightThruster", rightSideRH);

    /* ---------------- SENSORS ---------------- */
    odometry = new sf::Odometry("Odometry", ODOMETRY_RATE_HZ);

    sf::ColorCamera* camera = new sf::ColorCamera(
        "FrontCamera", 640, 480, CAMERA_HFOV_DEG, 10.0);
    camera->InstallNewDataHandler([this](sf::ColorCamera* cam) { PublishCameraImage(cam); });

    /* ---------------- ROBOT ---------------- */
    sf::FeatherstoneRobot* boat = new sf::FeatherstoneRobot("AUV");
    boat->DefineLinks(hull);
    boat->BuildKinematicStructure();

    boat->AddLinkActuator(leftThruster, "Hull",
        sf::Transform(sf::IQ(), sf::Vector3(-0.85, -0.50, 0.0)));
    boat->AddLinkActuator(rightThruster, "Hull",
        sf::Transform(sf::IQ(), sf::Vector3(-0.85, 0.50, 0.0)));
    boat->AddLinkActuator(boostLeftThruster, "Hull",
        sf::Transform(sf::IQ(), sf::Vector3(-0.45, -0.28, 0.0)));
    boat->AddLinkActuator(boostRightThruster, "Hull",
        sf::Transform(sf::IQ(), sf::Vector3(-0.45, 0.28, 0.0)));
    boat->AddLinkActuator(verticalLeftThruster, "Hull",
        sf::Transform(sf::Quaternion(0.0, -M_PI_2, 0.0), sf::Vector3(0.0, -0.45, 0.35)));
    boat->AddLinkActuator(verticalRightThruster, "Hull",
        sf::Transform(sf::Quaternion(0.0, -M_PI_2, 0.0), sf::Vector3(0.0, 0.45, 0.35)));

    boat->AddLinkSensor(odometry, "Hull", sf::I4());
    boat->AddVisionSensor(camera, "Hull",
        sf::Transform(sf::Quaternion(1.57, 0.0, 1.57), sf::Vector3(1.05, 0.0, -0.05)));

    AddRobot(boat, sf::Transform(sf::IQ(), sf::Vector3(0.0, 0.0, 1.5)));

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
        << "  W/S   : forward / reverse   A/D : turn left / right\n"
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
            SetThrusterCommands(0.0, 0.0, 0.0);
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
        SetThrusterCommands(0.0, 0.0, 0.0);
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
        float rightY = SDL_GameControllerGetAxis(controller, SDL_CONTROLLER_AXIS_RIGHTY) / 32767.0f;

        const float DEADZONE = 0.10f;
        if (std::fabs(leftX) < DEADZONE)  leftX = 0.0f;
        if (std::fabs(leftY) < DEADZONE)  leftY = 0.0f;
        if (std::fabs(rightY) < DEADZONE) rightY = 0.0f;

        const float forward = -leftY;
        const float yaw = leftX;
        SetThrusterCommands(
            std::clamp(forward + yaw, -1.0f, 1.0f),
            std::clamp(forward - yaw, -1.0f, 1.0f),
            std::clamp(rightY, -1.0f, 1.0f));
        return;
    }

    float forward = 0.0f, yaw = 0.0f, vertical = 0.0f;
    if (keys[SDL_SCANCODE_W]) forward += 0.65f;
    if (keys[SDL_SCANCODE_S]) forward -= 0.65f;
    if (keys[SDL_SCANCODE_A]) yaw -= 0.40f;
    if (keys[SDL_SCANCODE_D]) yaw += 0.40f;
    if (keys[SDL_SCANCODE_R]) vertical -= 0.40f;
    if (keys[SDL_SCANCODE_F]) vertical += 0.40f;

    SetThrusterCommands(
        std::clamp(forward + yaw, -1.0f, 1.0f),
        std::clamp(forward - yaw, -1.0f, 1.0f),
        std::clamp(vertical, -1.0f, 1.0f));
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
 * Pulse left side, right side, then vertical, and measure the CHANGE in
 * body velocity from odometry. Convention required by the controller:
 *   left  > 0 -> forward (+vx)       right > 0 -> forward (+vx)
 *   vertical > 0 -> down (+heave, NED)
 * ========================================================= */
bool Task2Manager::RunActuatorCalibration(double dt, double bodyVx, double heaveVel, double yawRate)
{
    struct Step { double left, right, vertical, duration; int measure; };
    static const Step steps[] = {
        {0.0, 0.0, 0.0, 2.0, 0},                       // settle
        {CALIB_COMMAND, 0.0, 0.0, 3.0, 1},             // left side only
        {0.0, 0.0, 0.0, 4.0, 0},
        {0.0, CALIB_COMMAND, 0.0, 3.0, 2},             // right side only
        {0.0, 0.0, 0.0, 4.0, 0},
        {0.0, 0.0, CALIB_COMMAND, 2.5, 3},             // vertical
        {0.0, 0.0, -CALIB_COMMAND, 2.5, 0},            // undo the depth change
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

    ApplyThrust(s.left, s.right, s.vertical);   // RAW: no sign correction
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
    {
        leftSign = decide(bodyVx - calibStartVx, "left side surge");
        const double dYaw = yawRate - calibStartYawRate;
        if (leftSign * SignOf(dYaw) < 0.0)
            std::cout << "[CALIB] WARNING: left side turns the robot LEFT; check thruster positions." << std::endl;
    }
    else if (s.measure == 2)
    {
        rightSign = decide(bodyVx - calibStartVx, "right side surge");
        const double dYaw = yawRate - calibStartYawRate;
        if (rightSign * SignOf(dYaw) > 0.0)
            std::cout << "[CALIB] WARNING: right side turns the robot RIGHT; check thruster positions." << std::endl;
    }
    else if (s.measure == 3)
    {
        verticalSign = decide(heaveVel - calibStartHeave, "vertical (down)");
    }

    ++calibStep;
    calibStepStarted = false;

    if (calibStep >= N)
    {
        std::cout << "[CALIB] done: leftSign=" << leftSign << " rightSign=" << rightSign
                  << " verticalSign=" << verticalSign << std::endl;
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
    const double refAccel = refStep / dt;

    const double err = speedRef - bodyVx;
    const double unsat = SURGE_KFF * speedRef + SURGE_KA * refAccel +
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
        SetThrusterCommands(0.0, 0.0, 0.0);
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
    posXNow = posX;
    posYNow = posY;

    // ---------------- CALIBRATE ----------------
    if (motionState == MotionState::CALIBRATE)
    {
        if (RunActuatorCalibration(dt, bodyVx, heaveVel, yawRate))
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

    const bool targetFresh =
        targetDetected && lastDetectionTime >= 0.0 &&
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

    // Mixer (vehicle convention): left+, right- = turn right (+yaw, NED).
    SetThrusterCommands(surgeCommand + yawCommand, surgeCommand - yawCommand, heaveCommand);

    // ---------------- Telemetry ----------------
    // [0] state [1] simTime [2] x [3] y [4] depth [5] yaw [6] yawRef [7] yawRate
    // [8] depthRef [9] heaveVel [10] yawCmd [11] heaveCmd [12] surgeCmd
    // [13] fresh [14] confidence [15] e_x [16] e_y [17] hErr [18] vErr
    // [19] camHorizontalIsX [20] camHSign [21] camVSign [22] camHVerified [23] camVVerified
    // [24] leftSign [25] rightSign [26] verticalSign [27] maxOvershoot [28] lastAlignmentTime
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
            leftSign, rightSign, verticalSign, maxOvershoot, lastAlignmentTime,
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
void Task2Manager::SetThrusterCommands(
    sf::Scalar leftCommand,
    sf::Scalar rightCommand,
    sf::Scalar verticalCommand)
{
    ApplyThrust(leftCommand * leftSign, rightCommand * rightSign, verticalCommand * verticalSign);
}

void Task2Manager::ApplyThrust(double left, double right, double vertical)
{
    constexpr double HORIZONTAL_LIMIT = 0.60;
    constexpr double VERTICAL_LIMIT   = 0.60;
    constexpr double SLEW_RATE        = 1.5;   // thrust fraction per second

    const double desiredLeft  = std::clamp(left,  -1.0, 1.0) * HORIZONTAL_LIMIT;
    const double desiredRight = std::clamp(right, -1.0, 1.0) * HORIZONTAL_LIMIT;
    const double desiredVert  = std::clamp(vertical, -1.0, 1.0) * VERTICAL_LIMIT;

    const double dt = std::clamp<double>(lastStepDt, 0.0, 0.05);
    const double maxDelta = SLEW_RATE * dt;
    auto slew = [maxDelta](double current, double desired)
    {
        return current + std::clamp(desired - current, -maxDelta, maxDelta);
    };

    appliedLeft     = slew(appliedLeft, desiredLeft);
    appliedRight    = slew(appliedRight, desiredRight);
    appliedVertical = slew(appliedVertical, desiredVert);

    const double leftSp  = ThrustFractionToSetpoint(appliedLeft);
    const double rightSp = ThrustFractionToSetpoint(appliedRight);
    const double vertSp  = ThrustFractionToSetpoint(appliedVertical);

    if (leftThruster)          leftThruster->setSetpoint(leftSp);
    if (rightThruster)         rightThruster->setSetpoint(rightSp);
    if (boostLeftThruster)     boostLeftThruster->setSetpoint(leftSp);
    if (boostRightThruster)    boostRightThruster->setSetpoint(rightSp);
    if (verticalLeftThruster)  verticalLeftThruster->setSetpoint(vertSp);
    if (verticalRightThruster) verticalRightThruster->setSetpoint(vertSp);
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

    const unsigned int width = 640;
    const unsigned int height = 480;
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

    const double rawX = std::clamp(msg->point.x, 0.0, CAMERA_WIDTH);
    const double rawY = std::clamp(msg->point.y, 0.0, CAMERA_HEIGHT);
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

    const double pixelX = std::clamp(objectPixelX, 0.0, CAMERA_WIDTH);
    const double pixelY = std::clamp(objectPixelY, 0.0, CAMERA_HEIGHT);
    const double normalizedHorizontal = (pixelX - CAMERA_WIDTH * 0.5) / (CAMERA_WIDTH * 0.5);
    const double normalizedVertical   = (pixelY - CAMERA_HEIGHT * 0.5) / (CAMERA_HEIGHT * 0.5);

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