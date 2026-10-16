#pragma once

#include <core/SimulationManager.h>
#include <core/FeatherstoneRobot.h>
#include <entities/statics/Obstacle.h>
#include <actuators/Thruster.h>
#include <sensors/scalar/Odometry.h>
#include <sensors/vision/ColorCamera.h>

#ifdef Max
#undef Max
#endif

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <SDL2/SDL.h>

#include <chrono>

class Task2Manager : public sf::SimulationManager
{
public:
    // scenarioCase: 1-7 (see Task2Manager.cpp). 0 = use $TASK2_CASE, else the default case.
    explicit Task2Manager(sf::Scalar stepsPerSecond, int scenarioCase = 0);
    ~Task2Manager() override;

    void BuildScenario() override;
    void SimulationStepCompleted(sf::Scalar timeStep) override;

    // Commands are FRACTIONS OF MAX THRUST in [-1, 1], in the VEHICLE convention:
    //   left/right > 0  -> that side pushes the robot FORWARD
    //   vertical   > 0  -> pushes the robot DOWN (+Z, NED)
    // Measured actuator signs (startup calibration) are applied here.
    void SetThrusterCommands(
        sf::Scalar leftCommand,
        sf::Scalar rightCommand,
        sf::Scalar verticalCommand);

private:
    // Research motion states (numbers match /usv/control_debug[0])
    enum class MotionState : int
    {
        SEARCH = 0,
        YAW_ALIGN = 1,
        YAW_SETTLE = 2,
        VERTICAL_ALIGN = 3,
        VERIFY_CENTER = 4,
        APPROACH = 5,
        TARGET_LOST_HOLD = 6,
        CALIBRATE = 7,
        STOPPED = 8,
        TARGET_REACHED = 9,
        BACK_OFF = 10,
        SURFACING = 11,
        SURFACED = 12,
        RETREAT = 13
    };

    // Publishing
    void PublishOdometry();
    void PublishCameraImage(sf::ColorCamera* camera);

    // ROS callbacks
    void ObjectPositionCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg);
    void TargetDetectionCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg);
    void TargetBoxCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);

    // Simulation helpers
    void UpdateTrackedObject();
    void PrintOdometry();
    void CheckAutotestExit();

    // Control
    bool InitializeController();
    void UpdateController();
    void UpdateAutonomousControl(double dt);
    void ResetAutonomousControl();
    void EnterState(MotionState next);
    void BeginSurfacing(const char* reason, bool aborted);
    bool RunActuatorCalibration(double dt, double bodyVx, double heaveVel, double yawRate);
    void ProcessDetection(double yaw, double depth, double pitch);
    void UpdateCameraAxisCheck();
    double SurgeSpeedControl(double speedTarget, double bodyVx, double dt);
    void ApplyThrust(double left, double right, double vertical);  // raw, no sign correction

    // Scenario
    int activeCase = 1;

    // ROS
    rclcpp::Node::SharedPtr rosNode;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odomPublisher;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr imagePublisher;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr controlDebugPublisher;
    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr objectPositionSubscriber;
    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr targetDetectionSubscriber;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr targetBoxSubscriber;

    // Simulation entities
    sf::Odometry* odometry = nullptr;
    sf::Obstacle* trackedObject = nullptr;

    // Real-camera object (case 6)
    double objectPixelX = 320.0;
    double objectPixelY = 240.0;
    double objectConfidence = 0.0;
    bool objectDetected = false;

    // Latest Stonefish-camera detection (x = e_x, y = e_y in pixels, z = confidence)
    bool autonomousMode = false;
    bool targetDetected = false;
    double targetErrorX = 0.0;
    double targetErrorY = 0.0;
    double targetConfidence = 0.0;
    bool newDetectionPending = false;
    double lastDetectionTime = -1.0;     // sim time of the last confident detection
    double lastDetectorMsgTime = -1.0;   // sim time of ANY detector message (heartbeat)
    double autonomyStartTime = -1.0;     // sim time calibration finished
    double lastDetectorWarnTime = -1.0;

    // Latest bounding box of the detection (pixels)
    double bboxWidth = 0.0;
    double bboxHeight = 0.0;
    bool bboxClippedH = false;   // touches left/right image border
    bool bboxClippedV = false;   // touches top/bottom image border
    double lastBboxTime = -1.0;

    // Thrusters (boost pair lives in the .cpp anonymous namespace)
    sf::Thruster* leftThruster = nullptr;
    sf::Thruster* rightThruster = nullptr;
    sf::Thruster* verticalLeftThruster = nullptr;
    sf::Thruster* verticalRightThruster = nullptr;
    double appliedLeft = 0.0;
    double appliedRight = 0.0;
    double appliedVertical = 0.0;

    // Manual controller
    SDL_GameController* controller = nullptr;
    bool controllerConnected = false;

    // Timing
    sf::Scalar odometryTimer = 0.0;
    sf::Scalar lastStepDt = 0.01;
    double simTime = 0.0;
    double autotestMaxTime = -1.0;   // $TASK2_AUTOTEST_MAX_TIME, quits the app when done
    bool autotestQuitSent = false;

    // ---- Actuator sign calibration (measured from odometry at startup) ----
    double leftSign = 1.0;
    double rightSign = 1.0;
    double verticalSign = 1.0;
    int calibStep = 0;
    double calibStepTime = 0.0;
    bool calibStepStarted = false;
    double calibStartVx = 0.0;
    double calibStartHeave = 0.0;
    double calibStartYawRate = 0.0;

    // ---- Camera axis mapping (verified online from robot motion) ----
    bool camHorizontalIsX = true;   // false = image rotated 90 deg (axes swapped)
    double camHSign = 1.0;          // +: e_x > 0 means target to the RIGHT
    double camVSign = 1.0;          // +: e_y > 0 means target BELOW
    bool camHVerified = false;
    bool camVVerified = false;
    double evSx = 0.0, evSy = 0.0, evPsiSq = 0.0, evPsiAbs = 0.0;
    double evVz = 0.0, evZSq = 0.0, evZAbs = 0.0;
    bool havePrevDetection = false;
    double prevDetTime = 0.0, prevDetYaw = 0.0, prevDetDepth = 0.0;
    double prevErrX = 0.0, prevErrY = 0.0;

    // ---- Guidance / control state ----
    MotionState motionState = MotionState::TARGET_LOST_HOLD;
    double stateTime = 0.0;
    double settledTime = 0.0;
    double stallTime = 0.0;
    bool approachMoving = false;
    bool terminalLock = false;        // final approach on odometry, vision locked out
    double terminalStartTime = 0.0;
    double terminalWaitTime = 0.0;
    double goodTrackSince = -1.0;     // start of the current run of confident detections
    bool retreatGoalSet = false;
    bool brakeGuardActive = false;
    double searchTime = 0.0;
    double searchTurned = 0.0;
    double prevSearchYaw = 0.0;
    double searchDirection = 1.0;

    bool refsInitialized = false;
    double yawRef = 0.0;
    double depthRef = 1.5;

    bool haveTargetEstimate = false;   // memory of the target in WORLD terms
    double filteredTargetYaw = 0.0;
    double filteredTargetDepth = 1.5;
    int outlierCount = 0;

    // Range (camera -> target front face), from bbox size + odometry prediction
    bool haveRange = false;
    double filteredRange = 0.0;
    int rangeJumpCount = 0;
    double lostGap = 1e9;

    // Surge: position hold along the heading, cascaded with a speed loop
    double posXNow = 0.0;
    double posYNow = 0.0;
    bool holdValid = false;
    double holdX = 0.0;
    double holdY = 0.0;
    double speedRef = 0.0;
    double surgeIntegral = 0.0;

    double yawIntegral = 0.0;
    double depthIntegral = 0.0;
    double surgeCommand = 0.0;

    bool missionAborted = false;

    // ---- Metrics for the evaluation section ----
    double acquisitionStartTime = -1.0;
    double initialErrorSign = 0.0;
    double maxOvershoot = 0.0;
    int correctiveTurns = 0;
    double lastErrorSign = 0.0;
    double lastAlignmentTime = -1.0;
    double lostSinceTime = -1.0;
};
