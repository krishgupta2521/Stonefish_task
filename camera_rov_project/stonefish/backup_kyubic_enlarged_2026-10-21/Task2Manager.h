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

#include <array>
#include <deque>
#include <vector>
#include <chrono>

class Task2Manager : public sf::SimulationManager
{
public:
    // scenarioCase: 1-7 (see Task2Manager.cpp). 0 = use $TASK2_CASE, else the default case.
    explicit Task2Manager(sf::Scalar stepsPerSecond, int scenarioCase = 0);
    ~Task2Manager() override;

    void BuildScenario() override;
    void SimulationStepCompleted(sf::Scalar timeStep) override;

    // Normalised axis commands in [-1, 1] (body frame, NED):
    //   surge > 0 forward, sway > 0 right, heave > 0 down, roll > 0 right side
    //   down, yaw > 0 turn right. Mapped onto the six T200s by thrust allocation;
    //   the signs measured at startup are applied here.
    void SetAxisCommands(double surge, double sway, double heave, double roll, double yaw);

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
        RETREAT = 13,
        ROV_MANUAL = 14      // ROV mode (piloted): telemetry only
    };

    // Publishing
    void PublishOdometry();
    void PublishCameraImage(sf::ColorCamera* camera);

    // ROS callbacks
    void ObjectPositionCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg);
    void TargetDetectionCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg);
    void TargetBoxCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
    void VisionFlowCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);

    // Simulation helpers
    void UpdateTrackedObject();
    void PlaceTargetInView();   // ROV: sim target at the real camera's image position
    void PrintOdometry();
    void CheckAutotestExit();

    // Control
    bool InitializeController();
    void UpdateController();
    void UpdateAutonomousControl(double dt);
    void ResetAutonomousControl();
    void EnterState(MotionState next);
    void BeginSurfacing(const char* reason, bool aborted);
    bool RunActuatorCalibration(double dt, double bodyVx, double heaveVel, double yawRate,
                                double depthHold);
    void ProcessDetection(double yaw, double depth, double pitch);
    void UpdateCameraAxisCheck();
    double SurgeSpeedControl(double speedTarget, double bodyVx, double dt,
                             double accel = 0.0, double decel = 0.0);   // 0 = mission ramps
    void BuildAllocation();
    void ApplyAxisCommands(const double cmd[5]);   // raw, no sign correction
    // Depth observer (Kalman filter on depth, vertical speed, acceleration bias)
    void ObserverPredict(double dt, double bodyVx);
    bool ObserverUpdate(int index, double measurement, double variance, bool gate = true);
    void UpdateDepthObserver(double dt, double pitchRate, double bodyVx, double depthTruth,
                             double heaveVelTruth);
    void AppendThrustTelemetry(std::vector<double>& data, double depth, double heaveVel);
    void UpdateRovControl(const Uint8* keys);
    bool ObserveTargetHeight(double targetDz, double lever);
    // Navigation without IMU / DVL: heading and forward-speed observers
    void UpdateNavigation(double dt);
    void ObserveTargetRange(double range);

    // Scenario
    int activeCase = 1;
    bool rovMode = false;                // case 8: piloted, no autonomy

    // ROV mode
    bool rovHeadingHold = false;
    double rovHeadingRef = 0.0;
    double rovFrontGap = -1.0;           // from the bulb's box size, -1 = unknown
    double rovLastGuardPrint = -1e9;
    double rovLastGapBoxTime = -1.0;
    int rovGapFartherCount = 0;
    double rovViewCorrX = 0.0;           // view matching: correction of the target placement
    double rovViewCorrY = 0.0;
    bool rovTargetDepthLimited = false;
    bool rovTargetFrozen = true;         // target fixed in the world (real object still)
    double rovFrozenX = 1e9, rovFrozenY = 1e9;   // real image position when frozen (none yet)
    double rovViewOffsetX = 0.0, rovViewOffsetY = 0.0;   // sim view position - real, while moving
    double rovStillRefX = 0.0, rovStillRefY = 0.0, rovStillTime = 0.0;
    bool rovDepthHold = false;
    double rovHeaveTrim = 0.0;           // learned hover thrust (camera only, held while turning)
    double rovDepthRef = 0.0;

    // ROS
    rclcpp::Node::SharedPtr rosNode;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odomPublisher;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr imagePublisher;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr controlDebugPublisher;
    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr objectPositionSubscriber;
    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr targetDetectionSubscriber;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr targetBoxSubscriber;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr visionFlowSubscriber;

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

    // Thrusters: 6 x T200 in kyubic::THRUSTERS order, and the thrust allocation
    std::array<sf::Thruster*, 6> thrusters{};
    std::array<double, 6> appliedThrust{};   // fraction of max thrust, after limit and slew
    double allocation[6][5] = {};            // pseudo-inverse: thrust per axis wrench
    double axisScale[5] = {};

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
    double surgeSign = 1.0;
    double yawSign = 1.0;
    double heaveSign = 1.0;
    int calibStep = 0;
    double calibStepTime = 0.0;
    bool calibStepStarted = false;
    double calibStartVx = 0.0;
    double calibStartHeave = 0.0;
    double calibStartYawRate = 0.0;
    double calibDepthRef = -1.0;

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
    double searchTurnedModel = 0.0;      // the same turn from the thrust model (camera only)
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
    double surgeRefAccel = 0.0;    // filtered rate of the speed reference

    double yawIntegral = 0.0;
    double depthIntegral = 0.0;
    double surgeCommand = 0.0;

    bool missionAborted = false;

    // ---- Depth from the camera (no depth sensor) ----
    bool useVisionDepth = true;          // false: simulator truth ($TASK2_DEPTH_SOURCE=sensor)
    double visionFlowB = 0.0;            // vertical image shift [px/s] (visual_heave_estimator.py)
    double visionFlowFeatures = 0.0;
    double visionFlowTime = -1.0;        // sim time of the latest flow message
    bool visionFlowNew = false;
    double visionVz = 0.0;               // vertical speed estimate [m/s], + = down
    double visionDepth = 0.0;            // integrated depth estimate [m]
    bool visionDepthOk = false;          // flow arriving: depth estimate is being updated
    double surfaceStillTime = 0.0;       // SURFACING: time the robot has stopped rising
    bool surfacePassive = false;         // SURFACING: vertical thrusters off, settling check
    bool verticalFault = false;          // vertical control not trusted: vertical thrusters off
    double kfX[3] = {0.0, 0.0, 0.0};     // observer state: depth [m], vz [m/s], accel. bias [m/s^2]
    double kfP[3][3] = {};
    double heaveForceCmd = 0.0;          // commanded vertical thrust through the T200 curve [N], lagged
    double targetAnchor = 0.0;           // target depth in the observer's frame [m]
    bool targetAnchorValid = false;
    int anchorRejects = 0;
    double lastTargetUpdateTime = -1.0;  // sim time of the last accepted target height measurement
    double flowVz = 0.0;                 // vertical speed from the camera flow alone [m/s]
    double flowVzSlow = 0.0;             // the same, averaged over the rocking period
    double heaveForcePerCmd = 0.0;       // vertical thrust [N] for a heave command of 1
    double heaveTrim = 0.0;              // vertical thrust that holds the robot level (learned)
    double depthRefSlewed = 0.0;         // depth reference after the vertical speed limit

    // ---- Navigation without IMU / DVL (camera + thrust models) ----
    struct Kf3
    {
        double x[3] = {0.0, 0.0, 0.0};   // position-like, rate, acceleration bias
        double P[3][3] = {{0.01, 0, 0}, {0, 0.0025, 0}, {0, 0, 1e-4}};
        void Predict(double dt, double accel, double dAccelDv, double qAccel, double qBias);
        bool Update(int i, double z, double var, bool gate);
    };
    bool useSensorNav = false;           // true: simulator IMU/odometry ($TASK2_NAV_SOURCE=sensor)
    Kf3 yawKf;                           // [heading, yaw rate, bias]
    Kf3 surgeKf;                         // [distance travelled, forward speed, bias]
    double surgeForceCmd = 0.0, yawTorqueCmd = 0.0;   // commanded, through the thrust curves, lagged
    double swayForceCmd = 0.0;
    double swayVelModel = 0.0;           // sideways speed from the thrust model [m/s]
    double surgeVelModel = 0.0;          // forward speed from the thrust model alone [m/s]
    double yawRateModel = 0.0;           // turn rate from the thrust model alone [rad/s]
    double navX = 0.0, navY = 0.0;       // dead-reckoned position (observer frame)
    double navYaw = 0.0, navYawRate = 0.0, navVx = 0.0;
    double navRoll = 0.0, navPitch = 0.0, navRollRate = 0.0, navPitchRate = 0.0;
    double visionFlowC = 0.0, visionFlowD = 0.0;   // horizontal shift [px/s], divergence [1/s]
    bool visionFlowHorizNew = false;
    double flowYawRate = 0.0, flowSurgeSign = 0.0;   // camera-only measurements (calibration)
    double rangeAnchor = 0.0;
    bool rangeAnchorValid = false;
    int rangeAnchorRejects = 0;
    double lastRangeUpdateTime = -1.0;
    bool rangeHoldValid = false;         // stationary hold on the camera range (no DVL)
    double rangeGoalGap = -1.0;          // BACK_OFF / RETREAT goal on the camera range
    std::deque<std::pair<double, double>> rangeHistory;   // (sim time, camera range)
    double rangeRateSpeed = 0.0, rangeRateTime = -1.0;   // camera-measured closing speed
    double rangeHoldGap = 0.0;

    // ---- Timing for the system log (wall clock, epoch seconds) ----
    double controlWallTime = 0.0;        // start of the latest control step
    double thrusterWallTime = 0.0;       // thruster setpoints applied
    double detectionRxWallTime = 0.0;    // /stonefish/target/detection received
    double detectionImageStamp = 0.0;    // its image header stamp (frame identity)
    double objectRxWallTime = 0.0;       // /object/position (real camera) received
    double objectImageStamp = 0.0;
    double targetMovedWallTime = 0.0;    // real-camera detection applied to the sim target
    double targetMovedImageStamp = 0.0;
    double pilotInput[4] = {0.0, 0.0, 0.0, 0.0};   // ROV: forward, sideways, vertical, turn

    // ---- Metrics for the evaluation section ----
    double acquisitionStartTime = -1.0;
    double initialErrorSign = 0.0;
    double maxOvershoot = 0.0;
    int correctiveTurns = 0;
    double lastErrorSign = 0.0;
    double lastAlignmentTime = -1.0;
    double lostSinceTime = -1.0;
};
