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

#include <cmath>

#include <cstring>

#include <iostream>

#include <algorithm>

/* =========================================================

 * CONSTRUCTOR

 * ========================================================= */

Task2Manager::Task2Manager(sf::Scalar stepsPerSecond)

    : sf::SimulationManager(

          stepsPerSecond,

          sf::Solver::SI,

          sf::CollisionFilter::EXCLUSIVE)

{

}

/* =========================================================

 * DESTRUCTOR

 * ========================================================= */

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

    /* =====================================================

     * MATERIALS

     * ===================================================== */

    CreateMaterial(

        "AUVMaterial",

        1000.0,

        0.3);

    CreateMaterial(

        "TargetMaterial",

        1000.0,

        0.3);

    CreateMaterial(

        "Rock",

        2700.0,

        0.3);

    /* =====================================================

     * MATERIAL INTERACTIONS

     * ===================================================== */

    SetMaterialsInteraction(

        "AUVMaterial",

        "AUVMaterial",

        0.5,

        0.2);

    SetMaterialsInteraction(

        "AUVMaterial",

        "TargetMaterial",

        0.5,

        0.2);

    SetMaterialsInteraction(

        "AUVMaterial",

        "Rock",

        0.5,

        0.2);

    SetMaterialsInteraction(

        "TargetMaterial",

        "Rock",

        0.5,

        0.2);

    /* =====================================================

     * ROBOT LOOKS

     * ===================================================== */

    CreateLook(

        "tracked_object_yellow",

        sf::Color::RGB(1.0f, 0.0f, 0.0f),

            0.6f,

            0.0f,

            0.0f

);

    CreateLook(

        "robot_yellow",

        sf::Color::RGB(

            1.0f,

            0.85f,

            0.0f),

        0.6f,

        0.0f,

        0.0f

);

    CreateLook(

        "robot_black",

        sf::Color::RGB(

            0.03f,

            0.03f,

            0.03f),

        0.3f,

        0.1f,

        0.0f);

    CreateLook(

        "robot_grey",

        sf::Color::RGB(

            0.25f,

            0.27f,

            0.30f),

        0.45f,

        0.25f,

        0.0f);

    CreateLook(

        "camera_blue",

        sf::Color::RGB(

            0.05f,

            0.20f,

            0.65f),

        0.15f,

        0.7f,

        0.0f);

    CreateLook(

        "propeller",

        sf::Color::RGB(

            0.05f,

            0.05f,

            0.05f),

        0.3f,

        0.0f,

        0.0f);

    CreateLook(

        "TargetRed",

        sf::Color::RGB(

            1.0f,

            0.05f,

            0.05f),

        0.5f,

        0.0f,

        0.0f);

    CreateLook(

        "seabed",

        sf::Color::RGB(

            0.45f,

            0.35f,

            0.20f),

        0.9f,

        0.0f,

        0.0f);

    /* =====================================================

     * OCEAN

     * ===================================================== */

    EnableOcean(0.0);

    getOcean()->setWaterType(0.2);

    /*

     * Water current.

     *

     * X direction:

     * 0.1 m/s

     */

    getOcean()->AddVelocityField(

        new sf::Uniform(

            sf::Vector3(

                0.0,

                0.0,

                0.0)));

    getOcean()->EnableCurrents();

    /*

     * Sun position.

     */

    getAtmosphere()->SetSunPosition(

        0.0,

        60.0);

    /* =====================================================

     * SEABED

     *

     * Stonefish uses NED coordinates:

     *

     * Z = 0 m  -> water surface

     * Z = 5 m  -> seabed

     *

     * The obstacle is centered at Z=5.25 with a

     * thickness of 0.5 m.

     *

     * Therefore its upper surface is at Z=5.0.

     * ===================================================== */

    sf::Obstacle* seabed =

        new sf::Obstacle(

            "Seabed",

            sf::Vector3(

                30.0,

                30.0,

                0.5),

            sf::I4(),

            "Rock",

            "seabed");

    AddStaticEntity(

        seabed,

        sf::Transform(

            sf::IQ(),

            sf::Vector3(

                0.0,

                0.0,

                5.25)));

    // =====================================================

// CUSTOM 3D OBJECT

// =====================================================

std::string objectPath =

    "/home/krish/Documents/Development/kyushu_project/camera_rov_project/stonefish/models/yellow_sensor_model.obj";

trackedObject =

    new sf::Obstacle(

        "TrackedObject",

        objectPath,

        0.5,

        sf::I4(),

        objectPath,

        0.5,

        sf::I4(),

        true,

        "TargetMaterial",

        "tracked_object_yellow");

AddStaticEntity(

    trackedObject,

    sf::Transform(

        sf::Quaternion(0.0,3.141592653589793, 0.0),

        sf::Vector3(

            3.0,

            0.0,

            1.5

        )

    )

);

    /* =====================================================

     * ROBOT PHYSICS SETTINGS

     * ===================================================== */

    sf::PhysicsSettings hullPhysics;

    hullPhysics.mode =

        sf::PhysicsMode::SUBMERGED;

    hullPhysics.collisions =

        true;

    hullPhysics.buoyancy =

        true;

    /* =====================================================

     * MAIN ROBOT BODY

     *

     * Dimensions:

     *

     * X = 2.4 m

     * Y = 1.3 m

     * Z = 0.7 m

     * ===================================================== */

    sf::Box* mainBody =

        new sf::Box(

            "MainBody",

            hullPhysics,

            sf::Vector3(

                2.4,

                1.3,

                0.7),

            sf::I4(),

            "AUVMaterial",

            "robot_yellow");

    /* =====================================================

     * FRONT SENSOR HOUSING

     * ===================================================== */

    sf::Box* frontHousing =

        new sf::Box(

            "FrontHousing",

            hullPhysics,

            sf::Vector3(

                0.35,

                0.80,

                0.45),

            sf::I4(),

            "AUVMaterial",

            "robot_black");

    /* =====================================================

     * TOP SENSOR MAST

     *

     * Cylinder axis is Z.

     * In NED, negative Z is upward.

     * ===================================================== */

    sf::Cylinder* sensorMast =

        new sf::Cylinder(

            "SensorMast",

            hullPhysics,

            0.12,

            0.45,

            sf::I4(),

            "AUVMaterial",

            "robot_black");

    /* =====================================================

     * TOP SENSOR

     * ===================================================== */

    sf::Cylinder* topSensor =

        new sf::Cylinder(

            "TopSensor",

            hullPhysics,

            0.16,

            0.12,

            sf::I4(),

            "AUVMaterial",

            "camera_blue");

    /* =====================================================

     * LEFT PROTECTIVE RAIL

     * ===================================================== */

    sf::Box* leftRail =

        new sf::Box(

            "LeftRail",

            hullPhysics,

            sf::Vector3(

                1.9,

                0.10,

                0.25),

            sf::I4(),

            "AUVMaterial",

            "robot_black");

    /* =====================================================

     * RIGHT PROTECTIVE RAIL

     * ===================================================== */

    sf::Box* rightRail =

        new sf::Box(

            "RightRail",

            hullPhysics,

            sf::Vector3(

                1.9,

                0.10,

                0.25),

            sf::I4(),

            "AUVMaterial",

            "robot_black");

    /* =====================================================

     * BOTTOM PROTECTION BAR

     * ===================================================== */

    sf::Box* bottomBar =

        new sf::Box(

            "BottomBar",

            hullPhysics,

            sf::Vector3(

                1.8,

                0.9,

                0.12),

            sf::I4(),

            "AUVMaterial",

            "robot_black");

    /* =====================================================

     * COMPOUND ROBOT BODY

     *

     * All these components become one rigid body.

     * ===================================================== */

    sf::Compound* hull =

        new sf::Compound(

            "Hull",

            hullPhysics,

            mainBody,

            sf::I4());

    /* =====================================================

     * ADD FRONT HOUSING

     * ===================================================== */

    hull->AddExternalPart(

        frontHousing,

        sf::Transform(

            sf::IQ(),

            sf::Vector3(

                1.15,

                0.0,

                0.0)));

    /* =====================================================

     * ADD SENSOR MAST

     * ===================================================== */

    hull->AddExternalPart(

        sensorMast,

        sf::Transform(

            sf::IQ(),

            sf::Vector3(

                0.0,

                0.0,

                -0.55)));

    /* =====================================================

     * ADD TOP SENSOR

     * ===================================================== */

    hull->AddExternalPart(

        topSensor,

        sf::Transform(

            sf::IQ(),

            sf::Vector3(

                0.0,

                0.0,

                -0.85)));

    /* =====================================================

     * ADD LEFT RAIL

     * ===================================================== */

    hull->AddExternalPart(

        leftRail,

        sf::Transform(

            sf::IQ(),

            sf::Vector3(

                0.0,

                -0.72,

                0.0)));

    /* =====================================================

     * ADD RIGHT RAIL

     * ===================================================== */

    hull->AddExternalPart(

        rightRail,

        sf::Transform(

            sf::IQ(),

            sf::Vector3(

                0.0,

                0.72,

                0.0)));

    /* =====================================================

     * ADD BOTTOM BAR

     * ===================================================== */

    hull->AddExternalPart(

        bottomBar,

        sf::Transform(

            sf::IQ(),

            sf::Vector3(

                -0.1,

                0.0,

                0.43)));

    /* =====================================================

     * NEUTRAL BUOYANCY

     *

     * Make the robot approximately neutrally buoyant.

     * ===================================================== */

    sf::Scalar neutralMass =

        hull->getVolume()

        * getOcean()->getLiquid().density;

    hull->ScalePhysicalPropertiesToArbitraryMass(

        neutralMass);

    /* =====================================================

     * PROPELLER PHYSICS

     * ===================================================== */

    sf::PhysicsSettings propellerPhysics;

    propellerPhysics.mode =

        sf::PhysicsMode::SUBMERGED;

    propellerPhysics.collisions =

        true;

    propellerPhysics.buoyancy =

        false;

    /* =====================================================

     * LEFT HORIZONTAL PROPELLER

     * ===================================================== */

    std::shared_ptr<sf::Polyhedron>

        leftPropeller =

            std::make_shared<sf::Polyhedron>(

                "LeftPropeller",

                propellerPhysics,

                sf::GetDataPath()

                    + "propeller.obj",

                1.0,

                sf::I4(),

                "AUVMaterial",

                "propeller");

    /* =====================================================

     * RIGHT HORIZONTAL PROPELLER

     * ===================================================== */

    std::shared_ptr<sf::Polyhedron>

        rightPropeller =

            std::make_shared<sf::Polyhedron>(

                "RightPropeller",

                propellerPhysics,

                sf::GetDataPath()

                    + "propeller.obj",

                1.0,

                sf::I4(),

                "AUVMaterial",

                "propeller");

    /* =====================================================

     * LEFT VERTICAL PROPELLER

     * ===================================================== */

    std::shared_ptr<sf::Polyhedron>

        verticalLeftPropeller =

            std::make_shared<sf::Polyhedron>(

                "VerticalLeftPropeller",

                propellerPhysics,

                sf::GetDataPath()

                    + "propeller.obj",

                1.0,

                sf::I4(),

                "AUVMaterial",

                "propeller");

    /* =====================================================

     * RIGHT VERTICAL PROPELLER

     * ===================================================== */

    std::shared_ptr<sf::Polyhedron>

        verticalRightPropeller =

            std::make_shared<sf::Polyhedron>(

                "VerticalRightPropeller",

                propellerPhysics,

                sf::GetDataPath()

                    + "propeller.obj",

                1.0,

                sf::I4(),

                "AUVMaterial",

                "propeller");

    /* =====================================================

     * ROTOR DYNAMICS

     * ===================================================== */

    std::shared_ptr<sf::MechanicalPI>

        leftRotorDynamics =

            std::make_shared<sf::MechanicalPI>(

                1.0,

                10.0,

                5.0,

                5.0);

    std::shared_ptr<sf::MechanicalPI>

        rightRotorDynamics =

            std::make_shared<sf::MechanicalPI>(

                1.0,

                10.0,

                5.0,

                5.0);

    std::shared_ptr<sf::MechanicalPI>

        verticalLeftRotorDynamics =

            std::make_shared<sf::MechanicalPI>(

                1.0,

                10.0,

                5.0,

                5.0);

    std::shared_ptr<sf::MechanicalPI>

        verticalRightRotorDynamics =

            std::make_shared<sf::MechanicalPI>(

                1.0,

                10.0,

                5.0,

                5.0);

    /* =====================================================

     * THRUST MODELS

     * ===================================================== */

    std::shared_ptr<sf::FDThrust>

        leftThrustModel =

            std::make_shared<sf::FDThrust>(

                    0.18,

                    0.80,

                    0.80,

                    0.05,

                true,

                getOcean()->getLiquid().density);

    std::shared_ptr<sf::FDThrust>

        rightThrustModel =

            std::make_shared<sf::FDThrust>(

                0.18,

                0.80,

                0.80,

                0.05,

                true,

                getOcean()->getLiquid().density);

    std::shared_ptr<sf::FDThrust>

        verticalLeftThrustModel =

            std::make_shared<sf::FDThrust>(

                0.18,

                0.80,

                0.80,

                0.05,

                true,

                getOcean()->getLiquid().density);

    std::shared_ptr<sf::FDThrust>

        verticalRightThrustModel =

            std::make_shared<sf::FDThrust>(

                0.18,

                0.80,

                0.80,

                0.05,

                true,

                getOcean()->getLiquid().density);

    /* =====================================================

     * FOUR THRUSTERS

     *

     * Reduced maximum setpoint from 105 to 40

     * to make manual control easier and reduce

     * aggressive flipping.

     * ===================================================== */

    leftThruster =

        new sf::Thruster(

            "LeftThruster",

            leftPropeller,

            leftRotorDynamics,

            leftThrustModel,

            0.18,

            true,

            150.0,

            false,

            true);

    rightThruster =

        new sf::Thruster(

            "RightThruster",

            rightPropeller,

            rightRotorDynamics,

            rightThrustModel,

            0.18,

            true,

            150.0,

            false,

            true);

    verticalLeftThruster =

        new sf::Thruster(

            "VerticalLeftThruster",

            verticalLeftPropeller,

            verticalLeftRotorDynamics,

            verticalLeftThrustModel,

            0.18,

            true,

            150.0,

            false,

            true);

    verticalRightThruster =

        new sf::Thruster(

            "VerticalRightThruster",

            verticalRightPropeller,

            verticalRightRotorDynamics,

            verticalRightThrustModel,

            0.18,

            true,

            150.0,

            false,

            true);

    /* =====================================================

     * ODOMETRY

     * ===================================================== */

    odometry =

        new sf::Odometry(

            "Odometry",

            5.0);

    /* =====================================================

     * CAMERA

     * ===================================================== */

    sf::ColorCamera* camera =

        new sf::ColorCamera(

            "FrontCamera",

            640,

            480,

            70.0,

            10.0);

    camera->InstallNewDataHandler(

        [this](sf::ColorCamera* cam)

        {

            PublishCameraImage(cam);

        });

    /* =====================================================

     * CREATE FEATHERSTONE ROBOT

     * ===================================================== */

    sf::FeatherstoneRobot* boat =

        new sf::FeatherstoneRobot(

            "AUV");

    boat->DefineLinks(

        hull);

    boat->BuildKinematicStructure();

    /* =====================================================

     * HORIZONTAL THRUSTER POSITIONS

     *

     * Symmetrical around Y.

     *

     * This reduces unwanted roll torque.

     * ===================================================== */

    boat->AddLinkActuator(

        leftThruster,

        "Hull",

        sf::Transform(

            sf::IQ(),

            sf::Vector3(

                -0.85,

                -0.50,

                0.0)));

    boat->AddLinkActuator(

        rightThruster,

        "Hull",

        sf::Transform(

            sf::IQ(),

            sf::Vector3(

                -0.85,

                0.50,

                0.0)));

    /* =====================================================

     * VERTICAL THRUSTER POSITIONS

     *

     * Local +X of the thruster is rotated vertically.

     *

     * Both thrusters receive identical commands.

     *

     * This minimizes roll/pitch torque.

     * ===================================================== */

    boat->AddLinkActuator(

        verticalLeftThruster,

        "Hull",

        sf::Transform(

            sf::Quaternion(

                0.0,

                -M_PI_2,

                0.0),

            sf::Vector3(

                0.0,

                -0.45,

                0.35)));

    boat->AddLinkActuator(

        verticalRightThruster,

        "Hull",

        sf::Transform(

            sf::Quaternion(

                0.0,

                -M_PI_2,

                0.0),

            sf::Vector3(

                0.0,

                0.45,

                0.35)));

    /* =====================================================

     * ODOMETRY SENSOR

     * ===================================================== */

    boat->AddLinkSensor(

        odometry,

        "Hull",

        sf::I4());

    /* =====================================================

     * FRONT CAMERA

     * ===================================================== */

    boat->AddVisionSensor(

        camera,

        "Hull",

        sf::Transform(

            sf::Quaternion(

                1.57,

                0.0,

                1.57),

            sf::Vector3(

                1.05,

                0.0,

                -0.05)));

    /* =====================================================

     * ADD ROBOT TO SIMULATION

     *

     * Start:

     *

     * X = 0

     * Y = 0

     * Z = 1.5 m

     * ===================================================== */

    AddRobot(

        boat,

        sf::Transform(

            sf::IQ(),

            sf::Vector3(

                0.0,

                0.0,

                1.5)));

    /* =====================================================

     * ROS 2 NODE

     * ===================================================== */

    rosNode =

        rclcpp::Node::make_shared(

            "task2_baseline");

    /* =====================================================

     * ODOMETRY PUBLISHER

     * ===================================================== */

    odomPublisher =

        rosNode->create_publisher<

            nav_msgs::msg::Odometry>(

                "/usv/odom",

                10);

    /* =====================================================

     * CAMERA PUBLISHER

     * ===================================================== */

    imagePublisher =

        rosNode->create_publisher<

            sensor_msgs::msg::Image>(

                "Stonefish/camera/image_raw",

                10);

    // =====================================================

// OBJECT POSITION SUBSCRIBER

// =====================================================

objectPositionSubscriber =

    rosNode->create_subscription<

        geometry_msgs::msg::PointStamped>(

            "/object/position",

            10,

            std::bind(

                &Task2Manager::ObjectPositionCallback,

                this,

                std::placeholders::_1));

// =====================================================
// AUTONOMOUS TARGET DETECTION SUBSCRIBER
// =====================================================

targetDetectionSubscriber =

    rosNode->create_subscription<

        geometry_msgs::msg::PointStamped>(

            "/stonefish/target/detection",

            10,

            std::bind(

                &Task2Manager::TargetDetectionCallback,

                this,

                std::placeholders::_1));



    /* =====================================================

     * CONTROLLER

     * ===================================================== */

    InitializeController();

    /* =====================================================

     * STARTUP MESSAGE

     * ===================================================== */

    std::cout

        << "\n"

        << "========================================\n"

        << "       UNDERWATER AUV SIMULATION\n"

        << "========================================\n"

        << "\n"

        << "Environment:\n"

        << "  Water surface : Z = 0.0 m\n"

        << "  AUV start     : Z = 1.5 m\n"

        << "  Tracked object: Z = 3.0 m\n"

        << "  Seabed        : Z = 5.0 m\n"

        << "\n"

        << "Controller:\n"

        << "  Left stick X  : Yaw\n"

        << "  Left stick Y  : Forward / Reverse\n"

        << "  Right stick Y : Ascend / Descend\n"

        << "  B             : Emergency stop\n"

        << "\n"

        << "Keyboard fallback:\n"

        << "  W             : Forward\n"

        << "  S             : Reverse\n"

        << "  A             : Turn left\n"

        << "  D             : Turn right\n"

        << "  R             : Ascend\n"

        << "  F             : Descend\n"

        << "  SPACE         : Stop\n"

        << "\n"

        << "========================================\n"

        << std::endl;

}

/* =========================================================

 * INITIALIZE XBOX CONTROLLER

 * ========================================================= */

bool Task2Manager::InitializeController()

{

    if (SDL_WasInit(

            SDL_INIT_GAMECONTROLLER) == 0)

    {

        if (SDL_InitSubSystem(

                SDL_INIT_GAMECONTROLLER |

                SDL_INIT_JOYSTICK) != 0)

        {

            std::cerr

                << "SDL controller initialization failed: "

                << SDL_GetError()

                << std::endl;

            return false;

        }

    }

    int controllerCount =

        SDL_NumJoysticks();

    std::cout

        << "SDL controllers detected: "

        << controllerCount

        << std::endl;

    for (int i = 0;

         i < controllerCount;

         ++i)

    {

        if (SDL_IsGameController(i))

        {

            controller =

                SDL_GameControllerOpen(i);

            if (controller != nullptr)

            {

                controllerConnected =

                    true;

                std::cout

                    << "Controller connected: "

                    << SDL_GameControllerName(

                           controller)

                    << std::endl;

                return true;

            }

        }

    }

    std::cout

        << "No SDL Game Controller found.\n"

        << "Keyboard control will be used."

        << std::endl;

    return false;

}

/* =========================================================

 * CONTROLLER UPDATE

 * ========================================================= */

void Task2Manager::UpdateController()

{

    // =====================================================
    // GLOBAL EMERGENCY STOP
    // =====================================================

    // Xbox B button always stops the AUV, including in
    // autonomous mode.
    if (controllerConnected && controller != nullptr)

    {

        SDL_GameControllerUpdate();

        if (SDL_GameControllerGetButton(
                controller,
                SDL_CONTROLLER_BUTTON_B))

        {

            SetThrusterCommands(0.0, 0.0, 0.0);

            return;

        }

    }

    // Keyboard SPACE is also a global emergency stop.
    SDL_PumpEvents();

    const Uint8* keys = SDL_GetKeyboardState(nullptr);

    if (keys[SDL_SCANCODE_SPACE])

    {

        SetThrusterCommands(0.0, 0.0, 0.0);

        return;

    }

    // =====================================================
    // AUTONOMOUS VISUAL SERVOING
    // =====================================================

    if (autonomousMode)

    {

        // No target -> stop.
        if (!targetDetected)

        {

            SetThrusterCommands(0.0, 0.0, 0.0);

            return;

        }

        constexpr double CAMERA_WIDTH = 640.0;
        constexpr double CAMERA_HEIGHT = 480.0;

        // Convert pixel error into [-1, +1].
        double normalizedX =

            targetErrorX / (CAMERA_WIDTH / 2.0);

        double normalizedY =

            targetErrorY / (CAMERA_HEIGHT / 2.0);

        normalizedX = std::clamp(

            normalizedX, -1.0, 1.0);

        normalizedY = std::clamp(

            normalizedY, -1.0, 1.0);

        // Conservative gains for the first autonomous test.
        constexpr double YAW_GAIN = 0.30;
        constexpr double VERTICAL_GAIN = 0.25;

        constexpr double MAX_YAW = 0.30;
        constexpr double MAX_VERTICAL = 0.20;

        constexpr double ALIGNMENT_THRESHOLD_X = 0.15;
        constexpr double ALIGNMENT_THRESHOLD_Y = 0.15;

        // Positive image X error means target is right.
        // With the existing differential-thrust convention,
        // positive yaw turns the AUV right.
        double yaw =

            YAW_GAIN * normalizedX;

        yaw = std::clamp(

            yaw, -MAX_YAW, MAX_YAW);

        // Positive image Y error means target is below
        // the camera center. Existing positive vertical
        // command produces descent.
        double vertical =

            VERTICAL_GAIN * normalizedY;

        vertical = std::clamp(

            vertical, -MAX_VERTICAL, MAX_VERTICAL);

        // Move forward only after the target is reasonably
        // centered. This prevents the AUV from charging at
        // a target while it is still far off-axis.
        double forward = 0.0;

        if (std::fabs(normalizedX) < ALIGNMENT_THRESHOLD_X &&
            std::fabs(normalizedY) < ALIGNMENT_THRESHOLD_Y)

        {

            forward = autonomousForwardSpeed;

        }

        double leftCommand =

            forward + yaw;

        double rightCommand =

            forward - yaw;

        leftCommand = std::clamp(

            leftCommand, -1.0, 1.0);

        rightCommand = std::clamp(

            rightCommand, -1.0, 1.0);

        SetThrusterCommands(

            leftCommand,

            rightCommand,

            vertical);

        return;

    }

    // =====================================================
    // EXISTING MANUAL CONTROLLER
    // =====================================================

    if (controllerConnected && controller != nullptr)

    {

        SDL_GameControllerUpdate();

        float leftX =

            SDL_GameControllerGetAxis(
                controller,
                SDL_CONTROLLER_AXIS_LEFTX) / 32767.0f;

        float leftY =

            SDL_GameControllerGetAxis(
                controller,
                SDL_CONTROLLER_AXIS_LEFTY) / 32767.0f;

        float rightY =

            SDL_GameControllerGetAxis(
                controller,
                SDL_CONTROLLER_AXIS_RIGHTY) / 32767.0f;

        const float DEADZONE = 0.10f;

        if (std::fabs(leftX) < DEADZONE)
            leftX = 0.0f;

        if (std::fabs(leftY) < DEADZONE)
            leftY = 0.0f;

        if (std::fabs(rightY) < DEADZONE)
            rightY = 0.0f;

        float forward = -leftY;
        float yaw = leftX;
        float vertical = rightY;

        float leftCommand =

            forward + yaw;

        float rightCommand =

            forward - yaw;

        leftCommand = std::clamp(

            leftCommand, -1.0f, 1.0f);

        rightCommand = std::clamp(

            rightCommand, -1.0f, 1.0f);

        vertical = std::clamp(

            vertical, -1.0f, 1.0f);

        SetThrusterCommands(

            leftCommand,

            rightCommand,

            vertical);

        return;

    }

    // Keyboard fallback.
    float forward = 0.0f;
    float yaw = 0.0f;
    float vertical = 0.0f;

    if (keys[SDL_SCANCODE_W])
        forward += 0.65f;

    if (keys[SDL_SCANCODE_S])
        forward -= 0.65f;

    if (keys[SDL_SCANCODE_A])
        yaw -= 0.40f;

    if (keys[SDL_SCANCODE_D])
        yaw += 0.40f;

    if (keys[SDL_SCANCODE_R])
        vertical -= 0.40f;

    if (keys[SDL_SCANCODE_F])
        vertical += 0.40f;

    float leftCommand = std::clamp(

        forward + yaw, -1.0f, 1.0f);

    float rightCommand = std::clamp(

        forward - yaw, -1.0f, 1.0f);

    SetThrusterCommands(

        leftCommand,

        rightCommand,

        std::clamp(vertical, -1.0f, 1.0f));
}


/* =========================================================

 * SET THRUSTER COMMANDS

 * ========================================================= */

void Task2Manager::SetThrusterCommands(

    sf::Scalar leftCommand,

    sf::Scalar rightCommand,

    sf::Scalar verticalCommand)

{

    if (leftThruster != nullptr)

    {

        leftThruster->setSetpoint(

            leftCommand);

    }

    if (rightThruster != nullptr)

    {

        rightThruster->setSetpoint(

            rightCommand);

    }

    /*

     * Both vertical thrusters receive

     * exactly the same command.

     *

     * This prevents unnecessary roll torque.

     */

    if (verticalLeftThruster != nullptr)

    {

        verticalLeftThruster->setSetpoint(

            verticalCommand);

    }

    if (verticalRightThruster != nullptr)

    {

        verticalRightThruster->setSetpoint(

            verticalCommand);

    }

}

/* =========================================================

 * PUBLISH ODOMETRY

 * ========================================================= */

void Task2Manager::PublishOdometry()

{

    if (!odometry ||

        !odomPublisher)

    {

        return;

    }

    nav_msgs::msg::Odometry msg;

    msg.header.stamp =

        rosNode->get_clock()->now();

    msg.header.frame_id =

        "world";

    msg.child_frame_id =

        "auv";

    /* =====================================================

     * POSITION

     * ===================================================== */

    msg.pose.pose.position.x =

        odometry->getLastValue(0);

    msg.pose.pose.position.y =

        odometry->getLastValue(1);

    msg.pose.pose.position.z =

        odometry->getLastValue(2);

    /* =====================================================

     * ORIENTATION

     * ===================================================== */

    msg.pose.pose.orientation.x =

        odometry->getLastValue(6);

    msg.pose.pose.orientation.y =

        odometry->getLastValue(7);

    msg.pose.pose.orientation.z =

        odometry->getLastValue(8);

    msg.pose.pose.orientation.w =

        odometry->getLastValue(9);

    /* =====================================================

     * LINEAR VELOCITY

     * ===================================================== */

    msg.twist.twist.linear.x =

        odometry->getLastValue(3);

    msg.twist.twist.linear.y =

        odometry->getLastValue(4);

    msg.twist.twist.linear.z =

        odometry->getLastValue(5);

    /* =====================================================

     * ANGULAR VELOCITY

     * ===================================================== */

    msg.twist.twist.angular.x =

        odometry->getLastValue(10);

    msg.twist.twist.angular.y =

        odometry->getLastValue(11);

    msg.twist.twist.angular.z =

        odometry->getLastValue(12);

    odomPublisher->publish(

        msg);

}

/* =========================================================

 * PUBLISH CAMERA IMAGE

 * ========================================================= */

void Task2Manager::PublishCameraImage(

    sf::ColorCamera* camera)

{

    if (!camera ||

        !imagePublisher)

    {

        return;

    }

    void* dataPointer =

        camera->getImageDataPointer();

    if (dataPointer == nullptr)

    {

        return;

    }

    const unsigned int width =

        640;

    const unsigned int height =

        480;

    const unsigned int channels =

        3;

    sensor_msgs::msg::Image msg;

    msg.header.stamp =

        rosNode->get_clock()->now();

    msg.header.frame_id =

        "front_camera";

    msg.height =

        height;

    msg.width =

        width;

    msg.encoding =

        "rgb8";

    msg.is_bigendian =

        false;

    msg.step =

        width * channels;

    const size_t dataSize =

        static_cast<size_t>(

            width)

        * height

        * channels;

    msg.data.resize(

        dataSize);

    std::memcpy(

        msg.data.data(),

        dataPointer,

        dataSize);

    imagePublisher->publish(

        msg);

}

/* =========================================================

 * PRINT ODOMETRY

 * ========================================================= */

void Task2Manager::PrintOdometry()

{

    if (!odometry)

    {

        return;

    }

    /* =====================================================

     * POSITION

     * ===================================================== */

    const double x =

        odometry->getLastValue(0);

    const double y =

        odometry->getLastValue(1);

    const double z =

        odometry->getLastValue(2);

    /* =====================================================

     * VELOCITY

     * ===================================================== */

    const double vx =

        odometry->getLastValue(3);

    const double vy =

        odometry->getLastValue(4);

    const double vz =

        odometry->getLastValue(5);

    /* =====================================================

     * QUATERNION

     * ===================================================== */

    const double qx =

        odometry->getLastValue(6);

    const double qy =

        odometry->getLastValue(7);

    const double qz =

        odometry->getLastValue(8);

    const double qw =

        odometry->getLastValue(9);

    /* =====================================================

     * YAW

     * ===================================================== */

    const double yaw =

        std::atan2(

            2.0 *

            (qw * qz + qx * qy),

            1.0 -

            2.0 *

            (qy * qy + qz * qz));

    std::cout

        << "[AUV ODOM] "

        << "X=" << x

        << " m  "

        << "Y=" << y

        << " m  "

        << "Depth(Z)=" << z

        << " m  "

        << "Vx=" << vx

        << "  "

        << "Vy=" << vy

        << "  "

        << "Vz=" << vz

        << "  "

        << "Yaw="

        << yaw * 180.0 / M_PI

        << " deg"

        << std::endl;

}

/* =========================================================

 * SIMULATION STEP COMPLETED

 * ========================================================= */

void Task2Manager::SimulationStepCompleted(

    sf::Scalar timeStep)

{

    /* =====================================================

     * ROS 2

     * Receive the newest detector message before
     * calculating the next thruster command.

     * ===================================================== */

    if (rclcpp::ok())

    {

        rclcpp::spin_some(

            rosNode);

    }

    /* =====================================================

     * UPDATE CONTROLLER

     * ===================================================== */

    UpdateController();

    // Synchronize the simulated target with the real camera, but keep
    // it around the configured target anchor position instead of
    // mapping image coordinates directly to the AUV origin.
    UpdateTrackedObject();

    /* =====================================================

     * ODOMETRY TIMER

     * ===================================================== */

    odometryTimer += timeStep;

    if (odometryTimer >= 0.2)

    {

        odometryTimer = 0.0;

        PrintOdometry();

        PublishOdometry();

    }

}

// =========================================================

// OBJECT POSITION CALLBACK

// =========================================================

void Task2Manager::ObjectPositionCallback(

    const geometry_msgs::msg::PointStamped::SharedPtr msg)

{

    if (!msg)

    {

        return;

    }

    // Store image coordinates

    objectPixelX = msg->point.x;

    objectPixelY = msg->point.y;

    objectDetected = true;

    std::cout

        << "[REAL OBJECT] "

        << "Pixel X=" << objectPixelX

        << "  "

        << "Pixel Y=" << objectPixelY

        << std::endl;

}

// =========================================================
// AUTONOMOUS TARGET DETECTION CALLBACK
// =========================================================

void Task2Manager::TargetDetectionCallback(
    const geometry_msgs::msg::PointStamped::SharedPtr msg)
{
    if (!msg)
        return;

    targetErrorX = msg->point.x;
    targetErrorY = msg->point.y;
    targetConfidence = msg->point.z;

    targetDetected = (targetConfidence >= 0.5);

    std::cout
        << "[YOLO TARGET] "
        << "Error X=" << targetErrorX
        << " Error Y=" << targetErrorY
        << " Confidence=" << targetConfidence
        << " Detected=" << targetDetected
        << std::endl;

}


void Task2Manager::UpdateTrackedObject()

{

    if (trackedObject == nullptr || !objectDetected)

    {

        return;

    }

    /* =====================================================
     * REAL-OBJECT -> STONEFISH TARGET MAPPING
     *
     * The AUV starts at approximately:
     *     (0.0, 0.0, 1.5)
     *
     * The target anchor is intentionally kept in front of the AUV:
     *     (3.0, 0.0, 1.5)
     *
     * The real camera still controls the target movement, but only
     * as an offset around this anchor. Therefore, when the real
     * object is at the image centre, the simulated target remains
     * 3 m in front of the AUV instead of jumping to the AUV.
     * ===================================================== */

    constexpr double imageWidth = 640.0;
    constexpr double imageHeight = 480.0;

    // Target anchor / nominal separation from the AUV.
    constexpr double targetAnchorX = 4.0;
    constexpr double targetAnchorY = 0.0;
    constexpr double targetAnchorZ = 1.5;

    // Maximum movement of the simulated target around the anchor.
    // Keeping these smaller than the anchor distance prevents the
    // target from crossing through the AUV during normal tracking.
    constexpr double targetMotionX = 1.5;
    constexpr double targetMotionY = 1.0;

    const double pixelX =
        std::clamp(objectPixelX, 0.0, imageWidth);

    const double pixelY =
        std::clamp(objectPixelY, 0.0, imageHeight);

    // Convert the image centre to zero offset.
    // x: left = negative, right = positive.
    // y: above = positive, below = negative in Stonefish Y mapping.
    const double normalizedX =
        (pixelX - imageWidth * 0.5) / (imageWidth * 0.5);

    const double normalizedY =
        (imageHeight * 0.5 - pixelY) / (imageHeight * 0.5);

    const double simulationX =
        targetAnchorX +
        normalizedX * targetMotionX;

    const double simulationY =
        targetAnchorY +
        normalizedY * targetMotionY;

    const double simulationZ = targetAnchorZ;

    trackedObject->setTransform(
        sf::Transform(
            sf::Quaternion(0.0, M_PI, 0.0),
            sf::Vector3(
                simulationX,
                simulationY,
                simulationZ)));

    std::cout
        << "[STONEFISH OBJECT] "
        << "Anchor=(" << targetAnchorX
        << ", " << targetAnchorY
        << ", " << targetAnchorZ << ") "
        << "Target=(" << simulationX
        << ", " << simulationY
        << ", " << simulationZ << ")"
        << std::endl;
}
