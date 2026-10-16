#include <rclcpp/rclcpp.hpp>

#include <cctype>
#include <cstdlib>
#include <string>
#include <vector>

#include <core/GraphicalSimulationApp.h>

#include "Task2Manager.h"


int main(int argc, char* argv[])
{
    // =====================================================
    // Initialize ROS 2 BEFORE creating the Stonefish manager
    // =====================================================

    rclcpp::init(argc, argv);


    // =====================================================
    // Test case: "./Task2Baseline 3" or "./Task2Baseline --case 3"
    // (0 = use $TASK2_CASE, else the default case)
    // =====================================================

    int scenarioCase = 0;
    const std::vector<std::string> args = rclcpp::remove_ros_arguments(argc, argv);
    for (size_t i = 1; i < args.size(); ++i)
    {
        if (args[i] == "--case" && i + 1 < args.size())
            scenarioCase = std::atoi(args[++i].c_str());
        else if (!args[i].empty() && std::isdigit(static_cast<unsigned char>(args[i][0])))
            scenarioCase = std::atoi(args[i].c_str());
    }


    // =====================================================
    // Stonefish rendering settings
    // =====================================================

    sf::RenderSettings settings;

    settings.windowW = 1200;
    settings.windowH = 900;

    settings.aa =
        sf::RenderQuality::HIGH;

    settings.shadows =
        sf::RenderQuality::HIGH;

    settings.ao =
        sf::RenderQuality::HIGH;

    settings.atmosphere =
        sf::RenderQuality::HIGH;

    settings.ocean =
        sf::RenderQuality::HIGH;

    settings.verticalSync = true;


    // =====================================================
    // Stonefish helper settings
    // =====================================================

    sf::HelperSettings helpers;

    helpers.showFluidDynamics = false;
    helpers.showCoordSys = false;
    helpers.showBulletDebugInfo = false;
    helpers.showSensors = false;
    helpers.showActuators = false;
    helpers.showForces = false;


    // =====================================================
    // Create simulation manager
    // =====================================================

    Task2Manager* simulationManager =
        new Task2Manager(500.0, scenarioCase);


    // =====================================================
    // Create Stonefish application
    // =====================================================

    sf::GraphicalSimulationApp app(
        "Task2Baseline",
        std::string(DATA_DIR_PATH),
        settings,
        helpers,
        simulationManager);


    // =====================================================
    // Run simulation
    // =====================================================

    app.Run();


    // =====================================================
    // Shutdown ROS 2 cleanly
    // =====================================================

    if (rclcpp::ok())
    {
        rclcpp::shutdown();
    }


    return 0;
}