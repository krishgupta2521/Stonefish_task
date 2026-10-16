#include <SDL2/SDL.h>
#include <iostream>

int main()
{
    if (SDL_Init(SDL_INIT_GAMECONTROLLER | SDL_INIT_JOYSTICK) != 0)
    {
        std::cerr << "SDL_Init failed: "
                  << SDL_GetError() << std::endl;
        return 1;
    }

    int count = SDL_NumJoysticks();

    std::cout << "Controllers detected: "
              << count << std::endl;

    for (int i = 0; i < count; ++i)
    {
        std::cout << i << ": "
                  << SDL_GameControllerNameForIndex(i)
                  << std::endl;
    }

    if (count == 0)
    {
        std::cout << "No controller detected." << std::endl;
        SDL_Quit();
        return 0;
    }

    SDL_GameController* controller =
        SDL_GameControllerOpen(0);

    if (!controller)
    {
        std::cerr << "Could not open controller: "
                  << SDL_GetError() << std::endl;

        SDL_Quit();
        return 1;
    }

    std::cout << "\nController opened successfully!\n";
    std::cout << "Move the sticks / press buttons.\n";
    std::cout << "Press Ctrl+C to stop.\n\n";

    while (true)
    {
        SDL_Event event;

        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_CONTROLLERAXISMOTION)
            {
                std::cout
                    << "Axis "
                    << static_cast<int>(event.caxis.axis)
                    << " = "
                    << event.caxis.value
                    << std::endl;
            }

            if (event.type == SDL_CONTROLLERBUTTONDOWN)
            {
                std::cout
                    << "Button "
                    << static_cast<int>(event.cbutton.button)
                    << " DOWN"
                    << std::endl;
            }
        }

        SDL_Delay(10);
    }

    SDL_GameControllerClose(controller);
    SDL_Quit();

    return 0;
}
