#include <SFML/Graphics.hpp>

#include "global.h"
#include "cudamain.h"
#include "vec3d.h"
#include "camera.h"

#include "mandelbulb.h"

#ifndef NDEBUG

#include <stdio.h>
#define log(fmt, ...) fprintf(stderr, fmt "\n", __VA_ARGS__)

#else

#define log(fmt, ...)

#endif

sf::Uint8* pixels;

int main() {
    sf::RenderWindow window(sf::VideoMode(W, H), "Mandelbulb");

    init();

    Camera camera(0, 0, -4, 1, 0, 0);
    Mandelbulb figure;

    pixels = (sf::Uint8*) malloc(W * H * 4 * sizeof(sf::Uint8));
    sf::Texture texture;
    texture.create(W, H);
    sf::Sprite sprite(texture);

    bool aPressed = false;
    bool dPressed = false;
    bool wPressed = false;
    bool sPressed = false;
    bool spacePressed = false;
    bool lShiftPressed = false;
    bool leftPressed = false;
    bool rightPressed = false;
    bool upPressed = false;
    bool downPressed = false;

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            switch (event.type) {
            case sf::Event::Closed:
                window.close();
                break;
            case sf::Event::KeyPressed:
                switch (event.key.code) {
                case sf::Keyboard::Escape:
                    window.close();
                    break;
                case sf::Keyboard::A:
                    aPressed = true;
                    break;
                case sf::Keyboard::D:
                    dPressed = true;
                    break;
                case sf::Keyboard::W:
                    wPressed = true;
                    break;
                case sf::Keyboard::S:
                    sPressed = true;
                    break;
                case sf::Keyboard::Space:
                    spacePressed = true;
                    break;
                case sf::Keyboard::LShift:
                    lShiftPressed = true;
                    break;
                case sf::Keyboard::Left:
                    leftPressed = true;
                    break;
                case sf::Keyboard::Right:
                    rightPressed = true;
                    break;
                case sf::Keyboard::Up:
                    upPressed = true;
                    break;
                case sf::Keyboard::Down:
                    downPressed = true;
                    break;
                case sf::Keyboard::R:
                    camera = Camera(0, 0, -4, 1, 0, 0);
                    break;
                case sf::Keyboard::N:
                    texture.copyToImage().saveToFile("screenshot.jpg");
                    break;
                }
                break;
            case sf::Event::KeyReleased:
                switch (event.key.code) {
                case sf::Keyboard::Escape:
                    window.close();
                    break;
                case sf::Keyboard::A:
                    aPressed = false;
                    break;
                case sf::Keyboard::D:
                    dPressed = false;
                    break;
                case sf::Keyboard::W:
                    wPressed = false;
                    break;
                case sf::Keyboard::S:
                    sPressed = false;
                    break;
                case sf::Keyboard::Space:
                    spacePressed = false;
                    break;
                case sf::Keyboard::LShift:
                    lShiftPressed = false;
                    break;
                case sf::Keyboard::Left:
                    leftPressed = false;
                    break;
                case sf::Keyboard::Right:
                    rightPressed = false;
                    break;
                case sf::Keyboard::Up:
                    upPressed = false;
                    break;
                case sf::Keyboard::Down:
                    downPressed = false;
                    break;
                }
                break;
            }
        }

        if (aPressed) {
            camera.moveSide(-shift);
        }
        if (dPressed) {
            camera.moveSide(shift);
        }
        if (wPressed) {
            camera.moveForward(shift);
        }
        if (sPressed) {
            camera.moveForward(-shift);
        }
        if (spacePressed) {
            camera.pos.y -= shift;
        }
        if (lShiftPressed) {
            camera.pos.y += shift;
        }
        if (leftPressed) {
            camera.rotateY(angleShift);
        }
        if (rightPressed) {
            camera.rotateY(-angleShift);
        }
        if (upPressed) {
            camera.rotateX(-angleShift);
        }
        if (downPressed) {
            camera.rotateX(angleShift);
        }

        log("x: %f, y: %f, z: %f, angleX: %f, angleY: %f", camera.pos.x, camera.pos.y, camera.pos.z, camera.angleX, camera.angleY);

        update(pixels, figure, camera);
        texture.update(pixels);

        window.clear();
        window.draw(sprite);
        window.display();
    }

    destruct();

    return 0;
}