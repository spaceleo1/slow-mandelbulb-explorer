#include <SFML/Graphics.hpp>

#include <stdio.h>

#include "global.h"
#include "cudamain.h"
#include "vec3d.h"
#include "camera.h"

#include "mandelbulb.h"

sf::Uint8* pixels;

int main() {
    sf::RenderWindow window(sf::VideoMode(W, H), "Mandelbulb");

    init();

    Camera camera(4, 0, 0, 1, 0, acos(-1) / 2.0f);
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
                case sf::Keyboard::R:
                    camera = Camera(0, 0, -4, 1, 0, 0);
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
                }
                break;
            }
        }

        if (aPressed) {
            float shiftX = cosf(camera.angleY) * shift;
            float shiftZ = sinf(camera.angleY) * shift;
            camera.pos.x -= shiftX;
            camera.pos.z -= shiftZ;
        }
        if (dPressed) {
            float shiftX = cosf(camera.angleY) * shift;
            float shiftZ = sinf(camera.angleY) * shift;
            camera.pos.x += shiftX;
            camera.pos.z += shiftZ;
        }
        if (wPressed) {
            float shiftX = -sinf(camera.angleY) * shift;
            float shiftZ = cosf(camera.angleY) * shift;
            camera.pos.x += shiftX;
            camera.pos.z += shiftZ;
        }
        if (sPressed) {
            float shiftX = -sinf(camera.angleY) * shift;
            float shiftZ = cosf(camera.angleY) * shift;
            camera.pos.x -= shiftX;
            camera.pos.z -= shiftZ;
        }
        if (spacePressed) {
            camera.pos.y -= shift;
        }
        if (lShiftPressed) {
            camera.pos.y += shift;
        }
        if (leftPressed) {
            camera.angleY -= angleShift;
        }
        if (rightPressed) {
            camera.angleY += angleShift;
        }

        printf("x: %f y: %f z: %f angleX: %f angleY: %f\n", camera.pos.x, camera.pos.y, camera.pos.z, camera.angleX, camera.angleY);

        update(pixels, figure, camera);
        texture.update(pixels);

        window.clear();
        window.draw(sprite);
        window.display();
    }

    destruct();

    return 0;
}