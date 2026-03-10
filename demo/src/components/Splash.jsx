import React, { useEffect } from 'react';
import { useApplication } from '@pixi/react';
import { Assets, Texture, Container, Sprite, Graphics, Ticker } from 'pixi.js';

export function Splash({ width, height, numRings = 4 }) {
    const { app } = useApplication();
    // Create containers for edges and houses (peeps)
    const edgesContainer = new Container();
    const peepsContainer = new Container();
    app.stage.addChild(edgesContainer);
    app.stage.addChild(peepsContainer);

    let peeps = [];
    let edges = [];

    const ticker = Ticker.shared;

    ticker.add((delta) => {
                    // Update each peep's movement.
                    peeps.forEach((peep) => {
                        peep.update(delta);
                    });
                    // Update edge positions based on current positions of connected houses.
                    edges.forEach((edge) => {
                        const { sprite, from, to } = edge;
                        const dx = to.sprite.x - from.sprite.x;
                        const dy = to.sprite.y - from.sprite.y;
                        const distance = Math.sqrt(dx * dx + dy * dy);
                        sprite.x = from.sprite.x;
                        sprite.y = from.sprite.y;
                        sprite.rotation = Math.atan2(dy, dx);
                        sprite.width = distance;
                    });
                });
    useEffect(() => {
        if (!app) return; // Wait for the app to be available

        // Function to add a house (peep) with random rotation and subtle movement parameters.
        const addPeep = (x, y) => {
            const texture = Texture.from('/assets/house_color.png');
            const sprite = new Sprite(texture);
            sprite.anchor.set(0.5);
            sprite.scale.set(0.5);
            sprite.position.set(x, y);

            // Capture initial values for movement/rotation.
            const initX = x;
            const initY = y;
            const initRotation = (Math.random() - 0.5) * (Math.PI - 0.4);
            const moveRadius = 5 + Math.random() * 20;
            const swing = 0.05 + Math.random() * 0.45;
            let angle = Math.random() * (2 * Math.PI);
            const speed = (0.05 + Math.random() * 0.95) / 60;

            // Optionally flip horizontally 50% of the time.
            if (Math.random() < 0.5) {
                sprite.scale.x *= -1;
            }

            // Instead of using 'this', use closure variables in the update function.
            const peep = {
                sprite,
                update(delta) {
                    angle += speed * delta.deltaTime; // delta is now a number (make sure you're using the correct delta property as needed)
                    const newX = initX + Math.cos(angle) * moveRadius;
                    const newY = initY + Math.sin(angle) * moveRadius;
                    const newRotation = initRotation + Math.cos(angle) * swing;
                    sprite.x = newX;
                    sprite.y = newY;
                    sprite.rotation = newRotation;
                }
            };

            peeps.push(peep);
            peepsContainer.addChild(sprite);
            return peep;
        };

        // Function to add an edge between two houses.
        const addEdge = (fromPeep, toPeep) => {
            const edgeTexture = Texture.from('/assets/connection_solo.png');
            const sprite = new Sprite(edgeTexture);
            sprite.anchor.set(0, 0.5);
            sprite.height = 10;
            edges.push({ sprite, from: fromPeep, to: toPeep });
            edgesContainer.addChild(sprite);
        };

        const initialize = () => {
            // Clear existing peeps and edges
            peeps.forEach(peep => {
                peepsContainer.removeChild(peep.sprite);
            });
            edges.forEach(edge => {
                edgesContainer.removeChild(edge.sprite);
            });
            peeps = [];
            edges = [];

            // Create a ring of houses around the center.
            const createRing = (radius, count) => {
                const centerX = width / 2;
                const centerY = height / 2;
                const ringPeeps = [];
                for (let i = 0; i < count; i++) {
                    const angle = (2 * Math.PI * i) / count;
                    const x = centerX + radius * Math.cos(angle);
                    const y = centerY + radius * Math.sin(angle);
                    const peep = addPeep(x, y);
                    ringPeeps.push(peep);
                }
                // Connect each peep to its adjacent peeps in the ring
                for (let i = 0; i < ringPeeps.length; i++) {
                    const nextIndex = (i + 1) % ringPeeps.length;
                    addEdge(ringPeeps[i], ringPeeps[nextIndex]);
                }
            };

            // Create multiple rings starting from the outside.
            const ringRadii = [800, 700, 600, 500];
            for (let i = 0; i < numRings; i++) {
                createRing(ringRadii[i], 40 - i * 5);
            }

            // Connect all houses within a given radius.
            const connectAllWithinRadius = (radius) => {
                const r2 = radius * radius;
                for (let i = 0; i < peeps.length; i++) {
                    for (let j = i + 1; j < peeps.length; j++) {
                        const dx = peeps[i].sprite.x - peeps[j].sprite.x;
                        const dy = peeps[i].sprite.y - peeps[j].sprite.y;
                        if (dx * dx + dy * dy < r2) {
                            addEdge(peeps[i], peeps[j]);
                        }
                    }
                }
            };

            // Connect houses with edges; adjust the threshold (e.g., 250) as needed.
            connectAllWithinRadius(200);
        };

        Assets.load([
            '/assets/house_color.png',
            '/assets/connection_solo.png',
        ]).then(initialize);

        // Cleanup containers on unmount.
        return () => {
            peepsContainer.destroy({ children: true });
            edgesContainer.destroy({ children: true });
            // Clear the peeps and edges arrays.
            peeps = [];
            edges = [];
        };
    }, [app, width, height, numRings]);

    return null;
}
