import React, { useEffect, useRef, useState, forwardRef, useImperativeHandle } from 'react';
import { useApplication } from '@pixi/react';
import { Assets, Container, Sprite, Texture } from 'pixi.js';
import energySharingPuzzle from '../data/energySharingPuzzle.json';

export const EnergyNetwork = forwardRef(({ setGraphData, width, height }, ref) => {
    const { app } = useApplication();
    const edgesContainer = useRef(new Container());
    const nodesContainer = useRef(new Container());

    const nodes = [];
    const nodesRef = useRef(nodes);
    const edges = [];
    const edgesRef = useRef(edges);

    const [activeNodes, setActiveNodes] = useState([]);
    const activeNodesRef = useRef(activeNodes);

    const solarProfile = energySharingPuzzle.solar_profile;
    const agents = energySharingPuzzle.agents;

    useImperativeHandle(ref, () => ({
        clearAllConnections() {
            edgesContainer.current.removeChildren();
            activeNodesRef.current = [];
            edgesRef.current = [];
            // updateGraph();
        }
    }));

    // const updateGraph = () => {
    //     const totalConsumption = Array(24).fill(0);
    //     const totalProduction = Array(24).fill(0);
    //
    //     const connectedNodes = new Set();
    //     edgesRef.current.forEach(edge => {
    //         connectedNodes.add(edge.from.id);
    //         connectedNodes.add(edge.to.id);
    //     });
    //
    //     nodesRef.current.forEach(node => {
    //         const id = node.id;
    //         if (activeNodes.includes(agent => agent.id === id)) {
    //             const agent = agents.find(a => a.id === id);
    //             agent.consumption.forEach((val, h) => totalConsumption[h] += val);
    //             solarProfile.forEach((val, h) => totalProduction[h] += val * 0.5);
    //         }
    //     });
    //
    //     const points = Array.from({ length: 24 }, (_, h) => ({ x: h, y: totalConsumption[h] }));
    //     const solarPoints = Array.from({ length: 24 }, (_, h) => ({ x: h, y: totalProduction[h] }));
    //
    //     setGraphData([
    //         { label: 'Total Consumption', points, color: '#3498db' },
    //         { label: 'Total Solar Production', points: solarPoints, color: '#2ecc71' }
    //     ]);
    // };

    useEffect(() => {
        if (!app) return;

        // Add edgesContainer first to ensure edges are drawn underneath nodes
        app.stage.addChild(edgesContainer.current);
        app.stage.addChild(nodesContainer.current);

        nodesRef.current = [];
        edgesRef.current = [];

        const addNode = (x, y) => {
            const texture = Texture.from('/assets/house_color.png');
            const sprite = new Sprite(texture);
            sprite.anchor.set(0.5);
            sprite.scale.set(0.5);
            sprite.position.set(x, y);
            sprite.originalX = x; // Store original position
            sprite.interactive = true;
            sprite.buttonMode = true;
            nodesContainer.current.addChild(sprite);
            nodesRef.current.push(sprite);
            return sprite;
        }

        // Function to add an edge between two houses.
        const addEdge = (fromPeep, toPeep) => {
            const edgeTexture = Texture.from('/assets/connection_solo.png');
            const sprite = new Sprite(edgeTexture);
            sprite.anchor.set(0, 0.5);
            sprite.height = 10;
            edgesRef.current.push({ sprite, from: fromPeep, to: toPeep });
            edgesContainer.addChild(sprite);
        };

        activeNodesRef.current = activeNodes;
        console.log('Updated activeNodes:', activeNodes);

        Assets.load(['/assets/house_color.png', '/assets/connection_solo.png']).then(() => {
            const centerX = width / 2;
            const centerY = height / 2;
            const radius = 150;
            const count = agents.length;

            const spriteScale = Math.min(width, height) / 700;

            agents.forEach((agent, i) => {
                const angle = (2 * Math.PI * i) / count;
                const x = centerX + radius * Math.cos(angle);
                const y = centerY + radius * Math.sin(angle);
                const sprite = addNode(x, y);
                sprite.id = agent.id;
                sprite.tint = 0x3498db; // Set initial color to blue
                sprite.scale.set(spriteScale);
                sprite.on('pointerover', () => {
                //
                });

                sprite.on('pointerdown', () => {
                    console.log("Active Nodes: ", activeNodes);
                    // There are two cases:
                    // 1. If the node is already selected, we need to unselect it.
                    //    This will also remove the connection from the graph (i.e. filter out the node & delete the edges).
                    if (activeNodesRef.current.includes(agent)) {
                        console.log("Deactivating agent:", agent.id);
                        // Remove the node from the active nodes
                        activeNodesRef.current.splice(activeNodesRef.current.indexOf(agent), 1);
                        // We also need to remove the edges from the graph. Each edge is a triplet (from, to, sprite)
                        edgesRef.current.forEach(edge => {
                                if (edge.from.id === agent.id || edge.to.id === agent.id) {
                                    edgesRef.current = edgesRef.current.filter(e => e !== edge);
                                    edgesContainer.current.removeChild(edge.line);
                                }
                            }
                        )
                    }

                    // 2. If the node is not selected, we need to select it.
                    else {
                        console.log("Activating agent:", agent.id);
                        // Add the node to the active nodes
                        activeNodesRef.current.push(agent)
                        // Add edges from this node to all other active nodes
                        const activeSprites = nodesRef.current.filter(sprite => activeNodes.includes(sprite.id));
                        activeSprites.forEach(sprite => {
                            if (sprite.id !== agent.id) {
                                const edgeTexture = Texture.from('/assets/connection_solo.png');
                                const line = new Sprite(edgeTexture);
                                line.tint = 0x000000;
                                line.height = 2;
                                line.anchor.set(0, 0.5);
                                line.zIndex = 0;
                                const dx = sprite.x - x;
                                const dy = sprite.y - y;
                                line.x = x;
                                line.y = y;
                                line.width = Math.sqrt(dx * dx + dy * dy);
                                line.rotation = Math.atan2(dy, dx);
                                edgesContainer.current.addChild(line);
                                edgesRef.current.push({ from: sprite, to: agent, line });
                            }
                        });
                    }
                });

                sprite.id = agent.id;
            });
        });

        // return () => {
        //     nodesRef.current = [];
        //     edgesRef.current = [];
        //     // nodesContainer.current.removeChildren();
        //     // edgesContainer.current.removeChildren();
        //     // app.stage.removeChild(nodesContainer.current);
        //     // app.stage.removeChild(edgesContainer.current);
        // };
    }, [app, activeNodes]);

    return null;
});
